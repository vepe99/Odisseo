"""A galaxy-disc rollout on a device mesh, with a movie, at 10^7-10^8 particles.

WHY THIS FILE EXISTS
--------------------
`jaccpot.distributed.fmm` gives a *force* on a mesh and nothing else: there is no
leapfrog, no Verlet, no scan over steps. ODISSEO's four lanes are all single-device
and `odisseo.render_callback` streams out of the single-GPU fused `strict_run_v2`
scan. So a multi-GPU force at 10^7 particles has, today, nowhere to go. This is the
missing loop, plus the two-line renderer that goes with it.

WHAT IT DOES
------------
Kick-drift-kick leapfrog, ONE force evaluation per step, over a compiled-once
`make_force_evaluator`. Positions, velocities and accelerations stay sharded across
the mesh for the whole rollout; nothing but a `(res, res)` density grid and a
handful of scalars ever crosses to the host.

THE REORDERING TRAP
-------------------
`make_force_evaluator`'s output rows are in the per-device Morton order the tree
build produced, and that is NOT the input row order -- **even with no padding at
all**. Measured directly: at cap == count == 1024 on two devices, zero padding
rows, the returned `gid` still disagrees with `gid_flat` in essentially every
position. `scatter_to_input_order`'s docstring says the two "agree whenever no
device is padded", which is true of them as *maps* and false of the row order, and
that is an easy sentence to read the wrong way. Mapping a force back on that
assumption reads every particle's acceleration off a Morton neighbour: smooth,
plausible, and wrong by tens of percent (`docs/distributed_padding_force_defect.md`).

A rollout cannot afford a host-side scatter every step, so it realigns ON DEVICE.
The partition is frozen, so each device owns a fixed set of global ids and the tree
only permutes *within* a device -- which makes the realignment a per-device sort
with no communication. `_align_forces` does it inside a `shard_map`: one argsort of
the returned gids, then two gathers. `rank_in` is precomputed once on the host
because the input order never changes.

The alignment is CHECKED against `scatter_to_input_order` on the first step, which
is the independent implementation the repo already trusts. Never delete that check.

EXTERNAL POTENTIAL
------------------
The AGAMA SCM disc ICs (`tools/agama_generate_scm_disk_ic.py`) are in equilibrium in
*disc self-gravity + an analytic NFW halo*, and the halo is not sampled -- the IC
file carries `halo_mass_code` / `halo_rs_code` for exactly this reason. Evolving the
disc under self-gravity alone would not be the same system, so the NFW term is added
per particle. It is elementwise, so it shards for free.

    Phi(r) = -G M ln(1 + r/rs) / r          (AGAMA's NFW convention)
    a(r)   = -G M [ ln(1+x)/r^3 - 1/(r^2 (rs+r)) ] * pos

DIAGNOSTICS
-----------
Total momentum, total angular momentum and centre-of-mass drift, all exactly
conserved quantities for this system and all cheap. Total ENERGY is deliberately
absent: the distributed evaluator returns accelerations only, so a potential would
have to be estimated by a subsampled direct sum, and an estimator is not a
conservation check. Momentum and Lz are; they are what is reported.

USAGE
-----
    export CUDA_VISIBLE_DEVICES=$(autocvd -n 5 -l -q -o)
    python -u mesh_galaxy_run.py --ic disk_20m.npz --ndev 5 --steps 1000 \
        --dt 1e-3 --leaf 512 --theta 0.7 --order 6 --movie out.gif
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from functools import partial

import numpy as np


# --------------------------------------------------------------------------- #
# external potential
# --------------------------------------------------------------------------- #


def nfw_acceleration(pos, mass_code, rs_code, g):
    """Analytic NFW acceleration, AGAMA's ``mass``/``scaleRadius`` convention.

    Parameters
    ----------
    pos : Any
        ``[n, 3]`` positions.
    mass_code : float
        Halo ``mass`` parameter in code units.
    rs_code : float
        Halo scale radius in code units.
    g : float
        Gravitational constant.

    Returns
    -------
    Any
        ``[n, 3]`` accelerations.
    """
    import jax.numpy as jnp

    r2 = jnp.sum(pos * pos, axis=1)
    r = jnp.sqrt(jnp.maximum(r2, 1e-30))
    x = r / rs_code
    # ln(1+x)/r^3 - 1/(r^2 (rs+r)), series-expanded near the origin where both
    # terms diverge individually and their difference does not.
    small = x < 1e-3
    big = jnp.log1p(x) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
        r2 * (rs_code + r), 1e-30
    )
    # lim_{x->0} of the bracket is 1/(2 rs^3) * (1 - 4x/3 + ...)
    ser = (1.0 / (2.0 * rs_code**3)) * (1.0 - (4.0 / 3.0) * x)
    coeff = jnp.where(small, ser, big)
    return -(g * mass_code) * coeff[:, None] * pos


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def make_density_projector(mesh, res, axes, lo, hi):
    """Build a sharded 2D density projector returning a replicated ``(res, res)``.

    Every device histograms only the rows it owns and the partials are summed with
    one ``psum``, so the only thing that leaves the mesh is the grid itself -- the
    same design as ``odisseo.render_callback``, moved onto a mesh.

    Parameters
    ----------
    mesh : Any
        The device mesh, axis ``"gpus"``.
    res : int
        Grid resolution.
    axes : tuple
        The two position components to project onto.
    lo : Any
        Lower bound of the projection window, per axis.
    hi : Any
        Upper bound of the projection window, per axis.

    Returns
    -------
    Callable
        ``(pos, mass) -> (res, res)`` float32 grid, replicated.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    try:
        from jax.experimental.shard_map import shard_map
    except ImportError:  # pragma: no cover - newer jax
        from jax.experimental.shard_map import shard_map  # type: ignore

    a0, a1 = int(axes[0]), int(axes[1])
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)

    def local(pos, mass):
        u = (pos[:, a0] - lo[0]) / (hi[0] - lo[0])
        v = (pos[:, a1] - lo[1]) / (hi[1] - lo[1])
        iu = jnp.clip((u * res).astype(jnp.int32), 0, res - 1)
        iv = jnp.clip((v * res).astype(jnp.int32), 0, res - 1)
        inside = (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0) & (mass > 0)
        w = jnp.where(inside, mass, 0.0).astype(jnp.float32)
        grid = jnp.zeros((res, res), jnp.float32).at[iu, iv].add(w)
        return jax.lax.psum(grid, "gpus")

    fn = shard_map(
        local,
        mesh=mesh,
        in_specs=(P("gpus", None), P("gpus")),
        out_specs=P(),
        check_rep=False,
    )
    return jax.jit(fn)


# --------------------------------------------------------------------------- #
# the rollout
# --------------------------------------------------------------------------- #


def make_aligner(mesh):
    """Build a device-local realignment of the evaluator's output to input order.

    The frozen partition means device ``d`` owns a fixed set of global ids and the
    tree build only permutes within ``d``, so sorting by the returned gid is a
    device-local operation and the whole thing lives inside one ``shard_map`` with
    no collective.

    Parameters
    ----------
    mesh : Any
        The device mesh, axis ``"gpus"``.

    Returns
    -------
    Callable
        ``(values, gid_out, rank_in) -> values`` in input row order.

    Notes
    -----
    ``rank_in`` is ``[ndev * cap]`` int32: for input row ``i``, the rank of its
    global id among the ids its own device owns. It is an OPERAND rather than a
    closed-over constant so that a repartition -- which changes which device owns
    which id, and therefore every rank -- can swap it without retracing the
    aligner. Matches ``odisseo.mesh_coupling.make_aligner``.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    def local(v, gid_out, rk):
        ascending = v[jnp.argsort(gid_out.reshape(-1))]
        return ascending[rk]

    fn = shard_map(
        local,
        mesh=mesh,
        in_specs=(P("gpus", None), P("gpus", None), P("gpus")),
        out_specs=P("gpus", None),
        check_rep=False,
    )
    return jax.jit(fn)


def direct_sum_probe(pos_rows, mass_rows, targets, soft, g, chunk=1 << 17):
    """fp64 direct-sum self-gravity for ``targets``, against ALL sources.

    Host-side numpy in float64 on purpose. The alternative -- doing it on device --
    would need ``jax_enable_x64``, which is a PROCESS-WIDE switch: turning it on to
    measure the lane would also change the lane being measured. A minute of numpy is
    cheaper than that confound.

    The sum is over every source including those on other devices, and excludes the
    target's own row, matching what the evaluator computes.

    Parameters
    ----------
    pos_rows, mass_rows : ndarray
        Positions ``(n, 3)`` and masses ``(n,)`` in PARTITION ROW order -- the order
        the aligned acceleration comes back in.
    targets : ndarray
        Row indices to evaluate.
    soft, g : float
        Plummer softening and G, matching the solver's.

    Returns
    -------
    ndarray
        ``(len(targets), 3)`` accelerations in float64.
    """
    tp = pos_rows[targets].astype(np.float64)
    acc = np.zeros((len(targets), 3), np.float64)
    eps2 = float(soft) * float(soft)
    n_src = len(pos_rows)
    for s0 in range(0, n_src, chunk):
        s1 = min(s0 + chunk, n_src)
        sp = pos_rows[s0:s1].astype(np.float64)
        sm = mass_rows[s0:s1].astype(np.float64)
        d = sp[None, :, :] - tp[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", d, d) + eps2
        w = sm[None, :] / (r2 * np.sqrt(r2))
        # Drop the self term. Softening makes it finite rather than infinite, so it
        # would not show up as a nan -- just a silently wrong reference.
        loc = np.nonzero((targets >= s0) & (targets < s1))[0]
        if loc.size:
            w[loc, targets[loc] - s0] = 0.0
        acc += np.einsum("ij,ijk->ik", w, d)
    return float(g) * acc


def _verify_alignment(aligned, accel_raw, gid_out, gid_in, n):
    """Check the on-device realignment against the repo's own host-side scatter.

    ``scatter_to_input_order`` is the implementation the codebase already trusts and
    the one written precisely because hand-rolling this scatter caused a real force
    defect. Agreeing with it is the evidence that the fast path is the same map.

    Parameters
    ----------
    aligned : Any
        The device-realigned accelerations, in input row order.
    accel_raw : Any
        The evaluator's raw output.
    gid_out : Any
        The gid array the evaluator returned.
    gid_in : Any
        The gid_flat that went in.
    n : int
        Particle count.

    Raises
    ------
    RuntimeError
        If the two maps disagree anywhere.
    """
    from jaccpot.distributed.fmm import scatter_to_input_order

    ref = scatter_to_input_order(accel_raw, gid_out, n)   # [n, 3], global order
    got = np.asarray(aligned)                              # [n, 3], input row order
    gi = np.asarray(gid_in).reshape(-1).astype(np.int64)
    ref_rows = ref[gi]                                     # -> input row order
    if not np.array_equal(got, ref_rows):
        bad = int(np.sum(np.any(got != ref_rows, axis=1)))
        worst = float(np.max(np.abs(got - ref_rows)))
        raise RuntimeError(
            f"on-device realignment disagrees with scatter_to_input_order on "
            f"{bad} of {n} rows (max |diff| {worst:.3e}). Refusing to integrate."
        )


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", required=True, help="npz from tools/agama_generate_scm_disk_ic.py")
    ap.add_argument("--ndev", type=int, default=5)
    ap.add_argument("--leaf", type=int, default=512)
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--softening", type=float, default=None)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--m2l-chunk", type=int, default=65536)
    ap.add_argument("--nearfield-chunk", type=int, default=512)
    ap.add_argument("--repartition-every", type=int, default=0, help="0 = never")
    # The near-field accumulator. jaccpot's own default is "input", i.e. the fp32
    # sum this lane's error floor was traced to: widening ONLY the accumulator buys
    # 439x accuracy for 1.8 % time and 0 % memory. This script never passed the knob,
    # so every number it has produced so far was taken at the floor -- which is also
    # why loosening theta once looked like it BOUGHT accuracy. Default it to wide.
    ap.add_argument("--nearfield-backend", default="auto",
                    choices=("auto", "baseline", "pallas"),
                    help="Near-field P2P backend. 'auto' picks pallas on Ampere+. Exposed "
                         "because the force was found to be 45-50 %% wrong on every call "
                         "after the first once positions MOVE, for BOTH MACs, with every "
                         "guard silent -- the signature of a kernel reading memory it did "
                         "not write on this call. 'baseline' is the pure-JAX P2P; if it is "
                         "accurate where 'pallas' is not, the Pallas near-field owns it.")
    ap.add_argument("--nearfield-accum", default="wide", choices=("input", "wide"),
                    help="Near-field accumulator width. 'wide' = fp64 accumulate.")
    # Dehnen (2014) section 5 mass-dependent MAC. Under "dehnen_error" theta stops
    # gating the SELF walk entirely -- `--adaptive-eps` replaces it as the accuracy
    # knob -- while theta still gates the CROSS walk's geometry.
    ap.add_argument("--mac-type", default="dehnen",
                    choices=("bh", "engblom", "dehnen", "dehnen_error"),
                    help="Multipole acceptance criterion.")
    ap.add_argument("--adaptive-eps", type=float, default=None,
                    help="Relative force-accuracy target for --mac-type dehnen_error. "
                         "Mandatory under that criterion.")
    ap.add_argument("--mac-cross-criterion", dest="mac_cross_criterion",
                    action="store_true", default=True,
                    help="Let the criterion decide cross-domain pairs too (default).")
    ap.add_argument("--no-mac-cross-criterion", dest="mac_cross_criterion",
                    action="store_false",
                    help="Self-only ablation: cross walk stays geometric.")
    ap.add_argument("--probe", type=int, default=0,
                    help="Measure the SELF-GRAVITY force error at t=0 against an "
                         "fp64 direct sum over ALL sources, for this many randomly "
                         "chosen targets. 0 = skip. The subsampled reference means "
                         "rel_l2 is only comparable at the SAME --probe: the same "
                         "config reads 3.13e-3 at probe 192 and 4.42e-3 at 256.")
    ap.add_argument("--probe-every", type=int, default=0,
                    help="Re-score the force against a FRESH fp64 direct sum every N steps, "
                         "not only at t=0. Exists because every accuracy number this lane "
                         "ever produced was on the first force, and the geometric MAC was "
                         "found to be 0.5 % accurate on its first call and 30-54 % wrong on "
                         "every later call to the SAME input (identical accept mask, zero "
                         "overflow flags, zero non-finite -- no other guard sees it). The "
                         "reference is recomputed because positions have moved; ~225 s per "
                         "probe at 17.8M on the host. 0 = t=0 only.")
    ap.add_argument("--probe-seed", type=int, default=20260901,
                    help="Seed for the probe's target choice, so two arms compare on "
                         "exactly the same targets.")
    # Cap overrides, named as in jaccpot's own bench/distributed_ceiling_ladder.py.
    # `resolved_for` only DERIVES a cap that is None, so an explicit value survives it.
    # These exist because the criterion's derived caps do not fit at every scale: at
    # 21 M / 6 devices / leaf 512 its cross wavefront is 16 777 216 entries, which at
    # order 6 asks for a single 63.78 GiB buffer on a 40 GB card.
    #
    # Shrinking a cap is only safe BECAUSE the overflow flags are checked: an
    # under-provisioned buffer truncates the walk, which makes the run FASTER and the
    # force WRONG (trap 6). Never pass these without reading the flags back.
    ap.add_argument("--pair-queue", type=int, default=0,
                    help="Override max_pair_queue (self wavefront). 0 = derive.")
    ap.add_argument("--cross-queue", type=int, default=0,
                    help="Override cross_max_pair_queue. 0 = derive.")
    ap.add_argument("--cross-neighbors", type=int, default=0,
                    help="Override cross_max_neighbors_per_leaf. 0 = derive.")
    ap.add_argument("--cross-interactions", type=int, default=0,
                    help="Override cross_max_interactions_per_node. 0 = derive.")
    ap.add_argument("--restart-from", default="",
                    help="Resume from a snapshot written by --checkpoint-every instead of "
                         "starting at the IC. Steps continue from the snapshot's own step "
                         "counter, and the conservation baseline is carried IN the snapshot "
                         "so dL/L stays continuous across a restart rather than resetting "
                         "to zero. Exists because the force can return NaN intermittently "
                         "(once at step 10 in ~14 steps, not reproducible); with the "
                         "finiteness gate a run now aborts in one step, and this makes that "
                         "cost a restart instead of the whole rollout.")
    ap.add_argument("--snapshot-steps", default="",
                    help="Comma-separated step indices at which to write a NON-rolling "
                         "snapshot <prefix>_step<k>.npz (same format as the checkpoint). "
                         "For re-evaluating a specific step's positions in isolation: a "
                         "force that is wrong at steps 1-2 and right at 0 and 3 is either "
                         "position-dependent (a single fresh call on X1 reproduces it) or "
                         "call-order-dependent (it does not).")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="Write a full particle snapshot (positions + velocities in "
                         "ORIGINAL input order) every N steps, plus one at the end. "
                         "A rolling file, replaced atomically, so disk stays bounded. "
                         "0 = final snapshot only. Without this a long run keeps only "
                         "density grids and throws the evolved galaxy away.")
    ap.add_argument("--overflow-every", type=int, default=25,
                    help="Re-check the capacity flags every N steps. Caps are static "
                         "but pair counts GROW as the disc clusters, so an overflow "
                         "can switch on mid-run and silently truncate the force. "
                         "0 disables (first force only, the old behaviour).")
    ap.add_argument("--render-every", type=int, default=0, help="0 = no movie")
    ap.add_argument("--render-res", type=int, default=800)
    ap.add_argument("--render-extent", type=float, default=1.2)
    # A face-on view cannot show a bulge: projected on the disc plane the bulge sits
    # under the disc's own centre. Edge-on is the view that separates them, and both
    # come from ONE integration -- the projector is a device-local histogram plus a
    # psum, so a second view costs a grid, not a second rollout.
    ap.add_argument("--projection", default="xy",
                    help="Projection(s) for the movie: xy, xz, yz, or a "
                         "comma-separated list (e.g. 'xy,xz' for face-on AND "
                         "edge-on from a single pass).")
    ap.add_argument("--diag-every", type=int, default=10)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop gracefully after this much wall clock. 0 = no budget")
    ap.add_argument("--no-halo", action="store_true")
    ap.add_argument("--out-prefix", default="mesh_galaxy")
    ap.add_argument("--max-particles", type=int, default=0, help="0 = all in the IC")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"),
                    help="working precision. float32 has a near-field round-off floor "
                         "around 1e-3 at 10^7; float64 costs ~2.3x and reaches 1e-5.")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.fmm import (
        DistributedFMMConfig,
        make_force_evaluator,
        make_mesh,
        partition_for_devices,
    )

    t_wall0 = time.perf_counter()
    ic = np.load(args.ic)
    restart = np.load(args.restart_from) if args.restart_from else None
    # The snapshot carries state and the baseline; the IC still supplies the halo and
    # bulge parameters, so both are opened and the snapshot wins on the state only.
    state0 = np.asarray((restart if restart is not None else ic)["state0"])
    mass = np.asarray(ic["mass"])
    wdt = np.float32 if args.dtype == "float32" else np.float64
    pos = np.ascontiguousarray(state0[:, 0, :], dtype=wdt)
    vel = np.ascontiguousarray(state0[:, 1, :], dtype=wdt)
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    halo_m = float(ic["halo_mass_code"]) if "halo_mass_code" in ic.files else 0.0
    halo_rs = float(ic["halo_rs_code"]) if "halo_rs_code" in ic.files else 1.0
    rdisk = float(ic["rdisk_code"]) if "rdisk_code" in ic.files else 0.24

    n_all = pos.shape[0]
    # N must be ndev * k * leaf so that cap == count and no device is padded.
    quantum = args.ndev * args.leaf
    n_target = args.max_particles or n_all
    n = (min(n_target, n_all) // quantum) * quantum
    if n < quantum:
        raise SystemExit(f"IC has {n_all} particles, need at least {quantum}")
    if n != n_all:
        print(f"# trimming {n_all} -> {n} so N = ndev*k*leaf ({args.ndev}*{n//quantum}*{args.leaf})")
    mass = mass.astype(wdt)
    pos, vel, mass = pos[:n], vel[:n], mass[:n]

    soft = args.softening
    if soft is None:
        # mean in-plane spacing of the disc, a conventional choice
        soft = float(0.5 * rdisk / np.sqrt(n / 1e5))
    print(
        f"# N={n:,} ndev={args.ndev} leaf={args.leaf} theta={args.theta} order={args.order} dtype={args.dtype}\n"
        f"# G={g} halo_mass={halo_m} halo_rs={halo_rs} rdisk={rdisk} softening={soft:.5g}\n"
        f"# mac={args.mac_type} eps={args.adaptive_eps} cross_criterion={args.mac_cross_criterion} "
        f"accum={args.nearfield_accum} nearfield_backend={args.nearfield_backend}\n"
        f"# dt={args.dt} steps={args.steps} devices={[d.device_kind for d in jax.devices()][:args.ndev]}",
        flush=True,
    )

    if args.mac_type == "dehnen_error" and args.adaptive_eps is None:
        raise SystemExit(
            "--mac-type dehnen_error requires --adaptive-eps: under the criterion "
            "theta no longer gates the self walk, so leaving eps unset would silently "
            "hand accuracy to a knob that decides nothing."
        )
    cfg = DistributedFMMConfig(
        leaf_size=args.leaf,
        theta=args.theta,
        order=args.order,
        softening=soft,
        G=g,
        m2l_chunk=args.m2l_chunk,
        nearfield_chunk=args.nearfield_chunk,
        nearfield_accum=args.nearfield_accum,
        nearfield_backend=args.nearfield_backend,
        mac_type=args.mac_type,
        adaptive_eps=args.adaptive_eps,
        mac_cross_criterion=bool(args.mac_cross_criterion),
        **{
            k: v
            for k, v in (
                ("max_pair_queue", args.pair_queue),
                ("cross_max_pair_queue", args.cross_queue),
                ("cross_max_neighbors_per_leaf", args.cross_neighbors),
                ("cross_max_interactions_per_node", args.cross_interactions),
            )
            if v > 0
        },
    )
    mesh = make_mesh(args.ndev)
    part = partition_for_devices(
        pos, mass, args.ndev, leaf_size=args.leaf, partitioner=cfg.partitioner
    )
    cap = part["cap"]
    if cap * args.ndev != n:
        raise SystemExit(f"padding present: cap={cap} ndev={args.ndev} n={n}")
    cfg_r = cfg.resolved_for(cap, args.ndev)

    print(
        "# caps: self_queue={:,} self_nbr={:,} self_int={:,} | cross_queue={:,} "
        "cross_nbr={:,} cross_int={:,}".format(
            cfg_r.max_pair_queue, cfg_r.max_neighbors_per_leaf,
            cfg_r.max_interactions_per_node, cfg_r.cross_max_pair_queue,
            cfg_r.cross_max_neighbors_per_leaf,
            cfg_r.cross_max_interactions_per_node,
        ),
        flush=True,
    )
    t0 = time.perf_counter()
    evaluate = make_force_evaluator(cfg_r, args.ndev, cap, mesh, jit=True)
    print(f"# evaluator built in {time.perf_counter() - t0:.1f} s", flush=True)

    gid_flat = np.asarray(part["gid_flat"])
    order_ix = gid_flat.astype(np.int64)  # row -> original particle index
    # rank_in[i] = rank of row i's global id among the ids ITS OWN device holds.
    # The input order is frozen for the whole rollout, so this is computed once.
    rank_in = np.empty(args.ndev * cap, np.int32)
    for d in range(args.ndev):
        sl = slice(d * cap, (d + 1) * cap)
        rank_in[sl] = np.argsort(np.argsort(gid_flat[sl])).astype(np.int32)
    shard = NamedSharding(mesh, P("gpus", None))
    shard1 = NamedSharding(mesh, P("gpus"))
    X = jax.device_put(jnp.asarray(part["pos_flat"]), shard)
    M = jax.device_put(jnp.asarray(part["mass_flat"]), shard1)
    V = jax.device_put(jnp.asarray(vel[order_ix]), shard)
    GID = jnp.asarray(part["gid_flat"])
    COUNTS = jnp.asarray(part["counts"])
    RANK = jax.device_put(jnp.asarray(rank_in), shard1)

    use_halo = (not args.no_halo) and halo_m > 0.0

    align = make_aligner(mesh)

    # Everything a repartition replaces lives here, so `force` closes over ONE
    # mutable cell instead of five immutable ones. Nothing in it is traced -- the
    # jitted pieces are `evaluate` and `align`, both of which take these as
    # operands -- so swapping them costs no recompile.
    pstate = {"M": M, "GID": GID, "COUNTS": COUNTS, "RANK": RANK, "order_ix": order_ix}

    def force(x):
        a_raw, gid_o, diag = evaluate(x, pstate["M"], pstate["GID"], pstate["COUNTS"])
        a_self = align(a_raw, gid_o, pstate["RANK"])  # -> input row order, device-local
        a = a_self
        if use_halo:
            a = a + nfw_acceleration(x, halo_m, halo_rs, g)
        # `a_self` is returned separately so the probe can compare the part the FMM
        # actually computes. Adding the analytic halo to both sides of that
        # comparison would inflate the denominator and report an error several times
        # smaller than the solver's own.
        return a, gid_o, diag, a_raw, a_self

    # first evaluation, eagerly, so the padding guard and the overflow flags are
    # readable -- both are invisible once this is inside a jitted step.
    t0 = time.perf_counter()
    A, gid_o, diag, A_raw, A_self = force(X)
    jax.block_until_ready(A)
    print(f"# first force (compile incl.) {time.perf_counter() - t0:.1f} s", flush=True)
    t0 = time.perf_counter()
    # Verify the PERMUTATION, before any arithmetic touches it. A permutation is
    # exact, so array_equal is the right test here -- but only on a value that has
    # not been through an add: subtracting the halo term back off reintroduces
    # float32 rounding and turns an exact check into a ~1e-6 mismatch on 80 % of
    # rows, which looks like a broken map and is not one.
    _verify_alignment(align(A_raw, gid_o, pstate["RANK"]), A_raw, gid_o, pstate["GID"], n)
    print(f"# realignment verified against scatter_to_input_order "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    # Read the field NAMES from jaccpot rather than restating them. The local copy
    # was correct for the eleven fields that existed when it was written, but the
    # Dehnen criterion appended `force_scale_min`/`force_scale_max` -- and those two
    # are exactly trap 14's witness (a CONSTANT force scale means the criterion
    # silently fell back to `eps * 1`, which accepts far more and runs FASTER). A
    # hardcoded list cannot report a field it does not know about.
    from jaccpot.distributed.fmm import DIAG_FIELDS

    def decode_diag(diag_arr):
        """Per-device diagnostic rows -> {field: [per-device values]}."""
        arr = np.asarray(diag_arr)
        return {
            name: [float(v) for v in arr[:, i]]
            for i, name in enumerate(DIAG_FIELDS)
            if i < arr.shape[1]
        }

    def overflow_flags(dec):
        return {k: sum(v) for k, v in dec.items() if k.endswith("overflow")}

    diag0 = decode_diag(diag)
    ovf = overflow_flags(diag0)
    fs = (diag0.get("force_scale_min", []), diag0.get("force_scale_max", []))
    if args.mac_type == "dehnen_error" and fs[0] and fs[1]:
        lo, hi = min(fs[0]), max(fs[1])
        # A min equal to the max is the `jnp.ones(...)` fallback, not a force scale.
        if not (hi > lo):
            raise SystemExit(
                f"force scale is CONSTANT ({lo:g}); the criterion fell back to "
                f"eps*1 instead of eps*min_b f_b (trap 14) -- refusing to integrate"
            )
        print(f"# force_scale range [{lo:.4g}, {hi:.4g}] ({hi / max(lo, 1e-300):.1f}x spread)",
              flush=True)
    probe_history = []

    def run_probe(step_index, a_self_arr, a_raw_arr=None, gid_o_arr=None):
        """Score the SELF-GRAVITY force at this step against a fresh fp64 direct sum.

        Targets are fixed by ``--probe-seed`` so every probe scores the same particles;
        the reference is recomputed each time because they have moved. Row order: both
        ``X`` and the aligned acceleration are in partition row order, and the targets
        index rows, so a repartition between probes changes WHICH particles are scored
        -- acceptable for a trend, but do not compare a probe across a repartition to
        one before it particle-by-particle.
        """
        t0 = time.perf_counter()
        pos_rows = np.asarray(jax.device_get(X))[:n]
        mass_rows = np.asarray(jax.device_get(pstate["M"]))[:n]
        rngp = np.random.default_rng(int(args.probe_seed))
        targets = np.sort(rngp.choice(n, size=int(args.probe), replace=False))
        a_ref = direct_sum_probe(pos_rows, mass_rows, targets, soft, g)
        a_got = np.asarray(jax.device_get(a_self_arr))[:n][targets].astype(np.float64)
        num = np.linalg.norm(a_got - a_ref, axis=1)
        den = np.linalg.norm(a_ref, axis=1)
        rel = num / np.maximum(den, 1e-300)
        rel_l2 = float(np.linalg.norm(num) / max(np.linalg.norm(den), 1e-300))
        stats = {
            "step": int(step_index),
            "probe": int(args.probe), "probe_seed": int(args.probe_seed),
            "rel_l2": rel_l2, "rel_median": float(np.median(rel)),
            "rel_p99": float(np.quantile(rel, 0.99)), "rel_max": float(rel.max()),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        # Is the FORCE wrong, or only its MAPPING back to input rows? Two independent
        # checks, both only possible here because the evaluator's raw output and its
        # gid map are in hand:
        #  (1) the on-device aligner against jaccpot's own host scatter, on THIS step
        #      (it used to be checked once, at the first force);
        #  (2) the RAW force, in the evaluator's own Morton order, against a direct sum at
        #      the positions in THAT order. If (2) is accurate while the aligned force is
        #      not, the force is right and the mapping is the defect.
        if a_raw_arr is not None and gid_o_arr is not None:
            gid_in = np.asarray(jax.device_get(pstate["GID"]))
            gid_out = np.asarray(jax.device_get(gid_o_arr)).reshape(-1)
            try:
                _verify_alignment(a_self_arr, a_raw_arr, gid_o_arr, pstate["GID"], n)
                stats["alignment_ok"] = True
            except Exception as exc:  # noqa: BLE001
                stats["alignment_ok"] = False
                print(f"# PROBE step={step_index}: ALIGNMENT MISMATCH -- {str(exc)[:160]}",
                      flush=True)
            row_of_gid = np.empty(int(gid_in.max()) + 1, np.int64)
            row_of_gid[gid_in] = np.arange(len(gid_in))
            valid = gid_out >= 0
            morton_rows = row_of_gid[gid_out[valid]]
            pos_m = pos_rows[morton_rows]; mass_m = mass_rows[morton_rows]
            # score the same PARTICLES: map the row targets to their Morton positions
            morton_pos_of_row = np.full(n, -1, np.int64)
            morton_pos_of_row[morton_rows] = np.arange(morton_rows.size)
            tm = morton_pos_of_row[targets]; okm = tm >= 0
            a_ref_m = direct_sum_probe(pos_m, mass_m, tm[okm], soft, g)
            # A map check that does not use the map as its own witness. The aligner and
            # scatter_to_input_order both consume gid_o, so they can agree while gid_o is
            # wrong -- and a wrong gid_o also poisons the raw-Morton probe above, whose
            # reference is built at positions permuted by it. Independent test: if gid_o
            # is right, the positions it implies are the evaluator's Morton order, so their
            # keys -- recomputed here with the same encoding -- must be non-decreasing PER
            # DEVICE. Ties are equal keys and still pass; a mis-mapping breaks monotonicity.
            try:
                from yggdrax.morton import morton_encode_impl
                from yggdrax.distributed.partition import global_bounds as _gb
                lo = pos_rows.min(axis=0); hi = pos_rows.max(axis=0)
                span = hi - lo
                _bounds = (jnp.asarray(lo - span * 1e-6), jnp.asarray(hi + span * 1e-6))
                keys = np.asarray(morton_encode_impl(jnp.asarray(pos_m, dtype=np.float32), _bounds))
                per_dev = np.asarray(jax.device_get(pstate["COUNTS"])).astype(int)
                starts = np.concatenate([[0], np.cumsum(per_dev)])
                viol = 0
                for d in range(len(per_dev)):
                    kd = keys[starts[d]:starts[d] + per_dev[d]]
                    viol += int(np.count_nonzero(np.diff(kd) < 0))
                stats["gid_o_morton_violations"] = viol
                print(f"# PROBE step={step_index} gid_o implies Morton order with {viol} descents "
                      f"({'consistent' if viol == 0 else 'INCONSISTENT -- gid_o itself is suspect'}; "
                      f"bounds are a host approximation, so a handful of descents at device "
                      f"seams is noise, thousands is not)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"# PROBE step={step_index}: morton check skipped ({str(exc)[:100]})", flush=True)
            a_raw_h = np.asarray(jax.device_get(a_raw_arr)).reshape(-1, 3)
            a_raw_t = a_raw_h[np.flatnonzero(valid)[tm[okm]]].astype(np.float64)
            num_m = np.linalg.norm(a_raw_t - a_ref_m, axis=1)
            stats["rel_l2_raw_morton"] = float(np.linalg.norm(num_m) / max(np.linalg.norm(np.linalg.norm(a_ref_m, axis=1)), 1e-300))
            print(f"# PROBE step={step_index} RAW-in-Morton-order rel_l2={stats['rel_l2_raw_morton']:.4e}"
                  f"  aligned rel_l2={rel_l2:.4e}  alignment_ok={stats['alignment_ok']}", flush=True)
        probe_history.append(stats)
        print(f"# PROBE step={step_index} n={args.probe} rel_l2={rel_l2:.4e} "
              f"median={np.median(rel):.3e} p99={np.quantile(rel, 0.99):.3e} "
              f"max={rel.max():.3e} ({stats['seconds']:.1f} s)", flush=True)
        return stats

    probe_stats = run_probe(0, A_self, A_raw, gid_o) if args.probe > 0 else None
    print(f"# self_near={sum(diag0.get('self_near_pairs', [0])):,.0f} "
          f"cross_near={sum(diag0.get('cross_near_pairs', [0])):,.0f} overflow={ovf}", flush=True)
    if any(v > 0 for v in ovf.values()):
        raise SystemExit(f"capacity overflow on the first force: {ovf} -- refusing to integrate")

    # The KDK arithmetic is deliberately NOT fused into one jit with the force.
    # Fusing them makes XLA hold the evaluator's traversal buffers and the
    # integrator's temporaries in a single live range; at 21M on five cards that
    # overflows, one device fails an allocation, never joins the AllGather, and the
    # other four hang at the rendezvous forever -- a deadlock, not an OOM message.
    # Split into three dispatches and the peak is the max of the two, not the sum.
    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def drift(x, v, a, dt):
        vh = v + 0.5 * dt * a
        return x + dt * vh, vh

    @partial(jax.jit, donate_argnums=(0, 1))
    def kick(vh, a, dt):
        return vh + 0.5 * dt * a

    def kdk(x, v, a, dt):
        xn, vh = drift(x, v, a, dt)
        an, gid_n, dg, raw_n, a_self_n = force(xn)
        # The diagnostic vector is RETURNED, not dropped. It used to be discarded
        # here, which left `diag` in the step loop bound to the FIRST force forever --
        # so the periodic overflow check re-validated step 0 on every cadence and could
        # never see a capacity that overflowed later, which is the only reason the check
        # exists. Caps are static; pair counts are not.
        return xn, kick(vh, an, dt), an, dg, a_self_n, raw_n, gid_n

    def invariants(x, v, m):
        """Momentum, angular momentum, COM and KE -- reduced in float64 on the host.

        NOT a device float32 reduction, which is what this was. Summing 21 million
        float32 cross-products carries a tree-reduction error of about
        ``log2(N) * eps = 2.9e-06`` relative, and dL/L is the norm of a DIFFERENCE of
        two such sums -- so the diagnostic's own round-off lands at ~3e-06, exactly
        the range the measurements were in. Two runs with forces agreeing to seven
        significant figures were reporting dL/L 1.1e-07 against 2.6e-06, a 23x
        "difference" that was entirely reduction order.

        float64 on the host rather than `jnp.float64` on device, because x64 is a
        PROCESS-WIDE switch: enabling it to fix a diagnostic would also let python
        floats promote the lane's float32 arrays, and at ~31 GB of 40 GB per card that
        is not a safe thing to do by accident.

        Costs one 252 MB device_get per array. The loop already blocks on X every
        step, so nothing is pipelined away, and at `--diag-every 10` it is ~1 s per
        call against a 60 s step.
        """
        xh = np.asarray(jax.device_get(x), dtype=np.float64)[:n]
        vh = np.asarray(jax.device_get(v), dtype=np.float64)[:n]
        mh = np.asarray(jax.device_get(m), dtype=np.float64)[:n]
        pp = (mh[:, None] * vh).sum(axis=0)
        ll = (mh[:, None] * np.cross(xh, vh)).sum(axis=0)
        com = (mh[:, None] * xh).sum(axis=0) / mh.sum()
        ke = 0.5 * float((mh * (vh * vh).sum(axis=1)).sum())
        return pp, ll, com, ke

    n_reparts = 0
    snapshot_steps = {int(t) for t in str(args.snapshot_steps).split(",") if t.strip()}

    def save_state(step_index, final=False):
        """Snapshot the particles in the caller's ORIGINAL order.

        Written to a temporary path and renamed, because a 19-hour run replacing its
        own checkpoint is exactly the situation where a half-written file loses both
        the new snapshot and the previous one. Rolling (one file) rather than
        numbered: 21 million particles is ~500 MB a copy.

        ``order_ix[row] = original index``, so ``st[order_ix] = rows`` puts every
        particle back where the caller had it -- including after a repartition, which
        replaces that map.
        """
        oi = pstate["order_ix"]
        pos_rows = np.asarray(jax.device_get(X))[:n]
        vel_rows = np.asarray(jax.device_get(V))[:n]
        st = np.empty((n, 2, 3), pos_rows.dtype)
        st[oi, 0, :] = pos_rows
        st[oi, 1, :] = vel_rows
        suffix = "final" if final else "ckpt"
        if final is None:  # step-indexed, non-rolling
            suffix = f"step{int(step_index)}"
        dest = pathlib.Path(f"{args.out_prefix}_{suffix}.npz")
        # The temp name must ALSO end in .npz: np.savez appends the extension when the
        # path lacks it, so a ".npz.tmp" target is written as ".npz.tmp.npz" and the
        # rename then fails on a file that was never created.
        tmp = dest.with_name(f".{dest.name}.tmp.npz")
        payload = dict(
            state0=st, mass=mass, step=np.asarray(int(step_index)),
            # The baseline the run started from, so a restart reports dL/L against the
            # ORIGINAL state rather than against wherever it happened to resume.
            baseline_p0=np.asarray(p0, dtype=np.float64),
            baseline_l0=np.asarray(l0, dtype=np.float64),
            baseline_com0=np.asarray(com0, dtype=np.float64),
            baseline_lscale=np.asarray(float(lscale)),
            baseline_step0=np.asarray(int(step0)),
            t=np.asarray(float(step_index) * float(args.dt)),
            n_particles=np.asarray(int(n)),
            dt=np.asarray(float(args.dt)), softening=np.asarray(float(soft)),
            halo_mass_code=np.asarray(float(halo_m)),
            halo_rs_code=np.asarray(float(halo_rs)),
        )
        for key in ("component", "n_disk", "n_bulge", "bulge_mass_code",
                    "bulge_scale_code", "disk_mass_code", "rdisk_code", "hdisk_code"):
            if key not in ic.files:
                continue
            v = ic[key]
            # `--max-particles` and the ndev*leaf trim shorten the particle arrays but
            # not the IC's per-particle labels, so a full-length `component` would be
            # misaligned with `state0` by however much was dropped.
            payload[key] = v[:n] if (v.ndim == 1 and v.shape[0] > n) else v
        np.savez(tmp, **payload)
        tmp.replace(dest)
        print(f"# checkpoint step {step_index} -> {dest.name} "
              f"({dest.stat().st_size / 1e6:.0f} MB)", flush=True)

    def repartition(Xc, Vc):
        """Rebuild the decomposition from the CURRENT positions.

        A frozen partition is fine for a short rollout and wrong for a long one. The
        disc shears -- over a quarter orbit an inner particle completes ~5x more
        azimuth than an outer one -- so RCB domains stop being spatially compact,
        cross-domain near-field work climbs, and the static caps eventually overflow.

        ``cap`` depends only on ``(n, ndev, leaf_size)`` and never on positions, so
        the compiled evaluator is keyed on nothing that moves: this is a host
        permutation plus a ``device_put``, with no recompile. The aligner takes
        ``rank_in`` as an operand for the same reason.
        """
        oi = pstate["order_ix"]
        pos_rows = np.asarray(jax.device_get(Xc))[:n]
        vel_rows = np.asarray(jax.device_get(Vc))[:n]
        # Rows -> original particle order, which is the order `mass` is in.
        pos_orig = np.empty_like(pos_rows)
        pos_orig[oi] = pos_rows
        vel_orig = np.empty_like(vel_rows)
        vel_orig[oi] = vel_rows
        newp = partition_for_devices(
            pos_orig, mass, args.ndev, leaf_size=args.leaf, partitioner=cfg.partitioner
        )
        if newp["cap"] != cap:
            raise SystemExit(
                f"repartition changed cap {cap} -> {newp['cap']}: that would force a "
                f"recompile, and it is supposed to be position-independent"
            )
        gf = np.asarray(newp["gid_flat"])
        oi2 = gf.astype(np.int64)
        rk = np.empty(args.ndev * cap, np.int32)
        for d in range(args.ndev):
            sl = slice(d * cap, (d + 1) * cap)
            rk[sl] = np.argsort(np.argsort(gf[sl])).astype(np.int32)
        pstate["M"] = jax.device_put(jnp.asarray(newp["mass_flat"]), shard1)
        pstate["GID"] = jnp.asarray(gf)
        pstate["COUNTS"] = jnp.asarray(newp["counts"])
        pstate["RANK"] = jax.device_put(jnp.asarray(rk), shard1)
        pstate["order_ix"] = oi2
        return (jax.device_put(jnp.asarray(newp["pos_flat"]), shard),
                jax.device_put(jnp.asarray(vel_orig[oi2]), shard))

    _PROJECTION_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    projectors = {}
    frames = {}
    if args.render_every > 0:
        ext = args.render_extent
        names = [t.strip().lower() for t in str(args.projection).split(",") if t.strip()]
        names = [t for t in names if t in _PROJECTION_AXES] or ["xy"]
        for nm in names:
            projectors[nm] = make_density_projector(
                mesh, args.render_res, _PROJECTION_AXES[nm], (-ext, -ext), (ext, ext)
            )
            frames[nm] = []
        print(f"# rendering projections {names} at {args.render_res}^2, "
              f"extent +/-{ext}", flush=True)

    def snap():
        for nm, proj in projectors.items():
            frames[nm].append(np.asarray(proj(X, pstate["M"])))

    def flush_frames():
        for nm, fr in frames.items():
            if fr:
                np.savez_compressed(f"{args.out_prefix}_frames_{nm}.npz",
                                    frames=np.stack(fr))

    snap()

    p0, l0, com0, ke0 = [np.asarray(z) for z in invariants(X, V, pstate["M"])]
    mtot = float(np.sum(mass))
    lscale = float(np.sum(mass * np.linalg.norm(np.cross(pos, vel), axis=1)))
    step0 = 0
    if restart is not None:
        step0 = int(restart["step"]) if "step" in restart.files else 0
        if "baseline_l0" in restart.files:
            p0 = np.asarray(restart["baseline_p0"])
            l0 = np.asarray(restart["baseline_l0"])
            com0 = np.asarray(restart["baseline_com0"])
            lscale = float(restart["baseline_lscale"])
            print(f"# restart: resuming after step {step0}, conservation baseline carried "
                  f"from the original start", flush=True)
        else:
            print(f"# restart: resuming after step {step0}; the snapshot carries NO "
                  f"baseline, so dL/L is measured from HERE, not from the IC", flush=True)
    print(f"# t=0  |L|={np.linalg.norm(l0):.6e}  KE={ke0:.6e}  com={np.round(com0, 6)}", flush=True)

    rows = []
    step_times = []
    t_run0 = time.perf_counter()
    for it in range(step0 + 1, args.steps + 1):
        t0 = time.perf_counter()
        X, V, A, diag, A_self, A_raw, gid_o = kdk(X, V, A, args.dt)
        jax.block_until_ready(X)
        step_times.append(time.perf_counter() - t0)

        if args.repartition_every and it % args.repartition_every == 0:
            X, V = repartition(X, V)
            # A is in the OLD row order and must not survive the swap.
            A, gid_o, diag, A_raw, _ = force(X)
            jax.block_until_ready(A)
            _verify_alignment(
                align(A_raw, gid_o, pstate["RANK"]), A_raw, gid_o, pstate["GID"], n
            )
            n_reparts += 1
            print(f"  step {it:>6d}  repartitioned (#{n_reparts}), realignment "
                  f"re-verified", flush=True)

        if args.render_every and it % args.render_every == 0:
            snap()
            # Flush every 10 frames. A long run on shared cards can be stopped at
            # any moment, and a movie that only exists in RAM at step N is a movie
            # that does not exist.
            if len(next(iter(frames.values()))) % 10 == 0:
                flush_frames()
        if args.max_hours > 0 and (time.perf_counter() - t_run0) > args.max_hours * 3600:
            print(f"# wall-clock budget reached after {it} steps -- stopping cleanly", flush=True)
            args.steps = it
            if args.render_every:
                snap()
            break
        # Caps are static; pair counts are NOT. The disc clusters as it evolves (and
        # a live bulge concentrates faster than the disc does), so a capacity that
        # cleared on the first force can overflow at step 400 -- silently truncating
        # the near list, which makes the run read FASTER while the force goes wrong.
        # The diag vector comes back from every call, so this costs one host sync.
        if args.overflow_every and it % args.overflow_every == 0:
            ovf_now = overflow_flags(decode_diag(diag))
            if any(v > 0 for v in ovf_now.values()):
                pathlib.Path(f"{args.out_prefix}_diag.json").write_text(json.dumps({
                    "args": vars(args), "n": n, "cap": cap, "softening": soft,
                    "rows": rows, "step_times": [round(t, 4) for t in step_times],
                    "aborted_at_step": it, "overflow": ovf_now, "partial": True,
                }, indent=1))
                raise SystemExit(
                    f"capacity overflow at step {it}: {ovf_now} -- the force is "
                    f"truncated from here on. Grow the relevant cap (or loosen the "
                    f"criterion) and restart; diagnostics written."
                )
        # A NaN is the cheapest thing to detect and the most expensive to miss. This
        # run reported `dL/L=nan` at step 10 and then span four A100s for SEVENTEEN
        # HOURS without another line of output: once the state goes non-finite the
        # tree's bounding box does too, every leaf becomes every other leaf's
        # neighbour, and the traversal grinds forever at 100 % utilisation. There is
        # no cheaper guard than this and no more expensive omission.
        #
        # Three device reductions plus one host sync per step, against a ~155 s step.
        # The loop already blocks on X every step, so nothing is pipelined away.
        finite = bool(
            jnp.isfinite(A).all() & jnp.isfinite(X).all() & jnp.isfinite(V).all()
        )
        if not finite:
            bad = {
                "accel": int(np.count_nonzero(~np.isfinite(np.asarray(A)))),
                "pos": int(np.count_nonzero(~np.isfinite(np.asarray(X)))),
                "vel": int(np.count_nonzero(~np.isfinite(np.asarray(V)))),
            }
            dec = decode_diag(diag)
            fs = (dec.get("force_scale_min", []), dec.get("force_scale_max", []))
            pathlib.Path(f"{args.out_prefix}_diag.json").write_text(json.dumps({
                "args": vars(args), "n": n, "cap": cap, "softening": soft,
                "rows": rows, "step_times": [round(t, 4) for t in step_times],
                "probe": probe_stats, "probe_history": probe_history, "num_repartitions": n_reparts,
                "aborted_at_step": it, "nonfinite_counts": bad,
                "force_scale_min": fs[0], "force_scale_max": fs[1],
                "overflow": overflow_flags(dec), "partial": True,
            }, indent=1))
            # Save the state so the failure is inspectable rather than only reported.
            try:
                save_state(it, final=False)
            except Exception as exc:  # noqa: BLE001
                print(f"# (could not snapshot the failed state: {exc})", flush=True)
            raise SystemExit(
                f"NON-FINITE STATE at step {it}: "
                f"{bad['accel']} accel / {bad['pos']} pos / {bad['vel']} vel entries. "
                f"force_scale min={min(fs[0]) if fs[0] else float('nan'):.6g} "
                f"max={max(fs[1]) if fs[1] else float('nan'):.6g}. "
                f"Refusing to continue -- a non-finite state makes the tree degenerate "
                f"and the traversal never terminates. Diagnostics and a snapshot written."
            )
        if args.probe > 0 and args.probe_every and it % args.probe_every == 0:
            run_probe(it, A_self, A_raw, gid_o)
        if args.checkpoint_every and it % args.checkpoint_every == 0:
            save_state(it)
        if it in snapshot_steps:
            save_state(it, final=None)
        if args.diag_every and it % args.diag_every == 0:
            p, l, com, ke = [np.asarray(z) for z in invariants(X, V, pstate["M"])]
            row = {
                "step": it,
                "t": it * args.dt,
                "seconds": round(step_times[-1], 4),
                "dP_over_scale": float(np.linalg.norm(p - p0) / (mtot + 1e-300)),
                "dL_over_L": float(np.linalg.norm(l - l0) / (lscale + 1e-300)),
                "com_drift": float(np.linalg.norm(com - com0)),
                "ke": float(ke),
            }
            # Per-step pair counts and flags. Until now only the FIRST force's were kept,
            # which hid whether a step that computes a wrong force also walks a
            # different tree. Both MACs see identical positions each step, so a
            # position-dependent fault should show here as a count that moves.
            _dec_all = decode_diag(diag)
            for _f in ("self_near_pairs", "self_far_pairs", "cross_near_pairs",
                       "cross_far_pairs"):
                if _f in _dec_all:
                    row[_f] = float(sum(_dec_all[_f]))
            row["overflow_any"] = float(sum(overflow_flags(_dec_all).values()))
            if args.mac_type == "dehnen_error":
                _dec = decode_diag(diag)
                _lo, _hi = _dec.get("force_scale_min", []), _dec.get("force_scale_max", [])
                if _lo and _hi:
                    # A force scale collapsing toward 0 makes eq (16a)'s `eps * min_b f_b`
                    # collapse with it. Watching the FLOOR is how that gets caught before
                    # it turns into a non-finite accept mask.
                    row["force_scale_min"] = float(min(_lo))
                    row["force_scale_max"] = float(max(_hi))
            rows.append(row)
            pathlib.Path(f"{args.out_prefix}_diag.json").write_text(json.dumps({
                "args": vars(args), "n": n, "cap": cap, "softening": soft,
                "rows": rows, "step_times": [round(t, 4) for t in step_times],
                "probe": probe_stats,
                "partial": True, "num_repartitions": n_reparts,
            }, indent=1))
            print(
                f"  step {it:>6d}  t={row['t']:.4f}  {row['seconds']:7.2f}s  "
                f"dL/L={row['dL_over_L']:.3e}  com={row['com_drift']:.3e}  KE={ke:.6e}"
                + (f"  fs=[{row['force_scale_min']:.4g},{row['force_scale_max']:.4g}]"
                   if "force_scale_min" in row else ""),
                flush=True,
            )

    total = time.perf_counter() - t_run0
    med = float(np.median(step_times)) if step_times else float("nan")
    print(f"# {args.steps} steps in {total:.1f} s, median {med:.3f} s/step "
          f"({n / max(med, 1e-9):,.0f} particles/s)", flush=True)

    out = pathlib.Path(f"{args.out_prefix}_diag.json")
    out.write_text(json.dumps({
        "args": vars(args), "n": n, "cap": cap, "softening": soft,
        "median_step_s": med, "total_s": total, "first_diag": diag0,
        "rows": rows, "step_times": [round(t, 4) for t in step_times],
        "probe": probe_stats, "probe_history": probe_history, "num_repartitions": n_reparts,
        "wall_total_s": time.perf_counter() - t_wall0,
    }, indent=1))
    print(f"# wrote {out}", flush=True)

    save_state(args.steps, final=True)
    flush_frames()
    for nm, fr in frames.items():
        if fr:
            print(f"# wrote {args.out_prefix}_frames_{nm}.npz  ({len(fr)} frames)",
                  flush=True)


if __name__ == "__main__":
    main()
