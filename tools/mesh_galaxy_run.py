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


def make_aligner(mesh, rank_in):
    """Build a device-local realignment of the evaluator's output to input order.

    The frozen partition means device ``d`` owns a fixed set of global ids and the
    tree build only permutes within ``d``, so sorting by the returned gid is a
    device-local operation and the whole thing lives inside one ``shard_map`` with
    no collective.

    Parameters
    ----------
    mesh : Any
        The device mesh, axis ``"gpus"``.
    rank_in : Any
        ``[ndev * cap]`` int32. For input row ``i``, the rank of its global id among
        the ids its own device owns. Precomputed on the host; the input order is
        frozen for the whole rollout.

    Returns
    -------
    Callable
        ``(values, gid_out) -> values`` in input row order.
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
    return jax.jit(lambda v, g: fn(v, g, rank_in))


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
    ap.add_argument("--render-every", type=int, default=0, help="0 = no movie")
    ap.add_argument("--render-res", type=int, default=800)
    ap.add_argument("--render-extent", type=float, default=1.2)
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
    state0 = np.asarray(ic["state0"])
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
        f"# dt={args.dt} steps={args.steps} devices={[d.device_kind for d in jax.devices()][:args.ndev]}",
        flush=True,
    )

    cfg = DistributedFMMConfig(
        leaf_size=args.leaf,
        theta=args.theta,
        order=args.order,
        softening=soft,
        G=g,
        m2l_chunk=args.m2l_chunk,
        nearfield_chunk=args.nearfield_chunk,
    )
    mesh = make_mesh(args.ndev)
    part = partition_for_devices(
        pos, mass, args.ndev, leaf_size=args.leaf, partitioner=cfg.partitioner
    )
    cap = part["cap"]
    if cap * args.ndev != n:
        raise SystemExit(f"padding present: cap={cap} ndev={args.ndev} n={n}")
    cfg_r = cfg.resolved_for(cap, args.ndev)

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

    use_halo = (not args.no_halo) and halo_m > 0.0

    align = make_aligner(mesh, jnp.asarray(rank_in))

    def force(x):
        a_raw, gid_o, diag = evaluate(x, M, GID, COUNTS)
        a = align(a_raw, gid_o)          # -> input row order, device-local
        if use_halo:
            a = a + nfw_acceleration(x, halo_m, halo_rs, g)
        return a, gid_o, diag, a_raw

    # first evaluation, eagerly, so the padding guard and the overflow flags are
    # readable -- both are invisible once this is inside a jitted step.
    t0 = time.perf_counter()
    A, gid_o, diag, A_raw = force(X)
    jax.block_until_ready(A)
    print(f"# first force (compile incl.) {time.perf_counter() - t0:.1f} s", flush=True)
    t0 = time.perf_counter()
    # Verify the PERMUTATION, before any arithmetic touches it. A permutation is
    # exact, so array_equal is the right test here -- but only on a value that has
    # not been through an add: subtracting the halo term back off reintroduces
    # float32 rounding and turns an exact check into a ~1e-6 mismatch on 80 % of
    # rows, which looks like a broken map and is not one.
    _verify_alignment(align(A_raw, gid_o), A_raw, gid_o, GID, n)
    print(f"# realignment verified against scatter_to_input_order "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
    d = np.asarray(diag)
    names = [
        "cross_far_pairs", "cross_near_pairs", "cross_queue_overflow",
        "cross_far_overflow", "cross_near_overflow", "self_far_pairs",
        "self_near_pairs", "self_queue_overflow", "self_far_overflow",
        "self_near_overflow", "l2l_level_overflow",
    ]
    diag0 = {k: [float(v) for v in d[:, i]] for i, k in enumerate(names) if i < d.shape[1]}
    ovf = {k: sum(v) for k, v in diag0.items() if k.endswith("overflow")}
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
        an, _, _, _ = force(xn)
        return xn, kick(vh, an, dt), an

    @jax.jit
    def invariants(x, v, m):
        p = jnp.sum(m[:, None] * v, axis=0)
        l = jnp.sum(m[:, None] * jnp.cross(x, v), axis=0)
        com = jnp.sum(m[:, None] * x, axis=0) / jnp.sum(m)
        ke = 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1))
        return p, l, com, ke

    project = None
    frames = []
    if args.render_every > 0:
        ext = args.render_extent
        project = make_density_projector(
            mesh, args.render_res, (0, 1), (-ext, -ext), (ext, ext)
        )
        frames.append(np.asarray(project(X, M)))

    p0, l0, com0, ke0 = [np.asarray(z) for z in invariants(X, V, M)]
    mtot = float(np.sum(mass))
    lscale = float(np.sum(mass * np.linalg.norm(np.cross(pos, vel), axis=1)))
    print(f"# t=0  |L|={np.linalg.norm(l0):.6e}  KE={ke0:.6e}  com={np.round(com0, 6)}", flush=True)

    rows = []
    step_times = []
    t_run0 = time.perf_counter()
    for it in range(1, args.steps + 1):
        t0 = time.perf_counter()
        X, V, A = kdk(X, V, A, args.dt)
        jax.block_until_ready(X)
        step_times.append(time.perf_counter() - t0)

        if args.render_every and it % args.render_every == 0:
            frames.append(np.asarray(project(X, M)))
            # Flush every 10 frames. A long run on shared cards can be stopped at
            # any moment, and a movie that only exists in RAM at step N is a movie
            # that does not exist.
            if len(frames) % 10 == 0:
                np.savez_compressed(f"{args.out_prefix}_frames.npz",
                                    frames=np.stack(frames))
        if args.max_hours > 0 and (time.perf_counter() - t_run0) > args.max_hours * 3600:
            print(f"# wall-clock budget reached after {it} steps -- stopping cleanly", flush=True)
            args.steps = it
            if args.render_every:
                frames.append(np.asarray(project(X, M)))
            break
        if args.diag_every and it % args.diag_every == 0:
            p, l, com, ke = [np.asarray(z) for z in invariants(X, V, M)]
            row = {
                "step": it,
                "t": it * args.dt,
                "seconds": round(step_times[-1], 4),
                "dP_over_scale": float(np.linalg.norm(p - p0) / (mtot + 1e-300)),
                "dL_over_L": float(np.linalg.norm(l - l0) / (lscale + 1e-300)),
                "com_drift": float(np.linalg.norm(com - com0)),
                "ke": float(ke),
            }
            rows.append(row)
            pathlib.Path(f"{args.out_prefix}_diag.json").write_text(json.dumps({
                "args": vars(args), "n": n, "cap": cap, "softening": soft,
                "rows": rows, "step_times": [round(t, 4) for t in step_times],
                "partial": True,
            }, indent=1))
            print(
                f"  step {it:>6d}  t={row['t']:.4f}  {row['seconds']:7.2f}s  "
                f"dL/L={row['dL_over_L']:.3e}  com={row['com_drift']:.3e}  KE={ke:.6e}",
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
        "wall_total_s": time.perf_counter() - t_wall0,
    }, indent=1))
    print(f"# wrote {out}", flush=True)

    if frames:
        np.savez_compressed(f"{args.out_prefix}_frames.npz", frames=np.stack(frames))
        print(f"# wrote {args.out_prefix}_frames.npz  ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
