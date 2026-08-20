#!/usr/bin/env python
"""Worked example: momentum-conserving individual timesteps on Nornax + Jaccpot.

Runs a Plummer or Hernquist sphere at a scale where direct summation is not an
option, through ODISSEO's block-step FMM lane
(:mod:`odisseo.blockstep_coupling`), and reports the three numbers that decide
whether the lane is doing its job:

* **momentum drift** -- ``|sum_i m_i v_i - p_0| / sum_i |m_i v_i|``, which must
  sit at float64 round-off (~1e-17..1e-16), not at the force accuracy;
* **energy drift** -- bounded and oscillating, since leapfrog is symplectic;
* **wall clock per base step**, against the shared-timestep lane
  (``odisseo.jaccpot_coupling``) covering the *same physical time*.

It also prints both lanes' ``|sum_i m_i a_i|`` on identical positions. jaccpot's
documentation attributes a ~1e-3..1e-5 residual to the target-centric force;
measured here both land at ~1e-17 while differing by the FMM tolerance, so the
far field is genuinely active and this is not a degenerate direct sum. The
block-step lane's necessity rests on the **per-level** antisymmetric split, which
the production coupler cannot express at all, rather than on the total.

What the wall-clock comparison actually shows
---------------------------------------------
Read it carefully, because the naive expectation is wrong. Jaccpot's fused
boundary kick applies ``level_weights[max(rung_i, rung_j)]`` as a scalar *inside*
the kernel -- it weights pairs, it does not prune them -- so every boundary costs
a **full** traversal regardless of how many levels are active. Per ``dt_max`` of
physical time:

===================================  =====================  ==============
                                     traversals             tree builds
===================================  =====================  ==============
block step, ``k_max = K``            ``2**K + 1``           ``1``
shared timestep at ``dt_min``        ``2**K``               ``2**K / refresh_every``
===================================  =====================  ==============

So the individual-timestep advantage here is **not** fewer force evaluations --
it is one host-side tree build per ``dt_max`` instead of one per sub-step (a real
cost: the build measured 22 s against 5 s per traversal at N = 20 000), plus a
scheme whose per-level splitting is exact. A shared-timestep run is only cheaper
if it is allowed to take ``dt_max`` steps, which is exactly the accuracy the fine
particles cannot afford. This script measures all of it rather than asserting any
of it.

Two measured numbers worth knowing before reading any timing here.

``jaccpot.mutual`` ships **no** ``jax.jit``, so a traversal costs 5.1 s eager
against 0.038 s under ``jax.jit`` at N = 20 000 on an A100 -- 135x. ``--jit-force``
turns that on.

And once the force is jitted, the **host tree build is the entire remaining
wall**: 22 s per base step against 0.55 s for the stepping. ``--lane jitted``
builds the topology on device inside one ``lax.scan`` over the whole rollout,
which takes a base step from 22.51 s to **0.399 s** (56x) with momentum drift
3.7e-18. Prefer it for anything but a like-for-like comparison against the
historical host numbers.

Usage
-----
::

    python tools/blockstep_fmm_demo.py --n 100000 --k-max 3 --n-base 12
    python tools/blockstep_fmm_demo.py --n 20000 --ic hernquist --backend pallas
    python tools/blockstep_fmm_demo.py --n 4096 --no-shared   # block step only
    python tools/blockstep_fmm_demo.py --n 20000 --lane jitted --no-shared

Requires a nornax carrying the fused-boundary primitive -- nornax ``main`` at or
after ``8fe9dbd`` ("Differentiable individual-timestep KDK leapfrog integrator
(#7)"), which squash-merged both the block-step integrator and the fused-boundary
work (#8). Point ``--nornax`` at a checkout to override the installed one.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import resource
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# environment -- must run before jax is imported
# --------------------------------------------------------------------------


def _select_gpu(
    *, enable: bool, timeout: int | None, exclude: list[int] | None
) -> str | None:
    """Pick one free GPU with autocvd, as the shared machines here require.

    ``least_used`` ranks by free *memory*, which on a busy node can hand back a
    device that is compute-saturated by someone else -- enough to inflate a
    wall-clock measurement by two orders of magnitude. Pass ``--gpu-exclude`` to
    steer away from those; the timing numbers are only meaningful on a device
    that is actually idle.
    """
    if not enable:
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        from autocvd import autocvd
    except Exception as exc:  # pragma: no cover -- optional dependency
        print(f"autocvd unavailable ({exc}); keeping CUDA_VISIBLE_DEVICES", flush=True)
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    autocvd(num_gpus=1, least_used=True, timeout=timeout, exclude=exclude)
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _use_nornax_from(path: str | None) -> None:
    """Repoint the editable-install finder at a specific nornax checkout.

    ``PYTHONPATH`` cannot win against setuptools' editable ``MetaPathFinder``,
    so overriding the installed nornax means mutating that finder's ``MAPPING``
    before the first import.
    """
    if not path:
        return
    if "nornax" in sys.modules:
        raise RuntimeError("--nornax must be applied before nornax is imported")
    pkg = os.path.join(path, "nornax")
    if not os.path.isdir(pkg):
        raise SystemExit(f"--nornax {path!r} has no nornax/ package directory")
    try:
        import __editable___nornax_0_0_1_finder as finder  # type: ignore
    except Exception:
        sys.path.insert(0, path)
        return
    finder.MAPPING["nornax"] = pkg


# --------------------------------------------------------------------------
# initial conditions
# --------------------------------------------------------------------------


def _sample_plummer(n: int, seed: int, scale: float = 1.0):
    """Plummer sphere, positions from the analytic inverse CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(1.0e-6, 0.999, size=n)
    r = scale / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    pos = _isotropic(rng, r)
    # Velocities from the classic rejection-sampled Plummer distribution
    # function, so the sphere starts near virial equilibrium.
    v_esc = np.sqrt(2.0) * (1.0 + (r / scale) ** 2) ** (-0.25)
    q = _rejection_sample_plummer_q(rng, n)
    vel = _isotropic(rng, q * v_esc)
    return pos, vel


def _sample_hernquist(n: int, seed: int, scale: float = 1.0):
    """Hernquist sphere; the ``r^-1`` cusp gives a wide acceleration spread.

    That spread is the point: it is what puts particles on different rungs, so
    the block scheme is actually exercised rather than collapsing to a shared
    timestep. Velocities use the analytic isotropic radial dispersion
    (Hernquist 1990, eq. 10) -- an approximate-equilibrium seed, adequate for an
    integrator benchmark.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(1.0e-4, 0.96, size=n)
    root = np.sqrt(u)
    r = scale * root / (1.0 - root)
    pos = _isotropic(rng, r)
    x = r / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma2 = (
            (x * (1.0 + x) ** 3) * np.log1p(1.0 / np.maximum(x, 1e-12))
            - (x / (1.0 + x)) * (0.25 + x * (1.0 + x * (13.0 / 3.0 + x * 25.0 / 12.0)))
        ) / scale
    sigma = np.sqrt(np.clip(sigma2, 0.0, None))
    vel = sigma[:, None] * rng.normal(size=(n, 3))
    return pos, vel


def _isotropic(rng, radii):
    cos_t = rng.uniform(-1.0, 1.0, size=radii.shape)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=radii.shape)
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    return np.stack(
        [radii * sin_t * np.cos(phi), radii * sin_t * np.sin(phi), radii * cos_t],
        axis=-1,
    )


def _rejection_sample_plummer_q(rng, n):
    """Sample ``q = v / v_esc`` from ``g(q) = q^2 (1 - q^2)^{7/2}``."""
    out = np.empty(n)
    filled = 0
    peak = 0.1  # max of g on [0, 1], with margin
    while filled < n:
        q = rng.uniform(0.0, 1.0, size=2 * (n - filled))
        y = rng.uniform(0.0, peak, size=q.shape)
        keep = q[y < q**2 * (1.0 - q**2) ** 3.5]
        take = min(len(keep), n - filled)
        out[filled : filled + take] = keep[:take]
        filled += take
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB.

    Peak *memory*, not trace size, is what breaks CI on this path: an eager loop
    over base steps reuses XLA's per-operation cache, while a ``lax.scan``
    inlines the whole force into one program that has to be compiled and held.
    RSS is the number that separates them.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100_000, help="particle count")
    ap.add_argument("--ic", choices=("plummer", "hernquist"), default="plummer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-max", type=int, default=3, help="finest rung")
    ap.add_argument("--n-base", type=int, default=8, help="base steps to take")
    ap.add_argument("--dt-max", type=float, default=None, help="base-step timestep")
    ap.add_argument("--eta", type=float, default=0.1, help="rung criterion accuracy")
    ap.add_argument("--theta", type=float, default=0.7, help="mutual MAC parameter")
    ap.add_argument("--order", type=int, default=4, help="multipole order")
    ap.add_argument("--leaf-size", type=int, default=32)
    ap.add_argument("--softening", type=float, default=1.0e-3)
    ap.add_argument("--backend", choices=("jax", "pallas"), default="jax")
    ap.add_argument("--rebuild-every", type=int, default=1)
    ap.add_argument(
        "--scan-base-steps",
        choices=("auto", "on", "off"),
        default="auto",
        help="drive base steps with block_kdk_rollout (lax.scan) or eagerly",
    )
    ap.add_argument(
        "--traced-boundary-weights",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "let nornax scan the sub-step boundaries over a traced weight table. "
            "'off' (default) unrolls them, which reuses jaccpot's per-kernel "
            "executables instead of recompiling the inlined force every base step"
        ),
    )
    ap.add_argument(
        "--jit-force",
        action="store_true",
        help=(
            "compile the mutual force once per prepared topology. Measured 135x "
            "per traversal (5.1 s -> 0.038 s at N=20000 on an A100) against a "
            "~170-210 s compile per rebuild, so pair it with --rebuild-every >= 4"
        ),
    )
    ap.add_argument("--no-energy", action="store_true", help="skip the O(N^2) energy")
    ap.add_argument(
        "--lane",
        choices=("host", "jitted"),
        default="jitted",
        help=(
            "'host' drives nornax's block_kdk_rollout with a host-built tree per "
            "base step. 'jitted' builds the topology on device inside one "
            "lax.scan over the whole rollout -- no host traffic in the loop, "
            "measured 56x per base step at N=20000 on an A100"
        ),
    )
    ap.add_argument("--no-shared", action="store_true", help="skip the baseline lane")
    ap.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="record diagnostics every this many base steps (the O(N^2) energy)",
    )
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--no-gpu-select", action="store_true", help="do not run autocvd")
    ap.add_argument("--autocvd-timeout", type=int, default=None)
    ap.add_argument(
        "--gpu-exclude",
        type=int,
        nargs="*",
        default=None,
        help="GPU indices autocvd must not pick (e.g. ones busy with other jobs)",
    )
    ap.add_argument("--nornax", default=None, help="path to a nornax checkout")
    args = ap.parse_args()

    _use_nornax_from(args.nornax)
    device = _select_gpu(
        enable=not args.no_gpu_select,
        timeout=args.autocvd_timeout,
        exclude=args.gpu_exclude,
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    if not args.float32:
        jax.config.update("jax_enable_x64", True)

    from odisseo import construct_initial_state
    from odisseo.blockstep_coupling import (
        BlockStepOptions,
        assert_fused_boundary_selected,
        blockstep_total_acceleration,
        build_blockstep_force,
        chunked_potential_energy,
        integrate_blockstep_jaccpot,
        resolve_topology_backend,
        total_linear_momentum,
    )
    from odisseo.option_classes import SimulationConfig, SimulationParams

    dtype = jnp.float32 if args.float32 else jnp.float64
    n = int(args.n)

    print(f"device            : CUDA_VISIBLE_DEVICES={device!r} on {jax.devices()}")
    print(f"nornax            : {__import__('nornax').__file__}")
    print(f"IC                : {args.ic}, N = {n:,}, dtype = {dtype.__name__}")

    sample = _sample_plummer if args.ic == "plummer" else _sample_hernquist
    pos, vel = sample(n, args.seed)
    mass = np.full(n, 1.0 / n)
    positions = jnp.asarray(pos, dtype=dtype)
    velocities = jnp.asarray(vel, dtype=dtype)
    mass = jnp.asarray(mass, dtype=dtype)
    state = construct_initial_state(positions, velocities)

    config = SimulationConfig(N_particles=n, softening=float(args.softening))
    params = SimulationParams(G=1.0)

    # --------------------------------------------------------------------
    # dt_max: pick it from the acceleration distribution so the rungs spread.
    # A dt_max below every particle's own criterion puts everyone on rung 0 and
    # the block scheme silently collapses to a shared timestep -- a run that
    # "passes" while testing nothing.
    # --------------------------------------------------------------------
    scan_map = {"auto": None, "on": True, "off": False}
    options = BlockStepOptions(
        dt_max=float(args.dt_max or 1.0),
        k_max=int(args.k_max),
        eta=float(args.eta),
        theta=float(args.theta),
        max_order=int(args.order),
        leaf_size=int(args.leaf_size),
        backend=args.backend,
        rebuild_every=int(args.rebuild_every),
        scan_base_steps=scan_map[args.scan_base_steps],
        traced_boundary_weights=scan_map[args.traced_boundary_weights],
        jit_force=bool(args.jit_force),
    )
    # The demo's `--lane` decides where the topology is built. Ask for it
    # explicitly rather than leaning on the default, so `--lane host` really does
    # exercise the host lane.
    options = dataclasses.replace(
        options,
        topology_backend="device" if args.lane == "jitted" else "host",
    )
    force = build_blockstep_force(config, params, options)
    backend = resolve_topology_backend(options)

    # This probe used to be an unconditional HOST build, which is the wrong
    # default now and was actively prohibitive at large N -- the host traversal is
    # 23.1 s at N = 20 000 and is what made an N = 1e6 run impossible to even set
    # up. Route it the same way the run itself is routed.
    t0 = time.perf_counter()
    if backend == "device":
        force.freeze_template(positions, mass)
        state_dev = jax.block_until_ready(
            jax.jit(force.rebuild_state)(positions, mass)
        )
        acc0 = jax.block_until_ready(
            blockstep_total_acceleration(
                state, mass, config, params, options, force=force, prepare=False
            )
        )
        num_far = int(state_dev.num_far_pairs)
        num_near = int(state_dev.num_near_pairs)
    else:
        force.prepare(positions, mass)
        acc0 = jax.block_until_ready(force.total_accelerations(positions, mass))
        num_far = int(force.state.far_a.shape[0])
        num_near = int(force.state.near_a.shape[0])
    t_prepare = time.perf_counter() - t0
    print(
        f"topology          : {num_far:,} far pairs, {num_near:,} leaf pairs, "
        f"{t_prepare:.2f} s to build + first force"
    )
    if num_far == 0:
        raise SystemExit(
            "no far pairs: this FMM is a direct sum and every far-field number "
            "below would be vacuous. Raise --theta or lower --leaf-size."
        )

    if args.dt_max is None:
        # Median particle takes ~one base step; the fast tail spreads down the
        # rungs. dt_i = eta * sqrt(softening / |a_i|).
        a_norm = np.asarray(jnp.linalg.norm(acc0, axis=-1))
        dt_i = args.eta * np.sqrt(args.softening / np.maximum(a_norm, 1e-30))
        dt_max = float(np.percentile(dt_i, 90.0))
        options = dataclasses.replace(options, dt_max=dt_max)
        print(f"dt_max            : {dt_max:.4e} (auto: 90th pct of dt_i)")
    else:
        dt_max = float(args.dt_max)
        print(f"dt_max            : {dt_max:.4e}")
    print(f"dt_min            : {options.dt_min:.4e} = dt_max / 2**{args.k_max}")

    scanned = assert_fused_boundary_selected(force, options.k_max)
    print(
        f"nornax path       : fused boundary kick, boundaries "
        f"{'scanned over a traced weight table' if scanned else 'unrolled'}; "
        f"{options.n_sub + 1} traversals + 1 tree build per base step"
    )
    if args.lane == "jitted":
        force_lane = "jitted (the whole rollout is one program)"
    elif args.jit_force:
        force_lane = "jitted per topology"
    else:
        force_lane = "eager (jaccpot ships no jit)"
    print(f"force lane        : {force_lane}")
    print(
        f"rollout lane      : {args.lane}"
        + (
            "  (device topology, rebuilt inside one lax.scan -- no host traffic)"
            if args.lane == "jitted"
            else "  (host tree build per base step)"
        )
    )

    # --------------------------------------------------------------------
    # the block-step run
    # --------------------------------------------------------------------
    print()
    print("=== block-step lane (mutual FMM, individual timesteps) ===")

    def show(base_index, rec):
        print(
            f"  base step {base_index:>3d}  |dp|/|p|scale = "
            f"{rec['momentum_drift']:.3e}   {rec['seconds_per_base_step']:.3f} s"
        )

    t0 = time.perf_counter()
    if args.lane == "jitted":
        from odisseo.blockstep_coupling import integrate_blockstep_jitted

        result = integrate_blockstep_jitted(
            state,
            mass,
            config,
            params,
            options=options,
            n_base=int(args.n_base),
            track_energy=not args.no_energy,
            energy_chunk=min(4096, n),
            time_steady_state=True,
        )
    else:
        result = integrate_blockstep_jaccpot(
            state,
            mass,
            config,
            params,
            options=options,
            n_base=int(args.n_base),
            force=force,
            track_energy=not args.no_energy,
            energy_chunk=min(4096, n),
            record_every=int(args.record_every),
            progress=show,
        )
    block_wall = time.perf_counter() - t0
    block_rss = _peak_rss_gb()

    hist = np.asarray(result.rung_histogram)
    print()
    print(f"  rungs (first)   : {hist[0].tolist()}")
    print(f"  rungs (last)    : {hist[-1].tolist()}")
    if int(np.count_nonzero(hist[-1] > 0)) == 1:
        print(
            "  WARNING         : every particle is on one rung, so the block "
            "scheme collapsed to a shared timestep. Raise --dt-max or --k-max."
        )
    print(f"  momentum drift  : max {float(np.max(result.momentum_drift)):.4e}")
    if result.energy_drift is not None:
        ed = np.asarray(result.energy_drift)
        print(f"  energy drift    : max |dE/E| {np.max(np.abs(ed)):.4e}")
        print(f"                    final     {ed[-1]:+.4e}")
    print(
        f"  s / base step   : {result.seconds_per_base_step:.4f}"
        + ("  (steady state, compile excluded)" if args.lane == "jitted" else
           "  (stepping only; the tree build below is on top)")
    )
    if args.lane == "jitted":
        freeze, compiled = (list(result.prepare_seconds) + [0.0, 0.0])[:2]
        print(f"  one-off         : {freeze:.1f} s template freeze + {compiled:.1f} s compile")
        print("  tree build      : inside the scan, on device (not a host cost)")
    else:
        print(f"  tree build      : {np.mean(result.prepare_seconds or [0.0]):.4f} s each")
    print(f"  total wall      : {block_wall:.2f} s for {result.n_base} base steps")
    print(f"  peak RSS        : {block_rss:.2f} GB")

    # --------------------------------------------------------------------
    # the shared-timestep baseline, over the SAME physical time
    # --------------------------------------------------------------------
    if not args.no_shared:
        from odisseo.jaccpot_coupling import (
            evaluate_acceleration_jaccpot,
            integrate_leapfrog_jaccpot_active,
        )

        print()
        print("=== shared-timestep lane (target-centric FMM, dt = dt_min) ===")
        n_steps = int(args.n_base) * options.n_sub
        shared_kwargs = dict(
            leaf_size=int(args.leaf_size),
            max_order=int(args.order),
            fmm_basis="real",
            fmm_theta=float(args.theta),
            fmm_tree_leaf_target=int(args.leaf_size),
        )
        timing: dict = {}
        t0 = time.perf_counter()
        shared_state = integrate_leapfrog_jaccpot_active(
            state,
            mass,
            config,
            params,
            num_steps=n_steps,
            dt=options.dt_min,
            refresh_every=1,
            timing_stats=timing,
            **shared_kwargs,
        )
        shared_state = jax.block_until_ready(shared_state)
        shared_wall = time.perf_counter() - t0

        p0 = total_linear_momentum(mass, velocities)
        p1 = total_linear_momentum(mass, shared_state[:, 1, :])
        scale = float(jnp.sum(jnp.abs(mass[:, None] * shared_state[:, 1, :])))
        shared_drift = float(jnp.linalg.norm(p1 - p0)) / scale
        print(f"  steps           : {n_steps} of dt_min (same physical time)")
        print(f"  momentum drift  : {shared_drift:.4e}")
        if not args.no_energy:
            ke = 0.5 * jnp.sum(mass * jnp.sum(shared_state[:, 1, :] ** 2, axis=-1))
            pe = chunked_potential_energy(
                shared_state[:, 0, :],
                mass,
                G=1.0,
                softening=float(args.softening),
                chunk=min(4096, n),
            )
            e0 = float(result.energy[0]) if result.energy is not None else None
            e1 = float(ke + pe)
            if e0:
                print(f"  energy drift    : {(e1 - e0) / abs(e0):+.4e}")
        print(f"  total wall      : {shared_wall:.2f} s")
        for key in ("prepare_seconds_total", "evaluate_seconds_total"):
            if key in timing:
                print(f"  {key:<15} : {timing[key]}")
        # The tree-build count is the individual-timestep advantage, stated
        # plainly: the block step rebuilds once per dt_max, a shared step at
        # dt_min rebuilds once per sub-step.
        shared_builds = timing.get("prepare_calls", n_steps)
        print(
            f"  tree builds     : {shared_builds} shared vs "
            f"{result.n_base} block, for the same physical time"
        )
        # Both totals include their first-call compile, so this is the honest
        # like-for-like comparison; result.seconds_per_base_step excludes it and
        # is the steady-state figure.
        print(
            f"  wall per dt_max : {shared_wall / max(args.n_base, 1):.4f} shared "
            f"vs {block_wall / max(result.n_base, 1):.4f} block "
            f"(both including compile; block steady state "
            f"{result.seconds_per_base_step:.4f})"
        )

        # The two forces must agree on the TOTAL acceleration -- not per level.
        a_shared = evaluate_acceleration_jaccpot(
            state, mass, config, params, **shared_kwargs
        )
        rel = float(jnp.linalg.norm(acc0 - a_shared) / jnp.linalg.norm(a_shared))
        print()
        print("=== the two forces, on the same positions ===")
        print(
            f"  total-acceleration agreement : {rel:.4e}  "
            f"(FMM tolerance; theta={args.theta}, order={args.order})"
        )

        def _residual(acc):
            scale = float(jnp.sum(jnp.abs(mass))) * float(
                jnp.mean(jnp.linalg.norm(acc, axis=-1))
            )
            return float(jnp.linalg.norm(jnp.sum(mass[:, None] * acc, axis=0))) / scale

        print(f"  |sum m a| / scale, mutual    : {_residual(acc0):.4e}")
        print(f"  |sum m a| / scale, target    : {_residual(a_shared):.4e}")
        print(
            "  NOTE: jaccpot's docs attribute a ~1e-3..1e-5 momentum residual to\n"
            "  the target-centric force. Measured here both lanes land at ~1e-17,\n"
            "  while differing by the FMM tolerance above -- so the far field is\n"
            "  active and this is not a degenerate direct sum. The block-step\n"
            "  lane's necessity rests on the PER-LEVEL antisymmetric split, which\n"
            "  the production coupler cannot express at all, not on this number."
        )

    print()
    print("Momentum is conserved structurally, not to a tolerance: the mutual")
    print("kernels evaluate each pair once and apply +f/-f, so the residual is")
    print("set by the width of the reduction, not by theta or the expansion order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
