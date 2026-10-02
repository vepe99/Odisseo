#!/usr/bin/env python
"""T3.0 -- jaccpot's production per-STEP cost (refresh + evaluate) via strict_run_v2.

``compare_force.py`` times ``eval_fn`` alone, which excludes the MAC walk that
pkdgrav3's ``gravity()`` includes.  jaccpot's production loop is
``strict_run_v2``: a device-resident velocity-Verlet scan that refreshes the
prepared state every step.  Its per-step wall time, and the ``refresh_*``
stage timers it accumulates, are the honest counterpart of pkdgrav3's
``domain_decompose + build_tree + gravity`` per step.

Runs one warm-up call (compile), then ``--steps`` steps in one call and
``--steps`` more in a second call, and reports wall per step plus the timer
deltas per step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import GpuMonitor, pick_idle_gpus, set_cuda_visible  # noqa: E402
from compare_force import apply_fast_lane_env  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402

STAGE_KEYS = [
    "refresh_total_seconds", "refresh_input_seconds", "refresh_tree_build_seconds",
    "refresh_tree_upward_seconds", "refresh_upward_compute_seconds",
    "refresh_dual_downward_seconds", "refresh_dual_setup_seconds",
    "refresh_dual_artifact_build_seconds", "refresh_dual_select_interactions_seconds",
    "refresh_dual_far_pair_plan_seconds", "refresh_dual_m2l_compute_seconds",
    "refresh_dual_l2l_compute_seconds", "refresh_nearfield_seconds",
    "refresh_nearfield_radix_payload_seconds", "refresh_evaluate_seconds",
    "refresh_compile_or_sync_suspect_seconds",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=256)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.6, 1.0])
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--vel-sigma", type=float, default=0.4,
                    help="isotropic Gaussian velocity dispersion (G=M=a=1 Plummer: ~0.4); "
                         "with --dt 0.01 particles move ~0.4 % of the scale radius per step, so the "
                         "refresh genuinely re-walks; 0 = frozen positions (motion-gated regime)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1)
    set_cuda_visible(devices)
    apply_fast_lane_env(args.n)

    import jax
    import jax.numpy as jnp
    from jaccpot import (FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig,
                         NearFieldConfig, TreeConfig)

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    M = jnp.asarray(mass, jnp.float32)
    vel = np.random.default_rng(7).normal(0.0, args.vel_sigma, (args.n, 3)).astype(np.float32)
    state0 = jnp.stack([jnp.asarray(pos, jnp.float32), jnp.asarray(vel)], axis=1)
    results = []
    for theta in args.thetas:
        s = FastMultipoleMethod(
            preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta,
            G=1.0, softening=1e-7, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                                       farfield=FarFieldConfig(mode="auto"),
                                       nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
            fixed_order=args.order)

        def snap():
            d = s.get_runtime_diagnostics() or {}
            return {k: float(d.get(k) or 0.0) for k in STAGE_KEYS}

        def run(state, prepared, steps):
            t0 = time.perf_counter()
            out = s.strict_run_v2(state=state, masses=M, dt=args.dt, num_steps=steps,
                                  refresh_every=1, leaf_size=args.leaf, max_order=args.order,
                                  theta=theta, prepared_state=prepared, return_prepared_state=True)
            jax.block_until_ready(out[0])
            return out, time.perf_counter() - t0

        # warm-up: prepare + compile
        (state, prepared, _), t_cold = run(state0, None, 1)
        d0 = s.get_runtime_diagnostics() or {}
        print(f"\ntheta {theta}: cold 1-step call {t_cold:.2f} s (prepare + compile); "
              f"compile_count={d0.get('strict_runner_compile_count')}", flush=True)
        calls = []
        with GpuMonitor(devices) as mon:
            for rep in range(3):
                before = snap()
                (state, prepared, _), wall = run(state, prepared, args.steps)
                after = snap()
                delta = {k: (after[k] - before[k]) / args.steps for k in STAGE_KEYS}
                calls.append(dict(rep=rep, steps=args.steps, wall_per_step_s=wall / args.steps, **delta))
        cont = mon.summary()
        d = s.get_runtime_diagnostics() or {}
        print(f"{'rep':>3} {'ms/step':>8} {'refresh':>8} {'tree+up':>8} {'dual':>7} {'walk':>7} "
              f"{'m2l':>6} {'near':>7} {'eval':>7} {'unattr':>7}   flags={cont.flags or '-'}")
        for c in calls:
            walk = c["refresh_dual_select_interactions_seconds"] + c["refresh_dual_artifact_build_seconds"] \
                + c["refresh_dual_setup_seconds"]
            print(f"{c['rep']:>3} {c['wall_per_step_s']*1e3:>8.1f} {c['refresh_total_seconds']*1e3:>8.1f} "
                  f"{c['refresh_tree_upward_seconds']*1e3:>8.1f} {c['refresh_dual_downward_seconds']*1e3:>7.1f} "
                  f"{walk*1e3:>7.1f} {c['refresh_dual_m2l_compute_seconds']*1e3:>6.1f} "
                  f"{c['refresh_nearfield_seconds']*1e3:>7.1f} {c['refresh_evaluate_seconds']*1e3:>7.1f} "
                  f"{c['refresh_compile_or_sync_suspect_seconds']*1e3:>7.1f}")
        info = dict(theta=theta, leaf=args.leaf, order=args.order, n=args.n, dt=args.dt,
                    vel_sigma=args.vel_sigma,
                    cold_s=t_cold, calls=calls, contention=cont.as_dict(),
                    counters={k: d.get(k) for k in (
                        "strict_runner_compile_count", "strict_runner_execute_count",
                        "strict_fused_fastlane_hits", "strict_fused_fastlane_attempts",
                        "strict_fused_fallback_count", "compiled_profile_refresh_reuse_tier_full",
                        "refresh_dual_planner_compile_count", "refresh_dual_planner_execute_count",
                        "recent_dual_neighbor_count")})
        print("  counters:", info["counters"])
        results.append(info)
        del s, prepared

    out = Path(args.out or (ROOT / "artifacts" / f"probe_step_plummer{args.n}_leaf{args.leaf}_v{args.vel_sigma:g}_dt{args.dt:g}.json"))
    out.write_text(json.dumps(dict(devices=devices, results=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
