#!/usr/bin/env python
"""T3.0 -- what does jaccpot's MAC walk cost per step, in production form?

``eval_fn`` (what ``compare_force.py`` times) excludes the walk that builds the
interaction lists; pkdgrav3's ``gravity()`` includes its walk.  jaccpot's per-step
production path is ``strict_prepare_refresh_and_evaluate`` (prepare once, then
refresh + evaluate every step), and only that path records the stage timers
(``refresh_*_seconds``; ``prepare_state`` on its own records nothing).

This probe runs K refresh+evaluate steps on slightly displaced positions (so the
walk is genuinely redone) and reports, per step, the wall time and the timer
deltas: input, tree+upward, dual (walk + far-field plan + M2L), near-field payload,
evaluate, and the unattributed remainder.  ``dual`` + ``nearfield`` is the
pkdgrav3-``gravity()``-equivalent addition to ``eval_fn``.
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
    "refresh_input_seconds",
    "refresh_tree_build_seconds",
    "refresh_tree_upward_seconds",
    "refresh_upward_compute_seconds",
    "refresh_dual_downward_seconds",
    "refresh_dual_setup_seconds",
    "refresh_dual_artifact_build_seconds",
    "refresh_dual_select_interactions_seconds",
    "refresh_dual_far_pair_plan_seconds",
    "refresh_dual_split_combined_seconds",
    "refresh_dual_raw_combined_seconds",
    "refresh_dual_m2l_compute_seconds",
    "refresh_dual_l2l_compute_seconds",
    "refresh_dual_downward_compute_seconds",
    "refresh_nearfield_seconds",
    "refresh_nearfield_radix_payload_seconds",
    "refresh_nearfield_target_blocks_seconds",
    "refresh_evaluate_seconds",
    "refresh_compile_or_sync_suspect_seconds",
    "refresh_total_seconds",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=256)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.6, 1.0])
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--displacement", type=float, default=1e-3)
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
    rng = np.random.default_rng(1)
    M = jnp.asarray(mass, jnp.float32)
    results = []
    for theta in args.thetas:
        s = FastMultipoleMethod(
            preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta,
            G=1.0, softening=1e-7, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                                       farfield=FarFieldConfig(mode="auto"),
                                       nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
            fixed_order=args.order)
        P = jnp.asarray(pos, jnp.float32)
        # the eval-only seam builds the fused state and sets the fused-mode flag
        prepared, ev = s.strict_fused_prepared_eval_fn(
            positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=theta)
        jax.block_until_ready(ev(prepared))

        def snap():
            d = s.get_runtime_diagnostics() or {}
            return {k: float(d.get(k) or 0.0) for k in STAGE_KEYS}

        steps = []
        cur = np.asarray(pos, np.float32)
        with GpuMonitor(devices) as mon:
            for k in range(args.steps):
                cur = cur + rng.normal(0.0, args.displacement, cur.shape).astype(np.float32)
                Pk = jnp.asarray(cur)
                before = snap()
                t0 = time.perf_counter()
                prepared, acc = s.strict_prepare_refresh_and_evaluate(
                    prepared, Pk, M, leaf_size=args.leaf, max_order=args.order, theta=theta)
                jax.block_until_ready(acc)
                wall = time.perf_counter() - t0
                after = snap()
                delta = {k2: after[k2] - before[k2] for k2 in STAGE_KEYS}
                steps.append(dict(step=k, wall_s=wall, **delta))
        cont = mon.summary()
        d = s.get_runtime_diagnostics() or {}
        info = dict(theta=theta, leaf=args.leaf, order=args.order, n=args.n,
                    steps=steps, contention=cont.as_dict(),
                    reuse=dict(
                        compiled_profile_refresh_reuse_tier_full=d.get("compiled_profile_refresh_reuse_tier_full"),
                        strict_runner_compile_count=d.get("strict_runner_compile_count"),
                        strict_fused_fastlane_hits=d.get("strict_fused_fastlane_hits"),
                        recent_dual_neighbor_count=d.get("recent_dual_neighbor_count"),
                    ))
        results.append(info)
        print(f"\ntheta {theta}: leaf {args.leaf} p {args.order} N {args.n}  flags={cont.flags or '-'}")
        print(f"{'step':>4} {'wall ms':>8} {'input':>7} {'tree+up':>8} {'dual':>8} {'  walk':>7} {'farplan':>8} "
              f"{'m2l':>7} {'near':>8} {'eval':>8} {'unattr':>8}")
        for st in steps:
            walk = st["refresh_dual_select_interactions_seconds"] + st["refresh_dual_artifact_build_seconds"] \
                + st["refresh_dual_setup_seconds"]
            print(f"{st['step']:>4} {st['wall_s']*1e3:>8.1f} {st['refresh_input_seconds']*1e3:>7.1f} "
                  f"{st['refresh_tree_upward_seconds']*1e3:>8.1f} {st['refresh_dual_downward_seconds']*1e3:>8.1f} "
                  f"{walk*1e3:>7.1f} {st['refresh_dual_far_pair_plan_seconds']*1e3:>8.1f} "
                  f"{st['refresh_dual_m2l_compute_seconds']*1e3:>7.1f} {st['refresh_nearfield_seconds']*1e3:>8.1f} "
                  f"{st['refresh_evaluate_seconds']*1e3:>8.1f} "
                  f"{st['refresh_compile_or_sync_suspect_seconds']*1e3:>8.1f}")
        print("  reuse:", info["reuse"])
        del s, prepared, ev

    out = Path(args.out or (ROOT / "artifacts" / f"probe_walk_plummer{args.n}_leaf{args.leaf}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(devices=devices, results=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
