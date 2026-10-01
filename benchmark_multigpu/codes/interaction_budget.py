#!/usr/bin/env python
"""T2.0 -- how much of every jaccpot force is a direct sum, exactly, per (leaf, theta).

The first comparison inferred from ``recent_dual_neighbor_count`` that jaccpot at
leaf 256 / theta 0.6 sums >50 % of N directly for every target.  This script
replaces that inference with the exact per-target budget read from the prepared
state (``common/budget.py``), alongside the wall time and the aggL2 error of the
*same* configuration, taken on a verified-idle GPU under a contention monitor.

Also records, per configuration, the prepare-stage timings that make up the MAC
walk (``refresh_dual_*``) so Track 3 can report jaccpot both eval-only and
eval+walk.

Run:
    JAX_ENABLE_X64=1 /export/home/tbuck/jaccpot/.venv/bin/python \
      benchmark_multigpu/codes/interaction_budget.py --n 200000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls  # noqa: E402
from compare_force import apply_fast_lane_env, rel_errors  # noqa: E402
from common.budget import jaccpot_direct_budget  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402

WALK_KEYS = (
    "refresh_dual_setup_seconds",
    "refresh_dual_artifact_build_seconds",
    "refresh_dual_select_interactions_seconds",
    "refresh_dual_far_pair_plan_seconds",
    "refresh_dual_split_combined_seconds",
    "refresh_dual_split_far_pairs_seconds",
    "refresh_dual_split_leaf_neighbors_seconds",
    "refresh_dual_raw_combined_seconds",
    "refresh_tree_build_seconds",
    "refresh_tree_upward_seconds",
    "refresh_nearfield_seconds",
    "refresh_total_seconds",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leafs", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--orders", type=int, nargs="+", default=[4],
                    help="expansion orders; several = the T2.2 truncation-vs-acceptance sweep")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--edge-cap", type=int, default=None,
                    help="override JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP")
    ap.add_argument("--set-env", nargs="+", default=[], metavar="KEY=VAL",
                    help="override fast-lane environment entries (applied after FAST_LANE_ENV)")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1)
    cvd = set_cuda_visible(devices)
    print(f"GPU: physical {devices} (CUDA_VISIBLE_DEVICES={cvd})", flush=True)
    env = apply_fast_lane_env(args.n)
    if args.edge_cap is not None:
        os.environ["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = str(args.edge_cap)
        env["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = str(args.edge_cap)
    for kv in args.set_env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
        env[k] = v

    import jax
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    if args.ic == "disk":
        pos, mass = IC_GENERATORS["disk"](subsample=args.n, seed=0)
    else:
        pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    n = int(pos.shape[0])
    P = jax.numpy.asarray(pos, jax.numpy.float32)
    M = jax.numpy.asarray(mass, jax.numpy.float32)

    ref = None
    if not args.no_reference:
        from common.reference import direct_accelerations

        print(f"N={n}: float64 direct reference ...", flush=True)
        ref = direct_accelerations(pos, mass, G=1.0, softening=args.softening, block_size=2048)

    rows = []
    hdr = (f"{'leaf':>5} {'p':>2} {'theta':>5} {'ms':>8} {'iqr':>6} {'aggL2':>10} {'leaves':>6} "
           f"{'nbr/leaf':>8} {'occ':>6} {'direct/tgt':>10} {'share':>6} {'p90':>8} {'self':>4} {'flags'}")
    print(hdr, flush=True)
    for leaf, order, theta in ((l, p, t) for l in args.leafs for p in args.orders for t in args.thetas):
        if True:
            s = FastMultipoleMethod(
                preset="large_n_gpu", runtime_path="large_n", basis="real",
                theta=theta, G=1.0, softening=args.softening,
                working_dtype=jax.numpy.float32,
                advanced=FMMAdvancedConfig(
                    tree=TreeConfig(mode="static_radix", leaf_target=leaf),
                    farfield=FarFieldConfig(mode="auto"),
                    nearfield=NearFieldConfig(mode="auto"),
                    mac_type="dehnen",
                ),
                fixed_order=order,
            )
            # prepare-stage (MAC walk, tree, upward) timings are only recorded while
            # this flag is on; strict_run_v2 sets it, the eval-only seam does not.
            s._refresh_timing_active = True
            try:
                prepared, ev = s.strict_fused_prepared_eval_fn(
                    positions=P, masses=M, leaf_size=leaf, max_order=order, theta=theta
                )
            except Exception as exc:
                msg = str(exc).splitlines()[0][:160]
                print(f"{leaf:>5} {order:>2} {theta:>5.2f}  FAILED: {msg}", flush=True)
                rows.append(dict(leaf=leaf, theta=theta, order=order, failed=msg))
                continue
            budget = jaccpot_direct_budget(prepared, n)
            out, timing, cont = timed_calls(
                lambda: ev(prepared), repeats=args.repeats, warmup=args.warmup,  # noqa: B023
                devices=devices, block=jax.block_until_ready,
            )
            diag = dict(s.get_runtime_diagnostics() or {})
            errs = rel_errors(np.asarray(out), ref) if ref is not None else None
            row = dict(
                leaf=leaf, theta=theta, order=order, n=n,
                force_eval_s=timing, contention=cont.as_dict(),
                errors=errs, budget=budget,
                fused_mode_active=bool(diag.get("strict_fused_mode_active")),
                recent_dual_neighbor_count=diag.get("recent_dual_neighbor_count"),
                recent_dual_far_pair_count=diag.get("recent_dual_far_pair_count"),
                static_radix_far_pair_count=diag.get("static_radix_far_pair_count"),
                walk_seconds={k: diag.get(k) for k in WALK_KEYS},
            )
            rows.append(row)
            print(
                f"{leaf:>5} {order:>2} {theta:>5.2f} {timing['min']*1e3:>8.2f} {timing['iqr']*1e3:>6.2f} "
                f"{(errs['aggL2'] if errs else float('nan')):>10.3e} {budget['num_leaves']:>6} "
                f"{budget['mean_neighbor_leaves_per_leaf']:>8.1f} {budget['occupancy_mean']:>6.1f} "
                f"{budget['direct_sources_per_target_mean']:>10.0f} "
                f"{budget['direct_share_of_N']:>6.3f} {budget['direct_sources_per_target_p90']:>8.0f} "
                f"{budget['self_entries']:>4} {','.join(cont.flags) or '-'}",
                flush=True,
            )
            del prepared, ev, s

    from common.env import capture_provenance

    out_path = Path(args.out or (ROOT / "artifacts" / f"interaction_budget_{args.ic}{n}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            dict(
                provenance=capture_provenance({"benchmark": "interaction_budget", "args": vars(args)}),
                ic=dict(name=args.ic, n=n, softening=args.softening),
                devices=devices,
                fast_lane_env=env,
                rows=rows,
            ),
            fh, indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
