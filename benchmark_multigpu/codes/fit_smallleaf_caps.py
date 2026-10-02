#!/usr/bin/env python
"""Phase 0.2 of the small-leaves plan: fit the fused lane's fixed capacities per leaf size.

The single-GPU fused lane (``strict_fused_prepared_eval_fn`` + ``strict_run_v2``)
sizes several buffers from environment variables that were tuned for leaf 256 at
N=200k.  At leaf 64 and 32 the far-pair list is 3-10x longer, the neighbour-edge
total ~2x, and the longest neighbour row is ``num_leaves - 1`` (a Plummer tail
leaf sees every other leaf), so leaf 32 has never run end to end.  This script
runs ONE configuration per process (the caps are read at import / solver
construction time, so a leaf sweep inside one process would share them) and
reports, for that configuration:

* the eager prepare's list sizes against every cap it has to fit under;
* whether the near-field kernel is the gather kernel (``leafpair_gather``) and
  not the masked all-pairs streaming kernel;
* aggL2 of the eager force against a float64 direct sum (subsampled targets);
* the #333 truncation check: a ``--steps``-step ``strict_run_v2`` from rest,
  the force the scan applied at x1 recovered from the trajectory
  (``a1 = (x2 - 2 x1 + x0) / dt^2``) against an eager prepare+evaluate AT x1.
  A capacity that saturates inside the compiled scan truncates the lists
  silently; this is the only check that sees it (see
  ``tests/integration/test_strict_run_v2_refresh_capacity.py``).

A configuration "fits" when the prepare raises nothing, the kernel layout is the
gather kernel, the scan raises nothing and the recovered force agrees with the
eager one to < 1e-2 (float32 positions limit the recovery to a few 1e-3).

No timing is taken here, so a shared card is fine: the GPU is chosen with
``autocvd`` (least used) unless ``CUDA_VISIBLE_DEVICES`` is set.

Run (jax-0.10.2 venv, finder repointed at the small-leaves worktree)::

    PYTHONPATH=.../sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-wt \
      /export/home/tbuck/jaccpot/.venv/bin/python codes/fit_smallleaf_caps.py \
      --leaf 64 --theta 0.6 --env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=2097152
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.budget import jaccpot_direct_budget
from common.ic import IC_GENERATORS
from compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    _nearfield_kernel_layout,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)


def _rel_l2(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _choose_device(allow_busy: bool) -> list[int]:
    from common.gpu_guard import pick_idle_gpus, set_cuda_visible

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [int(x) for x in cvd.split(",") if x.strip()]
    if allow_busy:
        # counts and parity need no idle card; autocvd picks the least-used one
        from autocvd import autocvd

        chosen = autocvd(num_gpus=1, least_used=True, set_env=True, progress=False)
        return [int(c) for c in chosen]
    chosen = pick_idle_gpus(1)
    set_cuda_visible(chosen)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer", choices=sorted(IC_GENERATORS))
    ap.add_argument("--leaf", type=int, required=True)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=3, help="strict_run_v2 steps for the truncation check")
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--ref-targets", type=int, default=4096)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL",
                    help="extra fast-lane env; applied on top of FAST_LANE_ENV_BY_LEAF[leaf]")
    ap.add_argument("--no-leaf-preset", action="store_true",
                    help="ignore FAST_LANE_ENV_BY_LEAF and start from the leaf-256 FAST_LANE_ENV")
    ap.add_argument("--max-neighbors", type=int, default=None,
                    help="TraversalOverrides(max_neighbors_per_leaf=...) -- explicit, bypasses the 2048 clamp")
    ap.add_argument("--max-interactions", type=int, default=None,
                    help="TraversalOverrides(max_interactions_per_node=...)")
    ap.add_argument("--max-pair-queue", type=int, default=None,
                    help="TraversalOverrides(max_pair_queue=...)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="pick the least-used card with autocvd instead of demanding an idle one")
    ap.add_argument("--skip-scan", action="store_true", help="only the eager prepare + accuracy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")

    devices = _choose_device(args.allow_busy)
    overrides: dict[str, str] = {}
    if not args.no_leaf_preset:
        overrides.update(fast_lane_overrides_for_leaf(args.leaf, args.n))
    overrides.update(dict(kv.split("=", 1) for kv in args.env))
    env = apply_fast_lane_env(args.n, overrides=overrides)
    traversal_overrides = {
        k: v
        for k, v in dict(
            max_neighbors_per_leaf=args.max_neighbors,
            max_interactions_per_node=args.max_interactions,
            max_pair_queue=args.max_pair_queue,
        ).items()
        if v is not None
    }
    # the leaf preset may also carry traversal overrides (they are not env vars)
    preset_trav = (FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {})
    if not args.no_leaf_preset:
        traversal_overrides = {**preset_trav, **traversal_overrides}

    import jax
    import jax.numpy as jnp
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TraversalOverrides,
        TreeConfig,
    )

    from common.reference import direct_accelerations

    print(f"devices={devices} leaf={args.leaf} p={args.order} theta={args.theta} N={args.n}")
    print("env overrides:", json.dumps(overrides, indent=None))
    print("traversal overrides:", traversal_overrides)

    pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    n = int(pos.shape[0])
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)

    runtime_cfg = RuntimePolicyConfig()
    if traversal_overrides:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(**{k: int(v) for k, v in traversal_overrides.items()})
        )
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=args.theta,
        G=1.0,
        softening=args.softening,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            runtime=runtime_cfg,
            mac_type="dehnen",
        ),
        fixed_order=args.order,
    )

    result: dict = dict(
        n=n, ic=args.ic, leaf=args.leaf, order=args.order, theta=args.theta,
        devices=devices, env_overrides=overrides, traversal_overrides=traversal_overrides,
        fast_lane_env=env, fits=False, stage="prepare",
    )
    num_leaves = -(-n // args.leaf)
    caps = dict(
        compact_far_pair_cap=int(env["JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP"]),
        neighbor_edge_cap=int(env["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"]),
        static_target_blocks_max_per_leaf=env["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"],
        num_leaves=int(num_leaves),
    )
    result["caps"] = caps

    def _fail(stage: str, exc: BaseException) -> int:
        full = "".join(traceback.format_exception(exc))
        first = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
        print(f"FAILED at {stage}: {first}", flush=True)
        for line in str(exc).splitlines()[1:16]:
            print("   ", line[:220], flush=True)
        result.update(stage=stage, failed=first, failed_full=full[-8000:])
        _write()
        return 2

    def _write() -> None:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            print(f"wrote {args.out}")

    # ---- eager prepare + eval (the exact lists) ----
    try:
        t0 = time.perf_counter()
        prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
            positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta
        )
        a_eager = np.asarray(jax.block_until_ready(eval_fn(prepared)), np.float64)
        prepare_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return _fail("prepare", exc)

    diag = dict(solver.get_runtime_diagnostics() or {})
    rows = np.asarray(prepared.neighbor_list.counts)
    layout = _nearfield_kernel_layout(prepared)
    validated = dict(getattr(solver._impl, "_strict_fused_validated_caps", None) or {})
    far_pairs = diag.get("static_radix_far_pair_count")
    budget = jaccpot_direct_budget(prepared, n)
    lists = dict(
        neighbor_rows_max=int(rows.max()),
        neighbor_edges_total=int(rows.sum()),
        neighbor_rows_over_2048=int((rows > 2048).sum()),
        far_pairs=None if far_pairs is None else int(far_pairs),
        direct_share_of_N=budget.get("direct_share_of_N"),
        active_leaves=diag.get("large_n_eval_active_leaf_count"),
        nearfield_kernel_layout=layout,
        fused_mode_active=bool(diag.get("strict_fused_mode_active")),
        prepare_s_incl_compile=prepare_s,
    )
    result["lists"] = lists
    result["validated_caps"] = validated
    headroom = dict(
        far_pairs_over_cap=None if far_pairs is None else float(far_pairs) / caps["compact_far_pair_cap"],
        edges_over_cap=float(rows.sum()) / caps["neighbor_edge_cap"],
        max_row_over_num_leaves=float(rows.max()) / max(1, num_leaves - 1),
    )
    result["headroom"] = headroom
    print(f"eager: rows max {rows.max()} (num_leaves-1 = {num_leaves-1}), edges {rows.sum()} "
          f"({headroom['edges_over_cap']:.2f} of cap {caps['neighbor_edge_cap']}), "
          f"far pairs {far_pairs} ({headroom['far_pairs_over_cap'] if far_pairs is not None else float('nan'):.2f} of cap "
          f"{caps['compact_far_pair_cap']}), layout {layout}, fused {lists['fused_mode_active']}, "
          f"direct {budget.get('direct_share_of_N'):.3f}N, prepare+compile {prepare_s:.1f} s", flush=True)
    print("validated caps:", json.dumps(validated, default=str), flush=True)

    # ---- accuracy vs float64 direct ----
    ref_idx = None
    if args.ref_targets and args.ref_targets < n:
        ref_idx = np.sort(np.random.default_rng(12345).choice(n, args.ref_targets, replace=False))
    a_ref = direct_accelerations(pos, mass, G=1.0, softening=args.softening,
                                 block_size=1024, target_indices=ref_idx)
    a_cmp = a_eager if ref_idx is None else a_eager[ref_idx]
    agg = _rel_l2(a_cmp, a_ref)
    result["errors"] = dict(aggL2=agg, ref_targets=None if ref_idx is None else int(len(ref_idx)))
    print(f"accuracy: aggL2 vs fp64 direct ({'all' if ref_idx is None else len(ref_idx)} targets) = {agg:.4e}",
          flush=True)

    ok = layout == "leafpair_gather" and lists["fused_mode_active"]
    if args.skip_scan:
        result.update(fits=ok, stage="prepare_only")
        _write()
        return 0 if ok else 1

    # ---- the #333 truncation check inside the compiled scan ----
    result["stage"] = "scan"
    try:
        state0 = jnp.stack([P, jnp.zeros((n, 3), jnp.float32)], axis=1)
        t0 = time.perf_counter()
        final, prepared_out, history = solver.strict_run_v2(
            state=state0, masses=M, dt=args.dt, num_steps=args.steps, refresh_every=1,
            leaf_size=args.leaf, max_order=args.order, theta=args.theta,
            prepared_state=None, return_prepared_state=True, return_history=True,
        )
        hist = np.asarray(jax.block_until_ready(history), np.float64)
        scan_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return _fail("scan", exc)

    xs = [pos.astype(np.float64)] + [hist[k, :, 0, :] for k in range(hist.shape[0])]
    a0_scan = 2.0 * (xs[1] - xs[0]) / args.dt**2
    err0 = _rel_l2(a0_scan, a_eager)
    errs = [err0]
    for k in range(1, min(len(xs) - 1, args.steps)):
        ak_scan = (xs[k + 1] - 2.0 * xs[k] + xs[k - 1]) / args.dt**2
        p_k, ev_k = solver.strict_fused_prepared_eval_fn(
            positions=jnp.asarray(xs[k], jnp.float32), masses=M,
            leaf_size=args.leaf, max_order=args.order, theta=args.theta,
        )
        a_k = np.asarray(jax.block_until_ready(ev_k(p_k)), np.float64)
        errs.append(_rel_l2(ak_scan, a_k))
        del p_k, ev_k
    traced = dict(getattr(solver._impl, "_strict_fused_traced_caps", None) or {})
    d2 = dict(solver.get_runtime_diagnostics() or {})
    refreshed_rows = np.asarray(prepared_out.neighbor_list.counts)
    scan = dict(
        steps=args.steps, dt=args.dt, scan_s_incl_compile=scan_s,
        recovered_force_rel_l2_by_step=errs,
        refreshed_rows_max=int(refreshed_rows.max()),
        refreshed_edges_total=int(refreshed_rows.sum()),
        traced_caps=traced,
        fallback_count=d2.get("strict_fused_fallback_count"),
        last_fallback_reason=d2.get("strict_fused_last_fallback_reason"),
        fastlane_hits=d2.get("strict_fused_fastlane_hits"),
        fastlane_attempts=d2.get("strict_fused_fastlane_attempts"),
        compile_count=d2.get("strict_runner_compile_count"),
    )
    result["scan"] = scan
    trunc_ok = all(e < 1e-2 for e in errs)
    ok = ok and trunc_ok and not d2.get("strict_fused_fallback_count")
    result.update(fits=ok, stage="done")
    print(f"scan: recovered-force rel L2 by step {['%.3e' % e for e in errs]}  "
          f"refreshed rows max {refreshed_rows.max()}  fallbacks {scan['fallback_count']}  "
          f"traced caps {json.dumps(traced, default=str)}", flush=True)
    print(f"FITS={ok}", flush=True)
    _write()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
