# Static-Radix Fused On-Device Checklist (Live)

Purpose:
- Track only remaining blockers for fully on-device, fully-jitted, vectorized strict static-radix FMM.
- Remove an item from `Pending` immediately when completed.

Run discipline:
- All gates must run with `micromamba run -n odisseo` and `--require-autocvd`.
- Canonical IC: `/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz`

## Pending
- Close the remaining long-horizon S3 gap where fused-on is still slower than fused-off under the production fixed policy.
- Remove remaining host-side sync points and non-jitted orchestration in strict-lane refresh/dual/downward path.
- Further reduce strict nearfield payload/state-pack overhead under fixed global caps, with scan-carry shapes staying invariant.
- Complete end-to-end strict fused device-directed orchestration for tree+upward+dual with no hot-loop Python control flow left in cadence-critical sections.
- Lock canonical production static-cap policy values (global fixed inputs) and keep S1/S2/S3 parity gates on those exact settings.

## Completed
- Enforce production fixed-policy direction (no autotune/adaptive sizing in benchmark lane).
  - Updated tooling defaults toward fixed/static policy in `tools/fused_audit_runner.py` and `tools/walltime_ab_compare.py`.
  - Production timing suite now run under fixed policy inputs plus explicit fixed neighbor cap.
- Remove strict fused concretization blockers in static-radix template refresh and large-N evaluate path.
  - Patched `yggdrax/yggdrax/_tree_impl.py` so static-radix template rebuild derives counts from static template shapes (`particle_indices`, `leaf_codes`, `left_child`) and avoids traced `int(...)` concretization points.
  - Patched `jaccpot/runtime/_large_n_types.py` and `jaccpot/runtime/_large_n_pipeline.py` to carry/use static `local_order` metadata in prepared state and evaluation.
- Stabilize strict fused neighbor-edge scan carry shape with fixed-cap policy support.
  - Patched `jaccpot/runtime/_large_n_pipeline.py` to support fixed neighbor-cap carry sizing with fail-fast overflow in strict fused static lane, preserving shape invariance for compiled scan execution.
- Re-ran full S1/S2/S3 fused audit gates under one consistent production policy.
  - Policy: `--fixed-policy`, `--fixed-neighbor-cap 262144`, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - S1 root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_140840/prod_policy_fixed_static_neighborcap262k_memmitig_S1/audit_summary.json` (`delta_variant_minus_baseline_seconds=2.0221763100125827`, fused active baseline, fallback `0`).
  - S2 root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_141133/prod_policy_fixed_static_neighborcap262k_memmitig_S2/audit_summary.json` (`delta_variant_minus_baseline_seconds=3.6222484330064617`, fused active baseline, fallback `0`).
  - S3 root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_140117/prod_policy_fixed_static_neighborcap262k_memmitig_S3/audit_summary.json` (`delta_variant_minus_baseline_seconds=-79.22771295797429`, fused active baseline, fallback `0`; S3 still performance-negative).
- Eliminate strict-fused bootstrap/tail host refresh calls from strict cadence path.
  - Patched `jaccpot/runtime/_fmm_impl.py` so `strict_run_v2` uses compiled bootstrap evaluate+integrate (`_strict_fused_evaluate_and_segment_compiled`) and compiled tail refresh+integrate via `_strict_fused_segment_batch_compiled` before any fallback.
  - Validation S1 audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_220942/20260527_after_compiled_bootstrap_tail_S1_S1`
  - Outcome: fused baseline `83.058s`, fused-off variant `79.514s`, delta `-3.543s`; strict fused remained active with fallback `0` and route counters show compiled fused refresh route usage (`runtime_strict_fused_device_refresh_route_count=2`).
- Replace strict runner host segment loop with a compiled strict-fused cadence batch path.
  - Patched `jaccpot/runtime/_fmm_impl.py` to add `JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP` (default on) and `_strict_fused_segment_batch_compiled`, which runs refresh+integrate segments in one compiled scan after an optional bootstrap segment.
  - Validation S1 audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_212426/20260527_after_compiled_segment_loop_S1_S1`
  - S1 outcome: fused baseline `81.282s`, fused-off variant `79.509s`, delta `-1.773s`; fused remained active with fallback `0`.
  - S2 walltime root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s2_after_compiled_segment_loop`
  - S2 outcome: fused baseline `237.488s`, fused-off variant `167.164s`, delta `-70.324s` (fused still substantially slower).
- Remove static-radix node-range host roundtrip in template refresh.
  - Patched `yggdrax/yggdrax/_tree_impl.py` so strict refresh uses on-device node-range reconstruction (`return_numpy=False`) instead of `device_get` to NumPy and re-upload.
  - Validation run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_yggdrax_on_device_node_ranges_S1`
  - Outcome: fused baseline `81.743s` (neutral/slightly worse vs current best `81.498s`), fused invariants still clean (`fused active`, `fallback 0`).
- Reduce strict fused host wrapper orchestration in per-segment refresh/evaluate path.
  - Implemented direct strict-fused branch in `strict_run_v2` that refreshes via `_refresh_large_n_same_topology` and evaluates via `evaluate_large_n_state`, bypassing generic `strict_prepare_refresh_and_evaluate` wrapper work on each segment.
  - Validation run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_strict_fused_direct_refresh_S1`
  - Outcome: fused baseline `81.498s`, fused-off variant `79.695s`, delta `-1.803s`; strict fused remained active with fallback `0`.
- Remove strict fused hot-path timing/accounting overhead from runtime refresh/runner loops in production mode.
  - Implemented `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING` (default on) and gated hot refresh timers in strict fused path across `_fmm_impl.py` and `_large_n_pipeline.py`.
  - Validation: runtime files pass `py_compile`; stage-timing diagnostics remain opt-in by setting `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING=0`.
  - Post-change S1 (autocvd, canonical IC) remains fused-active with zero fallback; latest clean audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_revert_S1` (baseline `84.006s`, variant `79.643s`, fused still slower by `4.363s`).
  - Rejected experiment (reverted): static template bound reuse from cached tree bounds increased wall time and was removed.
- Re-ran S1 autocvd gate on canonical IC after host-staging/topology reductions and confirmed no strict-fused regressions.
  - Best latest S1 walltime root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s1_after_initial_prepare_fused_propagation` (baseline `81.681s`, variant `79.334s`, delta `-2.347s`).
  - Matching diagnostic S1 audit: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_initial_prepare_fused_propagation_S1` (fused active, fallback `0`, delta `-1.808s`).

- Suppressed strict traced refresh host-side side effects in dual/downward hot path.
  - Patched `jaccpot/runtime/_fmm_impl.py` to add `suppress_host_side_effects` in strict traced refresh routing (`_refresh_large_n_same_topology` -> `_prepare_state_dual_and_downward` -> strict streamed fast helper), and gate host-only work: `_prepare_diag`, strict profile reload on hot key, strict shared env writes, planner/cache counter churn, and recent dual telemetry writes.
  - Persistent S1 audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_224928/20260527_after_suppress_strict_host_side_effects_S1_S1`
  - Outcome: fused baseline `81.229s`, fused-off variant `79.156s`, delta `-2.074s`; strict fused stayed active with fallback `0`.

- Resume implementation pass completed (2026-05-28): fixed-policy tooling now defaults to compiled strict fused + disallowed host fallback, fixed-policy runs require explicit `--fixed-neighbor-cap`, and strict runner perf path no longer accumulates scan history tensors when `return_history=false`.
  - Validation roots:
    - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M/audit_summary.json`
    - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M_repeat/audit_summary.json`
    - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M_nohist/audit_summary.json`
    - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap800768/audit_summary.json`
  - Outcome: fused stayed active with fallback `0`; strict fixed compiled lane currently remains slower than fused-off in S1.
- Diagnostic and revert cycle completed (2026-05-28):
  - Added strict-fused hot-timing diagnostic run at cap `800768` (`/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_static_fixed_compiled_diag_hot_timing_on_S1/audit_summary.json`), confirming fused still slower in S1 despite fused-active/no-fallback.
  - Tested strict fused segment-loop helper-call refactor (`/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_static_fixed_compiled_refactor_segcall_S1/audit_summary.json`); observed regression and reverted that refactor.
  - S3 long-run gate attempt under fixed compiled policy was started then terminated due unexpectedly long baseline runtime; no completed S3 result from that attempt.

## Update (2026-05-28 Evening)

### Newly Completed
- Fixed-policy tooling now defaults strict fused runs to device-only mode and records that flag in metadata.
  - `tools/walltime_ab_compare.py`
  - `tools/fused_audit_runner.py`
  - Added default: `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY=1`
  - Added metadata persistence for auditability.
- Strict fused runtime hot path now avoids per-call env parsing for compiled-loop/refresh switches and hard-bypasses M2L autotune when strict fused is active.
  - `jaccpot/runtime/_fmm_impl.py`

### Still Pending (confirmed by reruns)
- S1 fixed-policy parity gate still fails (`fused-on` slower by ~25-26s across repeated runs at neighbor cap `1048576`).
- Strict streamed fast-lane still not fully engaged in measured strict fused path (planner compiled-route counters remain active in fused-only probes).

## Update (2026-06-01)

### Newly Completed
- Added strict fast-lane blocker telemetry to active runtime/perf outputs and verified in two-step fused probes.
- Added traced-path guards to keep strict fused compiled cadence stable while collecting blocker reasons.

### Confirmed Remaining Blocker
- Strict fast-lane still does not engage under traced strict refresh; current blocker set is explicitly:
  - `split_build_disabled`
  - `compact_streamed_pairs_disabled`
  - `compact_streamed_tracer_unsupported`
- Upstream tracer-unsafeness in yggdrax bounded-count compact/split traversal control flow remains a gating issue for true fast-lane entry.
