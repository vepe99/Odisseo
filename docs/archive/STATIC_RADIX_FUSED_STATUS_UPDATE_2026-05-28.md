# Static-Radix Fused Status Update (2026-05-28)

## Context Read Before Continuation

Read in sequence:
- `docs/FUSED_PATH_RECOVERY_HANDOFF_2026-05-27.md`
- `docs/STATIC_RADIX_FUSED_STATUS_UPDATE_2026-05-27.md`
- `docs/STATIC_RADIX_FUSED_ON_DEVICE_CHECKLIST_2026-05-27.md`

## Policy Direction (Production)

We are now explicitly on a fixed/static policy direction:
- no autotune-driven sizing in production runs,
- capacities/chunking should be globally fixed and explicitly provided,
- objective remains full JIT viability and maximal on-device execution without adaptive host-side shape churn.

## Code Updates Completed In This Continuation

1. Static metadata propagation for strict fused refresh/eval:
- `yggdrax/yggdrax/_tree_impl.py`
  - `rebuild_static_radix_tree_from_template` now derives particle/internal sizing from static shape metadata:
    - particles from `template.particle_indices.shape[0]`,
    - expected leaves from `template.leaf_codes.shape[0]`,
    - internal node count from `template.left_child.shape[0]`.
  - Removes traced `int(...)` concretization points that blocked strict compiled fused refresh.

- `jaccpot/runtime/_large_n_types.py`
  - Added `local_order` to `LargeNPreparedState` static metadata (pytree aux/static path).

- `jaccpot/runtime/_large_n_pipeline.py`
  - `prepare_large_n_state` now stores static `local_order`.
  - `evaluate_large_n_state` now consumes static metadata (`local_order`, `max_leaf_size`) instead of traced scalar casts.

2. Fixed-cap shape invariance in strict fused neighbor-edge path:
- `jaccpot/runtime/_large_n_pipeline.py`
  - strict fused static lane now supports fixed neighbor cap carry shape with fail-fast on overflow,
  - keeps scan carry shape invariant in compiled strict fused path.
  - non-fused static behavior remains unchanged to avoid accidental memory inflation.

3. Tooling defaults aligned with fixed policy:
- `tools/fused_audit_runner.py`
- `tools/walltime_ab_compare.py`
  - defaults now prefer fixed/static policy lane for production benchmarking.

Validation:
- runtime/tooling files passed `py_compile` after patching.

## Full Retrace Findings (Current Code Path)

- strict fused no-host-fallback path now clears the prior concretization breakpoints,
- subsequent blocker surfaced as scan carry shape drift in neighbor edges (now patched with fixed-cap logic),
- long-horizon S3 requires explicit global neighbor-cap sizing to avoid cap-overflow failures,
- memory pressure in S3 variant improved using:
  - `TF_GPU_ALLOCATOR=cuda_malloc_async`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`

## Production-Policy S1/S2/S3 Timing Reruns (Completed)

Policy applied consistently:
- `--fixed-policy`
- `--fixed-neighbor-cap 262144`
- env: `TF_GPU_ALLOCATOR=cuda_malloc_async`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`

### S1
- audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_140840/prod_policy_fixed_static_neighborcap262k_memmitig_S1/audit_summary.json`
- `delta_variant_minus_baseline_seconds = 2.0221763100125827`
- fused status:
  - baseline fused active `true`, fallback `0`
  - variant fused active `false`, fallback `0`

### S2
- audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_141133/prod_policy_fixed_static_neighborcap262k_memmitig_S2/audit_summary.json`
- `delta_variant_minus_baseline_seconds = 3.6222484330064617`
- fused status:
  - baseline fused active `true`, fallback `0`
  - variant fused active `false`, fallback `0`

### S3
- audit root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_140117/prod_policy_fixed_static_neighborcap262k_memmitig_S3/audit_summary.json`
- `delta_variant_minus_baseline_seconds = -79.22771295797429`
- fused status:
  - baseline fused active `true`, fallback `0`
  - variant fused active `false`, fallback `0`
- gate outcome: fused-active-but-slower in S3 (major remaining blocker).

## Current Interpretation

- Strict fused path is now materially more robust for static compiled execution and remains fallback-free in the above runs.
- S1 and S2 under fixed policy are now favorable for fused-on in this run set.
- S3 remains strongly throughput-negative under the same production policy and is the primary unresolved performance blocker.

## Immediate Next Work

1. Continue full-path retrace to remove remaining host/device sync points and non-jitted orchestration in the strict fused refresh lane.
2. Reduce long-horizon S3 overhead in nearfield payload/carry and dual/downward orchestration under fixed global caps.
3. Re-run S1/S2/S3 with the same fixed production policy after each major change; keep fused-on vs fused-off deltas as gate criteria.

## Continuation Log (2026-05-28 Afternoon, Fixed-Compiled Policy Resume)

### Implemented

1. Fixed-policy tooling defaults moved to fully-jitted strict fused baseline behavior:
- `tools/fused_audit_runner.py`
- `tools/walltime_ab_compare.py`
- Changes:
  - `JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP=1` (default in fixed policy)
  - `JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL=1` (default in fixed policy)
  - `JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK=1` (default in fixed policy)
  - `--fixed-policy` now requires explicit `--fixed-neighbor-cap` (>0)
  - fixed-policy metadata now records explicit cap values in summary JSON.

2. Strict fused static carry sizing tightened in large-N prepare path:
- `jaccpot/runtime/_large_n_pipeline.py`
- Changes:
  - fused + static runtime now uses fixed/cached overflow capacity rather than per-step active-size carry by default path,
  - fused + static neighbor-edge capacity path removed adaptive cap-picking churn and keeps static cached/fixed capacity with fail-fast on exceed,
  - maintains shape-invariant padded carries for compiled strict fused segment scans.

3. Strict runner history accumulation overhead removed for perf mode:
- `jaccpot/runtime/_fmm_impl.py`
- Changes:
  - `_segment_scan` now supports `collect_history` static flag,
  - strict compiled bootstrap/segment/tail paths now pass `collect_history_local=bool(return_history)`,
  - perf runs (`return_history=false`) no longer materialize per-step history tensors in scan outputs.

Validation:
- `py_compile` passed for:
  - `/export/home/tbuck/Odisseo/tools/fused_audit_runner.py`
  - `/export/home/tbuck/Odisseo/tools/walltime_ab_compare.py`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

### Fixed-Cap Runtime Findings

- With strict fixed compiled no-host-fallback policy, `--fixed-neighbor-cap 262144` is insufficient in current S1 lane.
- Failure observed:
  - `active_edges=800768 cap=262144`
  - strict fused compiled segment batch fails as expected (host fallback disallowed).
- This confirms fixed cap must be selected per particle-count/runtime profile and audited explicitly.

### S1 Results After Resume (autocvd, canonical IC, fixed-policy)

Common env:
- `--fixed-policy`
- `TF_GPU_ALLOCATOR=cuda_malloc_async`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- strict fused fallback disallowed by fixed-policy defaults.

1. Cap `1048576`:
- root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M/audit_summary.json`
- baseline fused active `true`, fallback `0`
- `delta_variant_minus_baseline_seconds = -24.20005849399604`

2. Cap `1048576` repeat:
- root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M_repeat/audit_summary.json`
- baseline fused active `true`, fallback `0`
- `delta_variant_minus_baseline_seconds = -25.177057771012187`

3. Cap `1048576` + no-history scan optimization:
- root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap1M_nohist/audit_summary.json`
- baseline fused active `true`, fallback `0`
- `delta_variant_minus_baseline_seconds = -25.264034879015526`

4. Cap sensitivity (`800768`):
- root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_resume_static_fixed_compiled_S1_cap800768/audit_summary.json`
- baseline fused active `true`, fallback `0`
- `delta_variant_minus_baseline_seconds = -25.27408981701592`

Interpretation:
- Fully jitted strict fused path is stable and no-fallback under explicit fixed caps.
- S1 remains substantially slower than fused-off in this fixed-compiled/no-host-fallback configuration.
- Cap tightening from `1048576` to `800768` was neutral-to-slightly positive on baseline walltime but did not change pass/fail outcome.

## Continuation Log (2026-05-28 Late Afternoon, Diagnostics + Revert)

### Additional diagnostics run

- Diagnostic S1 with strict fused hot timing enabled:
  - root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_static_fixed_compiled_diag_hot_timing_on_S1/audit_summary.json`
  - cap: `--fixed-neighbor-cap 800768`
  - fused active `true`, fallback `0`
  - `delta_variant_minus_baseline_seconds = -24.332192578003742`

Key finding:
- Fused baseline remained slower despite comparable stage timings to fused-off.
- Dominant timed regions for both lanes remained tree/upward + dual/downward refresh work; strict fused wall gap persists in end-to-end runtime.

### Experiment and revert

- Tried refactor in `jaccpot/runtime/_fmm_impl.py` to call shared compiled refresh/eval helper from inside strict fused compiled segment scan body.
- Validation run root:
  - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_static_fixed_compiled_refactor_segcall_S1/audit_summary.json`
- Outcome:
  - regression (`delta_variant_minus_baseline_seconds = -26.31560111499857`), worse than prior cap-800768 run.
- Action:
  - refactor was reverted; retained prior no-history optimization and fixed-policy/static-cap enforcement updates.

### S3 gate attempt status

- Launched S3 fixed-policy run:
  - intended root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260528_static_fixed_compiled_resume_S3_cap1M`
- Run was terminated after unexpectedly long baseline runtime before completion; only `environment_snapshot.json` exists in that root.
- No S3 outcome recorded from this attempt.

## Continuation Log (2026-05-28 Evening, Device-Only Strict Pass)

### Implemented in this pass

1. Fixed-policy tooling lock extended to strict device-only mode:
- `tools/walltime_ab_compare.py`
- `tools/fused_audit_runner.py`
- Added default fixed-policy env:
  - `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY=1`
- Added metadata persistence of `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY` in run summaries/case metadata.

2. Strict fused runtime cleanup in `jaccpot/runtime/_fmm_impl.py`:
- Cached strict-fused env switches at init (remove per-call env parsing in hot strict path):
  - `JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP`
  - `JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL`
  - `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY`
- In strict fused mode, hard-bypass runtime M2L autotune path in downward chunk selection.
- Kept mixed-order far-pair split disabled in strict fused device-only route to avoid level-offset host materialization path.
- Tightened strict hot-path retention/orchestration decisions (interaction/traversal retention suppression, split-build forcing, adaptive/mixed-order suppression in strict device-only local branch) while preserving fallback-free strict behavior.

Validation:
- `py_compile` passed after each edit batch for:
  - `/export/home/tbuck/Odisseo/tools/walltime_ab_compare.py`
  - `/export/home/tbuck/Odisseo/tools/fused_audit_runner.py`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

### S1 fixed-policy reruns (A/B, fused-off baseline vs fused-on variant)

Common settings:
- `--fixed-policy`
- `--fixed-neighbor-cap 1048576`
- baseline env: `JACCPOT_STATIC_STRICT_FUSED_MODE=0`
- variant env: `JACCPOT_STATIC_STRICT_FUSED_MODE=1`
- device-only default active in fixed policy (`JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY=1`)

Run roots and deltas:
- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_device_only_recheck_20260528/walltime_ab_summary.json`
  - `delta_variant_minus_baseline_seconds = 25.160005436016945`
- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_device_only_recheck2_20260528/walltime_ab_summary.json`
  - `delta_variant_minus_baseline_seconds = 25.552635879983427`
- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_device_only_recheck3_20260528/walltime_ab_summary.json`
  - `delta_variant_minus_baseline_seconds = 25.614394808973884`
- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_device_only_recheck4_20260528/walltime_ab_summary.json`
  - `delta_variant_minus_baseline_seconds = 26.42787177397986`

Observed invariants remained clean:
- fused-on lane active: `runtime_strict_fused_mode_active=true`
- fallback count: `runtime_strict_fused_fallback_count=0`
- profile re-profiling: overflow/neighbor reprofile counters stayed `0`

### Focused fused-only probe diagnostics

- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_fused_probe_steps1_20260528/reports/galaxy_disk_profile_20260528_195142.json`
- `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_fused_probe_steps1b_20260528/reports/galaxy_disk_profile_20260528_195411.json`

Both probes show strict fused active and fallback-free, but still report planner compiled-route usage:
- `runtime_refresh_dual_planner_cache_misses=1`
- `runtime_refresh_dual_planner_compile_count=1`
- `runtime_refresh_dual_planner_compiled_route_count=1`

Interpretation:
- strict fused remains functionally stable and fallback-free,
- but S1 fused-on walltime remains ~25-26s slower than fused-off,
- and the intended strict streamed fast-lane bypass is still not fully engaged in the measured path.

### Current gate status

- Success criterion `fused-on <= fused-off` is still failing in S1 under current fixed device-only policy.
- No regression in correctness/fallback invariants; performance gap persists and remains the primary blocker.

## Continuation Log (2026-06-01, Fast-Lane Blocker Telemetry + Tracer Guards)

### What was added
- Added strict fast-lane blocker diagnostics in `jaccpot/runtime/_fmm_impl.py`:
  - `strict_fused_fastlane_diag_enabled`
  - `strict_fused_fastlane_attempts/hits/misses`
  - `strict_fused_fastlane_last_blockers`
  - `strict_fused_fastlane_block_counts`
- Propagated these diagnostics into active profile outputs via:
  - `odisseo/jaccpot_coupling.py`
  - `notebooks/scalability/galaxy_disk_fmm_large_n.py`

### What was observed
- Two-step fused probe with diagnostics (stable run):
  - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_fused_probe_fastlane_diag_steps2e_20260601/reports/galaxy_disk_profile_20260601_114752.json`
  - fast-lane attempts: `1`, hits: `0`, misses: `1`
  - blockers:
    - `split_build_disabled`
    - `compact_streamed_pairs_disabled`
    - `compact_streamed_tracer_unsupported`

### Tracer-safety fixes applied this cycle
- Guarded traced strict-refresh routing to avoid non-jittable compact/split branches that trigger `TracerBoolConversionError` in yggdrax bounded-count passes.
- Added planner-route tracer-safe bool conversion handling in `_prepare_state_dual_and_downward` when `suppress_host_side_effects` is active.

### Validation
- Stable two-step fused probe after guards:
  - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/s1_fused_probe_fastlane_diag_steps2f_20260601/reports/galaxy_disk_profile_20260601_115454.json`
  - runtime: `106.773s`
  - blockers remained explicitly reported (same set as above).

### Current interpretation
- We can now measure exact strict fast-lane blockers in the active path.
- Full fast-lane entry under traced compiled refresh is still blocked by tracer-unsafeness in compact traversal/split machinery (yggdrax bounded-count control flow) plus the resulting split/compact guards.
