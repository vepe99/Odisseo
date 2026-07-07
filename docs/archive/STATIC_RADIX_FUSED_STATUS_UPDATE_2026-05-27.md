# Static-Radix Fused Status Update (2026-05-27)

## Handoff Docs Read

Read top-to-bottom before resuming:
- `docs/STATIC_RADIX_FUSED_COMPACT_PACK_HANDOFF_2026-05-26.md`
- `docs/STATIC_RADIX_LARGE_N_COMPILED_STATE_HANDOFF_2026-05-26.md`

## Operational Updates

Updated `tools/walltime_ab_compare.py`:
- selects one free GPU with `autocvd` by default; `--no-autocvd` disables it; `--require-autocvd` fails hard if unavailable,
- streams subprocess output and emits heartbeat status lines via `--status-interval-seconds`,
- sets stable-lane defaults: `JACCPOT_LARGE_N_COMPILED_STATE_MODE=on` and `JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK=0`,
- supports explicit `--baseline-env KEY=VALUE` plus existing `--variant-env KEY=VALUE`,
- has optional `--profile-breakdown` for short diagnostic gates while keeping canonical walltime mode clean by default.

Regenerated missing canonical fixed IC:
- `/tmp/odisseo_fixed_agama_ic_200k.npz`
- shape: `state0=(200000, 2, 3)`, `mass=(200000,)`, dtype `float32`
- `ic_velocity_potential=nfw`, prograde fraction `1.0`, mass sum approximately `6.0`

## jaccpot Runtime Fixes

Patched `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`:
- populated `_StrictFusedRuntimeConfig.compact_nearfield_pack` from `JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK` with default off,
- stabilized fused scan carry for all `NodeNeighborList` array fields against the previous carry shape,
- preserved `None` fields as `None` when the previous carry used `None`; array fields get zero-like placeholders only when needed.

Syntax checks passed:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- `micromamba run -n odisseo python -m py_compile tools/walltime_ab_compare.py`

## Validation Results

Command family used single autocvd GPU (`CUDA_VISIBLE_DEVICES=9`):

```bash
micromamba run -n odisseo python tools/walltime_ab_compare.py \
  --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz \
  --out-root /tmp/odisseo_walltime_ab_fused_diag_200k_2_after_none_fix \
  --n-particles 200000 --num-steps 2 --state-dtype float32 \
  --leaf-size 256 --refresh-every 1 \
  --baseline-env JACCPOT_STATIC_STRICT_FUSED_MODE=on \
  --baseline-env JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=100000,200000,400000 \
  --variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=off \
  --profile-breakdown --require-autocvd --status-interval-seconds 30
```

Diagnostic 200k/2 after fixes:
- fused baseline active: `runtime_strict_fused_mode_active=true`
- fused fallback count: `0`
- fused device refresh route count: `1`
- fused planner bypassed count: `1`
- overflow reprofiles: `0`
- neighbor-edge reprofiles: `0`
- baseline script runtime: `101.560 s`
- variant fused-off script runtime: `75.345 s`

Walltime-only 200k/2 after fixes:
- summary: `/tmp/odisseo_walltime_ab_fused_smoke_200k_2_after_none_fix/walltime_ab_summary.json`
- fused-on script runtime: `100.782 s`
- fused-off script runtime: `76.633 s`
- wrapper delta variant-minus-baseline: `-24.135 s` (fused-off faster)

## Current Interpretation

Note: earlier same-day analysis that referenced `/tmp/odisseo_fixed_agama_ic_200k.npz` should be treated as provisional because `/tmp` artifacts are non-persistent. Canonical gating now uses the persistent IC at `/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz`.

Stable fused execution is restored for the 200k/2 gate with compact-pack off and compiled-state on. It is currently throughput-negative versus fused-off on the short 200k/2 walltime gate. Do not run the full 200k/20 oracle expecting a win until the fused active path overhead is understood.

## Next Recommended Step

Profile why active fused is slower now that it no longer falls back. Focus first on the fused device refresh route and neighbor-list carry stabilization overhead; the all-field shape fitting may be correct but too expensive or may preserve more payload than the fused route should carry.

## Autocvd-Gated Continuation (2026-05-27 Evening)

All continuation runs below used `micromamba run -n odisseo` plus `--require-autocvd` and pinned to one GPU:
- selected device: `CUDA_VISIBLE_DEVICES='9'`

Persistent canonical IC used:
- `/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz`

### S1 Audit (200k/2) with strict-fused v3 path

Run root:
- `/tmp/odisseo_fused_audit/20260527_180246/fused_static_device_refresh_v3_autocvd_S1`

Key outcomes:
- fused baseline: `runtime_strict_fused_mode_active=true`
- fused fallback count: `0`
- fused device refresh route count: `2`
- fused planner bypassed count: `2`
- baseline wall: `83.702 s`
- variant wall (fused off): `78.908 s`
- delta variant-minus-baseline: `-4.795 s` (fused-off still faster)

Compared with earlier non-autocvd v3 S1 run, fused baseline improved materially:
- fused baseline wall: `91.010 s` -> `83.702 s`

### S2 Walltime Gate (200k/20) with strict-fused on vs off

Run root:
- `/tmp/odisseo_walltime_ab/20260527_s2_autocvd`

Results:
- baseline (fused on) wall: `239.915 s`
- variant (fused off) wall: `176.353 s`
- delta variant-minus-baseline: `-63.562 s`

Interpretation:
- S2 gate still fails for fused-on throughput.
- Autocvd pinning reduced cross-run noise and improved reproducibility, but fused-on remains significantly slower.

### Immediate Plan Continuation

Priority remains host/device orchestration and refresh payload overhead in strict-fused mode:
1. eliminate strict-fused-specific nearfield carry padding overhead growth (`runtime_refresh_nearfield_neighbor_padding_seconds`),
2. trim strict-fused refresh payload/state pack shape to minimal required tensors,
3. keep static-template refresh route but continue reducing refresh-stage orchestration in tree/upward + dual-artifact path,
4. rerun S1 autocvd gate after each change, then recheck S2 walltime only after S1 shows stable gains.

## Live Execution Checklist

Active tracker for remaining fully on-device fused blockers:
- `docs/STATIC_RADIX_FUSED_ON_DEVICE_CHECKLIST_2026-05-27.md`

## Continuation Log (2026-05-27 Late Evening)

### Implemented

- Strict fused production timing gate added in `jaccpot/runtime/_fmm_impl.py` and `jaccpot/runtime/_large_n_pipeline.py`:
  - `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING` (default `1`) disables refresh hot-path timer/accounting calls in strict fused mode.
  - Diagnostics can be re-enabled with `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING=0`.

### Persistent-IC S1 Recheck (autocvd, canonical IC)

- Command family: `micromamba run -n odisseo python tools/walltime_ab_compare.py ... --require-autocvd`
- Canonical IC: `/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz`
- Run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s1_after_revert_bounds`
- Outcome:
  - fused baseline wall: `83.940 s`
  - fused-off variant wall: `79.754 s`
  - delta variant-minus-baseline: `-4.186 s`

### Diagnostic S1 Audit (autocvd)

- Run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_revert_S1`
- Invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- Key walltimes:
  - baseline: `84.006 s`
  - variant: `79.643 s`
  - delta variant-minus-baseline: `-4.363 s`

### Rejected/Removed Experiment

- Attempted static-template bound reuse from cached tree bounds in strict fused refresh increased fused baseline walltime to ~`103 s` on S1.
- Experiment was reverted; current code keeps bounds resolution behavior unchanged from pre-experiment path.

### Continuation Log (2026-05-27 Night, Host-Staging Trim)

Code changes:
- `jaccpot/runtime/_large_n_pipeline.py`
  - In strict fused mode, bypass host NumPy overflow compaction in nearfield speed-layout preparation (keeps fused path from host compaction round-trips).
- `jaccpot/runtime/_fmm_impl.py`
  - Removed explicit `jax.device_get` in dual-planner compiled-route bool extraction.

Validation:
- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`

S1 walltime recheck (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s1_after_host_staging_trim`
- fused baseline wall: `83.742 s`
- fused-off variant wall: `79.482 s`
- delta variant-minus-baseline: `-4.260 s`

S1 diagnostic audit (autocvd):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_host_staging_trim_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- walltimes:
  - baseline: `83.910 s`
  - variant: `79.826 s`
  - delta variant-minus-baseline: `-4.084 s`

Interpretation:
- strict fused remains stable and fallback-free.
- host-staging trims did not materially close the fused-vs-off walltime gap yet.

### Continuation Log (2026-05-27 Night, Initial-Prepare Fused Propagation)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - `prepare_state(..., fused_device_mode: bool = False)` now accepts fused-mode intent and forwards it to `prepare_large_n_state`.
  - `strict_prepare_refresh_and_evaluate` now forwards fused mode into the initial `prepare_state` call (when `prepared_state is None`), so the first strict segment follows fused route selection.
  - Static fused refresh topology branch simplified: fused route no longer depends on reuse-key probe/miss checks.

Validation:
- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`

S1 walltime recheck (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s1_after_initial_prepare_fused_propagation`
- fused baseline wall: `81.681 s`
- fused-off variant wall: `79.334 s`
- delta variant-minus-baseline: `-2.347 s`

S1 diagnostic audit (autocvd):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_initial_prepare_fused_propagation_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- walltimes:
  - baseline: `81.904 s`
  - variant: `80.096 s`
  - delta variant-minus-baseline: `-1.808 s`

Interpretation:
- This is the strongest improvement so far in the strict-fused S1 lane today.
- Fused remains slower than fused-off, but the gap is materially reduced versus earlier ~`-4.1` to `-4.8 s` range.

### Continuation Log (2026-05-27 Night, Fused Rematerialization Toggle)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - Added `JACCPOT_STATIC_STRICT_FUSED_DISABLE_REMATERIALIZE` (default `1`).
  - In `strict_run_v2`, strict fused mode now disables segment rematerialization by default (`state_curr = jnp.asarray(...)` between refresh segments), while leaving non-fused behavior unchanged.

S1 walltime recheck (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s1_after_disable_fused_rematerialize`
- fused baseline wall: `82.050 s`
- fused-off variant wall: `80.135 s`
- delta variant-minus-baseline: `-1.915 s`

Interpretation:
- This change is neutral-to-slightly-positive versus the pre-improvement baseline, but did not beat the best run from initial-prepare fused propagation (`81.681 s`).


### Continuation Log (2026-05-27 Night, Strict-Fused Direct Refresh/Evaluate Route)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - In `strict_run_v2`, added a strict-fused direct path that refreshes via `_refresh_large_n_same_topology` and evaluates via `evaluate_large_n_state`, bypassing per-segment `strict_prepare_refresh_and_evaluate` wrapper orchestration.
  - Preserved strict-runner profile counters and fail-fast behavior in this direct fused path.
  - Reverted the regressive static cache-key skip experiment before this run so static-radix cache-key behavior is back to the prior stable state.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_strict_fused_direct_refresh_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
  - `runtime_strict_fused_device_refresh_route_count=2`
  - `runtime_strict_fused_planner_bypassed_count=2`
- walltimes:
  - baseline: `81.498 s`
  - variant: `79.695 s`
  - delta variant-minus-baseline: `-1.803 s`

Interpretation:
- This is a measurable but modest improvement vs the immediate post-revert baseline (`81.869 s` -> `81.498 s`).
- Fused remains slower than fused-off, so the remaining major blocker is still the host-driven segment cadence itself (single compiled refresh+integrate loop not yet complete).


### Continuation Log (2026-05-27 Night, On-Device Static Node-Range Rebuild)

Code changes:
- `yggdrax/yggdrax/_tree_impl.py`
  - `rebuild_static_radix_tree_from_template` now keeps static node-range reconstruction on device.
  - Added `return_numpy` switch to `_static_radix_node_ranges_from_leaf_ranges`; strict refresh now uses `return_numpy=False` to avoid device_get -> NumPy -> device roundtrip.
- `jaccpot/runtime/_fmm_impl.py`
  - strict fused direct refresh/evaluate path retained from prior step.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/yggdrax/yggdrax/_tree_impl.py`
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_yggdrax_on_device_node_ranges_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- walltimes:
  - baseline: `81.743 s`
  - variant: `79.515 s`
  - delta variant-minus-baseline: `-2.228 s`

Interpretation:
- This removes a concrete host roundtrip in static-radix refresh, but wall-time did not beat the current best fused baseline (`81.498 s`).
- Next high-impact target remains the strict cadence loop itself: refresh+integrate must become a single compiled device loop.

### Nsight Capture Note (2026-05-27 Night)

- Nsight is available via explicit binary path: `/usr/local/cuda/bin/nsys`.
- `fused_audit_runner.py` works with `--nsys-bin /usr/local/cuda/bin/nsys`.
- The captures reported importer warnings (`Unknown driver API function index: 720`) but still emitted metrics in `audit_summary.json`.
- Latest nsys run root:
  - `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_after_strict_fused_direct_refresh_S1_nsys`


### Continuation Log (2026-05-27 Night, Compiled Strict-Fused Segment Cadence)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - Added `JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP` (default `1`).
  - Implemented `_strict_fused_segment_batch_compiled` in `strict_run_v2`: after optional bootstrap prepare segment, remaining full refresh+integrate segments run inside one compiled `lax.scan` carry loop.
  - Kept guarded fallback to the existing host-segment loop if compiled segment execution raises.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_212426/20260527_after_compiled_segment_loop_S1_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- walltimes:
  - baseline: `81.282 s`
  - variant: `79.509 s`
  - delta variant-minus-baseline: `-1.773 s`

S2 walltime gate (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/20260527_s2_after_compiled_segment_loop`
- walltimes:
  - baseline (fused on): `237.488 s`
  - variant (fused off): `167.164 s`
  - delta variant-minus-baseline: `-70.324 s`

Interpretation:
- Compiled segment batching is now in place, but it did not close the throughput gap yet.
- Fused-on improved slightly in S2 absolute walltime versus earlier runs, but fused-off improved more; net gap widened.
- Next blockers are still inside refresh-stage tree/upward/dual orchestration and residual host-driven bootstrap/tail/control-flow overhead.


### Continuation Log (2026-05-27 Night, Nsight After Compiled Segment Loop)

Nsight capture run:
- root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_213720/20260527_after_compiled_segment_loop_S1_nsys_S1`
- command path: `micromamba run -n odisseo ... fused_audit_runner.py --nsys-capture --nsys-bin /usr/local/cuda/bin/nsys --require-autocvd`
- selected device: `CUDA_VISIBLE_DEVICES='9'`

Results summary:
- baseline wall: `135.687 s`
- variant wall: `132.894 s`
- delta variant-minus-baseline: `-2.793 s`
- baseline nsys metrics: `gpu_active_percent=0.169`, `kernel_count=6441`, `host_idle_gap_max_ms=43057`
- variant nsys metrics: `gpu_active_percent=0.181`, `kernel_count=2618`, `host_idle_gap_max_ms=43289`
- gate flag: `flag_gpu_active_below_threshold=true`

Notes:
- Nsight still reports importer warnings (`Unknown driver API function index: 720`), but `audit_summary.json` metrics were emitted and consistent with prior captures.
- Despite compiled segment batching, GPU active fraction remains far below target; dominant blockers remain refresh-stage orchestration and synchronization cadence.


### Continuation Log (2026-05-27 Late Night, Compiled Bootstrap/Tail Strict Cadence)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - Restored strict runner method block integrity (`strict_run_segmented`, `strict_run_v2`) and added compiled strict-fused bootstrap/tail path.
  - Added `_strict_fused_evaluate_and_segment_compiled` so first strict segment (when `prepared_state is None`) evaluates+integrates in compiled mode after one prepare.
  - Tail segment path now attempts compiled refresh+integrate via `_strict_fused_segment_batch_compiled` (1 segment) before host fallback.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_220942/20260527_after_compiled_bootstrap_tail_S1_S1`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
- walltimes:
  - baseline: `83.058 s`
  - variant: `79.514 s`
  - delta variant-minus-baseline: `-3.543 s`
- counters:
  - `runtime_strict_fused_device_refresh_route_count=2`
  - `runtime_strict_fused_planner_bypassed_count=2`

Interpretation:
- Bootstrap/tail host refresh calls are removed from the strict fused cadence path, but throughput did not improve on this S1 run.
- Remaining dominant blockers are still tree/upward + dual artifact refresh orchestration and low GPU active utilization.


### Continuation Log (2026-05-27 Late Night, Dual Prelude Strict Short-Circuit)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - In `_prepare_state_dual_and_downward`, moved strict streamed fast-path short-circuit ahead of adaptive policy/cache-key scaffolding.
  - Strict streamed fast path now returns before `_interaction_cache_key` and adaptive policy state construction in strict fused cadence.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_222821/20260527_after_dual_prelude_short_circuit_S1_S1`
- walltimes:
  - baseline: `82.159 s`
  - variant: `80.043 s`
  - delta variant-minus-baseline: `-2.116 s`
- invariants: fused active, fallback `0`.

Interpretation:
- The host-prelude short-circuit is now in place but did not materially close the fused gap on S1.


### Continuation Log (2026-05-27 Late Night, Strict Profile Hot-Path Short-Circuit)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - In strict dual/downward orchestration, skip `_maybe_load_strict_cap_profile(...)` when strict fused is active and the strict context key is already stable.
  - Exact-profile fail-fast semantics remain unchanged.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_223212/20260527_after_strict_profile_hot_short_circuit_S1_S1`
- walltimes:
  - baseline: `82.970 s`
  - variant: `79.519 s`
  - delta variant-minus-baseline: `-3.451 s`
- invariants: fused active, fallback `0`.

Interpretation:
- Redundant strict-profile reload work is reduced in the hot path, but fused remains slower in S1.

### Continuation Log (2026-05-27 Late Night, Strict Traced Host Side-Effect Suppression)

Code changes:
- `jaccpot/runtime/_fmm_impl.py`
  - Added `suppress_host_side_effects` plumbing for strict traced fused refresh: `_refresh_large_n_same_topology` -> `_prepare_state_dual_and_downward` -> `_prepare_state_dual_and_downward_strict_streamed_fast`.
  - Gated strict traced host work in dual/downward hot path: `_prepare_diag` logging, strict profile hot reload, strict shared env one-shot writes, planner/cache hit-miss counter bookkeeping, and recent dual telemetry writes.
  - Preserved strict fail-fast checks and computational artifacts; changes are orchestration-side only.

Validation:
- `micromamba run -n odisseo python -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

S1 diagnostic audit (autocvd, canonical IC):
- persistent run root: `/export/home/tbuck/Odisseo/notebooks/scalability/runs/fused_audit/20260527_224928/20260527_after_suppress_strict_host_side_effects_S1_S1`
- walltimes:
  - baseline: `81.229 s`
  - variant: `79.156 s`
  - delta variant-minus-baseline: `-2.074 s`
- invariants:
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`
  - `runtime_large_n_overflow_profile_reprofiles=0`
  - `runtime_large_n_neighbor_edges_profile_reprofiles=0`

Interpretation:
- Strict fused remains slower, but the gap narrowed versus the prior `82.970s` baseline run.
- This confirms there was measurable host-side overhead in strict traced orchestration; next gains require moving remaining tree/upward + dual/downward scheduling and nearfield prep deeper into fused device execution.

## 2026-06-13 Current Status Pointer

- The strict production integrator now implements endpoint-correct velocity Verlet with acceleration carry: one bootstrap force plus one new endpoint FMM evaluation per step.
- Fused static payload is default-on with cap `32`; cap `16` fails eagerly and evolving-capacity status is carried through the compiled scan.
- Canonical 200k/1 median is `1.001s` payload-on versus `1.708s` opt-out.
- Canonical 200k/20 median is `1.128s / step`, fused active with fallback `0` and 20 endpoint force evaluations.
- Same-position force parity remains near `2e-7` relative L2, including evolved positions.
- Fresh Nsight still shows about `19.6k` measured-tail kernels and no hot-loop host/device transfers.
- Full details and artifacts: `docs/STATIC_RADIX_200K_GPU_PERF_HANDOFF_2026-06-10.md`.


## 2026-06-14 Nearfield Occupancy Promotion

- Occupancy sorting plus all-invalid tile skipping is now default-on in the strict radix payload evaluator.
- Canonical 200k/1 improved from `1.0048s` to `0.4710s` median over three measured runs.
- Direct force parity remains excellent: relative L2 `1.974e-7`, max absolute component `2.289e-5`.
- Default-on smoke measured `0.4746s`; explicit rollback measured `0.9962s`; fused active and fallback `0`.
- Rollback: set `JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT=0` and/or `JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES=0`.
- Full details: `docs/STATIC_RADIX_200K_GPU_PERF_HANDOFF_2026-06-10.md`.

## 2026-06-20 Componentwise Nearfield Default

- Promoted `JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS=1` into the fixed production policy.
- Canonical 200k/1 three-run median improved from `0.423086s` to `0.396896s` per step.
- Canonical 200k/20 timing improved from `0.427116s/step` to `0.399111s/step`, with fused active and fallback `0`.
- Direct force parity passed: relative L2 `5.45e-08`, max component `7.63e-06`.
- Nsight shows the win comes from lower heavy nearfield GPU busy time, not launch reduction: measured-tail kernels increase slightly, but GPU busy drops by about `114ms`.
- Rollback switch: `JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS=0`.
- Open blocker: 200k/20 final states are all NaN for both baseline and componentwise runs; this is pre-existing and must be handled as a separate finite-state production gate.

## 2026-06-20 NaN Stability Check

- Current 200k/20 timing artifacts are not production-correct: baseline and componentwise outputs both contain all-NaN `final_state` arrays.
- The likely trigger is the benchmark timestep (`t_end=2.0 Gyr`, `20` steps, `dt=0.1 Gyr`), not the componentwise nearfield path; a `t_end=0.2 Gyr` same-lane probe stayed finite but still produced severe high-radius/high-velocity outliers.
- The benchmark and simulator reports now include `t_end`, `dt`, final-state finite counts, NaN/Inf counts, and position/velocity norm summaries.
- Added `--require-finite-final-state` to both the simulator and A/B runner for explicit production-correctness gating.
- Next gate: isolate external-only vs self-only instability, then require finite final state before accepting any 200k/20 performance result.

## 2026-06-20 NFW Fix and Remaining Strict-Fused NaN

- Fixed the external NFW small-radius formula with analytic series expansions; focused potential regression passes.
- Stability split shows external-only is finite/sane, self-only is finite but violently ejects particles at `dt=0.1 Gyr`, and strict full fused still becomes all-NaN in the tight compiled production path.
- Generic/history mode is finite after the NFW fix, but it bypasses strict fused and cannot certify the production path.
- Next step is strict fused first-nonfinite-step attribution inside `jaccpot.strict_run_v2`, followed by timestep/softening/IC stability gates before more performance promotion.

## 2026-06-20 Strict Fused NaN Root Cause

- NaNs appear after the strict fused path corrupts velocities: step 1 is finite but bad, step 2 is catastrophic, step 3 introduces NaNs.
- ICs are not the direct source: direct self forces for worst particles are normal, and fresh FMM/non-hot refresh agree at endpoint positions.
- Root cause is unsafe cached compact far-pair reuse in the strict fused static-radix hot refresh path; the far-pair list changes after drift (`68272` cached vs `77002` fresh), so reused M2L pairs corrupt endpoint acceleration.
- Local `jaccpot` now fails fast by default for this unsafe reuse; legacy experiments must set `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1`.
- Next production work: implement scan-invariant fresh compact-pair rebuild with fixed-cap padded outputs, then add a moved-endpoint strict fused parity regression.

## 2026-06-20 Safe Rebuild Attempt Status

- Fresh compact-pair rebuild in strict fused scan is not yet production-correct: traced compact output is fixed-cap padded and M2L currently treats padded slots as valid.
- Node-interaction fallback is not tracer-safe due a yggdrax Python `bool(...)` on a traced overflow flag.
- Safe default remains fail-fast for stale compact-pair reuse; unsafe opt-in remains available only via `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1`.
- Next code target is masked/count-aware fixed-cap compact far pairs plus a moved-endpoint strict fused parity regression.


## 2026-06-20 Safe Fresh Compact-Pair Default

- Implemented count-aware fixed-cap compact far pairs: `CompactTaggedFarPairs` now carries `far_pair_count`, and jaccpot M2L masks padded/sentinel pairs before gathers and segment accumulation.
- Strict fused static-radix refresh now defaults to fresh compact-pair rebuild; stale compact-pair reuse remains available only with explicit unsafe opt-in.
- Focused 200k strict self-only default is finite/sane (`vel_max≈6.738`), while unsafe opt-in still corrupts (`vel_max≈200840`).
- Full ODISSEO 200k/3 with fixed policy and `--require-finite-final-state` passed: report `notebooks/scalability/reports/galaxy_disk_profile_20260620_200420.json`, `final_state_all_finite=true`, `final_state_nan_count=0`.
- Runtime impact is real: corrected warm focused self-only 200k/1 is `~1.49s` versus the corrupt unsafe path at `~0.40s`. Active M2L chunk skipping did not materially improve this, so the next work is Nsight attribution of the corrected default path.

## 2026-06-20 Corrected Default Nsight and Test Hardening

- Added and passed a jaccpot regression for padded compact far-pair M2L masking: `/export/home/tbuck/jaccpot/tests/integration/test_fmm.py::test_solidfmm_m2l_ignores_padded_compact_far_pairs`.
- The regression exposed and fixed a chunked scatter bug where invalid padded targets could overwrite validity for a real target after sorting.
- Corrected fused-on 200k/1 Nsight audit is saved under `/tmp/odisseo_nsys_corrected_safe_200k_1_20260620`.
- Fused-on measured step is `0.4425s`, fallback `0`, planner bypass count `2`, acceleration carry active, endpoint self-FMM evaluations `1`.
- Nsight tail has zero H2D/DtoH calls, `10837` kernels, and `2.74%` GPU active over the measured tail. The dominant launch families are repeated `loop_add_fusion_2`, `wrapped_compare`, `loop_compare_dynamic_slice_fusion`, `loop_compare_fusion_5`, and `loop_dynamic_slice_fusion_3`.
- Next optimization target: reduce fixed-cap traced refresh/M2L scan launch fragmentation around the `1563` M2L chunks and repeated loop/update kernels.

## 2026-06-21 Production Performance Update

- Flat compact far pairs are fixed-policy default-on at cap `131072`; canonical M2L chunks fell from `1563` to `32` with fail-fast overflow and rollback `JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS=0`.
- Static-radix node ranges are now reused by default because fixed Morton count buckets preserve identical ranges across refresh; rollback `YGGDRAX_STATIC_RADIX_REUSE_NODE_RANGES=0`.
- Corrected 200k/1 improved from `0.4425s`, `10837` tail kernels to `0.3119s`, `1504` tail kernels; no H2D/DtoH transfers, fused active, fallback `0`.
- Canonical 200k/20 is finite and physically sane at `6.6767s` total / `0.333835s per step`, with exactly 20 endpoint self-FMM evaluations.
- Cross-process 20-step elementwise trajectory comparison is chaotic even for identical settings; correctness gates remain moved-endpoint force parity, short-horizon parity, finite state, and distributional sanity.
- Next target is the remaining paired `392` dynamic-slice/reduce launch family, followed by `483` add/update launches. Full details: `docs/STATIC_RADIX_200K_GPU_PERF_HANDOFF_2026-06-10.md`.

## 2026-06-21 Nearfield Unroll Tuning Decision

- Tested pure-JAX nearfield unroll knobs for the remaining `392 + 392` dynamic-slice/reduce launch families.
- Tile size `8` is rejected: it slows canonical 200k/1 from `0.2896s` to `0.3567s` despite clean parity.
- Tile-scan unroll `2`, tile-scan unroll `4`, batch-scan unroll `2`, and combined tile-scan `4` plus batch-scan `2` all preserve parity and strict route gates, but only improve 200k/1 by about `0.5–1.3%`.
- No production default change: the signal is too small to promote without a material Nsight launch-count reduction.
- Current correctness fix remains fresh compact far-pair rebuild, active-count M2L masking, and velocity-Verlet acceleration carry; these unrolls preserve the fixed force path but are not force-error fixes.
- Next target is structural nearfield batching or a guarded Pallas prototype for the residual dynamic-slice/reduce launches. Full details: `docs/STATIC_RADIX_200K_GPU_PERF_HANDOFF_2026-06-10.md`.

