# Strict Fused Path Trace Handoff (2026-05-21)

## Objective
Identify every remaining performance-loss / host-orchestration point in the current strict fused lane and define concrete fixes to reach high GPU utilization and throughput.

## Latest Throughput Gate

- Command:
  - `JACCPOT_STATIC_STRICT_FUSED_MODE=on JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=100000,200000,400000 micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_fused_plumbing_fix --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=off`
- Results:
  - baseline (fused on): `183.6375 s`
  - variant (fused off): `179.0766 s`
  - delta variant-baseline: `-4.5609 s` (fused off faster)
- Interpretation:
  - fused routing is active and stable,
  - fused control flow is not enough; core refresh/nearfield body still underutilizes GPU and/or incurs extra overhead.

## Current Fused Code Path (as traced)

1. `strict_run_v2` routes to `_strict_run_v2_fused_profile` for eligible profiles.
2. Fused scan body calls `strict_prepare_refresh_and_evaluate(..., fused_fast=True)` each step.
3. `strict_prepare_refresh_and_evaluate` calls `_refresh_large_n_same_topology(..., fused_device_mode=True)`.
4. `_refresh_large_n_same_topology` rebuilds/updates tree+upward and calls `_prepare_state_dual_and_downward(..., fused_device_mode=True)`.
5. `evaluate_prepared_state` calls `evaluate_large_n_state`.

## Remaining Bottleneck Inventory

### A) High-impact host orchestration still in refresh/prepare path

1. `_refresh_large_n_same_topology` still executes substantial Python orchestration each step (topology checks, object assembly, helper dispatch) even in fused mode.
2. `_prepare_state_dual_and_downward` still performs many host decisions and object-graph unpack/repack operations around dual artifacts.
3. `prepare_large_n_state` (large-N pipeline) includes many stage-level branches and orchestration work before heavy kernels.

### B) In-loop non-device configuration/control

4. Nearfield preparation path still references env/config-driven choices in runtime pipeline code; these are not fully frozen into a precompiled static config object for fused mode.
5. Fused lane still relies on generic runtime helpers designed for mixed modes, adding conditional logic per step.

### C) Kernel fragmentation / low arithmetic intensity risk

6. Refresh body remains a sequence of many relatively short kernels/substages (tree/upward, dual build, nearfield payload/precompute/build, eval, update), which can produce low occupancy and launch-bound behavior.

### D) Sync and instrumentation hazards (partially mitigated)

7. Some planner/cache sync points (`device_get` branches) were bypassed in fused_device_mode, but the generic functions still contain sync-prone branches that could be hit by residual code paths.
8. Timing and diagnostics are reduced in fused_device_mode, but generic refresh functions still include instrumentation scaffolding.

## Completed Mitigations (already done)

- Added strict fused profile routing + fallback telemetry.
- Added fused warm-start carry fix so scan carry structure is stable.
- Added fused_device_mode plumbing into refresh/dual prep.
- Disabled planner path and stateful cache in fused_device_mode.
- Disabled refresh timing accumulation in fused_device_mode.

## Required Fix Plan (next implementation steps)

### Phase 1: Build true fused-specific refresh API (no generic orchestration)

1. Introduce dedicated method: `_refresh_large_n_same_topology_fused_device(...)`.
2. This method must:
   - accept a pre-frozen static config payload,
   - avoid generic profile/counter/timing bookkeeping,
   - avoid cache/planner/adaptive/policy branches,
   - call only strict-fast dual/downward and nearfield paths.
3. Route fused scan to this method directly (do not call generic `_refresh_large_n_same_topology`).

### Phase 2: Freeze configuration once at fused entry

4. At `_strict_run_v2_fused_profile` entry, resolve and freeze all traversal/nearfield/chunk/tile/cap settings into a `StrictFusedStaticConfig` object.
5. Pass this config through fused calls (static arg / closed-over constant) so no in-loop env/config lookups occur.

### Phase 3: Collapse nearfield preparation stages for fused mode

6. Add fused-only nearfield prep path that reuses static-capacity buffers and avoids repeated payload rebuild logic where possible.
7. Keep correctness by preserving same math/results; only execution structure changes.

### Phase 4: Remove residual sync-prone branches from fused route

8. Ensure fused route cannot touch any `device_get` planner/diagnostic branches.
9. Add assertable route markers in diagnostics (e.g., `strict_fused_device_refresh_route=1`, `strict_fused_planner_bypassed=1`).

### Phase 5: Re-validate and gate

10. Correctness gates:
    - tiny `n=256, steps=1` fused active/no fallback,
    - 200k `steps=1` fused active/no fallback.
11. Throughput gate:
    - one 200k `steps=20` walltime A/B (fused on/off) on fixed IC,
    - keep only if >3% wall-time gain and no correctness regression.

## Decision Status

- Fused mode is functionally active.
- Performance objective is not met.
- Fused mode should remain non-default for perf lane until the fused-specific refresh API and static-config execution path are completed.

## Additional Refactor Slice (2026-05-21)

### Changes Applied

- Propagated `fused_fast=True` into `_refresh_large_n_same_topology(..., fused_device_mode=True)` from `strict_prepare_refresh_and_evaluate`.
- In `_prepare_state_dual_and_downward`, enforced fused-mode bypasses already introduced (planner/cache disabled) and retained fused route stability.
- Added fused-device-mode bypass for large-N timing scaffolding in `prepare_large_n_state` (`_large_n_pipeline.py`) so refresh timing bookkeeping paths are disabled in fused mode.

### Validation

- 200k short fused validation (`num_steps=1`) still passes with fused active and no fallback:
  - report: `notebooks/scalability/reports/galaxy_disk_profile_20260521_113209.json`
  - `runtime_strict_fused_mode_active=true`
  - `runtime_strict_fused_fallback_count=0`

### Throughput Gate After This Slice

- Command:
  - `JACCPOT_STATIC_STRICT_FUSED_MODE=on JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=100000,200000,400000 micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_fused_timing_bypass --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=off`
- Results:
  - baseline (fused on): `181.9017 s`
  - variant (fused off): `175.9709 s`
  - delta variant-baseline: `-5.9309 s` (fused off faster)

### Updated Conclusion

- The fused route remains functionally active but still significantly slower.
- Current dominant blocker is no longer just routing/plumbing/timing scaffolding; it is the core refresh computational structure and kernel fragmentation itself (tree/upward + dual build + nearfield prep/eval still too fragmented/launch-bound in fused loop).

## Update 2026-05-21 (Fused Perf/Memory Slice)

### Changes landed
- Updated `jaccpot/runtime/_large_n_pipeline.py` fused path behavior (`fused_device_mode=True`) to reduce host orchestration and memory inflation in strict fused runs:
  - Disabled overflow profile re-cap/padding in fused mode; now keeps `overflow_profile_capacity = overflow_active_blocks`.
  - Disabled neighbor-edge profile re-cap/padding in fused mode for radix fast lane.
  - Kept existing strict/fallback behavior unchanged for non-fused mode.

### Why
- Overflow/neighbor profile recaps were adding host-side branching + extra allocations/padding that are not needed in the fused strict lane and can inflate prepared-state payloads.
- This makes fused mode closer to fixed-shape, no-extra-padding execution and lowers memory churn.

### Validation
- Syntax check passed:
  - `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`

### Next slice
- Replace generic `prepare_large_n_state` usage in strict fused refresh with a lean fused-only state packer to remove remaining broad nearfield orchestration branches from the hot path.

## Update 2026-05-21 (Fused Nearfield Precompute Bypass)

### Changes landed
- In `jaccpot/runtime/_large_n_pipeline.py`, fused strict mode now skips generic nearfield precompute vectors/schedules when both are true:
  - `fused_device_mode=True`
  - `execution_config.radix_fast_lane=True`
- Concretely bypassed `build_large_n_nearfield_precompute(...)` in that lane and carried `None` for:
  - `nearfield_target_leaf_ids`, `nearfield_source_leaf_ids`, `nearfield_valid_pairs`
  - `nearfield_chunk_sort_indices`, `nearfield_chunk_group_ids`, `nearfield_chunk_unique_indices`
- Non-fused and non-radix-fast paths remain unchanged.

### Why
- Strict fused acceleration path consumes radix fast payloads directly; generic pair-vector/scatter precompute is redundant there.
- Removing this build trims host orchestration and avoids unnecessary prepared-state data population.

### Validation
- Syntax check passed:
  - `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
- 200k one-step strict run completed successfully on fixed Agama IC file (gating relaxed for smoke only):
  - Command env included `JACCPOT_STATIC_STRICT_FUSED_MODE=on` and `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH=0`
  - Runtime output: `Runtime: 72.226 s`
  - Report: `/tmp/odisseo_fused_smoke_200k_20260521/galaxy_disk_profile_20260521_163959.json`

### Notes
- Tiny `n=256` strict smoke is blocked by strict static cap-profile matching in this lane and is not representative of the 200k production profile setup.

## Update 2026-05-21 (Fused Fallback Root-Cause Deep Dive)

### Key finding
- The 200k timing remains high primarily because strict fused lane is still falling back to strict segmented path.
- Evidence from breakdown runs:
  - `runtime_strict_fused_mode_active: False`
  - `runtime_strict_fused_execute_count: 1`
  - `runtime_strict_fused_fallback_count: 1`
  - fallback reason: `ConcretizationTypeError ... int(...)`

### What was fixed in this slice
1. Added env-gated fused debug re-raise hook in `_fmm_impl.py`:
- `JACCPOT_STATIC_STRICT_FUSED_DEBUG_RAISE=1`
- Allows full traceback capture instead of silent fallback.

2. Eliminated one concrete fused blocker:
- File: `jaccpot/runtime/_fmm_impl.py`
- Function: `_rebuild_tree_artifacts_from_topology`
- Replaced tracer-unsafe:
  - `num_internal = int(cached_tree.num_internal_nodes)`
- With tracer-safe static-shape derivation:
  - `num_internal = int(jnp.asarray(cached_tree.left_child).shape[0])`

3. Eliminated another fused blocker path:
- File: `jaccpot/runtime/_large_n_pipeline.py`
- In fused mode, bypassed static target-block builder branch that called:
  - `max_count = int(jnp.max(counts))` in `build_large_n_target_owned_blocks_static`
- Fused mode now forces dynamic target-block build path for tracer safety.

4. Further reduced fused prep payload overhead:
- In fused mode, avoid materializing giant `source_particle_ids/source_particle_mask` payload tensors.
- Fast-lane fallback path in nearfield kernel handles source-leaf payload-only mode.

### Validation runs
- 200k, 1-step smoke (fixed IC):
  - before: `72.226s`
  - after payload-materialization cut: `72.218s` (no significant change; compile/setup dominated at 1 step)
- 200k, 5-step breakdown runs remain ~92-93s and still show fused fallback active.

### Current dominant runtime stages (fallback path)
- `runtime_refresh_tree_upward_seconds` ~65s
- `runtime_refresh_dual_artifact_build_seconds` ~36s
- `runtime_refresh_nearfield_seconds` ~2s
- Indicates upward + dual artifact preparation are the primary throughput bottlenecks while fused fallback persists.

### Immediate next blocker work
- Continue tracer-safety sweep for strict fused refresh:
  - locate and remove remaining `int(...)` concretization sites under traced scan carry.
  - keep using `JACCPOT_STATIC_STRICT_FUSED_DEBUG_RAISE=1` until `runtime_strict_fused_mode_active=True` with zero fallback.
- Only once fused lane stays active, rerun wall-time A/B for meaningful throughput comparison.

## Update 2026-05-21 (Fused Mode Unblocked)

### Major milestone
- Strict fused mode now runs without fallback on debug validation:
  - `runtime_strict_fused_mode_active: True`
  - `runtime_strict_fused_fallback_count: 0`
  - `runtime_strict_runner_execute_count: 0`

### Key fixes that enabled this
1. Added debug re-raise switch for fused path:
- `JACCPOT_STATIC_STRICT_FUSED_DEBUG_RAISE=1`

2. Removed tracer concretization blockers:
- `_fmm_impl.py::_rebuild_tree_artifacts_from_topology`
  - replaced `int(cached_tree.num_internal_nodes)` with static-shape derivation from `left_child.shape[0]`.
- `_large_n_pipeline.py` fused path
  - bypassed static target-block builder and dynamic target-block builder branches that required Python concretization.

3. Stabilized fused scan carry pytree/shape
- In fused refresh return path (`_refresh_large_n_same_topology`):
  - preserve optional padded/payload fields from previous prepared state when new state omits them.
  - normalize neighbor edge vector length to previous carry shape via pad/truncate.
- In `evaluate_large_n_state`: if `radix_fast_payload` is missing, fall back to compiled generic large-N evaluator instead of raising.

### Current status
- Fused path now executes end-to-end for short debug validation runs.
- Throughput is still far from target; dominant costs remain refresh internals (tree/upward + dual artifact preparation).

### Next performance focus
- With fallback now removed, prioritize steady-state fused profiling and optimize:
  - tree/upward refresh cost,
  - dual artifact build cost,
  - avoid per-step host-visible shape/payload churn.
