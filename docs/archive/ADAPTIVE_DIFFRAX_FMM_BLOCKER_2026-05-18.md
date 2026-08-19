# Adaptive Diffrax FMM Blocker (2026-05-18)

## Current Status
- Odisseo now dispatches `FMM_ACC` by timestep mode:
  - fixed -> `integrate_leapfrog_jaccpot_active`
  - adaptive -> `integrate_diffrax_jaccpot_active`
- Adaptive config knobs are wired (`rtol/atol/min_dt/max_dt`, refresh cadence knobs).
- Reporting includes adaptive mode labels and adaptive counters.
- Unit dispatch test passes (`tests/test_integration_api.py`).
- Direct tiny adaptive smoke now completes (no tracer leak) with static-radix under current refactor.

## Updated Blocker Status
The previous immediate tracer-leak failure in tiny adaptive runs is currently mitigated by:
- side-effect-free Odisseo adaptive RHS (no Python prepared-state cache mutation),
- explicit jaccpot stateful-cache disable override in adaptive mode.

Remaining blocker is now shifted from "cannot run at all" to:
- proving this scales to production-size strict static-radix runs with acceptable compile/runtime overhead,
- then reducing overhead toward near-single-jit performance goals.

### New concrete finding (core-kernel scaffold, 2026-05-18 late session)

Adaptive scaffold benchmarking now works and shows a real micro speedup from the shared seam itself, but refresh reuse is still inactive:

- `adaptive_core_scaffold_exec_calls > 0`
- `adaptive_core_scaffold_refresh_calls = 0`
- `adaptive_full_prepare_calls` remains equal to RHS calls in tested micro runs.

Root-cause diagnostics:

- with `ODISSEO_FMM_ALLOW_TRACER_PREPARED_CACHE=0`:
  - `adaptive_core_prepared_drop_tracer > 0` (prepared states dropped every call)
  - `adaptive_core_prepared_non_large_n_seen = 0`
- with `ODISSEO_FMM_ALLOW_TRACER_PREPARED_CACHE=1`:
  - tracer drops vanish, but cached states become non-`LargeNPreparedState` in later RHS stages
  - `adaptive_core_prepared_non_large_n_seen > 0`
  - refresh still not used.

Interpretation:
- Python-side prepared-state caching inside diffrax RHS is not a stable carrier for reusable `LargeNPreparedState` across traced solver stages.
- The remaining unlock is to carry refresh-eligible runtime/prepared state in a tracer-safe compiled form (JAX-carried state), not via Python closure cache.

This confirms adaptive RHS still crosses stateful/runtime-caching logic in jaccpot that is not fully tracer-safe under outer JAX transforms (equinox/diffrax JIT stack).

## What Was Already Tried
1. Forced adaptive tree build to avoid LBVH fast-jit path:
   - `fmm_jit_tree=False` in adaptive coupler solver construction.
   - adaptive coupler default tree mode switched to `static_radix`.
2. Guarded one force-scale state mutation in jaccpot:
   - only store `_last_force_scale_nodes` when `allow_stateful_cache=True`.
3. Removed tracer-unsafe Python mutable RHS cache in Odisseo adaptive path:
   - no `prepared`/`prev_pos` Python dict mutation inside diffrax RHS anymore.
   - RHS currently does pure per-call `prepare_state -> evaluate_prepared_state`.
4. Attempted temporary tracer-detection monkeypatch from Odisseo:
   - removed again after pure-RHS refactor; still not sufficient as final fix.

## Conclusion
Adaptive “single giant JIT” requires a deeper jaccpot runtime refactor:
- strict separation of pure functional prepared/evaluate pipeline from any mutable host cache/state path,
- or a dedicated tracer-safe runtime entrypoint used by adaptive RHS.

## Next Implementation Slice (Recommended)
1. Keep explicit cache-policy mode and formalize it in jaccpot API (not private attr).
2. Reintroduce adaptive refresh cadence (every-k RHS or displacement threshold) using tracer-safe runtime state, not Python mutation.
3. Add adaptive strict static-radix performance smoke in Odisseo notebook flow (small N then medium N).
4. Run 200k validation lane and compare utilization/prepare overhead against fixed-step baseline.

## Repro Command
```bash
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 16 \
  --num-steps 1 \
  --adaptive-timestep \
  --fmm-adaptive-rtol 1e-2 \
  --fmm-adaptive-atol 1e-4 \
  --fmm-preset fast \
  --fmm-runtime-path auto \
  --fmm-tree-build-mode lbvh \
  --fmm-leaf-size 8 \
  --profile-breakdown \
  --output /tmp/adaptive_fmm_smoke16.npz
```

## 2026-05-19 Continuation Update (Static-Baseline First)

Context from production investigation:
- We explicitly prioritized the fixed static-radix path as the baseline because non-static lanes are not performance-comparable.
- Goal in this slice: run matched fixed vs adaptive at `n=200000` using identical ICs and single-GPU `autocvd` selection.

### New blockers found while executing static baseline

Two regressions in local jaccpot static/large-N path blocked clean baseline measurement until patched:

1. `NameError: cache_policy is not defined`
- File: `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- Function: `strict_prepare_refresh_and_evaluate`
- Cause: function used `cache_policy` when calling `_refresh_large_n_same_topology` but did not declare the keyword parameter.
- Hotfix applied locally: add `cache_policy: str = "auto"` to function signature.

2. `NameError: _env_bool is not defined`
- File: `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
- Cause: `_env_bool` was referenced in prepare-speed layout logic outside the scope where `_env_bool` is defined.
- Hotfix applied locally: replace that callsite with direct env parsing for `JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS`.

### Run status after hotfix

- Static strict smoke (`n=200000`, `num_steps=5`) completed:
  - Runtime: `102.868 s`
  - Report: `/tmp/odisseo_perf_compare/fixed/galaxy_disk_profile_20260519_132122.json`
- Static strict baseline (`n=200000`, `num_steps=10`) completed:
  - Runtime: `126.464 s`
  - Report: `/tmp/odisseo_perf_compare/fixed/galaxy_disk_profile_20260519_132534.json`
- Adaptive strict (`n=200000`) with `--adaptive-prepared-cache-mode python --fmm-adaptive-refresh-rhs-calls 8`:
  - `num_steps=10` run did not complete in a practical window and was terminated.
  - follow-up `num_steps=2` diagnostic run also remained long-running and was terminated for checkpointing.

Interpretation so far:
- The user concern is confirmed directionally: adaptive path remains far slower than fixed static baseline in production-like settings.
- We still need completed adaptive reports in the same lane to quantify whether slowdown is dominated by RHS call count vs refresh miss/rebuild behavior.

### Next exact step to resume

1. Re-run adaptive with the same IC file and strict static lane at very small horizon first (`num_steps=1` then `2`) until a report is produced.
2. Extract and compare these fields against fixed report:
   - `adaptive_rhs_evals_estimate`
   - `adaptive_full_prepare_calls`
   - `adaptive_refresh_prepare_calls`
   - `adaptive_core_scaffold_refresh_calls`
   - `adaptive_seconds`
3. If `adaptive_full_prepare_calls ~ adaptive_rhs_evals_estimate`, prioritize moving prepared/runtime state into JAX-carried solver state (not Python runtime cache).

### Repro commands used in this slice

Fixed baseline (strict static-radix):
```bash
JACCPOT_STATIC_STRICT_GPU_MODE=on \
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 10 --state-dtype float32 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-tree-build-mode static_radix \
  --fmm-refresh-every 1 --profile-breakdown \
  --report-dir /tmp/odisseo_perf_compare/fixed \
  --ic-source load --ic-input-path /tmp/odisseo_perf_compare/shared_ic.npz \
  --output /tmp/odisseo_perf_compare/fixed/final_state_fixed_10.npz
```

Adaptive probe (same IC and static lane):
```bash
JACCPOT_STATIC_STRICT_GPU_MODE=on \
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 2 --state-dtype float32 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-tree-build-mode static_radix \
  --adaptive-timestep --adaptive-prepared-cache-mode python \
  --fmm-adaptive-refresh-rhs-calls 8 \
  --fmm-adaptive-rtol 1e-3 --fmm-adaptive-atol 1e-6 \
  --profile-breakdown \
  --report-dir /tmp/odisseo_perf_compare/adaptive \
  --ic-source load --ic-input-path /tmp/odisseo_perf_compare/shared_ic.npz \
  --output /tmp/odisseo_perf_compare/adaptive/final_state_adaptive_2.npz
```

## 2026-05-19 Reliable Reporting + Early Scaling Cliff Checkpoint

### What was completed in this slice

1. Adaptive reporting reliability cleanup in Odisseo (`odisseo/jaccpot_coupling.py`)
- Added explicit reliable adaptive step-attempt fields:
  - `adaptive_step_attempts_estimate`
  - `adaptive_rejected_step_fraction`
- Added raw normalized diffrax stats dump:
  - `adaptive_diffrax_stats_raw`
- Kept backward compatibility while marking ambiguity:
  - `adaptive_rhs_evals_estimate` now mirrors step attempts
  - `adaptive_rhs_evals_estimate_deprecated` preserved
- Added reliability flag:
  - `adaptive_tracing_side_effect_counters_reliable=false`
- Clarified tracing-only prepare counters:
  - `adaptive_tracing_prepare_counter_full`
  - `adaptive_tracing_prepare_counter_refresh`

2. Fixed strict-lane summary diagnostics in Odisseo fixed path
- Added strict-run summary fields in timing reports:
  - `strict_production_lane_active`
  - `strict_runner_wall_seconds`
  - `runtime_strict_unaccounted_seconds`
  - `runtime_strict_refresh_share_of_wall`

3. Tiny-N static-radix blocker fix in local jaccpot
- File: `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py`
- Function: `build_large_n_target_owned_blocks_static`
- Fix: handle `neighbors.shape[0] == 0` by early returning all-invalid static blocks
- Effect: avoids out-of-range gather crash for tiny particle counts.

### Validation status

- `python3 -m py_compile odisseo/jaccpot_coupling.py` passed.
- `pytest -q tests/test_adaptive_core_kernel_flag.py tests/test_scalability_timing_gates.py` passed (`7 passed`).
- Tiny static-radix fixed/adaptive probes now complete at `n=256`.

### Key measured signals

Matched tiny lane (`n=256`, `num_steps=1`, static-radix, strict mode on, exact cap-profile match off):

- Fixed report:
  - `/tmp/odisseo_perf_compare/fixed/galaxy_disk_profile_20260519_213853.json`
  - runtime: `13.48 s`
- Adaptive report:
  - `/tmp/odisseo_perf_compare/adaptive/galaxy_disk_profile_20260519_213438.json`
  - runtime: `9.16 s`
  - `adaptive_num_accepted_steps = 75`
  - `adaptive_num_rejected_steps = 39`
  - `adaptive_rejected_step_fraction = 0.3421`
  - `adaptive_step_attempts_estimate = 114`
  - `adaptive_tracing_side_effect_counters_reliable = false`

Scaled points (`n=512`, `n=1024`) showed early runtime blow-up (runs were stopped after exceeding quick-diagnostic window), indicating an early scaling cliff.

### Interpretation

- Prior confusion about "3 full prepares" was valid: those counts are tracing-side and not execution-reliable.
- The reliable early signal is high adaptive step-attempt count and high reject fraction.
- Current dominant issue appears to be adaptive controller/attempt behavior multiplied by expensive FMM RHS cost per attempt, not directly the old Python prepare counter.

### Recommended next slice

Run a bounded tolerance sweep on static-radix adaptive lane to reduce reject pressure before deeper runtime refactors:

- particle counts: `n=256`, `n=512`
- tolerance grid:
  - `(rtol, atol) = (1e-2, 1e-4)`
  - `(rtol, atol) = (3e-2, 3e-4)`
  - `(rtol, atol) = (1e-1, 1e-3)`
- each run hard-stopped at ~90s wallclock
- record per run:
  - runtime
  - accepted/rejected counts
  - rejected fraction
  - step attempts
  - runtime per accepted step
  - runtime per attempt

Goal: identify a stable adaptive operating point and quantify how much loss is from step rejection dynamics versus deeper runtime-state limitations.
