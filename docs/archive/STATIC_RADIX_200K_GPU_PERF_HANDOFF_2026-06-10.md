# Static-Radix 200k GPU Performance Handoff (2026-06-10)

## Purpose
- Capture the current status of the ODISSEO + jaccpot strict static-radix 200k production-performance push.
- Preserve findings from Nsight, timing decomposition, and surgical optimization experiments.
- Define the next path toward the target: canonical 200k Agama disk, warm full ODISSEO simulator step `<0.5s`, strict fused active, fallback `0`, no hot-loop host round trips, and no Python cadence loop.

## Run Discipline
- Use `micromamba run -n odisseo`.
- Use `--require-autocvd` for benchmark/audit runs so only one free GPU is selected.
- Use canonical IC unless otherwise stated:
  - `/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz`
- Use fixed/static policy:
  - `--fixed-policy`
  - `--fixed-neighbor-cap 1048576` for current S1 unblock/profiling work.
- Treat compile time, IC loading, first profile miss, and report writing as outside the steady-state timing target.

## Current State Summary
- Strict fused static-radix route is active for canonical 200k runs.
- Host fallback remains `0` in the validated strict fused runs.
- Warm 200k/2 strict fused timing is still about `3.86s–3.91s` total, or about `1.93s–1.96s/step`.
- This is still far from the production target of `<0.5s` per full ODISSEO step.
- ODISSEO external force/integration overhead is negligible compared with jaccpot refresh/eval cost.
- The main remaining problem is not the ODISSEO wrapper; it is strict jaccpot refresh/upward/downward/eval orchestration and GPU launch fragmentation.

## Key Nsight Finding
- Successful graph-node Nsight capture:
  - `/tmp/odisseo_nsys_baseline_cuda_graph_node_200k_2/baseline_graph_node.nsys-rep`
- Derived summary:
  - `/tmp/odisseo_nsys_baseline_cuda_graph_node_200k_2/measured_tail_summary.json`
- Whole-process GPU active was only about `7.2%` because compile/import/warmup dominate the full trace window.
- Measured warm-step tail GPU active was about `80.7%`, so the steady measured region is not fully saturated.
- Measured tail had about `169k` GPU events/kernels for one `200k/2` run.
- Largest measured-tail idle gap was about `261 ms`; p95 gaps were tiny, so the core issue is launch fragmentation / graph fragmentation rather than one persistent host stall.
- Top measured-tail kernels from the graph-node capture:
  - `loop_multiply_fusion_1`: about `994 ms`, `8193` launches.
  - `input_reduce_fusion_3`: about `895 ms`, `4203` launches.
  - `input_reduce_fusion_1`: about `894 ms`, `4877` launches.
- CUDA API time was dominated by `cuGraphLaunch`:
  - about `21,699` calls and about `6.3s` whole-process API time.

## ODISSEO vs jaccpot Timing Split
- External-only strict timing mode:
  - Summary: `/tmp/odisseo_timing_mode_external_only_smoke_200k_2/walltime_ab_summary.json`
  - Baseline measured about `0.0024s / 2 steps`.
  - Conclusion: ODISSEO external potential/integration is not the bottleneck.
- jaccpot self-only strict timing mode:
  - Summary: `/tmp/odisseo_timing_mode_self_only_smoke_200k_2/walltime_ab_summary.json`
  - Baseline measured about `3.892s / 2 steps`.
  - Conclusion: full strict fused runtime is essentially jaccpot self-force cost.
- Full strict fused, refresh every step:
  - Summary: `/tmp/odisseo_retained_patches_smoke_200k_2/walltime_ab_summary.json`
  - Baseline measured about `3.909s / 2 steps`.
- Full strict fused, `refresh_every=2` diagnostic:
  - Summary: `/tmp/odisseo_refresh_every2_probe_200k_2/walltime_ab_summary.json`
  - Baseline measured about `2.152s / 2 steps`.
  - Conclusion: refresh cadence strongly affects runtime; refresh/upward/downward work is a primary multiplier.

## Standalone jaccpot Comparison
- Direct prepared-state jaccpot evaluate-only probe on canonical 200k showed repeated eval median about `0.705s`.
- That direct probe is much faster than ODISSEO strict fused refresh-every-step because it evaluates an already prepared state.
- ODISSEO strict fused is doing refresh/rebuild/upward/downward/local/eval cadence work, not only prepared-state force evaluation.
- This explains why standalone jaccpot “sub-second” timing does not directly translate to the current full ODISSEO step.

## Retained Code Changes From This Pass

### 1. Nsight/audit runner improvements
- `tools/fused_audit_runner.py`
  - Added warm/measure pass-through for benchmark capture.
  - Default Nsight trace now uses CUDA graph node tracing via `--cuda-graph-trace node`.
  - Added measured-tail GPU metrics derived from the final `script_runtime_seconds` window.
  - Added `measured_tail_gpu_active_percent`, measured-tail busy/span, host idle gaps, and kernel count to summary metrics.
  - Added fixed-policy metadata for `JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN`.

### 2. Walltime fixed-policy metadata/defaults
- `tools/walltime_ab_compare.py`
  - Fixed-policy environment now includes `JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN=1`.
  - Frozen baseline metadata records this knob.

### 3. Local expansion direct flatten
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
  - Added guarded `JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN` path in solid-FMM local expansion particle evaluation.
  - When enabled and no acceleration derivatives are requested, leaf-major local gradients are flattened directly instead of scatter-added into an already sorted contiguous particle buffer.
  - Small static-radix parity check passed exactly: `max_abs 0.0`, `rms 0.0`.
  - Canonical 200k/2 fixed-policy smoke with this default:
    - `/tmp/odisseo_fixed_policy_flatten_default_smoke_200k_2/walltime_ab_summary.json`
    - Strict fused active `true`, fallback `0`.
    - Measured about `3.862s / 2 steps`.
  - This is a real but modest improvement from prior `~3.91s / 2 steps`.

### 4. Default-off near/far diagnostic split
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
  - Added default-off diagnostic flags:
    - `JACCPOT_LARGE_N_EVAL_DISABLE_NEAR=1`
    - `JACCPOT_LARGE_N_EVAL_DISABLE_FAR=1`
  - Purpose: split final large-N evaluate branch cost during experiments.
  - Current result: disabling either near or far branch in this diagnostic did not materially reduce full strict fused runtime, indicating the remaining cost is dominated by refresh/upward/downward/compiled orchestration rather than just final near/far summation.

## Experiments Tried And Rejected

### Static nearfield/target-block knob sweeps
- Larger target batch/tile settings were tested:
  - `JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE=64`, `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE=16`
  - `JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE=128`, `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE=32`
- Results were tiny or negative:
  - `/tmp/odisseo_batch64_tile16_probe_200k_2/walltime_ab_summary.json`
  - `/tmp/odisseo_batch128_tile32_probe_200k_2/walltime_ab_summary.json`
- Not promoted.

### Scan unroll experiments
- `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL=4`
- `JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL=4`
- Accurate warm/measure run was about `3.877s / 2 steps`, only tiny improvement/noise-level.
- Not promoted.

### Dense radix payload materialization
- Tried opt-in materializing source-particle radix payload in strict fused mode.
- Also tried keeping the radix payload in strict fused carry.
- Combined dense+keep experiment did not improve runtime:
  - `/tmp/odisseo_keep_dense_radix_payload_probe_200k_2/walltime_ab_summary.json`
- Reverted.

### Disable strict fused rematerialization
- `JACCPOT_STATIC_STRICT_FUSED_DISABLE_REMATERIALIZE=1`
- Result was slightly worse:
  - `/tmp/odisseo_disable_remat_probe_200k_2/walltime_ab_summary.json`
- Not promoted.

### Analytic local-gradient evaluator
- Added and tested an analytic solid-FMM local-gradient evaluator to avoid per-particle autodiff.
- Direct evaluator parity matched to float noise.
- Prepared-state parity matched exactly in the tested static-radix case.
- Full 200k strict fused timing was slower/no better:
  - `/tmp/odisseo_analytic_local_grad_probe_200k_2/walltime_ab_summary.json`
- Reverted.

### Nearfield direct leaf flatten
- Added and tested a nearfield direct leaf-major flatten path analogous to the retained local-eval flatten.
- Static-radix parity matched exactly.
- Full 200k strict fused timing was worse:
  - `/tmp/odisseo_nearfield_direct_flatten_probe_200k_2/walltime_ab_summary.json`
- Reverted.

### Fixed template permutation reuse
- Added opt-in strict static-radix refresh path to reuse the template permutation and skip Morton encode/lexsort/reorder.
- Timing improvement was small/noise-level and semantic risk is high because particles move between refreshes:
  - `/tmp/odisseo_reuse_template_permutation_probe_200k_2/walltime_ab_summary.json`
- Reverted.

### Near-only / far-only evaluate split
- Near-only diagnostic:
  - `JACCPOT_LARGE_N_EVAL_DISABLE_FAR=1`
  - `/tmp/odisseo_eval_near_only_probe_200k_2/walltime_ab_summary.json`
  - Still about `3.895s / 2 steps`.
- Far-only diagnostic:
  - `JACCPOT_LARGE_N_EVAL_DISABLE_NEAR=1`
  - `/tmp/odisseo_eval_far_only_probe_200k_2/walltime_ab_summary.json`
  - Still about `3.912s / 2 steps`.
- Interpretation: these diagnostics did not isolate a cheap final near/far branch. The dominant cost remains refresh/upward/downward/compiled graph orchestration.

## Working Interpretation
- The current strict fused path is correctly avoiding host fallback and running under the strict fixed policy.
- The remaining performance gap is not caused by ODISSEO external forces, Python cadence loops, or obvious final near/far scatter overhead.
- Nsight points to huge kernel/CUDA graph launch density.
- Refresh cadence tests point to refresh/upward/downward recomputation as the main multiplier.
- The path to `<0.5s` probably requires reducing the compiled refresh graph itself, not another small wrapper or scatter tweak.

## Recommended Path Forward

### Phase 1: Profile refresh/upward/downward subgraphs directly
- Add diagnostic compile modes that skip or isolate these components inside strict `_strict_fused_segment_batch_compiled`:
  - tree/template rebuild only
  - upward only
  - downward/local M2L only
  - evaluate-only after refresh
- These should be default-off env flags, used only for profiling.
- Goal: determine which refresh subgraph produces the `loop_multiply_fusion_*` and `input_reduce_fusion_*` launch storm.

### Phase 2: Nsight graph-node capture on isolated modes
- Re-run `tools/fused_audit_runner.py` with:
  - `--nsys-capture`
  - `--nsys-bin /usr/local/cuda-12.4/bin/nsys`
  - default `--nsys-cuda-graph-trace node`
  - `--perf-warmup-runs 1 --perf-measure-runs 1`
- Compare measured-tail kernel counts and top kernels for each isolated mode.
- Goal: map top kernel names to refresh/upward/downward stages rather than guessing from final full-step traces.

### Phase 3: Reduce strict refresh graph fragmentation
- If upward dominates:
  - inspect mass moment / P2M / M2M loops for per-level scans or per-node small kernels.
  - consider fusing level loops or using larger batched kernels for static radix levels.
- If downward/M2L dominates:
  - inspect compact far-pair application and local accumulation for per-pair/per-level scans.
  - use fixed compact far pairs already cached, but reduce graph launch count in `_prepare_downward_with_artifacts` / M2L application.
- If tree/template rebuild dominates:
  - focus on Morton encode/lexsort/reorder and static topology rebuild alternatives, but fixed-permutation reuse alone was not enough and is semantically risky.

### Phase 4: Re-run production gates only after a real isolated win
- First gate: canonical 200k/2 strict fused probe.
- Then 200k/20 walltime with fixed policy.
- Then Nsight measured-tail capture.
- Promotion criteria remain:
  - strict fused active `true`
  - fallback `0`
  - blockers empty
  - no hot-loop host round trips
  - warm full ODISSEO step median trending toward `<0.5s`

## Commands Worth Reusing

### Canonical 200k/2 fixed-policy smoke
```bash
timeout 900 micromamba run -n odisseo python tools/walltime_ab_compare.py \
  --out-root /tmp/odisseo_fixed_policy_smoke_200k_2 \
  --n-particles 200000 \
  --num-steps 2 \
  --fixed-policy \
  --fixed-neighbor-cap 1048576 \
  --require-autocvd \
  --perf-warmup-runs 1 \
  --perf-measure-runs 1 \
  --profile-breakdown \
  --include-stdout-tail \
  --status-interval-seconds 30
```

### Nsight graph-node capture
```bash
timeout 900 micromamba run -n odisseo python tools/fused_audit_runner.py \
  --out-root /tmp/odisseo_nsys_s1_200k_2_graphnode \
  --run-class S1 \
  --fixed-policy \
  --fixed-neighbor-cap 1048576 \
  --require-autocvd \
  --nsys-capture \
  --nsys-bin /usr/local/cuda-12.4/bin/nsys \
  --perf-warmup-runs 1 \
  --perf-measure-runs 1 \
  --include-stdout-tail \
  --status-interval-seconds 30 \
  --audit-tag nsys_s1_graphnode_utilization
```

## Open Questions
- Which exact refresh substage owns the largest share of `loop_multiply_fusion_1` and `input_reduce_fusion_*` launches?
- Can upward/downward static-radix level loops be collapsed into fewer larger kernels?
- Can M2L/local accumulation use a more static packed representation that avoids thousands of tiny graph launches?
- Is a physically acceptable refresh cadence greater than `1` possible for production, or must every ODISSEO step recompute full refresh/downward state?
- Does a new jaccpot API for “prepared-eval with controlled topology/update cadence” make sense after isolated refresh profiling, or should we continue patching the current strict runner?

## 2026-06-10 Update: Strict Refresh Attribution Result

After adding `JACCPOT_STRICT_REFRESH_DIAG_MODE`, we ran Nsight graph-node captures for:

- `full`
- `tree_only`
- `upward_only`
- `downward_only`
- `eval_only`
- `integrator_only`

All runs used canonical 200k/2, fixed policy, explicit neighbor cap `1048576`, `--require-autocvd`, and Nsight graph-node tracing.

### Attribution Artifacts

- Aggregate JSON: `/tmp/odisseo_refresh_diag_attribution_200k_2.json`
- Aggregate CSV: `/tmp/odisseo_refresh_diag_attribution_200k_2.csv`
- Per-mode summaries: `/tmp/odisseo_refresh_diag_<mode>_200k_2/audit_summary.json`

### Key Measurements

| mode | 2-step seconds | tail GPU active | tail kernels | `cuGraphLaunch` calls | delta vs integrator |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | `4.040` | `79.5%` | `145632` | `21703` | `+1.711s` |
| `tree_only` | `2.527` | `64.8%` | `79031` | `11781` | `+0.198s` |
| `upward_only` | `2.544` | `65.8%` | `83119` | `11909` | `+0.216s` |
| `downward_only` | `2.576` | `65.2%` | `84322` | `11943` | `+0.248s` |
| `eval_only` | `3.915` | `80.6%` | `130907` | `19977` | `+1.587s` |
| `integrator_only` | `2.329` | `68.9%` | `69554` | `10217` | baseline |

Strict fused was active for the baseline cases and fallback remained `0`.

### Interpretation Change

The previous working hypothesis that strict refresh internals dominate was too broad. Refresh is still costly, but the isolated measurements show:

- Final `evaluate_large_n_state` is now the largest remaining incremental cost.
- `eval_only - integrator_only` is about `+1.59s / 2 steps`, adding about `61k` tail kernels and about `9.8k` `cuGraphLaunch` calls.
- Strict refresh internals add only about `+0.20–0.25s / 2 steps` over the integrator floor.
- Within refresh, tree/template rebuild is the largest safe secondary target; upward and downward/M2L deltas are comparatively small in this capture.

## Revised Performance Plan

### Target A: Final Evaluation Launch Reduction

Goal: reduce `eval_only` toward the `integrator_only` floor before spending more time on refresh internals.

1. Add default-off eval diagnostic modes inside `evaluate_large_n_state`, for example:
   - `JACCPOT_LARGE_N_EVAL_DIAG_MODE=full|near_only|far_only|local_only|near_zero|far_zero|permutation_only|zero`
2. Keep diagnostics profiling-only and shape-stable; nonphysical zero-force outputs are allowed only in diagnostic modes.
3. Surface the active eval mode in ODISSEO/walltime/audit reports.
4. Run 200k/2 fixed-policy probes and Nsight captures for the eval modes.
5. Map top kernels (`input_reduce_fusion_1`, `loop_multiply_fusion_1`, `input_scatter_fusion*`) to nearfield, local expansion, and permutation/scatter stages.
6. Optimize the stage with the largest isolated launch delta.

Likely optimization directions after attribution:

- Collapse repeated per-leaf/per-block final-eval scans into larger fixed-shape batched kernels.
- Reduce scatter/update fragmentation in final particle-order accumulation.
- Keep the validated local direct leaf flatten path on by default, but do not assume it is sufficient.
- Avoid more broad micro-optimizations until a diagnostic split proves the stage dominates.

### Target B: Static Tree/Template Rebuild

Goal: reduce the refresh tax of about `0.20–0.25s / 2 steps` without changing simulation semantics.

1. Add finer tree rebuild diagnostic splits after the eval target is understood:
   - Morton/key generation
   - sort/permutation
   - leaf metadata/counts
   - bounds and node metadata
   - mass/position payload reorder
2. Do not promote fixed-permutation reuse as production unless a separate physics/accuracy gate approves it.
3. Prefer caching invariant static-radix template metadata and avoiding redundant small metadata transforms.
4. Promote only if canonical full-mode timing improves and strict fused remains active with fallback `0`.

### Promotion Gates

Any candidate optimization must pass:

- `python3 -m py_compile` on touched runtime/tools.
- Canonical 200k/2 fixed-policy strict fused probe:
  - fused active `true`
  - fallback `0`
  - planner bypass positive
  - blockers empty
- Canonical 200k/20 fixed-policy walltime check.
- Nsight graph-node capture showing lower measured-tail kernel/graph-launch counts.
- No hot-loop host transfers, no scan-carry shape drift, no cap-dependent recompiles.

### Immediate Next Step

Start with `JACCPOT_LARGE_N_EVAL_DIAG_MODE` in `evaluate_large_n_state`, because final eval is the dominant remaining launch source.

## 2026-06-10 Implementation Start: Final-Eval Diagnostics

Initial implementation for Target A has begun.

### Added

- `JACCPOT_LARGE_N_EVAL_DIAG_MODE` in the radix fast-lane branch of `evaluate_large_n_state`.
- Supported modes:
  - `full`
  - `near_only`
  - `far_only`
  - `local_only`
  - `near_zero`
  - `far_zero`
  - `permutation_only`
  - `zero`
- Existing `JACCPOT_LARGE_N_EVAL_DISABLE_NEAR/FAR` flags remain supported.
- Solver/report metadata now exposes:
  - `large_n_eval_diag_mode`
  - `runtime_large_n_eval_diag_mode`

### Validation

- Compile checks passed for touched jaccpot, ODISSEO, notebook-driver, and tool files.
- Autocvd-backed canonical 200k/1 smoke completed with:
  - `JACCPOT_LARGE_N_EVAL_DIAG_MODE=permutation_only`
  - output root `/tmp/odisseo_eval_diag_permutation_smoke_200k_1`
  - report fields confirmed as `permutation_only`

### Next

Run fixed-policy 200k/2 probes for the eval modes, then Nsight graph-node captures for the modes that produce the largest deltas versus `integrator_only`.

### Correction: Eval Diagnostic Short-Circuit Placement

The first implementation placed `zero` / `permutation_only` inside `evaluate_large_n_state`. Nsight showed that this was too deep: entering the jitted strict fused eval call still retained the full eval graph, and `zero` showed the same top kernels and `cuGraphLaunch` count as `full`.

Fix applied:

- Short-circuit `JACCPOT_LARGE_N_EVAL_DIAG_MODE=zero` and `permutation_only` in the strict fused caller before invoking `evaluate_large_n_state`.
- Keep deeper `near_only` / `far_only` / `local_only` modes inside `evaluate_large_n_state`.

Corrected 200k/2 fixed-policy probes:

- `zero`: `/tmp/odisseo_eval_diag_modes_corrected_200k_2/zero`, `2.3465s / 2 steps`
- `permutation_only`: `/tmp/odisseo_eval_diag_modes_corrected_200k_2/permutation_only`, `2.3108s / 2 steps`

Interpretation:

- Caller-level eval bypass reaches the refresh/integrator floor.
- The production full-mode gap over corrected `zero` is about `1.7s / 2 steps`.
- The earlier `near_only`, `far_only`, and `local_only` probes all remained around `3.87–3.93s / 2 steps`, so both final nearfield and local/farfield eval paths independently trigger a large launch graph.
- Next useful Nsight captures should focus on corrected `zero` versus `near_only` and `far_only`, but note that one direct `near_only` Nsight import failed with an Nsight event-order importer error despite successful program execution.

## 2026-06-11 Update: Nearfield Split Diagnostics

Added a default-off diagnostic split for the radix fast-lane nearfield portion of final evaluation.

### Added

- `JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE=full|self_only|pairs_only|overflow_only|zero`
- Runtime/report metadata:
  - `large_n_nearfield_diag_mode`
  - `runtime_large_n_nearfield_diag_mode`
- JIT cache-key participation for the nearfield diagnostic mode, avoiding stale compiled graph reuse when the mode changes.
- Bounded GPU selection support in the benchmark tools:
  - `tools/walltime_ab_compare.py --autocvd-timeout-seconds <seconds>`
  - `tools/fused_audit_runner.py --autocvd-timeout-seconds <seconds>`

### Implementation Notes

- Production default remains `full`; no behavior changes unless the diagnostic env var is set.
- `zero` returns zero nearfield acceleration before entering the radix fast-lane nearfield kernels.
- `self_only` runs only intra-leaf self interactions and skips payload pair plus overflow additions.
- `pairs_only` skips intra-leaf self interactions and runs the primary payload pair path only.
- `overflow_only` skips the primary payload and isolates overflow payload / target-block overflow work when present.

### Validation

Compile checks passed for:

- `/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py`
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py`
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- `odisseo/jaccpot_coupling.py`
- `notebooks/scalability/galaxy_disk_fmm_large_n.py`
- `tools/walltime_ab_compare.py`
- `tools/fused_audit_runner.py`

### GPU Probe Status

Attempted bounded autocvd-gated 200k/2 probes under `/tmp/odisseo_nearfield_diag_modes_200k_2`, starting with `JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE=zero`.

Result:

- `autocvd` reported `0 / 1 GPU(s) available` for the full 120 second timeout.
- The run exited cleanly with `TimeoutError: Could not acquire 1 GPU(s) before timeout.`
- No timing result was produced yet; retry when the single GPU is free.

### Next Free-GPU Commands

Run nearfield split walltime probes first:

```bash
OUT=/tmp/odisseo_nearfield_diag_modes_200k_2
for mode in zero self_only pairs_only overflow_only; do
  JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE="$mode" \
  micromamba run -n odisseo python tools/walltime_ab_compare.py \
    --out-root "$OUT/$mode" \
    --n-particles 200000 \
    --num-steps 2 \
    --fixed-policy \
    --fixed-neighbor-cap 1048576 \
    --require-autocvd \
    --autocvd-timeout-seconds 120 \
    --perf-warmup-runs 1 \
    --perf-measure-runs 1 \
    --include-stdout-tail \
    --baseline-env JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE="$mode" \
    --variant-env JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE="$mode"
done
```

Then capture Nsight for the highest-delta mode against corrected `zero`, using graph-node tracing and the same fixed policy.

## 2026-06-12 Update: Final-Eval Attribution and Local-Gradient Experiments

### Nearfield Split Result

Strict-fused 200k/2 fixed-policy audit with variant forced back onto fused mode showed that nearfield is not the remaining bottleneck:

- Full baseline: `3.935s / 2 steps`, fused active `true`, fallback `0`, planner bypass `2`
- `JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE=zero`: `3.907s / 2 steps`, fused active `true`, fallback `0`, planner bypass `2`
- Delta: only about `0.03s / 2 steps`

Conclusion: the final-eval launch storm is not primarily the radix fast-lane nearfield path.

### Eval Diagnostic Sweep

Strict-fused 200k/2 fixed-policy eval-mode audits, with variant explicitly forced to fused mode:

| Variant mode | Variant time / 2 steps | Interpretation |
| --- | ---: | --- |
| `JACCPOT_LARGE_N_EVAL_DIAG_MODE=zero` | `2.400s` | Caller-level eval bypass floor |
| `far_zero` | `3.981s` | Disabling far inside eval does not prune graph enough |
| `far_only` | `3.900s` | Far/local eval entry remains expensive |
| `local_only` | `3.944s` | Local eval alone reproduces most full cost |

Conclusion: entering the local/far final-eval graph is the major cost. Branch-level flags inside `evaluate_large_n_state` do not meaningfully reduce launches, while caller-level `zero` does.

### Nsight Full vs Caller-Level Eval Zero

Nsight graph-node capture:

- Root: `/tmp/odisseo_eval_zero_nsys_200k_2_20260612/20260612_101631/eval_zero_nsys_fused_on_manual`
- Full baseline:
  - `4.097s / 2 steps`
  - measured-tail kernels: `145628`
  - total kernels: `297098`
  - `cuGraphLaunch`: `21703`
  - measured-tail GPU active: `80.27%`
- Caller-level eval `zero`:
  - `2.536s / 2 steps`
  - measured-tail kernels: `84309`
  - total kernels: `174432`
  - `cuGraphLaunch`: `11943`
  - measured-tail GPU active: `66.24%`

Delta attributable to final eval entry:

- `+1.56s / 2 steps`
- `+61319` measured-tail kernels
- `+9760` `cuGraphLaunch` calls

Top additional kernels in full vs eval-zero:

- `input_reduce_fusion_3`: `+8190` launches, `+1864.8 ms`
- `loop_multiply_fusion_1`: `+8190` launches, `+1009.3 ms`
- `input_scatter_fusion`: `+8066` launches, `+56.5 ms`
- `input_scatter_fusion_1`: `+8066` launches, `+50.3 ms`
- Several smaller slice/select/scatter fusions also add about `8k` launches each.

This points at the leaf/local-expansion final-eval graph, not ODISSEO wrapper orchestration.

### Local-Gradient Candidate Experiments

Added a default-off local-eval experiment:

- `JACCPOT_LOCAL_EVAL_ANALYTIC_GRAD=1`
- Implements analytic complex local-gradient evaluation without `jax.value_and_grad`.
- CPU parity against the existing autodiff batch path:
  - gradient max abs error about `3e-8` to `5e-8` for order `4` float32 tests.

Performance results:

- Analytic gradient + potential path:
  - baseline `4.043s / 2 steps`
  - variant `3.912s / 2 steps`
  - improvement about `3.2%`
- Analytic gradient-only path for `return_potential=False`:
  - baseline `3.938s / 2 steps`
  - variant `3.897s / 2 steps`
  - improvement about `1.0%`
- Inline scalar analytic path under the outer leaf `vmap`:
  - baseline `3.941s / 2 steps`
  - variant `3.937s / 2 steps`
  - effectively noise

Conclusion: autodiff/potential calculation is not the main launch-count source. Do not promote `JACCPOT_LOCAL_EVAL_ANALYTIC_GRAD` as a production optimization yet; keep it only as a default-off diagnostic/experiment unless later evidence changes.

### Refined Next Target

The next meaningful optimization should attack final local-eval graph structure, not nearfield and not autodiff micro-optimizations.

Concrete next investigation:

1. Confirm whether local eval is operating over static padded tree/leaf capacity rather than active leaves only.
   - The eval-added launch count has strong `~8190 = 2 × 4095` structure.
   - Current runtime reports active dual leaf count `782` and node count `1563`, so the `4095`-like launch pattern suggests a padded-capacity loop or graph-node expansion.
2. Add explicit profile fields for local eval shapes:
   - `large_n_eval_leaf_nodes_shape`
   - `large_n_eval_local_coefficients_shape`
   - `large_n_eval_active_leaf_count`
   - `large_n_eval_max_leaf_size`
3. If padded capacity is confirmed, add an experimental strict-fused-only local-eval compaction/batching path that evaluates only active leaves while preserving static output shapes.
4. If padded capacity is not the issue, lower/inspect only `_evaluate_local_expansions_for_particles` HLO and target the repeated `input_reduce_fusion_3` / `loop_multiply_fusion_1` source directly.
5. Promotion remains gated on >10% full-mode improvement, fused active `true`, fallback `0`, and lower Nsight kernel/graph-launch counts.

## 2026-06-12 Follow-up: Production Solid-FMM Local Eval Without Autodiff

### Production Change

The large-N solid-FMM local evaluation path no longer calls `jax.value_and_grad` in production acceleration-only eval.

Changed behavior:

- Full-particle solid-FMM L2P now uses analytic local-gradient equations by default for `max_acc_derivative_order=0`.
- Targeted solid-FMM L2P now also uses analytic gradient/potential equations for `max_acc_derivative_order=0`.
- The old autodiff helpers remain defined for reference/generic compatibility, but the large-N production eval path no longer routes through them.

Validation:

- `py_compile` passed for touched jaccpot and ODISSEO files.
- CPU parity against the previous autodiff implementation:
  - gradient-only max abs error: `5.93e-08`
  - gradient+potential max abs error: `5.93e-08`
  - potential max abs error: `1.11e-16`

### Shape Diagnostics Added

Runtime/report fields now expose local-eval shapes:

- `large_n_eval_leaf_nodes_shape`
- `large_n_eval_local_coefficients_shape`
- `large_n_eval_local_centers_shape`
- `large_n_eval_active_leaf_count`
- `large_n_eval_max_leaf_size`
- `large_n_eval_leaf_particle_slots`
- plus `runtime_...` variants in ODISSEO reports.

Canonical 200k/2 strict-fused diagnostic result:

- `runtime_large_n_eval_leaf_nodes_shape = [782]`
- `runtime_large_n_eval_local_coefficients_shape = [1563, 25]`
- `runtime_large_n_eval_local_centers_shape = [1563, 3]`
- `runtime_large_n_eval_active_leaf_count = 782`
- `runtime_large_n_eval_max_leaf_size = 256`
- `runtime_large_n_eval_leaf_particle_slots = 200192`

This refutes the earlier hypothesis that final L2P is directly looping over a `4095` padded leaf capacity. The `~8190` launch pattern likely comes from the local-eval graph structure / recurrence expansion, not from leaf-capacity over-evaluation.

### Performance Result After Autodiff Removal

Autocvd-gated 200k/2 strict-fused probe:

- Root: `/tmp/odisseo_prod_analytic_local_eval_audit_200k_2_20260612`
- Baseline: `3.8665s / 2 steps`, fused active `true`, fallback `0`, planner bypass `2`
- Duplicate strict-fused lane: `3.9421s / 2 steps`, fused active `true`, fallback `0`, planner bypass `2`

Interpretation: removing autodiff is the correct production cleanup, but not enough by itself to solve the launch-count bottleneck.

### Flat Analytic L2P Candidate

Added default-off candidate:

- `JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC=1`

Purpose:

- Evaluate all leaf-major particle slots in one flattened analytic L2P batch instead of nested leaf -> particle vmaps.

Validation:

- CPU parity against nested analytic path: exact for the tested synthetic leaf-major state.

Performance:

- Root: `/tmp/odisseo_flat_analytic_local_eval_audit_200k_2_20260612`
- Baseline: `3.9476s / 2 steps`
- Variant `JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC=1`: `3.9468s / 2 steps`
- Fused active `true`, fallback `0`, planner bypass `2`

Conclusion: flatting leaf-major slots at this level does not reduce the launch storm. Do not promote this flag.

### Next Kernel-Launch Target

The next optimization should replace the current recurrence-heavy complex L2P implementation with a static order-4 batched polynomial/derivative kernel that avoids repeated small `scatter`, `slice`, and `reduce` graph fragments from `complex_R_solidfmm` / derivative coefficient construction.

Concrete next steps:

1. Lower/inspect HLO for `_evaluate_local_expansions_for_particles` with canonical shapes `[782, 256, 3]` and order `4`.
2. Identify which expressions create `input_reduce_fusion_3` and `loop_multiply_fusion_1`.
3. Implement an order-4 specialized analytic L2P kernel for solid-FMM complex coefficients with static coefficient layout `[25]`.
4. Keep generic order path as fallback, but route canonical `large_n_gpu`, `solidfmm`, `order=4`, `return_potential=False` through the specialized kernel.
5. Promote only if Nsight shows a meaningful reduction in measured-tail kernels / `cuGraphLaunch` calls and the canonical 200k/2 full-mode improves by >10%.

## 2026-06-12 Follow-up: Solid-FMM Recurrence/XLA Comparison

### Question
- We checked whether the production local-eval path is still using autodiff and whether the solid-FMM recurrence can be written in a more XLA-friendly way before considering Pallas.
- Solid-fmm style FMM uses analytic recurrence relations for regular/singular harmonics. It does not need autodiff in production.
- The current jaccpot production local-gradient path now avoids autodiff, but the first analytic implementation still built the recurrence through a generic packed complex table that XLA lowers into many scatter/slice/gather operations.

### HLO finding
- The existing nested analytic local eval is better than the flat analytic candidate:
  - `JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC=1` generated a larger optimized HLO with many more fusions/scatters/reduces.
  - The flat variant remained performance-neutral/noisy and should stay default-off.
- The important issue found in the nested recurrence was dtype widening:
  - local-eval inputs are `f32` positions and `c64` coefficients
  - `complex_R_solidfmm` converted offsets to `f64` and harmonics to `c128`
  - optimized HLO showed large `c128[782,256,...]` and `f64[782,256,...]` intermediates inside `regular_solid_harmonic_gradient_coefficients`

### Guarded candidate implemented
- Added dtype-preserving regular solid harmonics:
  - `/export/home/tbuck/jaccpot/jaccpot/operators/complex_harmonics.py`
  - `complex_R_solidfmm_preserve_dtype`
- Added dtype-preserving analytic local-gradient helpers:
  - `/export/home/tbuck/jaccpot/jaccpot/operators/complex_ops.py`
  - `regular_solid_harmonic_gradient_coefficients_preserve_dtype`
  - `evaluate_local_complex_grad_analytic_preserve_dtype`
- Wired the full-particle solid-FMM local-eval branch behind:
  - `JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE=1`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- This remains default-off because the valid fused benchmark did not improve warm timing.

### Validation
- Compile checks passed for touched jaccpot files:
  - `complex_harmonics.py`
  - `complex_ops.py`
  - `_fmm_impl.py`
- Single-sample parity against the widened analytic path:
  - old widened output dtype: `float64`
  - dtype-preserving output dtype: `float32`
  - max absolute gradient delta: `6.58e-08`
  - gradient-coefficient max absolute delta after cast: `1.86e-09`
- Optimized GPU HLO with `JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE=1`:
  - `c128`: `0`
  - `f64`: `0`
  - `c64`: `621`
  - `f32`: `214`
  - `input_reduce_fusion`: `0`
  - `loop_multiply_fusion`: `13`
- This proves the candidate removes accidental double precision from the float32 local-eval recurrence, even though it does not reduce full simulator walltime enough.

### Benchmark results
- Invalid first profile A/B:
  - baseline fused-on: `3.935s / 2 steps`
  - variant: `3.153s / 2 steps`
  - invalid because `walltime_ab_compare.py --fixed-policy` intentionally sets the variant fused mode to `off` unless overridden; variant had `runtime_strict_fused_mode_active=false`.
- Valid strict-fused A/B command included:
  - `--variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=on`
  - `--variant-env JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE=1`
- Valid strict-fused result:
  - baseline: `3.9435s / 2 steps`, `1.9718s / step`
  - dtype-preserve variant: `3.9740s / 2 steps`, `1.9870s / step`
  - fused active: `true` for both
  - fallback count: `0` for both
  - planner bypass count: `2` for both
- Decision: keep `JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE=1` as a diagnostic/experimental path only; do not promote it as production optimization.

### Pallas status and next direction
- Pallas is available in the current environment:
  - JAX `0.9.0`
  - module `jax.experimental.pallas`
- A Pallas L2P kernel is plausible but more invasive. It should only target the final local-eval graph if Nsight confirms the remaining local-eval kernel storm is still dominated by materializing/contracting harmonic-gradient tensors.
- Candidate Pallas shape:
  - one program/block over leaf particle slots
  - gather leaf coefficient row and center
  - compute order-4 regular harmonic recurrence in registers
  - contract 25 complex coefficients into 3 real gradient components
  - write/scatter acceleration for valid particle slots
- Before Pallas, a less invasive XLA rewrite remains possible: generate explicit order-4 real/imag scalar recurrence formulas that avoid building the `(p+1,p+1)` table and avoid pack/scatter/gather machinery.

### Next concrete plan
- Do not spend more effort on dtype-only changes; they clean HLO but do not move the valid fused walltime.
- Next optimization target should be an explicit order-4 local-gradient contraction path, default-off:
  - no autodiff
  - no complex128/f64 widening
  - no table `.at[].set` recurrence materialization
  - no packed derivative gather maps in the hot L2P path
- Compare explicit-order4 XLA against current nested analytic with:
  - CPU/JAX parity
  - optimized HLO counts
  - valid fused-on 200k/2 profile timing
  - Nsight kernel/graph-launch deltas
- If explicit-order4 XLA still lowers poorly, then prototype Pallas L2P as the next surgical experiment.

## 2026-06-12 Follow-up: Explicit Order-4 Local-Gradient Candidate

### Implemented guarded candidate
- Added an explicit order-4 scalar local-gradient contraction path in jaccpot:
  - `/export/home/tbuck/jaccpot/jaccpot/operators/complex_ops.py`
  - `evaluate_local_complex_grad_order4_unrolled`
- Wired the full-particle solid-FMM local-eval branch behind:
  - `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- This path is default-off and only applies to `order == 4`. Other orders fall back to the dtype-preserving analytic helper.

### Intent
- Avoid the generic recurrence lowering that builds a `(p+1,p+1)` table through `.at[].set`, packs it, derives gradients through gather maps, and then contracts.
- Compute the order-4 regular harmonics as scalar expressions and directly contract local coefficients into the three gradient components.
- Preserve production semantics while removing the XLA-unfriendly table/scatter/gather local-eval machinery.

### Validation
- Compile checks passed for touched jaccpot files.
- Single-sample parity:
  - max abs vs widened analytic path: `4.55e-08`
  - max abs vs dtype-preserving analytic path: `0.0`
- Batched parity over 64 random float32/c64 samples:
  - max abs vs dtype-preserving analytic path: `0.0`

### Optimized HLO result
- With `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1` on the canonical local-eval lowering:
  - HLO chars: `121749`
  - fusions: `50`
  - scatters: `17`
  - reductions: `0`
  - gathers: `28`
  - slices: `256`
  - `input_reduce_fusion`: `0`
  - `loop_multiply_fusion`: `0`
  - `c128`: `0`
  - `f64`: `0`
- This is a much cleaner local-eval HLO than the dtype-preserving generic recurrence, which still had `loop_multiply_fusion=13`, `scatter=103`, and `reduce=14`.

### Valid strict-fused 200k/2 benchmark
- Command included:
  - `--variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=on`
  - `--variant-env JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1`
- Result:
  - baseline: `3.9258s / 2 steps`, `1.9629s / step`
  - order4-unrolled variant: `3.9209s / 2 steps`, `1.9605s / step`
  - fused active: `true` for both
  - fallback count: `0` for both
  - planner bypass count: `2` for both
- Decision: despite the much cleaner HLO, full ODISSEO warm-step timing did not materially improve. Keep the candidate default-off until Nsight proves it reduces local-eval kernel/graph-launch counts in the full simulator path.

### Interpretation
- The local-eval HLO cleanup is real, but the remaining 200k bottleneck is likely dominated by refresh internals outside this isolated L2P lowering, or by CUDA graph launch structure that this rewrite does not change enough in the full fused step.
- Next attribution should compare baseline vs `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1` with Nsight graph-node tracing, but optimization effort should shift back toward tree/upward/downward refresh subgraphs unless Nsight shows a meaningful local-eval launch/runtime delta.

### Nsight check for order4-unrolled L2P
- Ran `tools/fused_audit_runner.py` with Nsight graph-node tracing:
  - output root: `/tmp/odisseo_l2p_order4_unrolled_nsys_200k_2_20260612`
  - baseline: strict fused full path
  - variant env: `JACCPOT_STATIC_STRICT_FUSED_MODE=on`, `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1`
- Baseline:
  - measured runtime: `4.1519s / 2 steps`
  - strict fused active: `true`
  - fallback count: `0`
  - planner bypass count: `2`
  - total kernels: `296948`
  - measured-tail kernels: `145609`
  - measured-tail GPU active: `79.15%`
  - hot-loop H2D/D2H transfers: `0/0`
- Variant:
  - measured runtime: `4.1750s / 2 steps`
  - strict fused active: `true`
  - fallback count: `0`
  - planner bypass count: `2`
  - total kernels: `296938`
  - measured-tail kernels: `145549`
  - measured-tail GPU active: `78.60%`
  - hot-loop H2D/D2H transfers: `0/0`
- Delta variant minus baseline:
  - total kernels: `-10`
  - measured-tail kernels: `-60`
  - measured-tail GPU active: `-0.55 percentage points`
  - measured runtime: `+0.0231s / 2 steps`
- Conclusion:
  - The explicit order-4 L2P rewrite substantially cleans the isolated optimized HLO, but it barely changes full-path kernel count and does not improve measured walltime.
  - Local L2P recurrence cleanup is not the lever for the remaining launch storm.
  - Return focus to strict refresh internals: tree build/upward sweep and downward/M2L artifact/compute paths.
  - Keep both `JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE=1` and `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED=1` default-off diagnostics unless a future broader refactor makes them useful.

## 2026-06-12 Strict Refresh Detail Diagnostics Implementation

Implemented the dormant strict-refresh detail diagnostic enum in `jaccpot/runtime/_fmm_impl.py` so the reported modes now change the compiled strict static-radix graph instead of only changing report metadata.

Changes:
- `JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE != full` now disables final self-force evaluation in the strict compiled refresh/eval paths, returning zero self acceleration for profiling-only modes.
- Added detail-mode cache-key participation so switching detail modes cannot accidentally reuse the wrong compiled strict runner.
- Added static-radix early-return aliases:
  - `tree_sort_only`, `tree_metadata_only` reuse the existing tree-only cut.
  - `p2m_only`, `m2m_only` reuse the existing tree+upward cut.
- Added solid-FMM downward cuts:
  - `downward_artifacts_only` returns after interaction/downward initialization.
  - `m2l_only` runs M2L accumulation and skips L2L/final eval.
  - `l2l_only` skips M2L, seeds nonphysical local coefficients, runs L2L, and skips final eval.
- Added runtime/report counters for cached static compact far-pair reuse:
  - `runtime_static_radix_compact_pair_reuse_hits`
  - `runtime_static_radix_compact_pair_reuse_misses`

Validation:
- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py odisseo/jaccpot_coupling.py tools/walltime_ab_compare.py tools/fused_audit_runner.py`
- Canonical 200k/2, fixed policy, `--require-autocvd`, fixed neighbor cap `1048576`:
  - Full strict fused: `4.017s / 2 steps`; fused active `true`; fallback `0`; planner bypass `2`; compact-pair reuse `1 hit / 0 misses`.
  - `m2l_only`: `2.291s / 2 steps`; fused active `true`; fallback `0`.
  - `l2l_only`: `2.389s / 2 steps`; fused active `true`; fallback `0`.
  - `p2m_only`: `2.393s / 2 steps`; fused active `true`; fallback `0`.
  - `integrator_only`: `2.190s / 2 steps`; fused active `true`; fallback `0`.

Interpretation:
- The strict-refresh detail cuts are now usable for Nsight graph-node attribution.
- The first walltime cuts imply a large fixed ODISSEO/integrator/external-potential floor of about `2.19s / 2 steps` in this timing configuration.
- Relative to that floor, isolated refresh subgraphs add only about `0.10–0.20s / 2 steps` in these coarse detail cuts, while full strict fused is about `4.0s / 2 steps`.
- This points away from tree/upward/downward refresh alone as the dominant remaining walltime source and back toward final force evaluation/nearfield/external integration work, pending Nsight graph-node confirmation.

Artifacts:
- `/tmp/odisseo_strict_refresh_detail_full_200k_2_20260612/walltime_ab_summary.json`
- `/tmp/odisseo_strict_refresh_detail_m2l_only_200k_2_20260612/walltime_ab_summary.json`
- `/tmp/odisseo_strict_refresh_detail_l2l_only_200k_2_20260612/walltime_ab_summary.json`
- `/tmp/odisseo_strict_refresh_detail_p2m_only_200k_2_20260612/walltime_ab_summary.json`
- `/tmp/odisseo_strict_refresh_diag_integrator_only_200k_2_20260612/walltime_ab_summary.json`

Next steps:
- Run Nsight graph-node captures for `full`, `integrator_only`, `p2m_only`, `m2l_only`, and `l2l_only` using the new detail cuts.
- Add or run an `eval_only`/nearfield-only attribution pass to separate local far-field eval, nearfield leaf eval, and external potential/integrator work.
- If Nsight confirms the walltime deltas, prioritize final evaluation/nearfield launch reduction before deeper P2M/M2M/M2L/L2L refresh batching.

## 2026-06-12 Robust Timing Recheck + Nsight Attribution

Rechecked the diagnostic timings because the earlier per-mode numbers appeared non-additive. The important correction is that these modes are cumulative graph variants, not exclusive substage timers. Every mode still includes the ODISSEO scan/integrator/external-potential floor unless explicitly disabled, so the modes should be compared by subtraction against an appropriate floor, not summed.

Robust walltime probes, canonical 200k/2, fixed policy, fixed neighbor cap `1048576`, `--require-autocvd`, `perf_warmup_runs=1`, `perf_measure_runs=3`:
- Full strict fused: measured runs `[3.966, 4.080, 4.056]s`, median `4.056s / 2 steps`, fused active `true`, fallback `0`, planner bypass `4`.
- `integrator_only`: measured runs `[2.216, 2.220, 2.217]s`, median `2.217s / 2 steps`, fused active `true`, fallback `0`.
- `eval_only`: measured runs `[3.873, 3.977, 4.078]s`, median `3.977s / 2 steps`, fused active `true`, fallback `0`.

Interpretation:
- The timing is robust enough to trust the high-level attribution: `eval_only` is essentially as expensive as full mode.
- Tree/upward/downward refresh is not the dominant walltime in the current measured ODISSEO path.
- The large fixed floor means earlier `p2m_only`/`m2l_only`/`l2l_only` numbers around `2.3–2.4s / 2 steps` should not be summed; they mostly share the same floor.

Nsight graph-node captures:
- Full vs `eval_only`: `/tmp/odisseo_nsys_full_vs_eval_only_200k_2_20260612/audit_summary.json`
- `eval_only` vs `integrator_only`: `/tmp/odisseo_nsys_eval_only_vs_integrator_200k_2_20260612/audit_summary.json`

Key Nsight metrics from `eval_only` vs `integrator_only` measured tail:
- `eval_only`: `4.021s`, GPU active `80.8%`, GPU busy `3249ms`, kernels `130817`, H2D/D2H `0/0`.
- `integrator_only`: `2.355s`, GPU active `69.5%`, GPU busy `1637ms`, kernels `69518`, H2D/D2H `0/0`.
- Eval delta: about `+1.666s`, `+61299` measured-tail kernels, no hot-loop host/device transfers.

Top eval delta kernels by measured-tail duration:
- `input_reduce_fusion_1`: `+4096` launches, `+952ms`.
- `loop_multiply_fusion_1`: `+4094` launches, `+500ms`.
- `input_scatter_fusion`: `+4096` launches, `+30.6ms`.
- `input_scatter_fusion_1`: `+4096` launches, `+26.3ms`.
- `loop_slice_fusion`: `+4096` launches, `+16.3ms`.
- `loop_broadcast_select_fusion`: `+4096` launches, `+14.5ms`.

Conclusion:
- The actionable bottleneck is now final force evaluation, not strict refresh internals.
- The kernel storm appears to be a fixed 4096-iteration/chunked eval loop pattern. The next optimization target should identify whether those 4096 launches are nearfield leaf-pair eval, local/L2P eval, external acceleration composition, or a fused scan artifact, then batch/collapse that loop.
- Keep the refresh diagnostics because they are useful, but do not prioritize P2M/M2M/M2L/L2L launch reduction until eval launch reduction is addressed.

## 2026-06-13 Fused Payload Fast-Lane Experiment

Continued from the eval/Nsight attribution and traced the dominant measured-tail kernels back to the nearfield P2P path inside `_evaluate_tree_compiled_impl`, specifically `_compute_leaf_p2p_prepared_large_n_pairs_only_impl` / `_pair_contributions_batched`. The hot pattern was a 4096-iteration nearfield pair loop emitting large `loop_multiply_fusion` and `input_reduce_fusion` kernels.

Implemented a default-off fused payload experiment:
- `JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED=1` preserves static target-block payload fields in the strict fused scan carry instead of stripping them.
- Fused strict refresh can build the static source-leaf padded payload under that flag.
- Initial ODISSEO prepared-state reuse is normalized before measured warm runs so the scan carry pytree is invariant.
- Added report diagnostics for radix payload presence and source-leaf payload shape/slots.

Validation:
- Compile checks passed for touched runtime/tool files:
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py`
  - `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
  - `/export/home/tbuck/jaccpot/jaccpot/solver.py`
  - `odisseo/jaccpot_coupling.py`
  - `tools/walltime_ab_compare.py`
  - `tools/fused_audit_runner.py`
- Canonical 200k/2, fixed policy, fixed neighbor cap `1048576`, `--require-autocvd`, self-only timing:
  - Baseline: `3.410s / 2 steps`, fallback `0`, payload absent.
  - Payload route: `2.019s / 2 steps`, fallback `0`, payload present.
- Canonical 200k/1 shape probe:
  - Baseline: `1.687s / step`.
  - Payload route: `1.005s / step`.
  - Payload shape: `runtime_large_n_radix_payload_source_leaf_shape = [782, 32, 32]`.
  - Payload slots: `800768` source-leaf slots.
  - Source-particle payload remains `[0, 0, 0]`; this win is from the static source-leaf padded payload fallback, not the fully materialized source-particle payload.
- One-step output delta vs baseline:
  - max abs `4.58e-05`, RMS `3.86e-06` in final state.
  - Two-step max abs reached `9.26e-04`, RMS `3.80e-06`; needs a stricter force/parity check before promotion.

Nsight graph-node capture, canonical 200k/1:
- Artifact: `/tmp/odisseo_nsys_fused_payload_baseline_vs_variant_200k_1_20260613/audit_summary.json`
- Baseline measured tail:
  - runtime `1.771s`, kernels `76054`, GPU active `92.9%`, H2D/D2H `0/0`.
- Payload route measured tail:
  - runtime `1.045s`, kernels `19577`, GPU active `91.9%`, H2D/D2H `0/0`.
- Delta:
  - `-56477` measured-tail kernels.
  - `-0.727s` measured-tail span for one step.
  - no hot-loop host/device transfers introduced.

Interpretation:
- This is the first large launch-count reduction in the ODISSEO strict fused path: about `74%` fewer measured-tail kernels for one step and about `41%` lower 200k/2 warm walltime.
- The route is still default-off because the traced fused static block build skips the Python `int(jnp.max(counts))` capacity check inside the scan. For the canonical profile the static payload shape is exactly the desired tighter `800768` slot cap, but production promotion needs an equivalent shape-stable capacity assertion or a preflight guarantee that the static topology cannot exceed `[782, 32, 32]`.

Next steps:
- Add a safe capacity/preflight gate for the fused payload route before enabling it by default.
- Run direct force parity against the baseline/fallback path for the canonical IC, not only integrated final-state deltas.
- Run 200k/20 canonical timing with the payload route once parity passes.
- If parity is acceptable, promote `JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED=1` into the fixed production policy and keep source-particle payload materialization disabled unless a separate A/B proves it faster.

## 2026-06-13 Correct Velocity-Verlet, Payload Promotion, and Canonical Gates

### Integrator correction
- The canonical fixed-timestep path selects ODISSEO leapfrog/velocity Verlet, but the strict fused runner was reusing one self-FMM acceleration at both endpoints of each step.
- The strict runner now carries `(prepared_state, state, acceleration_current)` through one compiled scan.
- Each step drifts with `a_n`, refreshes and evaluates self gravity at the new positions, kicks with `0.5 * (a_n + a_{n+1}) * dt`, and carries `a_{n+1}`.
- Initial self acceleration is prepared outside measured runs, so steady-state timing contains exactly one new endpoint FMM evaluation per step.
- Strict static-radix production now fails fast unless `refresh_every=1`.
- Added evaluation-cadence diagnostics to the runtime and benchmark reports.

### Static payload cap and safety
- ODISSEO's automatic canonical static target-block cap is now `32`, matching jaccpot's validated default. Explicit user caps remain unchanged.
- Explicit cap `16` fails eagerly for the canonical IC with `num_leaves=782`, `block_size=32`, and an error recommending a larger `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF`.
- Cap `32` produces payload shape `[782, 32, 32]` with `800768` source-leaf slots.
- A device-carried capacity flag now spans the compiled scan with one post-scan check, preventing later topology evolution from silently truncating static target blocks without hot-loop host transfers.

### Correctness results
- Focused unit/API tests: `7 passed`.
- Initial canonical direct-force parity, payload versus opt-out: relative L2 `1.973e-7`; max absolute component `2.289e-5`; pass.
- At identical positions from the evolved 20-step opt-out trajectory: relative L2 `2.000e-7`; max absolute component `4.768e-7`; pass.
- Payload-on and opt-out trajectories diverge after 20 steps despite same-position force parity. State relative L2 reached about `4.05e-2`; this is chaotic amplification of tiny FP32 ordering differences, not a force-law or capacity failure. Long-horizon absolute trajectory identity is not a valid promotion oracle; use same-position force parity plus conservation/statistical gates.

### Performance results
- Formal canonical 200k/1, one warmup and three measured runs, self-only, autocvd:
  - payload default-on median: `1.001s / step`
  - explicit payload opt-out median: `1.708s / step`
  - acceleration carry active: `true`; endpoint self evaluations: `1`; fallback: `0`.
- Canonical 200k/20, one warmup and three measured runs:
  - payload measured runs: `[22.017, 22.552, 22.709]s`
  - payload median: `22.552s`, or `1.128s / step`
  - opt-out measured runs: `[38.042, 39.029, 39.346]s`
  - opt-out median: `39.029s`, or `1.951s / step`
  - endpoint self evaluations: `20`; fused active: `true`; fallback: `0`; capacity status: `true`.

### Corrected-path Nsight
- Artifact: `/tmp/odisseo_nsys_velocity_verlet_payload_vs_off_200k_1_20260613/audit_summary.json`.
- Payload measured runtime: `1.056s`.
- Payload measured tail remains about `19.6k` kernels, roughly `90.8%` GPU active, with zero H2D/DtoH transfers.
- Opt-out measured tail: `76064` kernels and `93.17%` GPU active.
- The payload launch reduction therefore survives endpoint-correct velocity Verlet. Next work remains attribution and batching of tree/upward/downward and residual nearfield kernels.

### Current production policy
- `JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED=1` is default; set it to `0` for rollback.
- Canonical target-block cap: `32`; canonical neighbor cap: `1048576`.
- Strict production requires `refresh_every=1`.


## 2026-06-14 Occupancy-Sorted Nearfield Promotion

- Nsight attribution showed the corrected full step was dominated by 100 padded nearfield tile evaluations, while tree, upward, and downward refresh together contributed only about `0.09s`.
- Canonical occupancy was sparse: `6931` real source blocks versus `25024` fixed-cap block slots. Sorting target leaves by active block count reduced batch padding, and `jax.lax.cond` now skips source-block tiles that are invalid for an entire target batch.
- The analytic P2P equations and fixed prepared-state shapes are unchanged. Source-leaf indices are remapped through the inverse leaf permutation before evaluation, and accelerations scatter through the correspondingly reordered particle IDs.
- Formal canonical 200k/1 autocvd A/B, one warmup and three measured runs:
  - old padded route median: `1.004817s / step`
  - occupancy-sorted skip route median: `0.470972s / step`
  - optimized measured runs: `[0.469173, 0.471010, 0.470972]s`
- Direct force parity passes: relative L2 `1.974e-7`; max absolute component `2.289e-5`.
- True default-on smoke: `0.474573s`; explicit opt-out: `0.996157s`; fused active, fallback `0`, capacity valid.
- Production defaults are now enabled. Independent rollback switches:
  - `JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT=0`
  - `JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES=0`
- Reports expose `runtime_large_n_radix_fast_occupancy_sort` and `runtime_large_n_radix_fast_skip_empty_tiles`.
- Retained artifacts:
  - `/tmp/odisseo_radix_occupancy_sort_skip_ab_200k_1_3runs_20260614/walltime_ab_summary.json`
  - `/tmp/odisseo_fused_payload_force_parity_occupancy_sort_skip_200k_20260614.json`
  - `/tmp/odisseo_radix_occupancy_default_on_vs_optout_200k_1_20260614/walltime_ab_summary.json`

## 2026-06-20 Componentwise Nearfield Reduction Promotion

### Summary
- Continued the pure-JAX nearfield optimization pass before starting any Pallas work.
- Tested and rejected two alternatives:
  - flattened source-particle reduction: `0.532s` versus `0.424s` baseline, rejected;
  - weighted-source/GEMM-style contraction: `0.450s` versus `0.419s` baseline, rejected.
- Promoted explicit Cartesian component reductions for the radix payload nearfield path:
  - new default: `JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS=1`;
  - rollback: set `JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS=0`.
- The implementation keeps the same analytic Newtonian pair equations and only changes the XLA expression shape from a trailing-vector `diff` tensor/reduction to explicit `dx`, `dy`, `dz` reductions.

### Validation
- Focused tests after promotion:
  - jaccpot nearfield route tests: `2 passed`;
  - ODISSEO strict velocity-Verlet/API tests: `7 passed`, one unrelated `shard_map` deprecation warning.
- Direct same-position force parity, payload default versus payload componentwise:
  - relative L2: `5.453680884670575e-08`;
  - max absolute component: `7.62939453125e-06`;
  - parity gate: pass.
- Canonical 200k/1, one warmup and three measured runs:
  - previous default median: `0.423086s / step`;
  - componentwise median: `0.396896s / step`;
  - measured componentwise runs: `[0.403354, 0.395122, 0.396896]s`;
  - improvement: about `6.2%`.
- Canonical 200k/20, one warmup and one measured run:
  - previous default: `8.542325s`, or `0.427116s / step`;
  - componentwise: `7.982216s`, or `0.399111s / step`;
  - endpoint self-FMM evaluations: `20`; fused active: `true`; fallback: `0`.
- True default-on smoke after promotion:
  - componentwise default: `0.393525s / step`;
  - explicit opt-out: `0.416741s / step`;
  - fixed policy reports `JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS=1`.

### Nsight
- Artifact: `/tmp/odisseo_nsys_componentwise_pairs_200k_1_20260620/audit_summary.json`.
- Componentwise measured-tail GPU busy: `350.804ms`.
- Delta versus previous default: about `-114ms` measured-tail GPU busy.
- Measured-tail kernels increased by about `390`, so this is another case where walltime improves by reducing heavy arithmetic even though launch count rises slightly.
- Measured-tail H2D/DtoH transfers stayed `0/0`.

### Caveat: 20-step finite-state gate
- The 200k/20 timing outputs contain all-NaN `final_state` arrays for both baseline and componentwise runs.
- This is not introduced by componentwise pairs because the baseline output is equally non-finite.
- Treat this as a separate production-correctness blocker for the canonical benchmark/integration setup.
- Performance and force-parity gates support the componentwise nearfield promotion, but a production-ready simulator still needs a finite-state 20-step gate with an appropriate timestep/IC/runtime configuration.

### Retained artifacts
- `/tmp/odisseo_componentwise_pairs_ab_200k_1_3runs_20260620/walltime_ab_summary.json`
- `/tmp/odisseo_componentwise_force_parity_200k_20260620.json`
- `/tmp/odisseo_componentwise_pairs_ab_200k_20_20260620/walltime_ab_summary.json`
- `/tmp/odisseo_nsys_componentwise_pairs_200k_1_20260620/audit_summary.json`
- `/tmp/odisseo_componentwise_default_vs_optout_200k_1_20260620/walltime_ab_summary.json`

## 2026-06-20 NaN Stability Finding and Benchmark Diagnostics

- The componentwise nearfield promotion is not the source of the 200k/20 NaNs: both the baseline payload path and the componentwise path produced all-NaN `final_state` arrays in `/tmp/odisseo_componentwise_pairs_ab_200k_20_20260620/walltime_ab_summary.json`.
- The canonical walltime runner had been timing `--num-steps 20` while inheriting the simulator default `--t-end-gyr 2.0`, i.e. `dt=0.1 Gyr`; the one-step case is finite but dynamically huge because it uses a single `2.0 Gyr` step.
- A same-lane `200k/20` probe with `--t-end-gyr 0.2` stayed finite, but showed extreme ejected-particle outliers, so the current issue is a serious stability/physics setup problem rather than a nearfield optimization regression.
- Added report/output metadata for `t_end_gyr`, `t_end_code`, `dt_gyr`, `dt_code`, final-state finite/NaN/Inf counts, and final position/velocity norm summaries in `notebooks/scalability/galaxy_disk_fmm_large_n.py`.
- Added `tools/walltime_ab_compare.py --t-end-gyr`, per-case `final_state_digest`, NaN-safe trajectory deltas, and `--require-finite-final-state` so A/B summaries can no longer hide non-finite outputs behind timing-only success.

Next stability checks before further promotion:

1. Run external-only and self-only `200k/20` probes at `t_end_gyr=2.0` and `0.2` to isolate whether the blow-up is external field, live self-gravity, or their combination.
2. Run `--initial-accel-report` on the canonical cached IC and inspect central acceleration/outlier percentiles.
3. Test smaller physical timesteps and/or larger softening before treating any `200k/20` walltime as production-correct.
4. Only resume launch-count optimization once the canonical correctness gate requires `final_state_all_finite=true`.

## 2026-06-20 NFW Stability Fix and Strict-Fused NaN Isolation

- Added an analytic small-`x` NFW formulation in `odisseo/potentials.py`:
  - `log1p(x)/x` uses the `1 - x/2 + x^2/3` series near zero;
  - `(log1p(x) - x/(1+x))/x^2` uses the `1/2 - 2x/3 + 3x^2/4` series near zero;
  - exactly zero radius returns finite zero vector acceleration instead of `0/0` NaNs.
- Added `tests/test_potentials.py::test_nfw_acceleration_is_finite_at_tiny_radius`; focused run passed.
- Direct tiny-radius check before the fix produced `NaN` at `r=0` and a huge wrong-sign acceleration at `r=1e-12`; after the fix it returns finite inward accelerations around the expected central NFW limit.
- Focused strict timing split at `200k/20`, `t_end=2.0 Gyr`, `dt=0.1 Gyr`:
  - external-only: finite and dynamically sane (`position_norm_max≈6.14`, `velocity_norm_max≈5.67`);
  - strict self-only: finite but catastrophically ejects particles (`position_norm_max≈3.75e5`, `velocity_norm_max≈1.98e5`);
  - strict full fused after NFW fix: still all-NaN in the tight compiled perf path.
- Generic/history mode after the NFW fix is finite, but it bypasses strict fused (`runtime_strict_fused_mode_active=false`), so it is not a valid correctness oracle for the production lane.

Current interpretation:

- The NFW singularity was a real bug and is now fixed.
- The remaining 200k/20 NaN is strict-fused-production-path-specific. The next diagnostic must collect first-nonfinite-step information inside `jaccpot` `strict_run_v2`/compiled scan, not via ODISSEO conservation history.
- Performance work should remain paused until strict fused full mode passes `--require-finite-final-state` for a physically meaningful timestep/cadence.

Artifacts:

- External-only vs self-only split: `/tmp/odisseo_stability_external_vs_self_200k_20_tend2_20260620/walltime_ab_summary.json`
- Finite-gated strict full failure after NFW fix: `/tmp/odisseo_nfw_safe_full_200k_20_tend2_20260620`
- Generic/history finite non-oracle: `/tmp/odisseo_history_full_200k_20_tend2_after_nfw_reports/galaxy_disk_profile_20260620_104658.json`

## 2026-06-20 Strict Fused NaN Root Cause: Unsafe Compact Far-Pair Reuse

- First-bad-step sweep at fixed `dt=0.1 Gyr` showed the strict fused path is finite but already corrupt after step 1 (`vel_max≈3.3e6` in the ODISSEO full path), worsens at step 2, and introduces NaNs at step 3.
- ICs are not the primary cause: worst step-1 particles have normal direct self accelerations (`~0.85–1.44`) at both initial and endpoint positions.
- Fresh FMM prepare/eval, non-hot same-topology refresh/eval, and direct subset evaluation agree at the step-1 endpoint positions.
- The corruption appears only when `strict_fused_mode_active=True` enables cached compact far-pair reuse inside the static-radix refresh hot path.
- Concrete evidence:
  - non-hot refresh at step-1 endpoint: `p99≈18.82`, `max≈29.50`;
  - hot refresh with cached compact far-pair reuse: `p99≈5384.97`, `max≈6.66e7`;
  - hot refresh with compact-pair reuse disabled outside the scan returns the correct `p99≈18.82`, `max≈29.50`;
  - far-pair count changes from cached `68272` to fresh `77002`, so the cached M2L interaction list is semantically stale after drift.
- Patched local `jaccpot` to fail fast by default when strict fused tries to reuse cached compact far pairs for moved static-radix positions. Unsafe legacy behavior now requires explicit `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1`, which reproduces the bad `vel_max≈2.0e5` self-only one-step result in the direct strict-run probe.

Implication:

- Existing unit tests prove important component parity, but they did not certify the full ODISSEO strict fused dynamic refresh/eval path. We need a new regression that compares strict fused endpoint refresh against fresh FMM/direct at moved positions.
- The production fix is not to reuse compact far pairs blindly. We need a scan-invariant fresh compact-pair rebuild path with fixed-cap padded outputs, or a validity key that proves the far-pair list is unchanged before reuse.

## 2026-06-20 Safe Rebuild Attempts After Compact-Pair Root Cause

Follow-up implementation attempts after identifying stale compact far-pair reuse:

- **Fresh compact-pair rebuild inside strict fused scan:** rebuilding pairs fresh avoids stale semantics, but the traced compact builder returns fixed-capacity arrays (`6402048` slots for the 200k profile) without an active mask/count in `CompactTaggedFarPairs`; downstream M2L treats padded slots as valid and produces NaNs.
- **Carry compact pairs as `None`:** attempted to keep scan carry invariant by stripping compact pairs from the input/output prepared state while rebuilding them internally. This still fails unless the refreshed output is also stripped, and when stripped the fixed-cap padded pairs still need masking before M2L.
- **Node-interaction safe path:** forcing non-compact node interactions would preserve count metadata conceptually, but yggdrax’s near-list count pass still uses Python `bool(count_wf_overflow)` on traced values, so it is not tracer-safe in the strict fused scan.
- Restored the safe default to a fail-fast guard. Unsafe legacy performance behavior requires `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1` and reproduces the corrupted one-step velocity.
- Added experimental flags for future work but left them default-off:
  - `JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD=1` for fixed-cap fresh compact-pair experiments; currently not production-correct because compact pairs lack mask/count.
  - `JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH=1` for node-interaction experiments; currently blocked by traced `bool(...)` in yggdrax near-list construction.

Required production fix:

1. Extend compact far-pair payloads with an active count or boolean mask and thread it through M2L/downward so padded fixed-cap slots are ignored on device.
2. Or implement a tracer-safe fixed-cap compact-pair builder that emits harmless masked pairs and a corresponding masked M2L application.
3. Add a moved-endpoint strict fused parity regression: initial prepared state -> drifted endpoint positions -> strict fused refresh/eval must match fresh FMM/direct subset before enabling the production scan.


## 2026-06-20 Safe Fresh Compact-Pair Rebuild Default

Implemented the production-correct follow-up to the stale compact far-pair reuse root cause:

- Extended `yggdrax.CompactTaggedFarPairs` with `far_pair_count` so traced fixed-cap compact far-pair arrays carry the number of active entries.
- Threaded the active count into jaccpot `_FarPairCOO` and solid-FMM downward/M2L setup.
- Masked padded compact far-pair slots in complex and real M2L full-batch/chunked paths so sentinel `-1` entries are never gathered or accumulated.
- Preserved strict scan-carry invariance by carrying the original compact-pair payload as a structure placeholder while using freshly rebuilt compact pairs internally for downward locals.
- Made `JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD` default-on for strict fused static-radix refresh. Legacy stale reuse remains explicit opt-in only via `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE=1`.

Validation:

- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py /export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py` passed.
- Focused 200k strict self-only one-step default now finishes finite/sane: `vel_max≈6.738`.
- Explicit unsafe legacy opt-in still reproduces corruption: `vel_max≈200840`, confirming stale compact-pair reuse is still not allowed for production.
- Focused 200k strict self-only three-step run is finite/sane through the old NaN onset window: `pos_max≈3.026`, `vel_max≈6.704`.
- Full ODISSEO canonical 200k/3, `t_end=0.3 Gyr`, autocvd GPU, fixed policy, cap `32`, `--require-finite-final-state` passed and saved `/tmp/odisseo_default_safe_200k_3/out.npz`; report `notebooks/scalability/reports/galaxy_disk_profile_20260620_200420.json` has `final_state_all_finite=true`, `final_state_nan_count=0`, `position_norm_max≈2.846`, `velocity_norm_max≈5.633`.

Runtime impact:

- Warm focused 200k strict self-only one-step corrected fresh path: `~1.49s`.
- Legacy unsafe reuse warm focused self-only one-step: `~0.40s`, but corrupted and non-production.
- Adding `lax.cond(start_idx < active_pair_count)` around chunked M2L work preserved correctness but did not materially improve warm runtime (`~1.49s`), suggesting the current overhead is dominated by fresh traversal/tree/downward rebuild or XLA still carries fixed-length scan/control overhead.
- Lowering `JACCPOT_LARGE_N_GPU_MIN_MEMORY_INTERACTIONS_PER_NODE` to `128` did not change the focused warm runtime, and initial prepare compact pairs are exact-length (`68272` active entries), so the remaining cost should be attributed with Nsight rather than assumed to be padded M2L alone.

Next optimization target:

1. Capture corrected default-on 200k/1 with Nsight graph-node tracing.
2. Re-run strict diagnostic cuts under the corrected integrator/default fresh path.
3. Attribute fresh traversal/tree, P2M/M2M, M2L/L2L, and nearfield deltas from corrected captures.
4. Only optimize compact-pair reuse again if we can prove a validity key that the far-pair list is unchanged after drift; otherwise keep fresh rebuild as the production-correct default.

## 2026-06-20 Corrected Default Nsight Pass and Regression Hardening

Regression hardening:

- Added `test_solidfmm_m2l_ignores_padded_compact_far_pairs` in `/export/home/tbuck/jaccpot/tests/integration/test_fmm.py`.
- The test compares an exact one-pair M2L accumulation against a padded compact-pair payload with `-1` sentinel slots and `active_count=1`.
- It covers both fullbatch and chunked M2L paths and asserts finite output plus equality to the exact active-pair result.
- Initial test run exposed a real chunked scatter bug: invalid padded targets sorted together with valid target `0` and could overwrite group validity. Fixed `_chunk_segment_scatter_add` by sorting on masked targets (`invalid -> int max`) before grouped reduction.
- Focused regression now passes: `micromamba run -n odisseo python -m pytest -q /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k 'padded_compact_far_pairs'`.

Corrected default Nsight pass:

- Command family: `tools/fused_audit_runner.py --nsys-capture --nsys-bin /usr/local/cuda-12.4/bin/nsys --nsys-cuda-graph-trace graph --fixed-policy --fixed-neighbor-cap 1048576 --require-autocvd --perf-warmup-runs 1 --perf-measure-runs 1 --n-particles 200000 --num-steps 1`.
- Artifact root: `/tmp/odisseo_nsys_corrected_safe_200k_1_20260620`.
- Fused-on baseline report: `/tmp/odisseo_nsys_corrected_safe_200k_1_20260620/baseline/reports/galaxy_disk_profile_20260620_202245.json`.
- Fused-on Nsight report: `/tmp/odisseo_nsys_corrected_safe_200k_1_20260620/baseline/nsys/baseline.nsys-rep`.
- Consolidated audit summary: `/tmp/odisseo_nsys_corrected_safe_200k_1_20260620/audit_summary.json`.

Fused-on corrected metrics:

- Warm measured step: `0.4425s` for 200k/1, fused active `true`, fallback `0`, planner bypass count `2`, acceleration carry active `true`, endpoint self-FMM evaluations `1`.
- No measured H2D/DtoH transfer calls.
- Tail GPU active: `2.74%`; tail GPU busy `11.89ms` over `433.9ms` span.
- Tail kernel count: `10837`.
- Static radix compact far-pair count in the strict traced path is fixed-cap padded: `6402048` slots, `1563` M2L chunks, with fresh compact-pair rebuild misses `1` and reuse hits `0`.
- Non-fused variant measured `0.8978s`, so corrected fused-on is now faster for 200k/1 despite the launch storm.

Top fused-on measured-tail launch sources by count:

- `loop_add_fusion_2`: `3588` launches, `3.65ms` kernel time.
- `wrapped_compare`: `1585` launches, `1.62ms` kernel time.
- `loop_compare_dynamic_slice_fusion`: `1563` launches, `1.84ms` kernel time.
- `loop_compare_fusion_5`: `1563` launches, `1.62ms` kernel time.
- `loop_dynamic_slice_fusion_3`: `1563` launches, `1.61ms` kernel time.
- `input_dynamic_slice_reduce_fusion`: `392` launches, `0.65ms` kernel time.

Interpretation:

- The correctness fix is now protected at the M2L padded-pair layer and the corrected fused-on path is again below the 0.5s 200k/1 target.
- The remaining bottleneck is launch count, not raw GPU math time: thousands of tiny loop/dynamic-slice/compare kernels dominate the tail.
- Next target should be the fixed-cap traced refresh/M2L scan: reduce the `1563` per-node/per-chunk loop family and the `3588` repeated add/update kernels. Nsight indicates this is still a CUDA graph launch fragmentation problem rather than host/device transfer overhead.

## 2026-06-21 Flat Compact Far Pairs and Static Node-Range Reuse

Implemented and promoted two correctness-preserving launch-reduction changes for the strict fused static-radix lane:

- Strict traced compact far pairs now use flat fixed-cap arrays controlled by `JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP`, with production cap `131072` and fail-fast overflow. The old per-node padded shape remains available with `JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS=0`.
- Static-radix refresh now reuses `template.node_ranges` by default. For fixed particle count and leaf size, these ranges index fixed Morton-sorted count buckets and do not change when particle permutation changes. Rollback: `YGGDRAX_STATIC_RADIX_REUSE_NODE_RANGES=0`.
- Fixed-policy benchmark/audit runners now set and report both controls explicitly.

Correctness hardening:

- Added strict fused moved-endpoint refresh/evaluate parity against fresh prepare/evaluate.
- Added an undersized compact far-pair cap failure regression.
- Retained the padded compact M2L masking regression for fullbatch and chunked paths.
- Focused jaccpot regressions pass: `padded_compact_far_pairs`, `moved_endpoint`, and `compact_far_pair_cap_fails`.
- Yggdrax static-radix refresh regression proves particle order changes while node ranges remain exactly equal.
- 200k/3 default-on and rollback runs are finite; relative state delta is `1.84e-7`, max absolute delta `4.14e-4`.
- Canonical 200k/20 default-on is finite and physically sane: position norm p50/p99/max `0.612/1.591/3.491`, velocity norm p50/p99/max `2.519/4.809/5.958`.
- Independent identical 200k/20 runs diverge by relative L2 `0.354`, comparable to default-on versus rollback (`0.393`). Therefore cross-process 20-step elementwise trajectory equality is not a valid oracle for this chaotic FP32 run; promotion uses moved-endpoint force parity, short-horizon parity, finiteness, and distributional sanity instead.

Performance and Nsight:

- Corrected pre-change baseline: `0.4425s/step`, `10837` measured-tail kernels.
- Flat compact cap only: `0.3831s/step`, `6239` tail kernels, M2L chunks `1563 -> 32`, zero H2D/DtoH.
- Flat cap plus node-range reuse: `0.3119s/step`, `1504` tail kernels, zero H2D/DtoH.
- Node-range reuse removes `4735` additional tail launches and improves the flat-cap result by `18.6%`; versus the corrected baseline the improvement is `29.5%`.
- Canonical 200k/20 measured `6.6767s`, or `0.333835s/step`; fused active `true`, fallback `0`, planner bypass `40`, endpoint self-FMM evaluations `20`.

Artifacts:

- Flat-cap Nsight: `/tmp/odisseo_nsys_flatcap_200k_1_20260621`.
- Flat-cap plus node-range reuse Nsight: `/tmp/odisseo_nsys_flatcap_range_reuse_200k_1_20260621`.
- Canonical measured 200k/20 audit: `/tmp/odisseo_flatcap_range_reuse_200k_20_20260621`.
- 200k/3 finite/parity gate: `/tmp/odisseo_range_reuse_gate_200k_3_20260621`.
- 200k/20 finite rollback gate: `/tmp/odisseo_range_reuse_gate_200k_20_20260621`.
- 200k/20 same-mode repeatability check: `/tmp/odisseo_range_reuse_repeatability_200k_20_20260621`.

Remaining measured-tail launch families after promotion:

- `loop_add_fusion`: `483` launches.
- `loop_dynamic_slice_fusion`: `392` launches.
- `input_dynamic_slice_reduce_fusion`: `392` launches.
- M2L chunk families: `32` launches each.

Next target: identify and batch the paired `392` dynamic-slice/reduce family, then the remaining `483` add/update launches. M2L is no longer the dominant launch source.

## 2026-06-21 Nearfield Pure-JAX Unroll Tuning Pass

Goal: test whether the remaining nearfield launch families can be reduced with existing static unroll knobs before introducing a larger rewrite or Pallas kernels.

Experiments:

- `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE=8` versus default `4`: rejected. Canonical 200k/1 slowed from `0.2896s` to `0.3567s`; parity remained clean with final-state max abs `1.53e-5` and relative L2 `6.27e-8`.
- `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL=2` versus default `1`: parity-clean but not promoted. Median improved from `0.289171s` to `0.285473s`; final-state max abs `1.53e-5`, relative L2 `4.49e-8`.
- Tile-scan unroll `4` versus `2`: parity-clean but not promoted. Median improved from `0.288403s` to `0.286332s`; final-state max abs `1.53e-5`, relative L2 `4.51e-8`.
- `JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL=2` versus default `1`: parity-clean but not promoted. Median improved from `0.288625s` to `0.287191s`; final-state max abs `1.53e-5`, relative L2 `4.63e-8`.
- Combined tile-scan unroll `4` plus batch-scan unroll `2`: parity-clean but not promoted. Median improved from `0.289384s` to `0.287016s`; final-state max abs `1.53e-5`, relative L2 `4.45e-8`.

Route/correctness gates for the accepted measurements:

- Strict fused active `true`, fallback `0`, planner bypass count `4`.
- Velocity-Verlet acceleration carry active; endpoint self-FMM evaluations equal measured steps.
- Flat compact far-pair cap remains `131072`, M2L chunks remain `32`.
- Final states are finite in all variants.

Decision:

- Do not change production defaults from these unroll knobs. The improvements are all under about `1.3%`, within expected benchmark noise, and none has yet demonstrated a material Nsight launch-count reduction.
- The force/correctness fix remains the fresh compact far-pair rebuild plus active-count M2L masking and velocity-Verlet acceleration carry. These unroll experiments preserve that corrected force path but do not further fix force error.
- Next useful step is a fresh Nsight capture of the current production default and/or the best unroll candidate only if kernel counts show a real reduction. Otherwise, move to a structural nearfield batching rewrite or a guarded Pallas prototype for the residual `392 + 392` dynamic-slice/reduce launch families.

Artifacts:

- Tile size 4 vs 8: `/tmp/odisseo_nearfield_tile4_vs8_200k_1_20260621`.
- Tile scan unroll 1 vs 2: `/tmp/odisseo_nearfield_unroll1_vs2_200k_1_20260621`.
- Tile scan unroll 2 vs 4: `/tmp/odisseo_nearfield_unroll2_vs4_200k_1_20260621`.
- Batch scan unroll 1 vs 2: `/tmp/odisseo_nearfield_batchunroll1_vs2_200k_1_20260621`.
- Combined unrolls: `/tmp/odisseo_nearfield_unroll_combo_200k_1_20260621`.

