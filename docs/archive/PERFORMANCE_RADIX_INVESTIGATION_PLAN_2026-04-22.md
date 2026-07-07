# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# ODISSEO + jaccpot Radix Fast-Lane Investigation Plan (2026-04-22)

## Why this plan
Current profiling shows `prepare_state` dominates runtime in the integrated ODISSEO path (`~20-30 s` per rebuild at `N=200000`), while standalone jaccpot reports much lower prepare time (target envelope `~0.4 s` for similar scale/config).  
Goal: close this gap by proving where the divergence happens and enforcing the fastest valid execution path.

## Scope
- In scope:
  - ODISSEO + jaccpot integration path for galaxy-disk large-N workflow
  - `prepare_state` performance root-cause analysis
  - runtime-path and compilation behavior audits
  - concrete optimization proposals and implementation candidates
- Out of scope for now:
  - adaptive refresh policy changes
  - model/physics changes unrelated to runtime-path performance

## Success criteria
1. Reproduce and explain the standalone vs integrated prepare-time gap.
2. Verify whether radix fast-lane (`large_n_gpu` + `large_n`) is consistently used.
3. Prioritize static-shape reuse for fixed `N`: keep tree/multipole data structures shape-stable across the run and only update numeric contents when positions change.
4. Identify avoidable recompilation/retracing or layout-conversion costs.
5. Produce prioritized optimization options with expected impact and risk.
6. Validate improvements with reproducible before/after reports (`N=200k`, `steps=20` and `steps=200`).

## Priority update (2026-04-23)
Top priority is now a fixed-`N` static-shape execution model:
- tree-structure and multipole containers should keep a reusable static shape for the whole simulation when particle count is fixed
- per-refresh work should prefer numeric-value updates over shape/topology reallocation/retracing
- ideal target: compile once at startup (or once per bounded variant set), then reuse compiled paths to end-of-run
- tree-topology and multipoles should be recomputed only when particle positions require it, without introducing new dynamic shapes

## Workstreams

### WS6: Static-Shape Tree/MultiPole Reuse (PRIORITY)
Objective: make fixed-`N` runs shape-stable so we minimize retracing/recompilation and reuse compiled kernels.

Tasks:
- Define and enforce a static-shape contract for:
  - tree containers
  - multipole containers
  - near-field/intermediate buffers with bounded capacities and masks as needed
- Prototype ODISSEO/jaccpot flow where per-refresh updates mutate values in preallocated structures.
- Detect and log any shape drift events (hard-fail in perf mode).
- Quantify startup compile cost vs steady-state prepare/evaluate cost after shape stabilization.

Deliverables:
- static-shape contract doc and assert hooks
- prototype implementation path (or upstream API proposal if blocked)
- compile-once feasibility report with measured hit rate

### WS1: Benchmark Harness Alignment
Objective: ensure apples-to-apples timing across standalone jaccpot and ODISSEO integration.

Tasks:
- Build a single harness that can run:
  - standalone jaccpot `prepare_state`/`evaluate_prepared_state`
  - ODISSEO coupler prepare/evaluate path
- Use identical:
  - particle data
  - dtype (`float32`)
  - device/GPU
  - solver config
  - warmup policy
- Record cold-start and warm-run timings separately.

Deliverables:
- `JSON`/`CSV` timing report with both paths side-by-side.

### WS2: Runtime-Path and Config Enforcement
Objective: guarantee the fastest intended path is active on every relevant call.

Tasks:
- Add explicit runtime audit fields to reports:
  - effective preset
  - effective runtime path
  - effective working dtype
  - key FMM config knobs
- Fail fast if expected values are not met in benchmark mode.

Deliverables:
- strict “effective config” report section
- optional assert mode for perf runs

### WS3: JIT/Compile Behavior Audit
Objective: separate pure execution cost from compilation/retracing overhead.

Tasks:
- Instrument first-call vs steady-state timings per segment.
- Track compile-like spikes by segment index.
- Check static argument variability that can trigger retracing.

Deliverables:
- compile/execute decomposition report
- list of retrace triggers (if any)

### WS4: Array Layout and Materialization Audit
Objective: detect costly array-representation transitions between integration and FMM prepare.

Tasks:
- Inspect state buffer properties before each prepare call.
- Benchmark effect of explicit rematerialization/casting at controlled points.
- Quantify host-device sync and copy penalties.

Deliverables:
- documented “state handoff contract” for best performance
- measured impact table for each materialization strategy

### WS5: Python Overhead and Loop Structure Audit
Objective: ensure hot path is dominated by compiled kernels, not Python orchestration.

Tasks:
- Validate no accidental per-step Python-heavy work in critical loop.
- Consolidate repeated operations under jitted scans where safe.
- Minimize repeated solver construction or non-essential object churn.

Deliverables:
- hot-path call graph summary
- shortlist of low-risk refactors

## Experiment matrix (initial)
Run on fixed GPU and environment:
- `N=200000`, `steps=20`, `refresh=4`
- `N=200000`, `steps=200`, `refresh=4`
- Variants:
  - cold process vs warm process
  - standalone jaccpot vs ODISSEO integration
  - optional toggles for JIT policy and materialization points
  - static-shape ON vs OFF (fixed-capacity/masked containers)
  - startup compile warmup then long-run reuse validation

## Reporting template per run
- command line
- GPU metadata
- effective FMM runtime config
- total runtime
- prepare/evaluate/update timing
- compile-like first-call cost vs steady-state
- notes on anomalies

## Prioritized implementation order
1. WS6 (static-shape tree/multipole reuse strategy for fixed `N`)
2. WS1 + WS2 (alignment and config truth)
3. WS3 + WS4 (compile/layout bottlenecks under static-shape constraints)
4. WS5 (loop/orchestration cleanups)

## Decision gates
- Gate A: if runtime-path mismatch exists, fix before any deeper optimization.
- Gate B: if static-shape contract cannot be enforced, stop and resolve API/data-structure blockers before further tuning.
- Gate C: if compile/retrace dominates, prioritize trace stability within static-shape constraints before lower-priority cleanups.
- Gate D: if execution is still dominated by rebuild after A/B/C, escalate upstream changes for reusable topology/multipole update APIs.

## Expected outputs for consolidation
- before/after table (`20-step` and `200-step`)
- confirmed bottleneck attribution
- recommended defaults and rationale
- implementation backlog for remaining high-impact work

## Execution Log

### 2026-04-22: WS1 + WS2 Initial Run (GPU 6)

Harness added:
- `notebooks/scalability/radix_fastlane_investigation.py`
- compares:
  - standalone jaccpot (`FastMultipoleMethod` minimal constructor)
  - ODISSEO coupler-built solver (`_build_fmm_solver`)
- reports:
  - effective runtime config from integration API resolver
  - cold single-call timing
  - steady-state per-state timing after warmup sweep

Artifacts:
- warmup-enabled baseline (`jit_tree=False`, `jit_traversal=False`)
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_v3_20260422_203549.json`
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_v3_20260422_203549.csv`
- jit-on comparison (`jit_tree=True`, `jit_traversal=True`)
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_v3_jit_on_20260422_204104.json`
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_v3_jit_on_20260422_204104.csv`

Key findings:
- Fast-lane assertions passed:
  - effective preset: `large_n_gpu`
  - effective runtime path: `large_n`
  - effective dtype: `float32`
- Steady-state prepare (3 benchmark states):
  - `jit OFF`:
    - direct: `~4.02 s` total prepare
    - coupler-builder: `~7.34 s` total prepare
  - `jit ON`:
    - direct: `~3.98 s` total prepare
    - coupler-builder: `~4.02 s` total prepare
- Interpretation:
  - the earlier large discrepancies can be dominated by compile/warmup asymmetry.
  - with controlled warmup and jit policy aligned, direct and coupler builder are close.
  - next step is to reconcile these microbench numbers with full integration-loop timings where prepare still dominates.

### 2026-04-22: Leaf-256 / Order-4 Alignment (GPU 6)

Constraint requested:
- keep `max_order=4`
- use `leaf_size=256` (matching fast-lane tuning direction)

Artifacts:
- aligned WS1/WS2 report:
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_leaf256_order4_20260422_210618.json`
  - `notebooks/scalability/reports/radix_fastlane_ws1ws2_gpu6_leaf256_order4_20260422_210618.csv`

Results (steady-state warmup-enabled rows, `N=200000`):
- direct jaccpot prepare per state:
  - `0.819 s`, `0.739 s`, `0.668 s`
- coupler-builder prepare per state:
  - `0.723 s`, `0.656 s`, `0.677 s`

WS3 first-seen segment trace (`N=200000`, `steps=20`, `refresh=4`, `leaf=256`, `order=4`, jit on):
- seg0: prepare `76.985 s`
- seg1: prepare `19.956 s`
- seg2: prepare `20.095 s`
- seg3: prepare `19.720 s`
- seg4: prepare `21.671 s`

Interpretation:
- we can approach sub-second prepare on repeated/warmed states with aligned settings.
- full integration still pays very large first-seen prepare cost per refreshed segment.
- this strongly indicates per-new-state topology/compile overhead remains the core blocker for end-to-end runtime.

### 2026-04-22: WS3 Step-1 Validation (GPU 6, isolated process)

Goal:
- validate only step 1 (input canonicalization inside `prepare_state`) against the `~20s` prepare anomaly.

What was tested:
- temporary jaccpot patch in `_prepare_state_input_arrays` to force canonicalized positions layout via row gather.
- benchmark setup held constant:
  - `N=200000`, `steps=20`, `refresh=4`
  - `preset=large_n_gpu`, `runtime_path=large_n`, `dtype=float32`
  - `leaf_size=256`, `max_order=4`, `jit_tree=True`, `jit_traversal=True`
  - GPU 6 (`CUDA_VISIBLE_DEVICES=6`)
- isolated A/B runs in fresh Python processes to avoid warm-cache bias:
  - `RAW_ONLY prepare=20.136 s`
  - `ASARRAY_ONLY prepare=19.763 s`

Additional observation:
- in-process second prepare on the same evolved state can drop to `~0.7-0.8 s`, indicating heavy one-time compile/cache effects.
- this explains earlier misleading `raw vs asarray` deltas when run sequentially in a single process.

Conclusion:
- step 1 did **not** resolve the cold evolved-state prepare bottleneck.
- the active issue is not just array-view layout; dominant cost remains first-seen prepare/compile behavior on new segment states.

Repository hygiene:
- temporary canonicalization patch was reverted in jaccpot (`runtime/_fmm_impl.py`) after validation.

### 2026-04-22: WS3 Compile-Stability Deep Dive (GPU 6)

Controlled setup:
- `N=200000`, `steps=20`, `refresh=4`, `leaf=256`, `order=4`
- `preset=large_n_gpu`, `runtime_path=large_n`, `dtype=float32`
- all segment states explicitly synchronized (`jax.block_until_ready`) before prepare benchmarking.

Key reproducible behavior:
- First fresh-process pass over 5 segment end-states shows one late heavy first-seen prepare:
  - example: `0.773, 1.313, 0.843, 0.842, 20.823 s`
- Repeating the exact same pass in the same process with fresh solver objects is fast:
  - example rep2/rep3: all states `~0.70-0.86 s`

Interpretation:
- steady-state prepare remains in the expected `~0.7-1.2 s` band.
- the `~20 s` event is a one-time first-seen compile/specialization for a later-state variant.

Observed per-state metadata (same run):
- tree topology sizes stayed fixed during the 5-segment sweep:
  - `parent_n=1563`, `leaf_n=782`
- near-field edge volume increased each segment:
  - `neighbors_n`: `178410 -> 184252 -> 193990 -> 199162 -> 212054`
- heavy first-seen event occurred at the largest seen neighbor volume in that pass.

What was tested and reverted:
- attempted fixed-capacity padding of overflow-tail target-block tensors in
  `jaccpot/runtime/_large_n_pipeline.py` to stabilize one dynamic shape.
- result: did **not** remove the first-seen `~20 s` event.
- patch reverted; file restored.

Practical implication for ODISSEO/jaccpot runs:
- without warmup, one-time compile spikes can appear when prepare first encounters a larger near-field regime.
- after those variants compile, prepare returns to the `~0.7-0.9 s` regime.

### 2026-04-23: WS6 Static-Shape Contract Smoke (GPU 0)

Goal:
- validate new static-shape instrumentation and fail-fast contract checks in ODISSEO coupler path.

Code-level additions in this session:
- Added new FMM config knobs:
  - `fmm_enforce_static_shape_contract`
  - `fmm_static_shape_warmup_prepares`
  - `fmm_rematerialize_between_refresh`
- Added prepared-state leaf dtype/shape signature tracking in `integrate_leapfrog_jaccpot_active`.
- Added shape-stability counters to timing stats:
  - `shape_signature_checks`
  - `shape_signature_unique_count`
  - `shape_signature_drift_events`
  - `shape_signature_stable`

GPU 0 runs:
- fail-fast contract run (`--fmm-enforce-static-shape-contract` enabled):
  - outcome: raised `RuntimeError` on prepared-state shape drift across refresh segments.
- diagnostic run (same config, enforcement disabled):
  - output: `/tmp/galaxy_gpu0_staticshape_smoke_noenforce.npz`
  - timing report:
    - `notebooks/scalability/reports/galaxy_disk_profile_20260423_103441.json`
    - `notebooks/scalability/reports/galaxy_disk_profile_20260423_103441.csv`
  - conservation report:
    - `notebooks/scalability/reports/galaxy_disk_conservation_20260423_103612.json`
    - `notebooks/scalability/reports/galaxy_disk_conservation_20260423_103612.csv`

Key diagnostics (`N=50000`, `steps=8`, `refresh=4`, warmup prepare=1):
- `shape_signature_checks=3`
- `shape_signature_unique_count=2`
- `shape_signature_drift_events=1`
- `shape_signature_stable=false`
- runtime:
  - `script_runtime_seconds=111.058 s`
  - `warmup_seconds=79.845 s`
  - `prepare_seconds=28.635 s` (`prepare_calls=2`)

Interpretation:
- fixed-`N` prepared-state shapes are still not fully stable across segment updates.
- static-shape guardrails are functioning correctly and catch drift early.
- next step remains WS6 core: remove the remaining dynamic-shape driver(s) while preserving runtime and conservation.

### 2026-04-23: WS6 Incremental Prepare Adapter (ODISSEO-side)

Goal:
- consume upcoming jaccpot incremental prepared-state APIs without breaking older jaccpot builds.

Code changes:
- updated `odisseo/jaccpot_coupling.py` to reuse previous prepared state across refresh segments and:
  - attempt `solver.refresh_prepared_state(...)` when available
  - fall back automatically to full `solver.prepare_state(...)` when unavailable/signature-mismatch
- added prepare-path diagnostics counters:
  - `refresh_prepare_attempts`
  - `refresh_prepare_successes`
  - `refresh_prepare_fallbacks`
  - `full_prepare_calls`
  - `refresh_prepare_method_available`

Validation:
- `tests/test_integration_api.py` passes after patch.

Usage implication:
- current ODISSEO runs remain backward-compatible with existing jaccpot.
- once jaccpot exposes stable incremental APIs, ODISSEO can use them immediately and report refresh-hit rate via timing stats.

### 2026-04-23: GPU 9 Validation with New jaccpot Incremental APIs

Goal:
- confirm ODISSEO actually exercises the new jaccpot `refresh_prepared_state` method.

Run:
- `CUDA_VISIBLE_DEVICES=9`
- `notebooks/scalability/galaxy_disk_fmm_large_n.py`
- `--mode perf --n-particles 10000 --num-steps 8 --profile-breakdown`
- `--fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-refresh-every 4`
- `--fmm-static-shape-warmup-prepares 1`

Artifacts:
- output: `/tmp/galaxy_gpu9_staticshape_api_smoke.npz`
- timing report:
  - `notebooks/scalability/reports/galaxy_disk_profile_20260423_141713.json`
  - `notebooks/scalability/reports/galaxy_disk_profile_20260423_141713.csv`

Key diagnostics:
- `refresh_prepare_method_available=true`
- `refresh_prepare_attempts=1`
- `refresh_prepare_successes=1`
- `refresh_prepare_fallbacks=0`
- `shape_signature_unique_count=2`
- `shape_signature_drift_events=1`
- `shape_signature_stable=false`

Interpretation:
- API wiring is now validated end-to-end (ODISSEO -> jaccpot refresh path).
- core WS6 blocker remains unresolved: prepared-state shape drift still occurs across refresh segments.

### 2026-04-23: WS6 Conservative Overflow-Profile Padding (jaccpot runtime)

Goal:
- stabilize the known drifting tensors (`nearfield_target_block_*` overflow tail) with minimal runtime risk.

Implemented in `jaccpot`:
- Added large-N overflow profile capacity logic in `runtime/_large_n_pipeline.py`:
  - profile cap tracks overflow block rows with configurable headroom and cap ladder.
  - only overflow target-block tensors are padded to profile capacity.
  - cap grows only when observed overflow exceeds current capacity (controlled reprofile).
- Added new large-N prepared-state metadata in `runtime/_large_n_types.py`:
  - `nearfield_target_block_overflow_profile_capacity`
  - `nearfield_target_block_overflow_active_blocks`
- Added runtime diagnostics in `runtime/_fmm_impl.py`:
  - `large_n_overflow_profile_cap`
  - `large_n_overflow_profile_reprofiles`
- Surfaced runtime diagnostics into ODISSEO timing stats in `odisseo/jaccpot_coupling.py`.

New env knobs (conservative defaults):
- `JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM` (default `2.0`)
- `JACCPOT_LARGE_N_OVERFLOW_PROFILE_CAP_OPTIONS`
  (default `64,128,256,512,1024,2048,4096,8192,16384,32768,65536`)

GPU 9 checks:
- multi-state jaccpot prepare validation:
  - profile cap reached `1024`
  - `large_n_overflow_profile_reprofiles=0` across subsequent nontrivial states
  - no shape drift observed between prepared states after cap establishment
- ODISSEO short perf run (`N=10000`, `steps=8`, `refresh=4`, warmup=1):
  - report: `notebooks/scalability/reports/galaxy_disk_profile_20260423_143723.json`
  - key counters:
    - `refresh_prepare_successes=1`
    - `runtime_large_n_overflow_profile_cap=4096`
    - `runtime_large_n_overflow_profile_reprofiles=0`
    - `shape_signature_unique_count=2`, `shape_signature_drift_events=1`

Interpretation:
- We now suppress repeated drift from overflow-tail growth after profile cap is established.
- Remaining drift event is the first transition from zero-overflow shape to the first nonzero profile-cap shape.
- Runtime impact in this short smoke is low-single-digit (~`+1.6%` vs prior comparable short run), consistent with conservative scope.

### 2026-04-23: Residual Drift Root-Cause Isolation (GPU 9)

Additional diagnostics added:
- ODISSEO timing report now includes post-warmup shape counters and signature-hash deltas.
- Debug payload `shape_signature_diff_post_warmup` reports added/removed dtype+shape entries.

Validated runs:
- strict bootstrap mode via
  - `JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP=4096`
- latest artifact:
  - `notebooks/scalability/reports/galaxy_disk_profile_20260423_150905.json`

Findings:
- `runtime_compiled_profile_transitions=0`
- `runtime_large_n_overflow_profile_reprofiles=0`
- residual post-warmup drift persists:
  - `shape_signature_drift_events_post_warmup=1`
- exact delta:
  - one `int64` 1D leaf changes shape (`23988 -> 19442`)

Interpretation:
- overflow-tail profile work is stable (no further profile transitions/reprofiles).
- remaining shape instability is now isolated to a topology-dependent edge-list-like payload.
- next WS6 task is explicit static-cap/masked padding for that edge-list family (likely neighbor/edge traversal tensors), not multipole overflow tails.

### 2026-04-23: WS6 Edge-List Static Cap (GPU 9) - Achieved Post-Warmup Stability

Implemented:
- Added radix fast-lane neighbor-edge profile capacity in `jaccpot/runtime/_large_n_pipeline.py`:
  - pads `neighbor_list.neighbors` to a tiered capacity profile.
  - tracks capacity growth/reprofile in runtime diagnostics.
- Added diagnostics fields in `jaccpot/runtime/_fmm_impl.py`:
  - `large_n_neighbor_edges_profile_cap`
  - `large_n_neighbor_edges_profile_reprofiles`
- Exposed new counters in ODISSEO timing report (`runtime_large_n_neighbor_edges_*`).

Conservative default tuning:
- `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM=1.0`
- `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_CAP_OPTIONS=4096,8192,12288,16384,20480,24576,28672,32768,49152,65536,98304,131072`

Validation run (GPU 9):
- command family: `galaxy_disk_fmm_large_n.py --mode perf --n-particles 10000 --num-steps 8 --fmm-refresh-every 4 --fmm-static-shape-warmup-prepares 1`
- with `JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP=4096`
- artifact:
  - `notebooks/scalability/reports/galaxy_disk_profile_20260423_154405.json`

Key results:
- `shape_signature_drift_events=0`
- `shape_signature_drift_events_post_warmup=0`
- `shape_signature_stable_post_warmup=true`
- `runtime_compiled_profile_transitions=0`
- `runtime_large_n_overflow_profile_reprofiles=0`
- `runtime_large_n_neighbor_edges_profile_reprofiles=0`
- runtime for this short smoke: `~98.8 s`

Interpretation:
- fixed-shape objective is now met for this validated lane after startup warmup.
- compile-profile transitions and reprofiles are eliminated in this short workload.
- runtime overhead remains present vs earlier non-edge-cap short runs, but reduced versus more aggressive edge headroom defaults.

### 2026-04-23: Acceptance Benchmark Checkpoint (Handoff Snapshot)

Status snapshot before next step:
- baseline acceptance run completed on GPU 9:
  - config: `N=200000`, `steps=20`, `refresh=4`, `preset=large_n_gpu`, conservation enabled (stride 1), warmup prepares 1
  - reports:
    - `/tmp/radix_acceptance_baseline/galaxy_disk_profile_20260423_155837.json`
    - `/tmp/radix_acceptance_baseline/galaxy_disk_conservation_20260423_160832.json`
  - key result: shape stable (`shape_signature_drift_events_post_warmup=0`)
  - runtime: `~312.19 s`
- paired locked-cap comparison run was launched with:
  - `JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP=883366`
  - `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP=3722466`
  - same benchmark settings/seed
  - output target: `/tmp/galaxy_gpu9_acceptance_locked_200k_20.npz`
  - report dir: `/tmp/radix_acceptance_locked/`
- at this checkpoint the locked-cap run is still in progress and final JSON/CSV are not yet recorded in this document.

### 2026-04-24: Acceptance Benchmark Completion + Regression Gates

Completed the previously launched locked-cap comparison:
- locked-cap timing report:
  - `/tmp/radix_acceptance_locked/galaxy_disk_profile_20260423_161528.json`
  - `/tmp/radix_acceptance_locked/galaxy_disk_profile_20260423_161528.csv`
- locked-cap conservation report:
  - `/tmp/radix_acceptance_locked/galaxy_disk_conservation_20260423_162429.json`
  - `/tmp/radix_acceptance_locked/galaxy_disk_conservation_20260423_162429.csv`

Baseline vs locked-cap summary (`N=200000`, `steps=20`, `refresh=4`, warmup prepares `1`):
- baseline runtime: `312.188 s`
- locked-cap runtime: `318.307 s` (`+1.96%`)
- both runs:
  - `shape_signature_drift_events_post_warmup=0`
  - `shape_signature_unique_count_post_warmup=1`
  - `runtime_compiled_profile_transitions=0`
  - `runtime_large_n_overflow_profile_reprofiles=0`
  - `runtime_large_n_neighbor_edges_profile_reprofiles=0`
  - `refresh_prepare_successes=4`, `refresh_prepare_fallbacks=0`
- conservation remained comparable/slightly improved in the locked-cap run:
  - `max_abs_dE_over_E0`: `0.108207 -> 0.108075`
  - `max_abs_dL_over_L0`: `0.091774 -> 0.091522`
  - `max_com_drift`: `0.021355 -> 0.021304`

Implemented in ODISSEO benchmark script:
- Added explicit perf regression gates to `notebooks/scalability/galaxy_disk_fmm_large_n.py`:
  - `--require-static-shape`
  - `--max-compiled-profile-transitions`
  - `--max-overflow-reprofiles`
  - `--max-neighbor-edge-reprofiles`
  - `--min-refresh-prepare-successes`
- These gates make the fixed-`N` static-shape objective executable in acceptance runs instead of relying on manual report inspection.

Recommended acceptance command shape for the next run:
```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 200 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-refresh-every 4 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 \
  --profile-breakdown \
  --conservation-report \
  --require-static-shape \
  --max-compiled-profile-transitions 0 \
  --max-overflow-reprofiles 0 \
  --max-neighbor-edge-reprofiles 0 \
  --min-refresh-prepare-successes 49
```

Next technical priority:
- run the gated `N=200000`, `steps=200` acceptance case
- if static/profile gates pass but prepare time still dominates, shift focus from shape drift to reducing warmup/refresh prepare execution cost inside jaccpot's large-N refresh path

### 2026-04-24: Fast-Lane Setting Alignment + Prepare Breakdown

Follow-up after reviewing the `318 s` locked-cap runtime:
- The acceptance script was still hard-coding `fmm_leaf_size=64` and `fmm_tree_leaf_target=64`.
- Earlier sub-second warm prepare measurements used the faster radix large-N setting `leaf_size=256`, `max_order=4`.
- This mismatch can dramatically increase leaf/edge bookkeeping and padded static-cap work.

Implemented in `notebooks/scalability/galaxy_disk_fmm_large_n.py`:
- default `--fmm-leaf-size` is now `256`
- default `--fmm-tree-leaf-target` follows `--fmm-leaf-size`
- added explicit CLI controls:
  - `--fmm-leaf-size`
  - `--fmm-tree-leaf-target`
  - `--fmm-max-order`
  - `--fmm-nearfield-edge-chunk-size`
- timing reports now echo these requested values.

Implemented in `odisseo/jaccpot_coupling.py`:
- added profiled prepare buckets:
  - `profiled_full_prepare_calls`
  - `profiled_refresh_prepare_calls`
  - `profiled_refresh_fallback_prepare_calls`
  - `profiled_full_prepare_seconds`
  - `profiled_refresh_prepare_seconds`
  - `profiled_refresh_fallback_prepare_seconds`
- these split the existing `prepare_seconds` into full prepare vs successful incremental refresh vs fallback prepare for the profiled integration loop.

Current constraint:
- GPU validation was not run in this session because all GPUs are currently occupied.
- CPU-only syntax/CLI checks are acceptable, but the meaningful acceptance rerun must wait for GPU availability.

Next GPU run should first repeat the `N=200000`, `steps=20`, `refresh=4` acceptance case with explicit `leaf=256/tree_leaf_target=256/order=4`, then run the gated `steps=200` case if the prepare buckets look sane.

### 2026-04-24: GPU 0 Leaf-256 Acceptance Diagnostic

Run completed on GPU 0:
```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-refresh-every 4 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 \
  --profile-breakdown \
  --conservation-report \
  --report-dir /tmp/radix_acceptance_leaf256_gpu0_20 \
  --output /tmp/galaxy_gpu0_acceptance_leaf256_200k_20.npz
```

Artifacts:
- profile:
  - `/tmp/radix_acceptance_leaf256_gpu0_20/galaxy_disk_profile_20260424_100504.json`
  - `/tmp/radix_acceptance_leaf256_gpu0_20/galaxy_disk_profile_20260424_100504.csv`
- conservation:
  - `/tmp/radix_acceptance_leaf256_gpu0_20/galaxy_disk_conservation_20260424_101505.json`
  - `/tmp/radix_acceptance_leaf256_gpu0_20/galaxy_disk_conservation_20260424_101505.csv`

Key timing result:
- `script_runtime_seconds=257.227`
- `warmup_seconds=125.573`
- `prepare_seconds=127.303`
- `evaluate_seconds=0.365`
- `update_seconds=2.716`
- `profiled_full_prepare_calls=1`
- `profiled_full_prepare_seconds=1.103`
- `profiled_refresh_prepare_calls=4`
- `profiled_refresh_prepare_seconds=126.200`
- average profiled refresh prepare: `31.55 s`

Static-shape/profile diagnostics:
- `shape_signature_drift_events_post_warmup=0`
- `shape_signature_unique_count_post_warmup=1`
- `refresh_prepare_successes=4`
- `refresh_prepare_fallbacks=0`
- `runtime_compiled_profile_transitions=0`
- `runtime_large_n_overflow_profile_reprofiles=0`
- `runtime_large_n_neighbor_edges_profile_reprofiles=0`
- `runtime_large_n_overflow_profile_cap=102830`
- `runtime_large_n_neighbor_edges_profile_cap=458600`

Conservation metrics:
- `max_abs_dE_over_E0=0.153768`
- `max_abs_dL_over_L0=0.132560`
- `max_com_drift=0.059733`

Interpretation:
- `leaf=256` reduces profile capacities substantially compared with the earlier leaf-64 locked-cap run, and the profiled full prepare is fast (`~1.1 s`).
- The dominant integration cost is now isolated to successful `refresh_prepared_state` calls (`~31.55 s` each), not full prepare, force evaluation, shape drift, fallback rebuild, or profile recompile/reprofile.
- `evaluate_prepared_state` remains fast (`~0.365 s` total over 5 calls), consistent with standalone jaccpot sweep expectations.
- The next optimization target is jaccpot's internal refresh implementation: add sub-stage timing inside `refresh_prepared_state` / radix large-N refresh to separate topology rebuild, neighbor edge rebuild/padding, multipole refresh, interaction scaffold update, and hidden compile/synchronization costs.

Code inspection follow-up:
- In current jaccpot (`/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`), `refresh_prepared_state(...)` delegates directly to `self.prepare_state(...)`.
- Therefore the ODISSEO-side refresh success counter only proves that the API path is called; it does **not** prove that an incremental refresh implementation is active.
- This explains the measured behavior:
  - full profiled prepare can be fast when the initial state/path is already warmed
  - each later "refresh" still performs full prepare-style work for the evolved state
  - static shape prevents profile drift/recompile, but it does not yet avoid rebuild execution cost
- Next jaccpot change should replace the internal `prepare_state(...)` call with real tiered refresh logic:
  1. same topology key: update sorted payloads and multipoles/locals only
  2. topology changed but capacity-compatible: rebuild topology/interactions into existing static-capacity containers
  3. capacity overflow: controlled full reprofile

## Incremental Refresh Recovery Plan (Continuous Memory)

### Summary
The slow ODISSEO+jaccpot path is not caused by force evaluation, prepared-state shape drift, or ODISSEO dispatch. ODISSEO reaches `refresh_prepared_state(...)`, but current jaccpot implements that method by delegating to full `prepare_state(...)`, so each refresh still pays full rebuild-style cost. The next priority is to instrument jaccpot refresh sub-stages first, then replace refresh internals with true tiered incremental behavior.

### Key Changes
- Add jaccpot refresh-stage timing diagnostics for the large-N radix path:
  - `refresh_total_seconds`
  - `refresh_input_seconds`
  - `refresh_tree_upward_seconds`
  - `refresh_dual_downward_seconds`
  - `refresh_nearfield_seconds`
  - `refresh_profile_accounting_seconds`
  - `refresh_compile_or_sync_suspect_seconds`
- Surface these through `get_runtime_diagnostics()` and ODISSEO timing reports as `runtime_refresh_*`.
- Keep existing ODISSEO refresh counters and prepare buckets; they already show full vs refresh-path outer costs.
- The instrumentation step must not change numerical behavior.

### Implementation Sequence
1. **jaccpot instrumentation only**
   - Instrument `refresh_prepared_state(...)` and `prepare_large_n_state(...)` sub-stages.
   - Use `time.perf_counter()` around existing stage calls.
   - Record cumulative seconds and call counts on the solver instance.
   - Reset counters in the existing runtime diagnostic reset path.
   - Return all timing fields from `get_runtime_diagnostics()`.

2. **ODISSEO report plumbing**
   - Add the new `runtime_refresh_*` fields to `odisseo/jaccpot_coupling.py`.
   - Keep benchmark CLI unchanged except reports now include the extra fields.

3. **GPU 0 diagnostic rerun**
   - First run profile-only, no conservation:
```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 20 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-refresh-every 4 \
  --fmm-leaf-size 256 --fmm-tree-leaf-target 256 --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 --profile-breakdown \
  --report-dir /tmp/radix_refresh_stage_timing_gpu0_20 \
  --output /tmp/galaxy_gpu0_refresh_stage_200k_20.npz
```
   - Run conservation separately only after profile timing is understood.

4. **True incremental refresh design**
   - Replace current `refresh_prepared_state -> prepare_state` delegation with tiered behavior:
     - same `topology_key`: refresh sorted positions/masses, upward multipoles, downward locals, and nearfield numeric payloads without rebuilding topology/interactions
     - topology changed but profile-compatible: rebuild topology/interactions into existing padded/static-capacity containers
     - capacity overflow: controlled full prepare/reprofile
   - Keep automatic fallback to full prepare for unsupported modes or invariant failures.

5. **Acceptance experiments**
   - `N=200000`, `steps=20`, `refresh=4`, `leaf=256`, profile-only.
   - Same run with conservation report.
   - `N=200000`, `steps=200`, `refresh=4`, profile-only.
   - Final gated run with:
     - `--require-static-shape`
     - `--max-compiled-profile-transitions 0`
     - `--max-overflow-reprofiles 0`
     - `--max-neighbor-edge-reprofiles 0`
     - `--min-refresh-prepare-successes 49`

### Success Criteria
- Instrumentation explains at least 95% of `profiled_refresh_prepare_seconds`.
- Before incremental implementation, measured refresh cost remains attributable and reproducible.
- After incremental implementation:
  - `profiled_refresh_prepare_seconds / refresh_prepare_calls` drops from `~31.55 s` toward the standalone warm prepare/evaluate envelope.
  - `evaluate_seconds` remains sub-second scale for `20`-step acceptance.
  - post-warmup shape drift remains zero.
  - compiled profile transitions and reprofiles remain zero for locked/static-cap runs.
  - conservation metrics do not regress relative to the accepted baseline for the same leaf/order configuration.

### Assumptions
- Scope remains limited to `large_n_gpu`, radix tree, `solidfmm`, fixed `N`, `leaf=256`, `max_order=4`.
- Instrumentation first is the chosen next step.
- Conservation reporting is treated as a separate validation phase because it performs additional FMM prepare/evaluate-potential passes and can obscure integration timing.

### 2026-04-24: Refresh Stage Timing Instrumentation Result

Implemented instrumentation:
- jaccpot now records refresh-stage timing counters:
  - `refresh_total_seconds`
  - `refresh_input_seconds`
  - `refresh_tree_upward_seconds`
  - `refresh_dual_downward_seconds`
  - `refresh_nearfield_seconds`
  - `refresh_profile_accounting_seconds`
  - `refresh_compile_or_sync_suspect_seconds`
  - `refresh_timing_calls`
- ODISSEO timing reports now surface these as `runtime_refresh_*`.

Validation:
- CPU syntax check passed:
  - `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py odisseo/jaccpot_coupling.py`
- Focused ODISSEO test passed:
  - `micromamba run -n odisseo python -m pytest tests/test_integration_api.py`

GPU 0 profile-only diagnostic:
```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 20 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-refresh-every 4 \
  --fmm-leaf-size 256 --fmm-tree-leaf-target 256 --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 --profile-breakdown \
  --report-dir /tmp/radix_refresh_stage_timing_gpu0_20 \
  --output /tmp/galaxy_gpu0_refresh_stage_200k_20.npz
```

Artifact:
- `/tmp/radix_refresh_stage_timing_gpu0_20/galaxy_disk_profile_20260424_103146.json`

Key results:
- `script_runtime_seconds=258.972`
- `warmup_seconds=125.033`
- `prepare_seconds=129.630`
- `profiled_refresh_prepare_seconds=128.541`
- `runtime_refresh_total_seconds=128.527`
- `runtime_refresh_timing_calls=4`
- stage split:
  - `runtime_refresh_input_seconds=0.0002`
  - `runtime_refresh_tree_upward_seconds=2.809`
  - `runtime_refresh_dual_downward_seconds=103.863`
  - `runtime_refresh_nearfield_seconds=21.849`
  - `runtime_refresh_profile_accounting_seconds=0.004`
  - `runtime_refresh_compile_or_sync_suspect_seconds=0.003`

Interpretation:
- The instrumentation explains effectively all refresh cost.
- The main bottleneck is dual/downward rebuild inside the current refresh path (`~25.97 s` per refresh).
- Secondary cost is nearfield/payload rebuild (`~5.46 s` per refresh).
- Tree/upward refresh is comparatively small (`~0.70 s` per refresh).
- Compile/sync residual is negligible, so this is real rebuild work, not hidden recompilation.

Next implementation target:
- first incremental slice should avoid `_prepare_state_dual_and_downward(...)` and nearfield rebuild when topology/interactions are reusable.
- If topology changes every refresh, the target becomes a capacity-compatible topology rebuild path that writes into static containers without retracing/reprofile, but the current data show the expensive code is the dual/downward stage rather than tree/upward.

### 2026-04-24: Dual/Downward Sub-Stage Timing Result

Implemented deeper instrumentation inside `_prepare_state_dual_and_downward(...)`:
- `refresh_dual_setup_seconds`
- `refresh_dual_artifact_build_seconds`
- `refresh_dual_far_pair_plan_seconds`
- `refresh_dual_m2l_autotune_seconds`
- `refresh_dual_select_interactions_seconds`
- `refresh_dual_downward_compute_seconds`
- `refresh_dual_finalize_seconds`
- `refresh_dual_residual_seconds`

Validation:
- CPU syntax check passed.
- Focused ODISSEO integration API tests passed.

GPU 0 profile-only rerun:
```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 20 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-refresh-every 4 \
  --fmm-leaf-size 256 --fmm-tree-leaf-target 256 --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 --profile-breakdown \
  --report-dir /tmp/radix_refresh_dual_stage_timing_gpu0_20 \
  --output /tmp/galaxy_gpu0_refresh_dual_stage_200k_20.npz
```

Artifact:
- `/tmp/radix_refresh_dual_stage_timing_gpu0_20/galaxy_disk_profile_20260424_104329.json`

Key results:
- `script_runtime_seconds=261.232`
- `profiled_refresh_prepare_seconds=130.501`
- `runtime_refresh_prepare_calls=4`
- `runtime_refresh_prepare_reuse_tier_full=4`
- `runtime_refresh_dual_downward_seconds=105.815`
- dual/downward split:
  - `runtime_refresh_dual_setup_seconds=0.041`
  - `runtime_refresh_dual_artifact_build_seconds=85.000`
  - `runtime_refresh_dual_far_pair_plan_seconds=0.001`
  - `runtime_refresh_dual_m2l_autotune_seconds=0.000009`
  - `runtime_refresh_dual_select_interactions_seconds=0.000008`
  - `runtime_refresh_dual_downward_compute_seconds=20.760`
  - `runtime_refresh_dual_finalize_seconds=0.013`
  - `runtime_refresh_dual_residual_seconds=0.00016`

Interpretation:
- The bottleneck is now localized to `_build_dual_tree_artifacts(...)`, not M2L autotune, far-pair planning, or unmeasured residual.
- `runtime_refresh_prepare_reuse_tier_full=4` confirms the profile fingerprint is unchanged for all refreshes.
- The first true incremental implementation slice should reuse the previous dual-tree artifact/neighbor-list structure for the full-reuse tier and recompute only upward multipoles/downward locals against the existing interaction scaffolding.
- If downward locals still cost `~5.19 s` per refresh after skipping artifact build, the second slice should target reuse/minimization in `_prepare_downward_with_artifacts(...)`.

### 2026-04-24: Fastest-Runtime Radix Lane Recovery

Finding:
- The ODISSEO integration was still inheriting the `large_n_gpu` minimum-memory split traversal default.
- That split builder is useful for huge standalone memory-fit cases, but it is not the fastest runtime path for the current ODISSEO `N=200000` acceptance case.
- Disabling the prepare-stage split traversal switches refresh prepare back to the single-traversal radix fast-lane behavior.

Implemented:
- Added a jaccpot runtime knob:
  - `RuntimePolicyConfig.prepare_stage_memory_split_enabled`
  - `FastMultipoleMethod(..., prepare_stage_memory_split_enabled=...)`
- Threaded the knob through ODISSEO:
  - `SimulationConfig.fmm_prepare_stage_memory_split_enabled`
  - `odisseo/jaccpot_coupling.py`
  - `odisseo/integration_api.py`
  - `notebooks/scalability/galaxy_disk_fmm_large_n.py`
- ODISSEO now defaults this knob to `False`, because these production runs explicitly prioritize fastest runtime over minimum-memory staging.
- Kept `retain_interactions=False`; the change does not force full far-pair interaction retention.

Validation:
- CPU syntax check passed:
  - `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/config.py /export/home/tbuck/jaccpot/jaccpot/solver.py /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py /export/home/tbuck/jaccpot/jaccpot/runtime/_interaction_cache.py odisseo/option_classes.py odisseo/jaccpot_coupling.py odisseo/integration_api.py notebooks/scalability/galaxy_disk_fmm_large_n.py`
- Focused ODISSEO test passed:
  - `micromamba run -n odisseo python -m pytest tests/test_integration_api.py`

GPU 0 verification without environment overrides:
```bash
CUDA_VISIBLE_DEVICES=0 micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 20 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n --fmm-refresh-every 4 \
  --fmm-leaf-size 256 --fmm-tree-leaf-target 256 --fmm-max-order 4 \
  --fmm-static-shape-warmup-prepares 1 --profile-breakdown \
  --report-dir /tmp/radix_refresh_fast_runtime_gpu0_20 \
  --output /tmp/galaxy_gpu0_refresh_fast_runtime_200k_20.npz
```

Artifact:
- `/tmp/radix_refresh_fast_runtime_gpu0_20/galaxy_disk_profile_20260424_113735.json`

Key results:
- `script_runtime_seconds=215.139`
- `warmup_seconds=164.418`
- `profiled_refresh_prepare_seconds=45.989`
- `profiled_refresh_prepare_calls=4`
- `evaluate_seconds=0.368`
- `fmm_prepare_stage_memory_split_enabled=False`
- `runtime_refresh_tree_upward_seconds=2.925`
- `runtime_refresh_dual_downward_seconds=21.403`
- `runtime_refresh_dual_artifact_build_seconds=1.293`
- `runtime_refresh_dual_downward_compute_seconds=20.052`
- `runtime_refresh_nearfield_seconds=21.627`
- `runtime_compiled_profile_transitions=0`
- `runtime_large_n_overflow_profile_reprofiles=0`
- `runtime_large_n_neighbor_edges_profile_reprofiles=0`

Comparison to split minimum-memory run:
- Previous split-path refresh prepare: `~134 s` over 4 refreshes.
- Fastest-runtime single traversal refresh prepare: `~46 s` over 4 refreshes.
- Dual artifact build dropped from `~87 s` to `~1.3 s`.
- Remaining refresh cost is now roughly half downward compute and half nearfield/payload rebuild.

Next target:
- Implement the true incremental refresh tier for same compiled profile/static capacity:
  - keep the single-traversal fastest-runtime path as the ODISSEO default.
  - avoid rebuilding nearfield numeric payloads when neighbor topology/profile is compatible.
  - investigate whether `_prepare_downward_with_artifacts(...)` can reuse static buffers or compiled plans so only multipole/local values are updated.

### 2026-04-27: Same-Topology Refresh Probe + M2L Chunk Sweep

Implemented:
- Added a conservative jaccpot `refresh_prepared_state(...)` same-topology tier:
  - derives a radix topology key from the previous prepared tree when the explicit key is absent
  - attempts to reuse the previous radix topology and nearfield scaffold only when the current topology key matches
  - includes an exact neighbor-list safety gate before reusing nearfield payloads
  - falls back to full large-N prepare behavior when topology or neighbor scaffold differs
- Added runtime diagnostics:
  - `large_n_same_topology_refresh_attempts`
  - `large_n_same_topology_refresh_hits`
  - `large_n_same_topology_refresh_misses`
  - miss reasons for `no_key`, `topology`, `neighbor`, and `traced`
- Surfaced these diagnostics through ODISSEO timing reports.
- Exposed `fmm_m2l_chunk_size` through ODISSEO config and the scalability CLI for targeted far-field sweeps.

Validation:
- CPU syntax checks passed for touched files.
- Focused ODISSEO integration tests passed:
  - `micromamba run -n odisseo python -m pytest tests/test_integration_api.py`

GPU diagnostic results:
- `N=50000`, `steps=8`, `refresh=4`, `leaf=256`, `order=4`:
  - same-topology attempts: `1`
  - hits: `0`
  - miss reason: `topology=1`
- `N=200000`, `steps=20`, `refresh=4`, profile-only:
  - report: `/tmp/radix_same_topology_refresh_gpu0_20/galaxy_disk_profile_20260427_105827.json`
  - runtime: `134.183 s`
  - refresh prepare: `30.667 s` over `4` refreshes
  - static/profile gates remained stable:
    - `runtime_compiled_profile_transitions=0`
    - `shape_signature_drift_events_post_warmup=0`

M2L chunk sweep:
- 50k smoke:
  - default refresh prepare: `10.592 s`
  - `m2l_chunk=512`: `10.502 s`
  - `m2l_chunk=2048`: `10.873 s`
  - `m2l_chunk=4096`: `10.132 s`
  - `m2l_chunk=8192`: `10.231 s`
- 200k check with `m2l_chunk=4096`:
  - report: `/tmp/radix_m2l4096_gpu8_200k_20/galaxy_disk_profile_20260427_113637.json`
  - runtime: `137.272 s`
  - refresh prepare: `33.015 s`
  - not better than default `30.667 s`

Interpretation:
- The current galaxy cadence changes radix Morton topology every refresh segment, so exact-topology payload-only reuse is not the dominant path.
- Static shapes remain stable, but topology-dependent numeric payloads still need to be rebuilt under the fixed-capacity profile.
- M2L chunk tuning is not a reliable improvement for the 200k acceptance case.
- Next implementation target should be capacity-compatible topology/nearfield refresh:
  - avoid repeated Python/host materialization in large-N nearfield payload construction
  - keep padded output capacities fixed
  - update topology-dependent arrays with compiled/static-shape kernels where possible
  - preserve the automatic full-prepare fallback for capacity overflow or invariant failure

### 2026-04-27: Nearfield Capacity-Compatible Refresh Slice

Implemented:
- Added large-N nearfield substage timing diagnostics:
  - `refresh_nearfield_leaf_groups_seconds`
  - `refresh_nearfield_precompute_seconds`
  - `refresh_nearfield_target_blocks_seconds`
  - `refresh_nearfield_block_sort_seconds`
  - `refresh_nearfield_speed_layout_seconds`
  - `refresh_nearfield_overflow_profile_seconds`
  - `refresh_nearfield_radix_payload_seconds`
  - `refresh_nearfield_neighbor_padding_seconds`
  - `refresh_nearfield_state_pack_seconds`
  - `refresh_nearfield_residual_seconds`
- Surfaced these fields through ODISSEO timing reports.
- Removed redundant target-block sorting when the target-owned block builder or payload offsets already guarantee leaf-major order.
- Changed the radix fast-lane default target-owned block size from `8` to `32`.

Validation:
- CPU syntax checks passed for touched jaccpot/ODISSEO files.
- Focused ODISSEO integration tests passed:
  - `micromamba run -n odisseo python -m pytest tests/test_integration_api.py`

50k nearfield diagnostic (`N=50000`, `steps=8`, `refresh=4`, GPU 1):
- Before sort skip, block size 8:
  - report: `/tmp/radix_nearfield_substage_gpu1_50k_8/galaxy_disk_profile_20260427_115525.json`
  - refresh prepare: `10.727 s`
  - nearfield: `3.562 s`
  - target blocks: `1.887 s`
  - block sort: `1.172 s`
- After sort skip, block size 8:
  - report: `/tmp/radix_nearfield_sortskip_gpu1_50k_8/galaxy_disk_profile_20260427_115932.json`
  - refresh prepare: `9.473 s`
  - nearfield: `2.146 s`
  - block sort: effectively zero
- New default, block size 32:
  - report: `/tmp/radix_default_block32_gpu1_50k_8/galaxy_disk_profile_20260427_120911.json`
  - runtime: `69.965 s`
  - refresh prepare: `9.272 s`
  - nearfield: `1.993 s`
  - overflow profile work: effectively zero

Target-block-size sweep notes:
- block size `16` did not improve over `8` after sort skip.
- block size `32` was best among tested values.
- block size `64` increased evaluation cost and was worse than `32`.

200k profile check (`N=200000`, `steps=20`, `refresh=4`, GPU 1, block size 32):
- report: `/tmp/radix_sortskip_block32_gpu1_200k_20/galaxy_disk_profile_20260427_120421.json`
- runtime: `132.599 s`
- refresh prepare: `28.019 s` over `4` refreshes
- nearfield: `9.819 s`
- static/profile gates remained stable:
  - `runtime_compiled_profile_transitions=0`
  - `shape_signature_drift_events_post_warmup=0`

Comparison to earlier 2026-04-27 200k default before this slice:
- runtime: `134.183 s -> 132.599 s`
- refresh prepare: `30.667 s -> 28.019 s`
- nearfield: `14.880 s -> 9.819 s`
- evaluation increased (`0.352 s -> 1.204 s`) but total runtime still improved.

Next target:
- target-block construction is now the dominant nearfield substage.
- Continue by moving target-owned block construction/overflow compaction toward a static-shape compiled path, or by reusing yggdrax-produced block payloads directly when available.

### 2026-04-27: Static-Capacity Target-Block Refresh

Implemented:
- Added a fixed-capacity target-owned block builder for radix large-N nearfield:
  - emits masked tensors with shape `(num_leaves, max_blocks_per_leaf, block_size)`
  - avoids dynamic target-block flattening/sorting/overflow compaction when capacity is sufficient
  - falls back to the existing dynamic builder if observed neighbor degree exceeds capacity
- Enabled the static target-block path by default for the radix fast lane.
- Default static capacity:
  - `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=32`
  - override options: `8,16,32,64,128`

50k validation (`N=50000`, `steps=8`, `refresh=4`, GPU 1):
- previous block32 default:
  - report: `/tmp/radix_default_block32_gpu1_50k_8/galaxy_disk_profile_20260427_120911.json`
  - runtime: `69.965 s`
  - refresh prepare: `9.272 s`
  - nearfield: `1.993 s`
- static target blocks:
  - report: `/tmp/radix_static_target_blocks_gpu1_50k_8/galaxy_disk_profile_20260427_122554.json`
  - runtime: `65.890 s`
  - refresh prepare: `7.246 s`
  - nearfield: `0.163 s`
  - target-block construction: `0.058 s`
  - static/profile gates remained stable

200k validation (`N=200000`, `steps=20`, `refresh=4`, GPU 1):
- previous block32 default:
  - report: `/tmp/radix_sortskip_block32_gpu1_200k_20/galaxy_disk_profile_20260427_120421.json`
  - runtime: `132.599 s`
  - refresh prepare: `28.019 s`
  - nearfield: `9.819 s`
- static target blocks:
  - report: `/tmp/radix_static_target_blocks_gpu1_200k_20/galaxy_disk_profile_20260427_123152.json`
  - runtime: `124.129 s`
  - refresh prepare: `18.679 s`
  - nearfield: `0.674 s`
  - target-block construction: `0.232 s`
  - static/profile gates remained stable:
    - `runtime_compiled_profile_transitions=0`
    - `shape_signature_drift_events_post_warmup=0`

Interpretation:
- This is the first true capacity-compatible topology-refresh improvement:
  dynamic target-block payload construction is replaced by fixed-capacity masked tensors.
- The bottleneck moved away from nearfield prepare:
  - nearfield prepare is now sub-second at 200k over 4 refreshes.
  - dominant refresh cost is again dual/downward compute (`~13.45 s` over 4 refreshes).
- Evaluation cost increased (`1.204 s -> 4.605 s`) because the fixed-capacity payload evaluates padded slots.
- Net 200k runtime still improved by `~8.47 s` for the 20-step profile.

Next target:
- Smaller static capacity check:
  - `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16` improves padded evaluation at 50k
    but is not capacity-compatible at 200k.
  - 50k/GPU 9:
    `/tmp/radix_static_target_blocks16_gpu9_50k_8/galaxy_disk_profile_20260427_150959.json`
    - runtime: `67.533 s`
    - refresh prepare: `7.524 s`
    - evaluate: `0.276 s`
    - nearfield: `0.155 s`
    - target-block construction: `0.061 s`
  - 200k/GPU 9:
    `/tmp/radix_static_target_blocks16_gpu9_200k_20/galaxy_disk_profile_20260427_151306.json`
    - runtime: `135.725 s`
    - refresh prepare: `28.334 s`
    - evaluate: `1.211 s`
    - nearfield: `9.580 s`
    - target-block construction: `7.492 s`
    - `runtime_large_n_overflow_profile_cap=32768`
  - Interpretation: capacity 16 overflows at 200k and falls back to the dynamic
    target-block construction path, effectively matching the earlier sort-skip
    nearfield cost. Keep default capacity at `32` for the 200k target.
- Focus next on the dual/downward topology-refresh path, now the dominant steady
  refresh cost under the valid static target-block capacity.

### 2026-04-27: Solid-FMM Dual/Downward Substage Split

Implemented:
- Added refresh diagnostics for the complex solid-FMM downward compute bucket:
  - `runtime_refresh_dual_m2l_compute_seconds`
  - `runtime_refresh_dual_l2l_compute_seconds`
  - `runtime_refresh_dual_final_symmetry_seconds`
  - `runtime_refresh_dual_source_motion_seconds`
- These diagnostics intentionally synchronize sub-stages during profiling, so
  use them to locate cost rather than as clean runtime baselines.

50k diagnostic (`N=50000`, `steps=8`, `refresh=4`, GPU 9):
- report:
  `/tmp/radix_solidfmm_dual_substage_gpu9_50k_8/galaxy_disk_profile_20260427_153220.json`
- runtime: `72.036 s`
- refresh prepare: `7.569 s`
- dual/downward: `4.591 s`
- downward compute: `4.410 s`
- M2L compute: `4.406 s`
- L2L compute: `0.001 s`
- final symmetry: `0.000 s`

200k diagnostic (`N=200000`, `steps=20`, `refresh=4`, GPU 9):
- report:
  `/tmp/radix_solidfmm_dual_substage_gpu9_200k_20/galaxy_disk_profile_20260427_153647.json`
- runtime: `127.898 s`
- refresh prepare: `19.533 s`
- dual/downward: `14.041 s`
- downward compute: `12.948 s`
- M2L compute: `12.921 s`
- L2L compute: `0.011 s`
- final symmetry: `0.001 s`
- static/profile gates remained stable:
  - `runtime_compiled_profile_transitions=0`
  - `shape_signature_drift_events_post_warmup=0`

Interpretation:
- The remaining solid-FMM refresh cost is overwhelmingly M2L translation.
- L2L propagation and final conjugate-symmetry cleanup are not material
  bottlenecks for the current 200k target.
- Keep the complex solid-FMM spherical-harmonic basis for accuracy; do not
  pursue the real-basis detour for this coupling path.

Next target:
- Optimize or reschedule the complex solid-FMM M2L accumulation itself while
  preserving the spherical basis and the existing static-capacity nearfield path.

### 2026-04-27: Reconcile Isolated jaccpot Fast Timing vs ODISSEO Loop

Corrected diagnosis:
- The isolated jaccpot fast-lane harness still reproduces fast warmed calls on
  GPU 9 with the same galaxy segment states:
  `/tmp/radix_fastlane_repro_gpu9/radix_fastlane_repro_gpu9_leaf256_order4_20260427_154917.json`
  - direct jaccpot warmed prepare total over 6 states: `4.743 s`
  - ODISSEO coupler-built warmed prepare total over 6 states: `3.925 s`
  - per-state ODISSEO coupler prepare rows: `0.622-0.687 s`
  - per-state ODISSEO coupler evaluate rows: `0.942-0.960 s`
- Therefore ODISSEO does not construct an inherently slow jaccpot solver.
- The full integration benchmark is slower because it first-sees a new tree
  topology at each refresh. Shape/profile stability is not enough: the topology
  and interaction payload values are different every refresh.

Full ODISSEO 200k profile with per-prepare events:
- report:
  `/tmp/radix_odisseo_nonintrusive_pairdiag_gpu9_200k_20/galaxy_disk_profile_20260427_160108.json`
- runtime: `139.136 s`
- warmup: `110.971 s`
- profiled prepare events:
  - full initial prepare: `0.539 s`
  - refresh 1: `6.978 s`
  - refresh 2: `4.508 s`
  - refresh 3: `4.457 s`
  - refresh 4: `4.507 s`
- latest interaction scale is normal, not inflated:
  - nodes: `1563`
  - leaves: `782`
  - neighbor entries: `304626`
  - far pairs: `42754`
  - M2L chunk size: `1024`

Refresh bypass A/B:
- Added opt-in ODISSEO environment switch:
  `ODISSEO_DISABLE_FMM_REFRESH_PREPARED_STATE=1`.
- report:
  `/tmp/radix_odisseo_plain_prepare_gpu9_200k_20/galaxy_disk_profile_20260427_160654.json`
- runtime: `135.546 s`
- post-warmup full-prepare events:
  - `0.556 s`, then `4.168 s`, `4.068 s`, `4.009 s`, `4.286 s`
- Interpretation: bypassing the refresh wrapper helps slightly but does not
  remove the first-seen topology cost. Plain prepare is also slow for new
  topologies in the full integration loop, while warmed repeats of those same
  topologies are fast in the isolated harness.

Instrumentation correction:
- The earlier `runtime_refresh_dual_m2l_compute_seconds` deep split used
  `block_until_ready` inside the M2L substage and can absorb queued upstream GPU
  work. It is now opt-in only via
  `JACCPOT_REFRESH_TIMING_SYNC_SUBSTAGES=1`.
- Default profiling should not interpret the deep M2L substage as a clean
  steady-state kernel time.

Fixed-depth probe:
- Exposed `fmm_tree_build_mode` through ODISSEO and the galaxy benchmark.
- `fixed_depth` is not currently viable as a drop-in static topology primitive:
  - 50k, leaf/target 256 OOMs during solid-FMM upward P2M
    (`13.83 GiB` allocation request).
  - 50k, leaf/target 1024 also OOMs
    (`16.90 GiB` allocation request).

Next target:
- Do not chase real-basis kernels or built-in fixed-depth as the primary route.
- Build a sparse static-topology radix/grid refresh path for the complex
  solid-FMM basis:
  - fixed spatial cells/topology under fixed bounds
  - fixed-capacity leaf particle payloads
  - reusable neighbor/far interaction graph
  - update only sorted particle payloads, multipoles, and M2L numerical locals
  - controlled overflow/reprofile when a cell capacity is exceeded
