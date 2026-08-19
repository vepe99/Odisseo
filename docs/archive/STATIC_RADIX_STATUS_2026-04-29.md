# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Static Radix Status - 2026-04-29

## Goal

Replace the spatial `capacity_fixed_depth` experiment with `static_radix`.

Implementation plan:

```text
docs/STATIC_RADIX_IMPLEMENTATION_PLAN_2026-04-29.md
```

`static_radix` means:

- recompute Morton codes from current particle positions,
- sort by Morton code,
- split sorted particles into fixed count buckets of size `leaf_size`,
- keep a deterministic balanced tree over those leaf buckets,
- preserve static array/data-structure shapes for fixed `N` and `leaf_size`.

This is intentionally not a fixed spatial-cell template.

## Branches

Created matched branches from the previous capacity-fixed work:

- `/export/home/tbuck/yggdrax`: `feat/static-radix`
- `/export/home/tbuck/jaccpot`: `feat/static-radix`
- `/export/home/tbuck/Odisseo`: `feat/static-radix`

All repos had dirty work before this branch was created. Do not clean unrelated
files blindly.

## Implementation Snapshot

### yggdrax

Implemented:

- `build_static_radix_tree(...)`
- `rebuild_static_radix_tree_from_template(...)`
- `Tree.from_particles(..., build_mode="static_radix")`

Removed public/source references to:

- `capacity_fixed_depth`
- `build_capacity_fixed_depth_tree`
- `rebuild_capacity_fixed_depth_tree_from_template`

Focused test result:

```text
micromamba run -n odisseo python -m pytest /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k 'static_radix' -o addopts=
3 passed, 21 deselected in 11.35s
```

### jaccpot

Implemented:

- accepts `tree_build_mode="static_radix"`,
- rejects `tree_build_mode="capacity_fixed_depth"`,
- routes prepare through yggdrax `static_radix`,
- exposes static radix refresh diagnostics:
  - `static_radix_refresh_hits`
  - `static_radix_refresh_misses`
  - `static_radix_profile_overflows`

Current static refresh implementation rebuilds the large-N prepared state and
counts it as a static refresh only when the refreshed prepared-state profile is
same-shape or capacity-compatible with the previous profile. This is correct
enough to remove the spatial-cell overflow failure, but still needs performance
work to avoid rebuilding more than necessary.

Focused test result:

```text
CUDA_VISIBLE_DEVICES=8 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
micromamba run -n odisseo python -m pytest /export/home/tbuck/jaccpot/tests/integration/test_fmm.py \
  -k 'static_radix or capacity_fixed_depth_tree_mode_is_removed' -o addopts=
2 passed, 57 deselected in 22.47s
```

### ODISSEO

Implemented:

- CLI accepts `--fmm-tree-build-mode static_radix`,
- CLI no longer accepts `capacity_fixed_depth`,
- for `static_radix`, benchmark config uses `--fmm-leaf-size` as the static
  bucket size and ignores `--fmm-tree-leaf-target`,
- reports include runtime static-radix refresh counters.

## Next Actions

1. Run compile checks for jaccpot and ODISSEO after final edits.
2. Run ODISSEO API regression:

```bash
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
micromamba run -n odisseo python -m pytest /export/home/tbuck/Odisseo/tests/test_integration_api.py -o addopts=
```

3. Run a small ODISSEO static-radix smoke:

```bash
CUDA_VISIBLE_DEVICES=8 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 20000 \
  --num-steps 2 \
  --t-end-gyr 0.2 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-max-order 4 \
  --fmm-prepare-stage-memory-split \
  --profile-breakdown \
  --report-dir /tmp/static_radix_gpu8_20k_2 \
  --output /tmp/static_radix_gpu8_20k_2.npz
```

Acceptance fields to inspect first:

- `runtime_static_radix_refresh_hits`
- `runtime_static_radix_refresh_misses`
- `runtime_compiled_profile_transitions`
- `shape_signature_stable_post_warmup`
- refresh timing buckets

## Smoke Update

The 20k/2-step ODISSEO static-radix smoke passed on GPU 8:

```text
Saved /tmp/static_radix_gpu8_20k_2.npz
Runtime: 42.539 s
Saved timing report JSON: /tmp/static_radix_gpu8_20k_2/galaxy_disk_profile_20260429_132614.json
```

Key report fields:

```text
profiled_prepare_events:
  full:    33.536 s
  refresh: 0.531 s
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_large_n_same_topology_refresh_hits: 1
runtime_large_n_same_topology_refresh_misses: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
runtime_refresh_tree_upward_seconds: 0.498 s
runtime_refresh_dual_downward_seconds: 0.005 s
runtime_refresh_nearfield_seconds: 0.024 s
runtime_refresh_prepare_reuse_tier_full: 1
```

The 200k/2-step acceptance smoke also passed on GPU 8:

```text
Saved /tmp/static_radix_gpu8_200k_2.npz
Runtime: 93.892 s
Saved timing report JSON: /tmp/static_radix_gpu8_200k_2/galaxy_disk_profile_20260429_132940.json
```

Key report fields:

```text
profiled_prepare_events:
  full:    82.616 s
  refresh: 0.613 s
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_large_n_same_topology_refresh_hits: 1
runtime_large_n_same_topology_refresh_misses: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
runtime_refresh_tree_upward_seconds: 0.522 s
runtime_refresh_dual_downward_seconds: 0.011 s
runtime_refresh_nearfield_seconds: 0.075 s
runtime_refresh_prepare_reuse_tier_full: 1
```

The 200k/20-step acceptance run passed on GPU 8:

```text
Saved /tmp/static_radix_gpu8_200k_20.npz
Runtime: 126.451 s
Saved timing report JSON: /tmp/static_radix_gpu8_200k_20/galaxy_disk_profile_20260429_133354.json
```

Key report fields:

```text
prepare_seconds: 98.600
evaluate_seconds: 25.228
update_seconds: 1.726
profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_refresh_fallback_prepare_calls: 0
profiled_full_prepare_seconds: 86.702
profiled_refresh_prepare_seconds: 11.897
runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_large_n_same_topology_refresh_hits: 19
runtime_large_n_same_topology_refresh_misses: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
runtime_refresh_tree_upward_seconds: 10.464
runtime_refresh_dual_downward_seconds: 0.157
runtime_refresh_nearfield_seconds: 1.177
runtime_refresh_prepare_reuse_tier_full: 19
```

This satisfies the first static-radix acceptance signal:

- no OOM,
- no capacity/spatial template overflow,
- no fallback full prepares,
- stable prepared-state shapes,
- no compiled-profile transitions,
- 19/19 refresh hits.

Remaining performance work:

- first full prepare is still about `86.7 s`,
- total evaluate time is about `25.2 s` for 20 calls,
- static refresh itself is now about `0.6 s` per call.

## Verification Update

After restoring the existing fixed-depth helper import in yggdrax and making
the jaccpot overflow fallback test explicitly disable static target blocks, the
broader focused suites are clean:

```text
JAX_ENABLE_X64=1 PYTHONPATH=/export/home/tbuck/yggdrax \
  micromamba run -n odisseo python -m pytest \
  /export/home/tbuck/yggdrax/tests/unit/test_tree.py -o addopts=

24 passed in 114.47s
```

```text
CUDA_VISIBLE_DEVICES=8 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
  micromamba run -n odisseo python -m pytest \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -o addopts=

59 passed in 531.48s
```

Earlier ODISSEO API regression remains:

```text
/export/home/tbuck/Odisseo/tests/test_integration_api.py
4 passed, 1 warning in 5.30s
```

Current blocker:

- No functional static-radix blocker from the focused suites or 200k/20
  acceptance run.
- The tree repos still need a final diff review and cleanup of unrelated local
  dirt before commit/PR staging.

Exact next action:

```text
Review diffs in yggdrax, jaccpot, and ODISSEO; separate static-radix edits from
pre-existing dirty files; then run the same three focused verification commands
once after final cleanup.
```

## PR Staging Update

Diff hygiene, final verification, commits, branch pushes, and PR creation are
complete.

Static-radix commits:

```text
yggdrax:  a7180a3 Add static radix tree build mode
jaccpot:  577b092 Integrate static radix FMM refresh path
ODISSEO:  8bf35f6 Expose static radix FMM integration
ODISSEO:  eecdf22 Add radix fast-lane investigation harness
```

Branches pushed:

```text
TobiBu/yggdrax:   feat/capacity-fixed-radix, feat/static-radix
TobiBu/jaccpot:   feat/capacity-fixed-radix, feat/static-radix
vepe99/Odisseo:   feat/capacity-fixed-radix, feat/static-radix
```

Open PRs, all targeting the base branch of this work:

```text
yggdrax:  https://github.com/TobiBu/yggdrax/pull/22
          base feat/capacity-fixed-radix <- head feat/static-radix

jaccpot:  https://github.com/TobiBu/jaccpot/pull/13
          base feat/capacity-fixed-radix <- head feat/static-radix

ODISSEO:  https://github.com/vepe99/Odisseo/pull/2
          base feat/capacity-fixed-radix <- head feat/static-radix
```

GitHub CLI note:

```text
gh was installed locally under /tmp/gh-cli for this server session.
Version: gh 2.92.0
Authenticated user: TobiBu
```

The older capacity/performance investigation docs remain intentionally
untracked in ODISSEO and are not part of the static-radix PR:

```text
docs/CAPACITY_FIXED_RADIX_IMPLEMENTATION_PLAN_2026-04-27.md
docs/CAPACITY_FIXED_RADIX_STATUS_2026-04-27.md
docs/PERFORMANCE_HANDOFF_2026-04-22.md
docs/PERFORMANCE_RADIX_INVESTIGATION_PLAN_2026-04-22.md
docs/Radix Large-N Recompile-Minimization .md
```

The radix fast-lane investigation harness was intentionally added to the
ODISSEO PR for future benchmarking:

```text
notebooks/scalability/radix_fastlane_investigation.py
```

## Final Pre-PR Verification

Final focused suites were run after cleanup and before PR creation:

```text
PYTHONPATH=/export/home/tbuck/yggdrax \
  micromamba run -n odisseo python -B -m pytest \
  /export/home/tbuck/yggdrax/tests/unit/test_tree.py -o addopts=

24 passed in 130.77s
```

```text
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
  micromamba run -n odisseo python -B -m pytest \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -o addopts=

59 passed in 576.69s
```

```text
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
  micromamba run -n odisseo python -B -m pytest \
  /export/home/tbuck/Odisseo/tests/test_integration_api.py -o addopts=

4 passed, 1 warning in 3.06s
```

Fresh small ODISSEO static-radix GPU smoke on GPU 0:

```text
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
micromamba run -n odisseo python -B \
  /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 20000 \
  --num-steps 2 \
  --t-end-gyr 0.2 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-max-order 4 \
  --fmm-prepare-stage-memory-split \
  --profile-breakdown \
  --require-static-shape \
  --max-compiled-profile-transitions 0 \
  --max-overflow-reprofiles 0 \
  --min-refresh-prepare-successes 1 \
  --report-dir /tmp/static_radix_gpu0_20k_2 \
  --output /tmp/static_radix_gpu0_20k_2.npz
```

Result:

```text
Saved /tmp/static_radix_gpu0_20k_2.npz
Runtime: 44.242 s
Saved timing report JSON: /tmp/static_radix_gpu0_20k_2/galaxy_disk_profile_20260429_203811.json
Saved timing report CSV : /tmp/static_radix_gpu0_20k_2/galaxy_disk_profile_20260429_203811.csv

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 1
profiled_refresh_fallback_prepare_calls: 0
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_large_n_same_topology_refresh_hits: 1
runtime_large_n_same_topology_refresh_misses: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: True
refresh_prepare_successes: 1
runtime_refresh_tree_upward_seconds: 0.5593071468174458
runtime_refresh_dual_downward_seconds: 0.005494195967912674
runtime_refresh_nearfield_seconds: 0.026852678507566452
```

## Current Blocker

No known functional static-radix blocker remains from the local focused tests,
the 200k/20 acceptance run, or the fresh GPU 0 smoke.

The only remaining local ODISSEO working-tree dirt is the intentionally
excluded older handoff/investigation docs listed above.

## Exact Next Actions

1. Review PRs in dependency order:

```text
1. yggdrax  #22
2. jaccpot  #13, depends on yggdrax #22
3. ODISSEO  #2, depends on yggdrax #22 and jaccpot #13
```

2. Address any CI/review comments without mixing in the intentionally excluded
older capacity/performance docs.

3. After merge, update dependency pins/submodule/source checkout assumptions
for any downstream environment that expects the new yggdrax and jaccpot APIs.

4. Continue performance work after functional merge:

```text
- first full prepare remains the dominant cost,
- 200k/20 evaluate time remains about 25 s,
- static refresh is about 0.6 s per refresh in the acceptance run,
- radix fast-lane investigation harness is now available for follow-up profiling.
```

## Post-Merge Performance Follow-Up - 2026-04-30

All static-radix PRs were merged. Local `jaccpot` was fast-forwarded to
`origin/feat/static-radix` commit `4c9a893`; ODISSEO remained synced on
`feat/static-radix` after the review-fix push.

Updated the radix fast-lane harness:

- added `--fmm-tree-build-mode`, defaulting to `static_radix`,
- passed matching tree/nearfield/runtime advanced config into the direct
  jaccpot solver path,
- moved cold-start timing before FMM-based benchmark-state generation,
- added `--cold-start-order` to prove which solver pays the first compile cost.

Verification:

```text
micromamba run -n odisseo python -m py_compile \
  notebooks/scalability/radix_fastlane_investigation.py
```

20k static-radix smoke:

```text
Saved JSON report: /tmp/radix_fastlane_static_smoke/static_radix_fastlane_20k_20260430_101352.json
Saved CSV report : /tmp/radix_fastlane_static_smoke/static_radix_fastlane_20k_20260430_101352.csv

direct_jaccpot: prepare=0.968s evaluate=0.214s total=1.181s over 2 states
odisseo_coupler_builder: prepare=1.040s evaluate=0.214s total=1.254s over 2 states
```

200k direct-first cold-start run:

```text
Saved JSON report: /tmp/radix_fastlane_static_200k_coldfirst/static_radix_fastlane_200k_coldfirst_20260430_102038.json
Saved CSV report : /tmp/radix_fastlane_static_200k_coldfirst/static_radix_fastlane_200k_coldfirst_20260430_102038.csv

cold_start_single_call:
  direct_jaccpot: prepare=74.674s evaluate=6.942s total=81.616s
  odisseo_coupler_builder: prepare=0.762s evaluate=0.906s total=1.668s

steady rows:
  direct_jaccpot: prepare=1.102s evaluate=1.815s total=2.918s over 2 states
  odisseo_coupler_builder: prepare=1.101s evaluate=1.817s total=2.918s over 2 states
```

200k coupler-first cold-start run:

```text
Saved JSON report: /tmp/radix_fastlane_static_200k_couplerfirst/static_radix_fastlane_200k_couplerfirst_20260430_102503.json
Saved CSV report : /tmp/radix_fastlane_static_200k_couplerfirst/static_radix_fastlane_200k_couplerfirst_20260430_102503.csv

cold_start_single_call:
  odisseo_coupler_builder: prepare=73.981s evaluate=6.964s total=80.945s
  direct_jaccpot: prepare=0.767s evaluate=0.917s total=1.684s

steady rows:
  direct_jaccpot: prepare=1.150s evaluate=1.819s total=2.969s over 2 states
  odisseo_coupler_builder: prepare=1.134s evaluate=1.820s total=2.953s over 2 states
```

Conclusion:

- the old `~80s` first full prepare is the cold large-N compile/startup bill,
  not an ODISSEO wrapper-specific cost,
- direct jaccpot and the ODISSEO coupler builder have indistinguishable
  steady-state static-radix prepare/evaluate timing at 200k,
- steady-state per-state cost at 200k is roughly `0.56s` prepare plus `0.91s`
  evaluate for `leaf_size=256`, `max_order=4`.

Exact next action:

```text
Profile the cold large-N compile/startup path inside jaccpot and decide whether
to hide it with an explicit warmup, split/report it separately, or reduce the
compiled surface area for first prepare/evaluate.
```

## Cold Prepare Stage Breakdown - 2026-04-30

Added temporary/profiling diagnostics for cold prepare stage deltas:

- ODISSEO harness now enables jaccpot runtime stage counters around cold
  `prepare_state`,
- jaccpot now exposes split dual-tree artifact substage counters:
  - split far-pair traversal,
  - split leaf-neighbor traversal,
  - raw combined traversal,
  - split dense-buffer materialization.

Verification:

```text
micromamba run -n odisseo python -m py_compile \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_interaction_cache.py \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py

micromamba run -n odisseo python -m py_compile \
  notebooks/scalability/radix_fastlane_investigation.py
```

20k split-substage smoke:

```text
Saved JSON report: /tmp/radix_fastlane_split_substage_20k/static_radix_split_substage_20k_20260430_113058.json
Saved CSV report : /tmp/radix_fastlane_split_substage_20k/static_radix_split_substage_20k_20260430_113058.csv

cold prepare:
  total:          31.407s
  tree/upward:    13.902s
  dual/downward:  15.800s
  nearfield:       1.640s
  unaccounted:     0.065s

dual artifact build:
  total:          15.739s
  split far:       7.524s
  split neighbor:  8.215s
```

200k split-substage run:

```text
Saved JSON report: /tmp/radix_fastlane_split_substage_200k/static_radix_split_substage_200k_20260430_113434.json
Saved CSV report : /tmp/radix_fastlane_split_substage_200k/static_radix_split_substage_200k_20260430_113434.csv

cold prepare:
  total:          75.071s
  tree/upward:    14.708s
  dual/downward:  59.104s
  nearfield:       1.192s
  unaccounted:     0.066s

dual artifact build:
  total:          48.804s
  split far:      25.374s
  split neighbor: 23.430s

downward compute:
  10.297s

cold evaluate:
  7.036s

steady rows:
  direct_jaccpot: prepare=1.179s evaluate=1.820s total=2.999s over 2 states
  odisseo_coupler_builder: prepare=1.174s evaluate=1.821s total=2.995s over 2 states
```

Split toggle result:

```text
20k with JACCPOT_PREPARE_STAGE_MEMORY_SPLIT_ENABLED=0:
  cold prepare: 26.319s
  dual artifact build: 10.269s

200k with JACCPOT_PREPARE_STAGE_MEMORY_SPLIT_ENABLED=0:
  cold prepare: 87.961s
  dual artifact build: 61.308s
```

Conclusion:

- the cold `~80s` wall time is not hidden Python overhead; it is almost fully
  accounted for inside jaccpot prepare stages,
- at 200k the dominant compile/startup cost is dual-tree artifact build,
  split almost evenly between far-pair traversal and leaf-neighbor traversal,
- disabling the split traversal helps at 20k but hurts at 200k, so the default
  split path remains the better large-N choice.

Exact next action:

```text
Investigate whether the far-pair and leaf-neighbor traversal kernels can share
compiled structure or be warmed explicitly before timed full prepare. The first
candidate is a deliberate warmup/reporting mode rather than changing the split
default, because steady-state timing is already good and no-split regresses at
200k.
```

## Warm Sweep Performance Target - 2026-04-30

Important target clarification:

- standalone jaccpot previously demonstrated subsecond warm full-FMM behavior at
  `N=200000` in its optimal large-N/radix configuration,
- ODISSEO should therefore not stop at "cold compile explained" or "refreshes
  are shape-stable",
- the warm ODISSEO+jaccpot target is:

```text
N = 200000
tree_build_mode = static_radix
leaf_size = 256
max_order = 4
runtime_path = large_n
preset = large_n_gpu

warm full FMM sweep = warm prepare_state + warm evaluate_prepared_state < 1.0 s
```

Current static-radix fast-lane harness numbers are still above that target:

```text
200k split-substage run, warm rows:
  direct_jaccpot:           prepare=1.179s evaluate=1.820s total=2.999s over 2 states
  odisseo_coupler_builder:  prepare=1.174s evaluate=1.821s total=2.995s over 2 states

approx per-state warm sweep:
  prepare ~= 0.59s
  evaluate ~= 0.91s
  total   ~= 1.50s
```

So the remaining performance goal is twofold:

1. Treat cold compile/startup separately and report or warm it deliberately.
2. Reduce the steady warm full sweep from roughly `1.5s` toward the standalone
   jaccpot optimum below `1s`.

Exact next action update:

```text
Reproduce the historical standalone jaccpot subsecond 200k configuration
side-by-side with the current ODISSEO fast-lane harness, then diff runtime
knobs and prepared-state profiles until the warm prepare+evaluate gap is
explained.
```

## Static Harness Guardrail And A/B Update - 2026-04-30

Guardrail from the 2026-04-30 working session:

- do not weaken `static_radix` into an adaptive/spatial rebuild path just to
  improve timings,
- preserve the fixed-shape harness because its core value is reusable prepared
  datastructures across changing particle distributions,
- performance work should tune large-N execution knobs and diagnostics around
  the fixed topology contract, not replace that contract.

Historical timing clarification:

- `/tmp/radix_fastlane_repro_gpu9/radix_fastlane_repro_gpu9_leaf256_order4_20260427_154917.json`
  recorded the reproduced standalone/coupler fast-lane sweep at `leaf_size=256`,
  `max_order=4`, `large_n_gpu`, `float32`.
- That reproduction showed warm ODISSEO coupler rows around
  `prepare=0.62-0.69s`, `evaluate=0.94-0.96s`, total `~1.58-1.63s` per state.
- Older integration profiles such as
  `/tmp/radix_m2l4096_gpu8_200k_20/galaxy_disk_profile_20260427_113637.json`
  reported `evaluate_seconds=0.355s` over `5` evaluate calls, but those profiles
  had expensive refresh prepares and did not represent the same isolated
  warm-sweep timing surface as the fast-lane harness.

New A/B runs:

```text
static_radix + jit_tree/jit_traversal:
  report: /tmp/radix_static_jit_true_200k/static_radix_jit_true_200k_20260430_120228.json
  CSV   : /tmp/radix_static_jit_true_200k/static_radix_jit_true_200k_20260430_120228.csv
  warm odisseo rows:
    state 0: prepare=0.621s evaluate=0.914s total=1.534s
    state 1: prepare=0.632s evaluate=0.910s total=1.543s

lbvh + jit_tree/jit_traversal:
  report: /tmp/radix_lbvh_jit_true_200k/lbvh_jit_true_200k_20260430_120441.json
  CSV   : /tmp/radix_lbvh_jit_true_200k/lbvh_jit_true_200k_20260430_120441.csv
  warm odisseo rows:
    state 0: prepare=0.567s evaluate=0.913s total=1.480s
    state 1: prepare=0.761s evaluate=0.892s total=1.653s
```

Conclusion:

- enabling the old `jit_tree/jit_traversal` flags does not recover subsecond
  full-sweep timing,
- `lbvh` and `static_radix` have essentially the same warm evaluate cost in the
  current harness, so the `~0.9s/state` evaluate gap is not caused by static
  radix topology reuse,
- `static_radix` is still the correct target harness: the longer integration run
  `/tmp/static_radix_gpu8_200k_20/galaxy_disk_profile_20260429_133354.json`
  had `19` static refresh hits, `0` misses, `0` overflows, and refresh prepares
  around `0.6s`.

Harness update:

- `notebooks/scalability/radix_fastlane_investigation.py` now exposes safe
  large-N performance knobs without changing static-radix semantics:
  - `--fmm-nearfield-edge-chunk-size`
  - `--large-n-target-block-size`
  - `--large-n-static-target-blocks`
  - `--large-n-static-target-blocks-max-per-leaf`
- the JSON report now records the effective `JACCPOT_LARGE_N_*` environment
  values so future timings are reproducible.

Exact next action:

```text
Run static_radix-only target-block/edge-chunk A/B experiments through the
harness knobs, keeping static refresh hits/misses/overflows as acceptance
guards. The next candidate set is target block size and static target-block
capacity, because evaluate remains the dominant warm-sweep gap.
```

## Static Target Blocks Restores Subsecond Warm Sweep - 2026-04-30

Command shape:

```text
micromamba run -n odisseo python notebooks/scalability/radix_fastlane_investigation.py \
  --n-particles 200000 \
  --num-steps 2 \
  --fmm-refresh-every 1 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-working-dtype float32 \
  --fmm-tree-build-mode static_radix \
  --leaf-size 256 \
  --max-order 4 \
  --segments-to-benchmark 1 \
  --cold-start-order coupler_first \
  --skip-steady-state-warmup \
  --assert-fast-lane \
  --large-n-static-target-blocks \
  --large-n-static-target-blocks-max-per-leaf 16
```

Report:

```text
JSON: /tmp/radix_static_targetblocks16_200k/static_targetblocks16_200k_20260430_124827.json
CSV : /tmp/radix_static_targetblocks16_200k/static_targetblocks16_200k_20260430_124827.csv
```

Effective large-N env recorded in the JSON:

```text
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=<unset/default>
```

Warm rows:

```text
direct_jaccpot:
  state 0: prepare=0.661s evaluate=0.234s total=0.895s
  state 1: prepare=0.604s evaluate=0.227s total=0.831s

odisseo_coupler_builder:
  state 0: prepare=0.628s evaluate=0.227s total=0.856s
  state 1: prepare=0.661s evaluate=0.227s total=0.888s
```

Conclusion:

- `static_radix` plus static target blocks meets the warm full-FMM sweep target
  at 200k: ODISSEO coupler warm sweeps are `0.856s` and `0.888s`,
- this keeps the fixed/static datastructure harness intact; the speedup comes
  from the nearfield target-block layout used by the large-N evaluate path,
- cold startup still remains expensive (`~83s` prepare for coupler-first cold
  in this run), and static target blocks increase cold nearfield state
  construction cost, so cold compile/startup should stay reported separately.

Exact next action:

```text
Promote the static target-block setting to the recommended 200k static_radix
benchmark configuration, then run a longer integration-profile validation
with the same knobs to confirm refresh hits/misses/overflows across changing
particle distributions.
```

## Production Config Wiring - 2026-04-30

Longer moving-particle validation with static target blocks:

```text
env JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1 \
    JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16 \
  micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
    --mode perf \
    --n-particles 200000 \
    --num-steps 20 \
    --fmm-refresh-every 1 \
    --fmm-preset large_n_gpu \
    --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix \
    --fmm-leaf-size 256 \
    --fmm-tree-leaf-target 256 \
    --fmm-max-order 4 \
    --fmm-nearfield-edge-chunk-size 256 \
    --profile-breakdown \
    --require-static-shape \
    --max-compiled-profile-transitions 0 \
    --max-overflow-reprofiles 0 \
    --max-neighbor-edge-reprofiles 0 \
    --min-refresh-prepare-successes 19
```

Report:

```text
JSON: /tmp/static_radix_targetblocks16_integration_200k_20/galaxy_disk_profile_20260430_142255.json
CSV : /tmp/static_radix_targetblocks16_integration_200k_20/galaxy_disk_profile_20260430_142255.csv
```

Validation result:

```text
total runtime including cold startup: 108.132s
prepare total: 94.993s over 20 calls
evaluate total: 10.699s over 20 calls
profiled cold full prepare: 82.181s
profiled refresh prepare: 12.812s over 19 calls

warm refresh prepare average: 0.674s
warm evaluate average estimate: ~0.231s
warm full FMM sweep estimate: ~0.905s

static_radix refresh hits: 19
static_radix refresh misses: 0
static_radix profile overflows: 0
compiled profile transitions: 0
overflow reprofiles: 0
neighbor-edge reprofiles: 0
post-warmup shape signatures: stable, 1 unique signature
```

Production wiring added after that validation:

- `SimulationConfig` now carries:
  - `fmm_large_n_target_block_size`
  - `fmm_large_n_static_target_blocks`
  - `fmm_large_n_static_target_blocks_max_per_leaf`
- `odisseo.jaccpot_coupling` applies these settings while calling jaccpot
  full prepare and refresh prepare, so production integrations no longer have
  to depend on ambient shell environment variables.
- `notebooks/scalability/galaxy_disk_fmm_large_n.py` exposes the settings via:
  - `--fmm-large-n-target-block-size`
  - `--fmm-large-n-static-target-blocks`
  - `--no-fmm-large-n-static-target-blocks`
  - `--fmm-large-n-static-target-blocks-max-per-leaf`

Config-driven smoke without shell env overrides:

```text
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 20000 \
  --num-steps 2 \
  --fmm-refresh-every 1 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --fmm-nearfield-edge-chunk-size 256 \
  --fmm-large-n-static-target-blocks \
  --fmm-large-n-static-target-blocks-max-per-leaf 16 \
  --profile-breakdown \
  --require-static-shape \
  --max-compiled-profile-transitions 0 \
  --max-overflow-reprofiles 0 \
  --max-neighbor-edge-reprofiles 0 \
  --min-refresh-prepare-successes 1
```

Report:

```text
JSON: /tmp/static_radix_config_targetblocks16_smoke_20k_2/galaxy_disk_profile_20260430_145132.json
```

Smoke result:

```text
static_radix refresh hits: 1
static_radix refresh misses: 0
static_radix profile overflows: 0
compiled profile transitions: 0
overflow reprofiles: 0
neighbor-edge reprofiles: 0
reported fmm_large_n_static_target_blocks_requested: true
reported fmm_large_n_static_target_blocks_max_per_leaf_requested: 16
```

Exact next action:

```text
Re-run the 200k/20 integration validation through the new config flags rather
than env overrides, then decide whether static target blocks with max-per-leaf
16 should become the default for static_radix + large_n_gpu at N>=200k.
```

## Config-Driven 200k Validation - 2026-04-30

The production-path rerun uses the new ODISSEO config/CLI flags directly, with
no `JACCPOT_LARGE_N_*` shell environment prefix:

```text
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --fmm-refresh-every 1 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --fmm-nearfield-edge-chunk-size 256 \
  --fmm-large-n-static-target-blocks \
  --fmm-large-n-static-target-blocks-max-per-leaf 16 \
  --profile-breakdown \
  --require-static-shape \
  --max-compiled-profile-transitions 0 \
  --max-overflow-reprofiles 0 \
  --max-neighbor-edge-reprofiles 0 \
  --min-refresh-prepare-successes 19
```

Report:

```text
JSON: /tmp/static_radix_config_targetblocks16_integration_200k_20/galaxy_disk_profile_20260430_145648.json
CSV : /tmp/static_radix_config_targetblocks16_integration_200k_20/galaxy_disk_profile_20260430_145648.csv
```

Result:

```text
total runtime including cold startup: 108.715s
prepare total: 95.787s over 20 calls
evaluate total: 10.613s over 20 calls
profiled cold full prepare: 83.194s
profiled refresh prepare: 12.593s over 19 calls

warm refresh prepare average: 0.663s
warm evaluate average estimate: ~0.232s
warm full FMM sweep estimate: ~0.895s

static_radix refresh hits: 19
static_radix refresh misses: 0
static_radix profile overflows: 0
compiled profile transitions: 0
overflow reprofiles: 0
neighbor-edge reprofiles: 0
post-warmup shape signatures: stable, 1 unique signature
```

The report confirms:

```text
fmm_large_n_static_target_blocks_requested = true
fmm_large_n_static_target_blocks_max_per_leaf_requested = 16
```

Conclusion:

- the production config path now reproduces the subsecond warm full-sweep
  behavior without ambient env requirements,
- `static_radix` remains stable across changing particle distributions,
- static target blocks with max-per-leaf `16` is the current recommended
  production setting for `static_radix + large_n_gpu` at `N=200000`.

Follow-up implementation:

- the target-block setting is now automatic for
  `static_radix + large_n_gpu` when `N_particles >= fmm_large_n_min_particles`,
- the automatic cap is `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16`,
- explicit config still wins:
  - `fmm_large_n_static_target_blocks=False` disables the automatic default,
  - `fmm_large_n_static_target_blocks_max_per_leaf=<int>` overrides the cap,
  - `fmm_large_n_target_block_size=<int>` still passes through to jaccpot.

Sanity check:

```text
SimulationConfig(
  N_particles=200000,
  fmm_preset="large_n_gpu",
  fmm_tree_build_mode="static_radix",
)
=> JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
=> JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16

SimulationConfig(..., fmm_large_n_static_target_blocks=False)
=> JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=0
```

Exact next action:

```text
Run one final 200k/20 integration gate without the explicit
--fmm-large-n-static-target-blocks flags, confirming the automatic default
is active and still keeps the warm full sweep below 1s.
```

## Automatic Default Smoke - 2026-04-30

Command shape, intentionally without explicit target-block flags:

```text
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 2 \
  --fmm-refresh-every 1 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --fmm-nearfield-edge-chunk-size 256 \
  --profile-breakdown \
  --require-static-shape \
  --max-compiled-profile-transitions 0 \
  --max-overflow-reprofiles 0 \
  --max-neighbor-edge-reprofiles 0 \
  --min-refresh-prepare-successes 1
```

Report:

```text
JSON: /tmp/static_radix_auto_targetblocks16_smoke_200k_2/galaxy_disk_profile_20260430_151259.json
```

Result:

```text
runtime including cold startup: 91.683s
profiled cold full prepare: 82.122s
profiled refresh prepare: 0.706s
evaluate total: 6.680s over 2 calls

static_radix refresh hits: 1
static_radix refresh misses: 0
static_radix profile overflows: 0
compiled profile transitions: 0
overflow reprofiles: 0
neighbor-edge reprofiles: 0
post-warmup shape signatures: stable
```

The auto-default smoke report was generated before the JSON gained an explicit
`fmm_large_n_effective_environment_overrides` field, but the stage counters and
the direct helper sanity check confirm the automatic static-target-block path.
Future reports include the effective override dictionary directly.

Production status:

```text
static_radix + large_n_gpu + N>=200k now defaults to:
  JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
  JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16

Explicit SimulationConfig values still override the default.
```
