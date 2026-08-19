# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Static Radix Optimization Status - 2026-05-05

## Context

The older `capacity_fixed_depth` work has been superseded by `static_radix`.
Current static radix behavior:

- Morton-sort current particles.
- Split sorted particles into fixed count buckets of `leaf_size`.
- Preserve fixed data-structure shapes for fixed `N` and `leaf_size`.
- Allow particle order, geometry, interaction contents, and numerical payloads
  to change across refreshes.

The recent jaccpot follow-up PR fixing static-radix workspace handling has been
merged. Local jaccpot was fast-forwarded to `origin/feat/static-radix`.

## Latest Known Full Baseline

Post-merge validation before today's optimization patch:

```text
report: /tmp/static_radix_postmerge_workspacefix_200k_20_gpu2/galaxy_disk_profile_20260505_115952.json
output: /tmp/static_radix_postmerge_workspacefix_200k_20_gpu2.npz
```

Key fields:

```text
total_seconds: 110.632
prepare_seconds: 99.849
evaluate_seconds: 8.345
update_seconds: 1.550

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_refresh_fallback_prepare_calls: 0
profiled_full_prepare_seconds: 86.811
profiled_refresh_prepare_seconds: 13.038

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

Automatic ODISSEO large-N static defaults were active:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16
```

Refresh timing breakdown from that run:

```text
runtime_refresh_total_seconds: 13.003
runtime_refresh_tree_upward_seconds: 10.899
runtime_refresh_nearfield_seconds: 1.872
runtime_refresh_dual_downward_seconds: 0.173
```

Interpretation: refresh stability is solved. The largest warm-refresh cost is
tree/upward refresh, followed by evaluate.

## Optimization Patch Under Test

Repository:

```text
/export/home/tbuck/jaccpot
branch: feat/static-radix
file: jaccpot/runtime/_fmm_impl.py
```

Change:

- Static-radix topology reuse now keys on fixed data-structure shape instead of
  the full sorted Morton-code stream.
- Cached static-radix topology rebuild now updates dynamic fields
  `particle_indices`, `morton_codes`, bounds, and leaf code metadata while
  reusing fixed parent/child/range topology arrays.

Rationale:

- For `static_radix`, the data-structure shape depends on particle count and
  `leaf_size`, not on the exact Morton-code values.
- The previous key path required hashing sorted Morton codes and usually failed
  generic topology reuse even though the static structure was compatible.
- Warm static refresh should still recompute Morton order and numerical payloads,
  but it should not rebuild or host-hash fixed topology arrays.

## Validation So Far

Passed:

```bash
python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py

CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k static_radix
```

Result:

```text
1 passed, 58 deselected in 24.51s
```

## Next Measurement

Ran the full ODISSEO 200k/20 gate again on GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
micromamba run -n odisseo python -B notebooks/scalability/galaxy_disk_fmm_large_n.py \
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
  --min-refresh-prepare-successes 19 \
  --report-dir /tmp/static_radix_shape_key_reuse_200k_20_gpu2 \
  --output /tmp/static_radix_shape_key_reuse_200k_20_gpu2.npz
```

Acceptance checks:

- `runtime_static_radix_refresh_hits == 19`
- `runtime_static_radix_refresh_misses == 0`
- `profiled_refresh_fallback_prepare_calls == 0`
- `runtime_compiled_profile_transitions == 0`
- `shape_signature_stable_post_warmup == true`

Performance fields to compare against baseline:

- `total_seconds`
- `profiled_refresh_prepare_seconds`
- `runtime_refresh_tree_upward_seconds`
- `evaluate_seconds`

## Shape-Key Reuse Result

Output:

```text
report: /tmp/static_radix_shape_key_reuse_200k_20_gpu2/galaxy_disk_profile_20260505_134402.json
output: /tmp/static_radix_shape_key_reuse_200k_20_gpu2.npz
```

Result:

```text
total_seconds: 109.012
prepare_seconds: 98.523
evaluate_seconds: 8.043
update_seconds: 1.516

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_refresh_fallback_prepare_calls: 0
profiled_full_prepare_seconds: 85.537
profiled_refresh_prepare_seconds: 12.987

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

Refresh timing comparison:

```text
baseline runtime_refresh_total_seconds:       13.003
new      runtime_refresh_total_seconds:       12.945

baseline runtime_refresh_tree_upward_seconds: 10.899
new      runtime_refresh_tree_upward_seconds: 10.786

baseline evaluate_seconds:                    8.345
new      evaluate_seconds:                    8.043
```

Interpretation:

- The optimization is behaviorally safe.
- It produces a small improvement, but not a large one.
- The dominant refresh cost remains tree/upward numeric work, not static topology
  reconstruction or topology-key hashing.

## Next Hypothesis

The next useful lever is finer profiling and optimization inside the static
refresh upward stage. Current aggregate timing folds together:

- static radix Morton encode/sort/reorder,
- geometry/mass-moment preparation,
- solid-FMM P2M/M2M upward payload refresh,
- synchronization overhead around those calls.

Add or inspect finer counters before trying a larger rewrite. If P2M/M2M is the
true cost, optimize solid-FMM upward for the fixed bucket topology. If
encode/sort/reorder is material, target the static-radix refresh path itself.

## Cold Prepare Stage Profiling

After lazy geometry and nearfield payload reuse, warm refresh is small enough
that the 20-step benchmark is dominated by the first full prepare. Added
ODISSEO-side per-prepare stage deltas:

- `profiled_prepare_events[*].stage_seconds`
- `profiled_prepare_stage_seconds_by_path`

Implementation note: ODISSEO uses the high-level jaccpot facade, so the profiling
scope activates timing on `solver._impl` when present.

Smoke validation:

```text
report: /tmp/static_radix_cold_timing_smoke2_20k_2_gpu2/galaxy_disk_profile_20260505_174117.json
total_seconds: 42.453
prepare_seconds: 33.731
```

The smoke report contains both a full prepare event and a refresh event with
stage deltas.

Full 200k/20 profile:

```text
report: /tmp/static_radix_cold_prepare_profile_200k_20_gpu2/galaxy_disk_profile_20260505_174533.json
output: /tmp/static_radix_cold_prepare_profile_200k_20_gpu2.npz

total_seconds: 97.427
prepare_seconds: 86.838
evaluate_seconds: 8.163
update_seconds: 1.536

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_full_prepare_seconds: 85.246
profiled_refresh_prepare_seconds: 1.591

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
```

Full prepare stage split:

```text
dual/downward total:          64.403
dual artifact build:          53.119
  split far pairs:            27.642
  split leaf neighbors:       25.476
tree/upward total:            16.265
  upward compute:             11.359
  tree build:                  4.283
nearfield payload:             4.502
  target blocks:               2.495
  speed layout:                0.868
  radix payload:               0.536
  leaf groups:                 0.297
dual downward compute:        11.281
```

Warm refresh stage split across 19 refreshes:

```text
tree/upward total:             0.482
compile/sync suspect:          0.210
dual downward compute:         0.115
dual finalize:                 0.041
```

Interpretation:

- Warm refresh is no longer the dominant 20-step cost.
- The first full prepare is dominated by cold dual-tree artifact construction,
  especially split far-pair and split leaf-neighbor builds.
- Upward compute remains visible, but it is now a secondary cold-prepare target.
- Nearfield prepared payload construction is smaller than dual artifact build,
  but target-block generation is still a measurable cold cost.

Next target:

- Investigate static-radix cold dual artifact construction. The main question is
  whether the static-radix fixed topology can build or cache the far-pair and
  leaf-neighbor scaffolds in a cheaper shape-stable way during the first prepare,
  rather than paying the full split dual-tree build cost up front.

## Split vs Raw Dual Build Check

Tested whether disabling the minimum-memory split dual build would reduce cold
prepare time:

```bash
JACCPOT_PREPARE_STAGE_MEMORY_SPLIT_ENABLED=0
```

Short 200k/2 run:

```text
report: /tmp/static_radix_unsplit_dual_smoke_200k_2_gpu2/galaxy_disk_profile_20260505_180109.json

total_seconds: 106.691
prepare_seconds: 97.091
evaluate_seconds: 7.119
profiled_full_prepare_seconds: 96.933
profiled_refresh_prepare_seconds: 0.158
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
```

Full prepare split:

```text
dual/downward total:          76.312
dual artifact build:          65.178
  raw combined traversal:     65.178
tree/upward total:            15.920
nearfield payload:             4.621
```

Interpretation:

- The raw combined traversal path is slower than the split path for this 200k
  static-radix large-N case.
- Keep the split path as the default. The next dual-build optimization needs to
  reduce or avoid work inside the split far-pair and leaf-neighbor builders,
  not simply switch back to the combined raw traversal.

## Shared Count-Pass Dual Split

Implemented a Yggdrax split-builder optimization:

- Added `build_compact_far_pairs_and_leaf_neighbor_lists`.
- The bounded explicit-traversal path now uses one shared dual-tree count walk
  with `collect_far=True` and `collect_near=True`.
- Far pairs and near neighbors are still filled in separate compact passes, so
  the low-memory split behavior is preserved.
- jaccpot's minimum-memory split path now calls the shared-count builder for the
  streamed compact-far-pairs + near-neighbor case.
- Added timing field:
  `refresh_dual_split_shared_far_near_seconds`.

Validation:

```text
python3 -m py_compile:
  yggdrax/yggdrax/_interactions_impl.py
  yggdrax/yggdrax/interactions.py
  yggdrax/yggdrax/__init__.py
  yggdrax/tests/unit/test_interactions_adapter.py
  jaccpot/jaccpot/runtime/_interaction_cache.py
  jaccpot/jaccpot/runtime/_fmm_impl.py
  Odisseo/odisseo/jaccpot_coupling.py
  Odisseo/notebooks/scalability/radix_fastlane_investigation.py

Yggdrax focused adapter tests: 8 passed in 45.92s
Yggdrax broader interaction tests: 18 passed in 66.74s
jaccpot static_radix slice: 1 passed, 58 deselected in 22.94s
ODISSEO API regression: 4 passed, 1 warning in 10.06s
```

200k/2 smoke:

```text
report: /tmp/static_radix_shared_count_smoke_200k_2_gpu2/galaxy_disk_profile_20260505_195327.json

total_seconds: 83.376
prepare_seconds: 73.830
profiled_full_prepare_seconds: 73.671
profiled_refresh_prepare_seconds: 0.159
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
```

200k/2 full prepare split:

```text
dual/downward total:                53.098
dual artifact build:                41.965
  shared far/near split builder:    41.965
tree/upward total:                  15.922
dual downward compute:              11.129
nearfield payload:                   4.566
```

Full 200k/20 gate:

```text
report: /tmp/static_radix_shared_count_200k_20_gpu2/galaxy_disk_profile_20260505_195530.json
output: /tmp/static_radix_shared_count_200k_20_gpu2.npz

total_seconds: 85.814
prepare_seconds: 75.139
evaluate_seconds: 8.212
update_seconds: 1.555

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_full_prepare_seconds: 73.535
profiled_refresh_prepare_seconds: 1.604

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

200k/20 full prepare split:

```text
dual/downward total:                53.068
dual artifact build:                41.783
  shared far/near split builder:    41.782
tree/upward total:                  15.639
dual downward compute:              11.283
upward compute:                     10.887
nearfield payload:                   4.752
tree build:                          4.171
```

Comparison against the cold-prepare profile before shared-count:

```text
total_seconds:                  97.427 -> 85.814  (-11.613)
profiled_full_prepare_seconds:  85.246 -> 73.535  (-11.711)
dual artifact build:            53.119 -> 41.783  (-11.336)
```

Interpretation:

- The missing count-pass reuse was a real cold-prepare cost.
- The optimization preserves refresh stability and static-shape behavior.
- The remaining cold wall is split between:
  - shared far/near split builder: ~41.8s,
  - tree/upward: ~15.6s,
  - dual downward compute: ~11.3s,
  - nearfield payload: ~4.8s.

Next target:

- Continue inside the shared split builder. The next likely lever is finer
  timing of the shared builder's one count pass vs the two fill passes. If the
  count pass dominates, pursue a fixed static-radix scaffold/count cache. If
  fill dominates, look at compact-fill kernel shape/capacity and whether the
  far or near fill can be fused without exceeding the memory envelope.

## Shared Split Substages and Combined Fill

Added finer timing fields for the shared split builder:

- `refresh_dual_split_shared_count_seconds`
- `refresh_dual_split_shared_far_fill_seconds`
- `refresh_dual_split_shared_near_fill_seconds`
- `refresh_dual_split_shared_combined_fill_seconds`

Initial 200k/2 substage profile with separate far/near fills:

```text
report: /tmp/static_radix_shared_count_substages_200k_2_gpu2/galaxy_disk_profile_20260505_213243.json

total_seconds: 83.031
profiled_full_prepare_seconds: 73.431

shared far/near total: 41.779
  count pass:          13.971
  far fill:            14.174
  near fill:           11.910
```

Implemented combined compact fill:

- The shared builder still performs one exact count pass.
- It now calls `_dual_tree_walk_compact_fill_impl` once with both
  `collect_far=True` and `collect_near=True`.
- This avoids running the same dual-tree traversal twice for far and near fill.
- Exact compact far and near output buffers are still used, so this avoids the
  raw combined traversal's large fixed per-node/per-leaf buffers.

200k/2 combined-fill smoke:

```text
report: /tmp/static_radix_shared_combined_fill_200k_2_gpu2/galaxy_disk_profile_20260505_215041.json

total_seconds: 74.389
prepare_seconds: 65.059
profiled_full_prepare_seconds: 64.899
profiled_refresh_prepare_seconds: 0.160
runtime_static_radix_refresh_hits: 1
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0

shared far/near total: 33.339
  count pass:          13.964
  combined fill:       17.691
```

Full 200k/20 combined-fill gate:

```text
report: /tmp/static_radix_shared_combined_fill_200k_20_gpu2/galaxy_disk_profile_20260505_215254.json
output: /tmp/static_radix_shared_combined_fill_200k_20_gpu2.npz

total_seconds: 77.763
prepare_seconds: 67.071
evaluate_seconds: 8.268
update_seconds: 1.545

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_full_prepare_seconds: 65.452
profiled_refresh_prepare_seconds: 1.618

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

200k/20 full prepare split:

```text
dual/downward total:                44.926
dual artifact build:                33.640
  shared far/near split builder:    33.640
    count pass:                     14.261
    combined fill:                  17.673
tree/upward total:                  15.704
dual downward compute:              11.282
nearfield payload:                   4.748
tree build:                          4.085
```

Comparison:

```text
before shared-count:
  total_seconds:                  97.427
  profiled_full_prepare_seconds:  85.246
  dual artifact build:            53.119

after shared count:
  total_seconds:                  85.814
  profiled_full_prepare_seconds:  73.535
  dual artifact build:            41.783

after combined fill:
  total_seconds:                  77.763
  profiled_full_prepare_seconds:  65.452
  dual artifact build:            33.640
```

Interpretation:

- The combined compact fill is another real cold-prepare win, reducing the
  200k/20 gate by about 8.1s versus shared-count separate fills.
- The remaining shared builder cost is now split between one count pass
  (~14.3s) and one combined fill (~17.7s).
- The next dual-build lever is no longer duplicate fill traversal; it is either
  reducing the count pass, reducing the combined fill, or avoiding a portion of
  both with a static-radix scaffold/cache.

Next target:

- Investigate whether static-radix can cache count/offset scaffolds safely for
  fixed topology and fixed traversal parameters, and whether counts remain
  stable enough across refreshes to support a capacity-first fill path without
  recounting.

## Traversal Process-Block Tuning

Checked one low-risk fill optimization first:

- Hypothesis: compact far-pair tags are unused by the non-adaptive streamed
  path, so skipping tag allocation could reduce combined-fill time.
- 200k/2 diagnostic result:

```text
report: /tmp/static_radix_untagged_combined_fill_200k_2_gpu2/galaxy_disk_profile_20260505_220930.json

total_seconds: 75.136
profiled_full_prepare_seconds: 65.625
shared far/near total: 33.925
  count pass:          13.980
  combined fill:       18.140
```

Interpretation:

- No measurable win; the result was slightly worse/noise versus the combined
  fill baseline.
- Backed out the no-tag experiment.

Added narrow diagnostic env overrides for the minimum-memory GPU traversal
seed:

- `JACCPOT_LARGE_N_GPU_MIN_MEMORY_PAIR_QUEUE`
- `JACCPOT_LARGE_N_GPU_MIN_MEMORY_PROCESS_BLOCK`
- `JACCPOT_LARGE_N_GPU_MIN_MEMORY_INTERACTIONS_PER_NODE`
- `JACCPOT_LARGE_N_GPU_MIN_MEMORY_NEIGHBORS_PER_LEAF`

Then tested process-block and queue tuning:

```text
process_block=256:
report: /tmp/static_radix_process_block_256_200k_2_gpu2/galaxy_disk_profile_20260505_221702.json

total_seconds: 71.705
profiled_full_prepare_seconds: 61.909
shared far/near total: 30.506
  count pass:          10.805
  combined fill:       17.904

process_block=256, pair_queue=262144:
report: /tmp/static_radix_queue_262k_block_256_200k_2_gpu2/galaxy_disk_profile_20260505_221853.json

total_seconds: 71.532
profiled_full_prepare_seconds: 61.496
shared far/near total: 30.320
  count pass:          10.847
  combined fill:       17.709

process_block=512:
report: /tmp/static_radix_process_block_512_200k_2_gpu2/galaxy_disk_profile_20260505_223142.json

total_seconds: 71.066
profiled_full_prepare_seconds: 61.136
shared far/near total: 30.052
  count pass:          10.771
  combined fill:       17.543

process_block=1024:
report: /tmp/static_radix_process_block_1024_200k_2_gpu2/galaxy_disk_profile_20260505_223506.json

total_seconds: 70.766
profiled_full_prepare_seconds: 61.152
shared far/near total: 29.956
  count pass:          10.666
  combined fill:       17.591
```

Interpretation:

- The useful lever is the process block, not the larger default queue.
- Raising the sub-million minimum-memory process block from `64` to `1024`
  cuts the shared count pass from roughly `14s` to `10.7s`.
- Kept the small default queue to preserve the memory envelope.

Implemented default:

- `_GPU_MINIMUM_MEMORY_PROCESS_BLOCK = 1024`
- Minimum-memory seed docs now describe a small queue seed plus streamed
  process-block floor.

Full 200k/20 gate with the new default and no traversal env overrides:

```text
report: /tmp/static_radix_process_block_1024_200k_20_gpu2/galaxy_disk_profile_20260505_223830.json
output: /tmp/static_radix_process_block_1024_200k_20_gpu2.npz

total_seconds: 73.406
prepare_seconds: 62.651
evaluate_seconds: 8.258
update_seconds: 1.515

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 19
profiled_full_prepare_seconds: 61.050
profiled_refresh_prepare_seconds: 1.601

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

200k/20 full prepare split:

```text
dual/downward total:                41.000
dual artifact build:                30.236
  shared far/near split builder:    30.235
    count pass:                     10.728
    combined fill:                  17.770
tree/upward total:                  16.055
nearfield payload:                   4.336
```

Comparison:

```text
after combined fill:
  total_seconds:                  77.763
  profiled_full_prepare_seconds:  65.452
  dual artifact build:            33.640

after process-block tuning:
  total_seconds:                  73.406
  profiled_full_prepare_seconds:  61.050
  dual artifact build:            30.236
```

Validation:

```text
python3 -m py_compile:
  jaccpot/jaccpot/runtime/_fmm_impl.py
  jaccpot/jaccpot/runtime/_interaction_cache.py
  yggdrax/yggdrax/_interactions_impl.py
  yggdrax/yggdrax/interactions.py

Yggdrax focused adapter tests: 8 passed in 48.98s
jaccpot static_radix slice:    1 passed, 2 skipped, 324 deselected in 29.48s
ODISSEO API regression:       4 passed, 1 warning in 7.00s
git diff --check:             clean for jaccpot and yggdrax
```

Next target:

- The count pass is now materially better, leaving combined fill (~17.8s) as
  the largest shared-builder substage. Next likely levers are fill-kernel shape
  and whether the static-radix topology can reuse a bounded scaffold for a
  capacity-first fill path without recounting.

## Full Galaxy Timing Check

Ran one more traversal process-block sweep point before switching to the real
galaxy timing:

```text
process_block=2048:
report: /tmp/static_radix_process_block_2048_200k_2_gpu2/galaxy_disk_profile_20260505_232335.json

total_seconds: 72.560
profiled_full_prepare_seconds: 62.313
shared far/near total: 30.332
  count pass:          10.967
  combined fill:       17.593
```

Interpretation:

- `2048` regressed versus `1024`; keep `_GPU_MINIMUM_MEMORY_PROCESS_BLOCK = 1024`.
- The scaffold/cache idea is less compelling for the normal static-radix path
  now because warm refreshes reuse the dual artifacts entirely. It would mainly
  help fallback/full-reprepare cases, not the main steady-state run.

Full 200k/200 galaxy-disk timing with the optimized static-radix defaults:

```text
report: /tmp/static_radix_optimized_galaxy_200k_200_gpu2/galaxy_disk_profile_20260505_232548.json
output: /tmp/static_radix_optimized_galaxy_200k_200_gpu2.npz

total_seconds: 100.161
prepare_seconds: 77.393
evaluate_seconds: 19.963
update_seconds: 1.810

profiled_full_prepare_calls: 1
profiled_refresh_prepare_calls: 199
profiled_full_prepare_seconds: 61.657
profiled_refresh_prepare_seconds: 15.736

runtime_static_radix_refresh_hits: 199
runtime_static_radix_refresh_misses: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true
```

Selected stage totals:

```text
full prepare:
  dual/downward total:              41.352
  dual artifact build:              30.265
    shared far/near split builder:  30.257
      count pass:                   10.890
      combined fill:                17.656
  tree/upward total:                15.813
  upward compute:                   11.038
  nearfield payload:                 4.417
  tree build:                        4.183

refresh aggregate over 199 calls:
  refresh prepare total:            15.736
  tree/upward total:                 4.378
  dual artifact build:               0.007
  dual downward compute:             1.081
  compile/sync suspect:              2.028
```

Comparison against earlier documented 20-step gates:

```text
capacity-fixed 200k/20:              116.690
static-radix targetblocks16 200k/20: 103.367
static-radix targetblock4 200k/20:    99.386
optimized static-radix 200k/200:     100.161
```

Interpretation:

- The optimized path now runs a 200-step galaxy simulation in about the same
  wall time as the earlier 20-step gates.
- Cold prepare is still the largest single cost, but it is amortized over the
  full run. Steady-state refresh is about `15.736 / 199 = 0.079s` per refresh.
- Evaluate is about `19.963 / 200 = 0.100s` per step, and is now a meaningful
  steady-state target if we want more end-to-end speed.
- Further cold dual-build work is still possible, but the full simulation says
  the next best user-visible target is probably steady-state evaluate/refresh,
  not more cold-only micro-optimization.

## Split Timing Instrumentation

Added counters:

- jaccpot `refresh_tree_build_seconds`
- jaccpot `refresh_upward_compute_seconds`
- ODISSEO report fields:
  - `runtime_refresh_tree_build_seconds`
  - `runtime_refresh_upward_compute_seconds`
- fast-lane investigation stage keys for the same fields.

Validation:

```bash
python3 -m py_compile \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py \
  /export/home/tbuck/Odisseo/odisseo/jaccpot_coupling.py \
  /export/home/tbuck/Odisseo/notebooks/scalability/radix_fastlane_investigation.py

CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k static_radix

micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/Odisseo/tests/test_integration_api.py
```

Results:

```text
jaccpot static_radix slice: 1 passed, 58 deselected in 25.93s
ODISSEO API regression:    4 passed, 1 warning in 7.74s
```

Full split-timing run:

```text
report: /tmp/static_radix_refresh_split_timing_200k_20_gpu2/galaxy_disk_profile_20260505_141256.json
output: /tmp/static_radix_refresh_split_timing_200k_20_gpu2.npz
```

Result:

```text
total_seconds: 112.973
prepare_seconds: 102.394
evaluate_seconds: 8.227
update_seconds: 1.544

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
profiled_refresh_fallback_prepare_calls: 0
runtime_compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true

runtime_refresh_total_seconds: 14.078
runtime_refresh_tree_upward_seconds: 11.938
runtime_refresh_tree_build_seconds: 0.344
runtime_refresh_upward_compute_seconds: 11.571
runtime_refresh_nearfield_seconds: 1.903
runtime_refresh_dual_downward_seconds: 0.173
runtime_refresh_compile_or_sync_suspect_seconds: 0.049
```

Interpretation:

- Static-radix tree build/reorder is not the remaining bottleneck.
- The dominant warm-refresh cost is the solid-FMM upward compute path.
- The shape-key reuse patch is safe but only a small win; it should be kept if
  we want cleaner static-radix semantics, but it is not the main performance
  lever.

## Upward Leaf Batch Sweep

Refresh-specific 200k static-radix micro-sweep:

```text
batch=64   refresh_avg=0.694171 tree_upward=2.269093 nearfield=0.487620 hits=4 misses=0
batch=128  refresh_avg=0.683441 tree_upward=2.240447 nearfield=0.440754 hits=4 misses=0
batch=256  refresh_avg=0.679174 tree_upward=2.239107 nearfield=0.411237 hits=4 misses=0
batch=2048 refresh_avg=0.672871 tree_upward=2.208146 nearfield=0.425043 hits=4 misses=0
```

Conclusion: keep the current `upward_leaf_batch_size=2048`; smaller batches do
not improve the representative refresh path.

## Next Step

Profile inside `prepare_solidfmm_complex_upward_sweep(...)`:

- geometry computation,
- mass moments,
- leaf P2M,
- internal M2M aggregation.

The current evidence says optimization should focus on P2M/M2M or avoiding
unneeded geometry/moment work for the static-radix refresh/evaluate contract.

## Upward Substage Profile - 2026-05-05

Added opt-in solid-FMM upward substage profiling behind:

```text
JACCPOT_PROFILE_UPWARD_STAGES=1
```

This flag intentionally calls `jax.block_until_ready(...)` after each upward
substage, so use these numbers for attribution only, not as production runtime.

New jaccpot diagnostics:

- `refresh_upward_geometry_seconds`
- `refresh_upward_mass_moments_seconds`
- `refresh_upward_p2m_seconds`
- `refresh_upward_m2m_seconds`
- `refresh_upward_source_motion_seconds`

Representative 200k static-radix refresh micro-profile on GPU 2:

```text
refresh_total_seconds: 3.367690
refresh_tree_upward_seconds: 2.873774
refresh_tree_build_seconds: 0.073849
refresh_upward_compute_seconds: 2.793856

refresh_upward_geometry_seconds: 2.532371
refresh_upward_mass_moments_seconds: 0.056575
refresh_upward_p2m_seconds: 0.022405
refresh_upward_m2m_seconds: 0.180057
refresh_upward_source_motion_seconds: 0.0

refresh_nearfield_seconds: 0.428550
refresh_dual_downward_seconds: 0.049919
static_radix_refresh_hits: 5
static_radix_refresh_misses: 0
```

Interpretation:

- The bottleneck is not leaf P2M.
- It is not internal M2M either.
- The dominant upward substage is exact tree geometry construction.
- Static-radix geometry currently uses exact per-leaf particle bounds because
  `use_morton_geometry=False`; this requires leaf particle gathers plus internal
  bounds aggregation.

Design implication:

- Warm static-radix refreshes with interaction-cache hits likely do not need
  exact geometry. The dual-tree artifacts are already cached, and `center_mode`
  is `com`, so multipole centers come from mass moments rather than AABB
  geometry.
- The next optimization should avoid exact geometry computation on cached
  static-radix refreshes, while preserving exact geometry for cold prepares and
  fallback/cache-miss traversal builds.

## Lazy Geometry Optimization

Implemented in jaccpot:

- `prepare_solidfmm_complex_upward_sweep(...)` accepts `defer_geometry`.
- Static-radix warm refresh defers geometry only when:
  - refresh timing is active, meaning this is a refresh path,
  - tree mode is `static_radix`,
  - upward center mode is `com`,
  - an interaction cache entry exists.
- `_build_dual_tree_artifacts(...)` accepts an optional `geometry_factory`.
  It performs the cache lookup first. If the cache hits, geometry is never
  materialized. If the cache misses, exact geometry is computed before traversal.

This keeps exact geometry for cold prepares and cache-miss/fallback traversal
builds while skipping it for the normal warm static-radix cache-hit path.

Validation:

```text
python3 -m py_compile:
  jaccpot/runtime/_fmm_impl.py
  jaccpot/runtime/_interaction_cache.py
  jaccpot/upward/solidfmm_complex_tree_expansions.py

jaccpot static_radix slice:
  1 passed, 58 deselected in 24.13s
```

Opt-in substage profile after lazy geometry:

```text
refresh_total_seconds: 0.795387
refresh_tree_upward_seconds: 0.327961
refresh_tree_build_seconds: 0.085855
refresh_upward_compute_seconds: 0.235996

refresh_upward_geometry_seconds: 0.000029
refresh_upward_mass_moments_seconds: 0.057103
refresh_upward_p2m_seconds: 0.017404
refresh_upward_m2m_seconds: 0.159376

refresh_nearfield_seconds: 0.399562
refresh_dual_downward_seconds: 0.052126
static_radix_refresh_hits: 5
static_radix_refresh_misses: 0
interaction_cache_hits: 5
interaction_cache_misses: 1
```

Full 200k/20 production gate after lazy geometry:

```text
report: /tmp/static_radix_lazy_geometry_200k_20_gpu2/galaxy_disk_profile_20260505_155055.json
output: /tmp/static_radix_lazy_geometry_200k_20_gpu2.npz
```

Result:

```text
total_seconds: 93.852
prepare_seconds: 83.621
evaluate_seconds: 7.832
update_seconds: 1.470

profiled_full_prepare_seconds: 81.101
profiled_refresh_prepare_seconds: 2.520
profiled_refresh_fallback_prepare_calls: 0

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_static_radix_profile_overflows: 0
runtime_interaction_cache_hits: 19
runtime_interaction_cache_misses: 1
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true

runtime_refresh_total_seconds: 2.489
runtime_refresh_tree_upward_seconds: 0.536
runtime_refresh_tree_build_seconds: 0.287
runtime_refresh_upward_compute_seconds: 0.229
runtime_refresh_upward_geometry_seconds: 0.0
runtime_refresh_nearfield_seconds: 1.744
runtime_refresh_dual_downward_seconds: 0.155
```

Runtime comparison:

```text
post-merge workspace fix baseline: 110.632s
shape-key reuse run:              109.012s
split-timing run:                 112.973s
lazy-geometry run:                 93.852s
```

Conclusion:

- Lazy geometry is the first major static-radix refresh optimization in this
  phase.
- Warm refresh prepare fell from roughly `13s / 19 = 0.686s` to
  `2.52s / 19 = 0.133s`.
- The next bottleneck is nearfield refresh (`1.744s` over 19 refreshes), not
  upward compute.

## Nearfield Payload Reuse

Implemented in jaccpot:

- Static-radix refresh now uses the topology-preserving refresh path instead of
  calling full `prepare_state(...)`.
- This path rebuilds numeric tree/upward/downward payloads, checks that the
  active neighbor list still matches, then reuses the previous prepared-state
  nearfield payloads:
  - leaf particle groups,
  - target-block layouts,
  - padded static target blocks,
  - radix fast-lane payload,
  - neighbor profile padding.
- Static-radix topology candidate generation no longer depends on generic
  `reuse_topology`; static refresh has its own fixed-shape contract.

Validation:

```text
python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py

CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k static_radix
```

Result:

```text
1 passed, 58 deselected in 23.74s
```

Short 200k refresh micro-profile:

```text
refresh_total_seconds: 0.245446
refresh_tree_upward_seconds: 0.185245
refresh_nearfield_seconds: 0.0
refresh_nearfield_leaf_groups_seconds: 0.0
refresh_nearfield_target_blocks_seconds: 0.0
refresh_nearfield_speed_layout_seconds: 0.0
refresh_nearfield_radix_payload_seconds: 0.0
refresh_dual_downward_seconds: 0.0
static_radix_refresh_hits: 5
static_radix_refresh_misses: 0
interaction_cache_hits: 5
interaction_cache_misses: 1
```

Full 200k/20 production gate:

```text
report: /tmp/static_radix_reuse_nearfield_200k_20_gpu2/galaxy_disk_profile_20260505_170643.json
output: /tmp/static_radix_reuse_nearfield_200k_20_gpu2.npz
```

Result:

```text
total_seconds: 93.573
prepare_seconds: 82.788
evaluate_seconds: 8.305
update_seconds: 1.670

profiled_full_prepare_seconds: 81.150
profiled_refresh_prepare_seconds: 1.638
profiled_refresh_fallback_prepare_calls: 0

runtime_static_radix_refresh_hits: 19
runtime_static_radix_refresh_misses: 0
runtime_interaction_cache_hits: 19
runtime_interaction_cache_misses: 1
runtime_compiled_profile_transitions: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0
runtime_large_n_overflow_profile_reprofiles: 0
shape_signature_stable_post_warmup: true

runtime_refresh_total_seconds: 0.748
runtime_refresh_tree_upward_seconds: 0.503
runtime_refresh_nearfield_seconds: 0.0
runtime_refresh_dual_downward_seconds: 0.0
```

Interpretation:

- Warm refresh prepare is now about `1.64s / 19 = 0.086s`.
- Static-radix warm refresh rebuilds no nearfield payload and no dual/downward
  traversal artifacts on cache hits.
- End-to-end runtime is now dominated by cold prepare (`~81s`) plus evaluate
  (`~8.3s` over 20 calls), not warm refresh.

Next optimization targets:

1. Cold prepare: dual-tree artifact build and first compile/startup cost.
2. Warm evaluate: currently about `8.3s / 20 = 0.415s` per call in the full
   integration run.
