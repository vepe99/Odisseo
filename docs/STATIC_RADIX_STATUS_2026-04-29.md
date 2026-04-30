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
