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
