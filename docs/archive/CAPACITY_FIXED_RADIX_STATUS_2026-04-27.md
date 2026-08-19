# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Capacity-Fixed Radix Status - 2026-04-27

## Goal

Implement a new `capacity_fixed_depth` radix mode for the ODISSEO/jaccpot/yggdrax FMM path:

- preserve existing `fixed_depth` semantics,
- build a sparse fixed radix topology with hard leaf capacity,
- keep solid-FMM/spherical-basis accuracy,
- refresh numerical payloads against the fixed topology without recompiling,
- validate first on GPU 9 with the 200k-particle, 20-step ODISSEO galaxy benchmark.

## Branches

Created matching feature branches:

- `/export/home/tbuck/yggdrax`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/jaccpot`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/Odisseo`: `feat/capacity-fixed-radix`

The working trees already had unrelated dirty files before this work. Do not clean or revert them blindly.

## Implementation Status

### yggdrax

Changed:

- `yggdrax/_tree_impl.py`
- `yggdrax/tree.py`
- `tests/unit/test_tree.py`

Implemented:

- New radix build mode `capacity_fixed_depth`.
- Sparse capacity refinement starting from fixed-depth Morton partitions.
- Hard check that occupied leaves do not exceed `leaf_size`.
- Template refresh helper:
  `rebuild_capacity_fixed_depth_tree_from_template(...)`
- Public wrapper:
  `build_capacity_fixed_depth_tree(...)`
- Unit tests for clustered capacity enforcement and topology-preserving template refresh.

Quick smoke result already observed:

```text
capacity_fixed_depth 142 60 64
capacity_fixed_depth 142 60 True
```

Meaning: build mode correct, 142 leaves, max leaf count 60 under capacity 64, and template refresh preserved topology structure.

### jaccpot

Changed:

- `jaccpot/runtime/_fmm_impl.py`
- `tests/integration/test_fmm.py`

Implemented:

- Accepted `capacity_fixed_depth` in tree mode validation.
- Routed the mode through yggdrax.
- Treated it as capacity-checked: `max_leaf_size > leaf_size` remains an error.
- Added refresh path that rebuilds particle ordering/ranges from the capacity-fixed template, then refreshes solid-FMM payloads.
- Added integration test for `prepare_state` + `refresh_prepared_state` using `capacity_fixed_depth`.

Quick GPU 9 smoke result already observed before GPUs became unavailable:

```text
prepare capacity_fixed_depth 128 129
refresh capacity_fixed_depth 128 129 1
```

Meaning: prepare/refresh used the new mode, max leaf stayed at 128, leaf count stayed 129, and same-topology refresh hits reached 1.

### ODISSEO

Changed:

- `notebooks/scalability/galaxy_disk_fmm_large_n.py`
- `odisseo/jaccpot_coupling.py`

Implemented:

- CLI now accepts:
  `--fmm-tree-build-mode capacity_fixed_depth`
- Timing/profile report now includes:
  `runtime_max_leaf_size`
  `runtime_max_leaves`
- Existing same-topology refresh counters are already surfaced and should show capacity-template refresh hits.

## Checks Already Run

Passed:

```bash
python3 -m py_compile /export/home/tbuck/yggdrax/yggdrax/_tree_impl.py /export/home/tbuck/yggdrax/yggdrax/tree.py
python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py
python3 -m py_compile /export/home/tbuck/Odisseo/odisseo/jaccpot_coupling.py /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py
python3 -m py_compile /export/home/tbuck/yggdrax/tests/unit/test_tree.py
python3 -m py_compile /export/home/tbuck/jaccpot/tests/integration/test_fmm.py
```

Not run yet because no GPU is free:

- targeted yggdrax pytest,
- targeted jaccpot pytest,
- ODISSEO integration tests,
- 200k/20-step GPU 9 benchmark.

## Tests To Run Next

### 1. yggdrax targeted unit tests

These should not require GPU, but run them first because they validate the new topology primitive.

```bash
micromamba run -n odisseo pytest /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k 'capacity_fixed_depth'
```

Expected:

- 2 tests pass.
- Max leaf count remains within capacity.
- Template refresh preserves structural arrays: parent/children/leaf codes/depths.

### 2. jaccpot targeted integration test

Run on GPU 9 when available:

```bash
CUDA_VISIBLE_DEVICES=9 micromamba run -n odisseo pytest /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k 'capacity_fixed_depth'
```

Expected:

- prepare succeeds with `tree_build_mode="capacity_fixed_depth"`,
- refresh succeeds through `refresh_prepared_state`,
- `max_leaf_size <= leaf_size`,
- topology leaf count stays fixed,
- `large_n_same_topology_refresh_hits >= 1`.

### 3. ODISSEO API regression test

```bash
micromamba run -n odisseo pytest /export/home/tbuck/Odisseo/tests/test_integration_api.py
```

Expected:

- existing API tests pass,
- no regressions from the new tree mode plumbing.

### 4. ODISSEO 200k/20-step acceptance benchmark on GPU 9

```bash
CUDA_VISIBLE_DEVICES=9 micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode capacity_fixed_depth \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --profile-breakdown \
  --report-dir /tmp/radix_capacity_fixed_gpu9_200k_20 \
  --output /tmp/galaxy_gpu9_capacity_fixed_200k_20.npz
```

Expected profile checks:

- no OOM,
- `runtime_max_leaf_size <= 256`,
- `shape_signature_stable_post_warmup == true`,
- `runtime_large_n_same_topology_refresh_hits > 0`,
- refresh events after initial prepare use the capacity-template path,
- runtime is meaningfully below the current LBVH/topology-rebuild baseline around 135 seconds.

## If A Test Fails

### yggdrax capacity test fails

Focus on:

- `build_capacity_fixed_depth_tree(...)`
- `_refine_leaf_partitions_by_capacity(...)`
- `rebuild_capacity_fixed_depth_tree_from_template(...)`

Most likely failure modes:

- mixed-depth Morton interval boundary bug,
- particles not covered by sparse template leaves after movement,
- empty leaf/internal range handling.

### jaccpot refresh test falls back

Focus on:

- `_refresh_large_n_same_topology(...)`
- `_rebuild_capacity_fixed_depth_tree_artifacts_from_template(...)`
- neighbor-list equality check

Most likely failure modes:

- capacity template refresh succeeds but neighbor list changes,
- `topology_key` handling is too strict or too loose,
- prepared-state profile still sees a shape drift.

### ODISSEO benchmark is still slow

Check the timing report fields first:

- `profiled_prepare_events`
- `runtime_refresh_tree_upward_seconds`
- `runtime_refresh_dual_downward_seconds`
- `runtime_refresh_nearfield_seconds`
- `runtime_large_n_same_topology_refresh_hits`
- `runtime_large_n_same_topology_refresh_misses`
- `runtime_large_n_same_topology_refresh_miss_topology`
- `runtime_large_n_same_topology_refresh_miss_neighbor`
- `runtime_max_leaf_size`

If refresh hits are high but runtime remains large, the next bottleneck is likely payload rebuild cost rather than topology recompilation.

## Current Stop Point

Implementation is partially verified by compile checks and two manual smoke tests.

The next concrete action is to run the four commands in **Tests To Run Next**, starting with the yggdrax targeted unit test and then the jaccpot GPU 9 targeted test once a GPU is free.

## Recovery Update - 2026-04-28

Session recovered on `/export/home/tbuck/Odisseo` branch `feat/capacity-fixed-radix`.
All three feature repos are on the same branch:

- `/export/home/tbuck/Odisseo`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/yggdrax`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/jaccpot`: `feat/capacity-fixed-radix`

Sandbox note: local sandboxed commands failed with:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Commands were therefore run with explicit approval/escalation.

### Dirty-state inventory at recovery

ODISSEO modified tracked files:

- `notebooks/galaxy_disc_sims.ipynb`
- `notebooks/scalability/galaxy_disk_fmm_large_n.py`
- `odisseo/__init__.py`
- `odisseo/integration_api.py`
- `odisseo/jaccpot_coupling.py`
- `odisseo/option_classes.py`
- `odisseo/potentials.py`
- `tests/test_integration_api.py`
- generated `__pycache__` files

ODISSEO untracked files include the capacity docs, prior performance docs, `notebooks/scalability/radix_fastlane_investigation.py`, report artifacts, and generated caches.

yggdrax modified tracked files:

- `tests/unit/test_tree.py`
- `yggdrax/_tree_impl.py`
- `yggdrax/tree.py`

jaccpot modified tracked files:

- `jaccpot/config.py`
- `jaccpot/runtime/_fmm_impl.py`
- `jaccpot/runtime/_interaction_cache.py`
- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/runtime/_large_n_types.py`
- `jaccpot/solver.py`
- `tests/integration/test_fmm.py`

jaccpot also has untracked `external/`.

### Tests run on GPU 2

The requested constraint for this recovery session is GPU 2 only.

The yggdrax targeted test initially failed before collection because the yggdrax pytest config injects coverage flags but the `odisseo` environment does not have `pytest-cov` active. Rerun with pytest addopts cleared:

```bash
CUDA_VISIBLE_DEVICES=2 micromamba run -n odisseo pytest -o addopts= /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k capacity_fixed_depth
```

Result:

```text
2 passed, 21 deselected in 15.13s
```

The jaccpot targeted integration test needed both JAX x64 and local source paths. Without x64 it failed importing yggdrax Morton uint64 constants. Without `PYTHONPATH`, it imported the wrong/incomplete yggdrax package and could not find `build_tree`.

Working command:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
micromamba run -n odisseo pytest -o addopts= /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k capacity_fixed_depth
```

Result:

```text
1 passed, 57 deselected in 32.74s
```

ODISSEO API regression:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
micromamba run -n odisseo pytest -o addopts= /export/home/tbuck/Odisseo/tests/test_integration_api.py
```

Result:

```text
4 passed, 1 warning in 3.19s
```

### 200k benchmark recovery note

An initial 200k/20-step benchmark was started on GPU 2 with the old/default padding:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode capacity_fixed_depth \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --profile-breakdown \
  --report-dir /tmp/radix_capacity_fixed_gpu2_200k_20 \
  --output /tmp/galaxy_gpu2_capacity_fixed_200k_20.npz
```

It ran silently for about 13.5 minutes, held about 8.3 GiB on GPU 2, and was interrupted with `SIGINT` after remembering the previous-session finding: the padding was too small for the 200k galaxy simulation. The interrupt traceback showed it was compiling the dual-tree walk from a refresh fallback path:

```text
refresh_prepared_state -> prepare_state -> prepare_large_n_state
  -> _prepare_state_dual_and_downward -> _dual_tree_walk_impl compile
```

This means the old/default-padding run was not a valid acceptance result.

Relevant padding/profile knobs found in current code:

- `JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION`, default `1.0`
- `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF`, default `32`
- `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM`, default `1.0`
- `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP`, default `0`
- `JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM`, default `2.0`
- `JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP`, default `0`

Current restarted GPU 2 acceptance attempt uses wider envelopes:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION=3.0 \
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=128 \
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM=2.0 \
JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM=3.0 \
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP=131072 \
JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP=65536 \
micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode capacity_fixed_depth \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 256 \
  --fmm-max-order 4 \
  --profile-breakdown \
  --report-dir /tmp/radix_capacity_fixed_gpu2_200k_20_pad3_blocks128 \
  --output /tmp/galaxy_gpu2_capacity_fixed_200k_20_pad3_blocks128.npz
```

Runtime observation: the process can be silent for a long time before writing
artifacts. Spot checks sometimes showed low GPU utilization, but interactive
monitoring also showed intermittent 100% GPU usage, so do not assume the run is
stuck merely because stdout is quiet or a single `nvidia-smi` sample is low.

Next session should first check whether that command finished and inspect:

- `/tmp/radix_capacity_fixed_gpu2_200k_20_pad3_blocks128`
- `/tmp/galaxy_gpu2_capacity_fixed_200k_20_pad3_blocks128.npz`

Important report fields:

- `runtime_max_leaf_size`
- `runtime_max_leaves`
- `runtime_max_nearfield_blocks`
- `runtime_max_nearfield_target_block_slots`
- `runtime_large_n_same_topology_refresh_hits`
- `runtime_large_n_same_topology_refresh_misses`
- `runtime_large_n_same_topology_refresh_miss_topology`
- `runtime_large_n_same_topology_refresh_miss_neighbor`
- `shape_signature_stable_post_warmup`
- `runtime_refresh_nearfield_neighbor_padding_seconds`
- `runtime_refresh_nearfield_target_blocks_seconds`

### Recompilation wiring recovered from previous session

There are two separate recompilation-control layers in the current code, and both
should be preserved while debugging the 200k run.

ODISSEO side:

- `odisseo/jaccpot_coupling.py` builds one solver config with one global
  `fixed_max_leaf_size` contract tied to the requested FMM leaf size.
- `_prepare_or_refresh_state(...)` calls `solver.refresh_prepared_state(...)`
  when a prior prepared state exists and falls back to full prepare only if the
  refresh API is unavailable or signature attempts fail.
- Several refresh call signatures are attempted intentionally. This keeps
  ODISSEO compatible with local jaccpot API changes while still preferring the
  refresh path.
- `_prepared_state_shape_signature(...)` records dtype/shape signatures for
  every array leaf of the prepared state.
- Timing reports include:
  `shape_signature_stable_post_warmup`,
  `shape_signature_diff_post_warmup`,
  `runtime_compiled_profile_transitions`,
  and the jaccpot refresh reuse tiers.

jaccpot side:

- `solver.refresh_prepared_state(...)` forwards directly to
  `runtime/_fmm_impl.py::refresh_prepared_state(...)`.
- `refresh_prepared_state(...)` computes a compiled-profile fingerprint before
  and after refresh.
- It first attempts `_refresh_large_n_same_topology(...)`. For
  `capacity_fixed_depth`, that path uses the capacity-fixed radix template
  helper to refresh particle order/ranges while preserving the sparse topology.
- If same-topology refresh returns `None`, it falls back to `prepare_state(...)`.
  That fallback is where the interrupted 200k default-padding run was compiling
  the dual-tree walk.
- After refresh/fallback, jaccpot classifies reuse as:
  `refresh_prepare_reuse_tier_full` for exact profile fingerprint match,
  `refresh_prepare_reuse_tier_topology` when the new profile fits inside the
  old padded capacity envelope, or
  `refresh_prepare_reuse_tier_overflow` when it exceeds the old envelope.
- `rebuild_topology_in_place(...)` is currently a wrapper around
  `refresh_prepared_state(...)` for the large-N GPU profile, so the same profile
  accounting applies.

Implication for the 200k galaxy run: fixed topology alone is not enough. The
prepared-state arrays also need stable or capacity-compatible padded shapes
across refreshes, especially neighbor edges, target-owned blocks, overflow
blocks, leaf particle slots, and max leaves/nodes. If the report shows
`shape_signature_stable_post_warmup == false` or nonzero
`runtime_refresh_prepare_reuse_tier_overflow`, increase the corresponding
padding/profile envelope rather than treating it as a tree correctness failure.

### Investigation Update - 2026-04-28

The long GPU 2 200k run was killed intentionally. It was not producing a useful
acceptance signal because we needed to understand why ODISSEO was much slower
than isolated jaccpot.

Key finding: the slow path is caused by capacity-fixed refresh misses. When
`_refresh_large_n_same_topology(...)` misses, jaccpot falls back into full
`prepare_state(...)`, which recompiles/rebuilds the expensive dual/downward path.
That is why even small ODISSEO diagnostics took more than two minutes.

Small diagnostics on GPU 2:

1. `20k`, `2` steps, default `t_end_gyr=2.0`, padding `3.0`, target `256`:

```text
runtime_large_n_same_topology_refresh_hits: 0
runtime_large_n_same_topology_refresh_miss_topology: 1
last_error: positions exceed template bounds
refresh prepare elapsed: ~53.9 s
```

2. `20k`, `2` steps, default `t_end_gyr=2.0`, padding `10.0`, target `256`:

```text
runtime_large_n_same_topology_refresh_hits: 0
runtime_large_n_same_topology_refresh_miss_topology: 1
last_error: capacity template refresh overflow, count=288, capacity=256
refresh prepare elapsed: ~53.7 s
```

3. `20k`, `2` steps, default `t_end_gyr=2.0`, padding `20.0`, target `128`,
after the yggdrax headroom patch:

```text
runtime_large_n_same_topology_refresh_hits: 0
runtime_large_n_same_topology_refresh_miss_topology: 1
last_error: capacity template refresh overflow, count=270, capacity=256
refresh prepare elapsed: ~54.9 s
```

This third run was deliberately harsh: with only 2 steps and the default
`t_end_gyr=2.0`, each step is 1 Gyr. The 200k/20 acceptance run uses 0.1 Gyr
steps.

4. `20k`, `2` steps, `--t-end-gyr 0.2`, padding `20.0`, target `128`, after the
yggdrax headroom patch:

```text
runtime_large_n_same_topology_refresh_hits: 1
runtime_large_n_same_topology_refresh_misses: 0
runtime_refresh_prepare_reuse_tier_full: 1
runtime_compiled_profile_transitions: 0
shape_signature_stable: true
shape_signature_stable_post_warmup: true
profiled full prepare elapsed: ~62.4 s
profiled refresh prepare elapsed: ~3.4 s
runtime_refresh_tree_upward_seconds: ~3.19 s
runtime_refresh_dual_downward_seconds: 0.0 s
runtime_refresh_nearfield_seconds: 0.0 s
```

This is the first recovered ODISSEO path that looks like the intended isolated
jaccpot behavior: refresh hits, no shape drift, no dual/downward rebuild, and no
nearfield rebuild on refresh.

Code change made in `/export/home/tbuck/yggdrax`:

- `yggdrax/_tree_impl.py`: `capacity_fixed_depth` now refines the template to
  `min(leaf_size, target_leaf_particles)` but still stores `leaf_size` as the
  refresh/runtime capacity. This lets `--fmm-tree-leaf-target 128` create
  movement headroom while `--fmm-leaf-size 256` remains the hard refresh cap.
- `tests/unit/test_tree.py`: added
  `test_capacity_fixed_depth_target_creates_refresh_headroom`.

Tests after this patch:

```bash
CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=1 \
micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k capacity_fixed_depth
```

Result:

```text
3 passed, 21 deselected in 20.91s
```

```bash
CUDA_VISIBLE_DEVICES=2 JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
micromamba run -n odisseo pytest -o addopts= \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k capacity_fixed_depth
```

Result:

```text
1 passed, 57 deselected in 33.73s
```

Recommended next 200k command should use the real acceptance timestep and the
recovered padding/headroom settings:

```bash
CUDA_VISIBLE_DEVICES=2 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION=20.0 \
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=128 \
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM=2.0 \
JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM=3.0 \
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_BOOTSTRAP_CAP=131072 \
JACCPOT_LARGE_N_OVERFLOW_PROFILE_BOOTSTRAP_CAP=65536 \
micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --t-end-gyr 2.0 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode capacity_fixed_depth \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 128 \
  --fmm-max-order 4 \
  --profile-breakdown \
  --report-dir /tmp/radix_capacity_fixed_gpu2_200k_20_pad20_target128 \
  --output /tmp/galaxy_gpu2_capacity_fixed_200k_20_pad20_target128.npz
```

Acceptance signals to check first:

- `runtime_large_n_same_topology_refresh_hits` should be near the number of
  post-initial refreshes.
- `runtime_large_n_same_topology_refresh_misses` should be zero or very low.
- `runtime_refresh_dual_downward_seconds` should stay near zero after the
  initial prepare.
- `runtime_refresh_nearfield_seconds` should stay near zero after the initial
  prepare.
- `runtime_compiled_profile_transitions == 0`.
- `shape_signature_stable_post_warmup == true`.

### GPU 8 Tuning Update - 2026-04-28 Evening

We resumed after the GPU 2 OOM and used GPU 8 for short 200k smoke runs.

Important environment note:

- GPU 8 was not totally empty: one Python process held about `1.9 GiB`.
- Runs used:
  - `CUDA_VISIBLE_DEVICES=8`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async`
- This avoided full JAX preallocation and made the runs coexist with the
  existing GPU 8 process.

#### ODISSEO memory-split fix

The 200k OOM was not caused by raw particle count. It happened in the initial
full prepare dual-tree interaction build. The critical finding was that ODISSEO
was forcing:

```text
fmm_prepare_stage_memory_split_enabled=False
```

This prevented jaccpot's `large_n_gpu` production profile from auto-enabling
the lower-peak split prepare path.

Code changed in `/export/home/tbuck/Odisseo`:

- `odisseo/option_classes.py`
  - default changed from `False` to `None`:
    `fmm_prepare_stage_memory_split_enabled: Optional[bool] = None`
- `notebooks/scalability/galaxy_disk_fmm_large_n.py`
  - added explicit CLI controls:
    - `--fmm-prepare-stage-memory-split`
    - `--no-fmm-prepare-stage-memory-split`
  - default is `None`, so jaccpot may choose the production low-memory default.

Compile check passed:

```bash
python3 -m py_compile \
  /export/home/tbuck/Odisseo/odisseo/option_classes.py \
  /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py
```

The ODISSEO API regression test was not rerun in this session because the
escalated command was rejected.

#### CPU tree-size diagnostic

A CPU-only diagnostic checked the exact 200k initial disk topology. It showed
that padding alone barely changes leaf/node count, while target headroom is the
larger topology multiplier.

Selected results:

```text
pad 1,  target 128: leaves 11369, nodes 22737, max_leaf 128, depth 4..13
pad 20, target 128: leaves 11635, nodes 23269, max_leaf 128, depth 4..16

pad 1,  target 256: leaves 7869, nodes 15737, max_leaf 256, depth 4..12
pad 20, target 256: leaves 8051, nodes 16101, max_leaf 256, depth 4..16
```

Interpretation:

- `--fmm-tree-leaf-target 128` increases topology size versus `256`.
- Large padding increases refinement depth and changes cell geometry.
- The OOM source is the dual-tree interaction/scaffold build, not raw tree
  storage.

#### GPU 8 short 200k smoke matrix

All short runs used:

```text
CUDA_VISIBLE_DEVICES=8
JAX_ENABLE_X64=1
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo
XLA_PYTHON_CLIENT_PREALLOCATE=false
TF_GPU_ALLOCATOR=cuda_malloc_async
JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION=5.0
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=64
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM=1.5
JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM=2.0
```

Common benchmark args:

```text
--mode perf
--n-particles 200000
--num-steps 2
--t-end-gyr 0.2
--fmm-preset large_n_gpu
--fmm-runtime-path large_n
--fmm-tree-build-mode capacity_fixed_depth
--fmm-refresh-every 1
--fmm-leaf-size 256
--fmm-max-order 4
--fmm-prepare-stage-memory-split
--profile-breakdown
```

Results:

1. `--fmm-tree-leaf-target 192`

Report:

```text
/tmp/radix_capacity_fixed_gpu8_200k_2_pad5_target192_splitauto/galaxy_disk_profile_20260428_211902.json
```

Outcome:

```text
script_runtime_seconds: 205.8936
same_topology_refresh_hits: 0
same_topology_refresh_misses: 1
last_error: capacity template refresh overflow: leaf=5873 count=260 capacity=256
refresh_prepare_reuse_tier_overflow: 1
compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
refresh prepare elapsed: 82.95 s
```

2. `--fmm-tree-leaf-target 160`

Report:

```text
/tmp/radix_capacity_fixed_gpu8_200k_2_pad5_target160_spliton/galaxy_disk_profile_20260428_212308.json
```

Outcome:

```text
script_runtime_seconds: 169.0257
same_topology_refresh_hits: 0
same_topology_refresh_misses: 1
last_error: capacity template refresh overflow: leaf=2889 count=259 capacity=256
refresh_prepare_reuse_tier_topology: 1
compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
refresh prepare elapsed: 72.85 s
```

3. `--fmm-tree-leaf-target 128`

Report:

```text
/tmp/radix_capacity_fixed_gpu8_200k_2_pad5_target128_spliton/galaxy_disk_profile_20260428_212658.json
```

Outcome:

```text
script_runtime_seconds: 116.6927
same_topology_refresh_hits: 1
same_topology_refresh_misses: 0
refresh_prepare_reuse_tier_full: 1
compiled_profile_transitions: 0
shape_signature_stable_post_warmup: true
runtime_refresh_tree_upward_seconds: 3.3852
runtime_refresh_dual_downward_seconds: 0.0
runtime_refresh_nearfield_seconds: 0.0
full prepare elapsed: 80.52 s
refresh prepare elapsed: 5.27 s
```

This is the first 200k ODISSEO run with the intended capacity-fixed behavior:

- no OOM,
- refresh hit,
- no capacity-template miss,
- no dual/downward rebuild on refresh,
- no nearfield rebuild on refresh,
- no compiled-profile transition,
- stable prepared-state shape signature.

#### Recommended next run

Run the full 200k/20-step acceptance case on GPU 8 using the winning short-run
settings:

```bash
CUDA_VISIBLE_DEVICES=8 \
JAX_ENABLE_X64=1 \
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot:/export/home/tbuck/Odisseo \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION=5.0 \
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=64 \
JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM=1.5 \
JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM=2.0 \
micromamba run -n odisseo python /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 200000 \
  --num-steps 20 \
  --t-end-gyr 2.0 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode capacity_fixed_depth \
  --fmm-refresh-every 1 \
  --fmm-leaf-size 256 \
  --fmm-tree-leaf-target 128 \
  --fmm-max-order 4 \
  --fmm-prepare-stage-memory-split \
  --profile-breakdown \
  --report-dir /tmp/radix_capacity_fixed_gpu8_200k_20_pad5_target128_spliton \
  --output /tmp/galaxy_gpu8_capacity_fixed_200k_20_pad5_target128_spliton.npz
```

Expected acceptance signals:

- `runtime_large_n_same_topology_refresh_hits` close to `19`,
- `runtime_large_n_same_topology_refresh_misses == 0` or very low,
- `runtime_refresh_dual_downward_seconds == 0.0` after initial prepare,
- `runtime_refresh_nearfield_seconds == 0.0` after initial prepare,
- `runtime_compiled_profile_transitions == 0`,
- `shape_signature_stable_post_warmup == true`,
- total runtime meaningfully below fallback-heavy runs.

Do not return to `padding=20` unless a bounds miss appears. The GPU 8 data says
`padding=5,target=128` is the first viable memory/performance point.

#### Performance Gap Investigation For Next Session

The successful GPU 8 `target=128` smoke fixed the catastrophic refresh fallback,
but it is still far from the pure jaccpot expectation.

Observed report:

```text
total_seconds: 116.69
prepare_seconds: 85.79
evaluate_seconds: 28.66
update_seconds: 1.45

full prepare elapsed: 80.52 s
refresh prepare elapsed: 5.27 s
runtime_refresh_tree_upward_seconds: 3.3852 s
runtime_refresh_compile_or_sync_suspect_seconds: 1.8498 s
runtime_refresh_dual_downward_seconds: 0.0 s
runtime_refresh_nearfield_seconds: 0.0 s
```

Interpretation:

- The capacity-fixed same-topology refresh is working.
- The expensive refresh fallback is gone.
- Remaining gap is still very large:
  - first full prepare is about `80 s`,
  - successful refresh prepare is about `5.3 s`,
  - full acceleration evaluate is about `14.3 s` per call.
- GPU 8 was shared, so timings may be noisy, but shared GPU use cannot by
  itself explain a subsecond pure-jaccpot path becoming tens of seconds.

Main hypothesis:

ODISSEO now has the correct topology-refresh behavior, but either:

- the exact galaxy disk geometry creates a much harder jaccpot interaction
  scaffold than the pure benchmark,
- ODISSEO is invoking a slower large-N evaluation configuration than the pure
  jaccpot benchmark,
- ODISSEO timing includes compile/synchronization costs that the pure benchmark
  excludes,
- or coupling/rematerialization/scatter around jaccpot is adding overhead.

Next session should run an apples-to-apples isolated jaccpot comparison using
the exact same ODISSEO initial positions and masses:

- `N=200000`
- same seed/disk parameters as `galaxy_disk_fmm_large_n.py`
- `preset="large_n_gpu"`
- `runtime_path="large_n"`
- `tree_build_mode="capacity_fixed_depth"`
- `leaf_size=256`
- `tree_leaf_target=128`
- `max_order=4`
- `JACCPOT_CAPACITY_FIXED_BOUNDS_PADDING_FRACTION=5.0`
- same low-memory env:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `TF_GPU_ALLOCATOR=cuda_malloc_async`
- preferably run on an empty GPU if available.

Measurements to collect separately:

1. `solver.prepare_state(...)` on the initial ODISSEO positions.
2. `solver.evaluate_prepared_state(...)` on that prepared state.
3. `solver.refresh_prepared_state(...)` after one ODISSEO-sized timestep.
4. `solver.evaluate_prepared_state(...)` after refresh.

Questions this must answer:

- Is isolated jaccpot also slow on the exact ODISSEO galaxy geometry?
- Does isolated jaccpot evaluate stay near subsecond when using
  `capacity_fixed_depth,target=128`?
- Is the `~14 s` evaluate time inside jaccpot or in ODISSEO coupling/timing?
- Is the `~3.4 s` refresh upward time caused by tree/template payload refresh,
  sorted payload rebuild, solid-FMM P2M/M2M work, or synchronization?
- Are we accidentally timing first-call compile in ODISSEO but comparing against
  warmed pure-jaccpot timings?

If isolated jaccpot is fast on the same positions:

- profile ODISSEO's `evaluate_prepared_state` wrapper/call site,
- inspect scatter/rematerialization and `block_until_ready` placement,
- compare solver config objects between ODISSEO and the pure benchmark,
- verify that the same large-N production contract is active:
  `minimum_memory`, streamed far pairs, no retained interactions, radix tree,
  solid-FMM basis.

If isolated jaccpot is also slow on the same positions:

- optimize jaccpot for the thin clustered galaxy geometry,
- inspect interaction counts:
  - `runtime_recent_dual_far_pair_count`
  - `runtime_recent_dual_neighbor_count`
  - `runtime_recent_dual_leaf_count`
  - `runtime_recent_dual_node_count`
- compare these counts to the pure benchmark case that reaches subsecond
  runtime,
- reduce evaluate cost before doing the full 20-step ODISSEO acceptance run.
