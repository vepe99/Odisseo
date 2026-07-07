# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Capacity-Fixed Radix Implementation Plan - 2026-04-27

## Objective

We need ODISSEO to call jaccpot's solid-FMM large-N path with a topology that is fixed across integration refreshes, but still memory-safe for clustered galaxy disks.

The existing `fixed_depth` mode is not memory-safe for the disk because it chooses a global Morton depth from average occupancy. At 200k particles with target 256, it produced leaves with tens of thousands of particles, which blows up solid-FMM P2M and nearfield buffers.

The new mode is:

```text
capacity_fixed_depth
```

It must mean:

- keep existing `fixed_depth` behavior unchanged,
- start from a fixed-depth Morton partition,
- refine occupied cells until every materialized leaf has `count <= leaf_size`,
- materialize only occupied refined leaves, not the full uniform grid,
- reuse that fixed sparse topology as a template during ODISSEO refreshes,
- update only numerical payloads and particle range/order data during refresh.

Primary acceptance target:

```text
GPU 9, ODISSEO galaxy disk, 200k particles, 20 steps,
solid-FMM basis, leaf_size=256, no OOM, stable shapes,
refresh hits after initial prepare, faster than the current ~135s path.
```

## Why This Is The Right Shape

jaccpot's isolated >4M-particle fast path uses the LBVH/radix lane, where `leaf_size` is effectively a hard capacity. That is why it avoids OOM.

ODISSEO's failed `fixed_depth` experiment was not failing because solid-FMM or jaccpot large-N is inherently too memory hungry. It failed because current yggdrax `fixed_depth` is average-depth based:

```text
depth = ceil(log_8(ceil(N / target_leaf_particles)))
```

For a thin clustered disk, this leaves central Morton cells massively over capacity. Therefore the correct fix is not more padding/tuning. The correct fix is a capacity-compatible fixed topology.

## Repository Plan

Use matching feature branches:

- `/export/home/tbuck/yggdrax`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/jaccpot`: `feat/capacity-fixed-radix`
- `/export/home/tbuck/Odisseo`: `feat/capacity-fixed-radix`

Merge order once verified:

1. yggdrax
2. jaccpot
3. ODISSEO

Important: all three repos had dirty work before this effort. Do not revert unrelated files. Commit only relevant source/tests/docs and exclude generated caches/reports.

## yggdrax Design

### New build mode

Add `capacity_fixed_depth` as a radix build mode.

Implementation intent:

1. Compute Morton codes under fixed bounds.
2. Sort particles by `(morton_code, original_index)`.
3. Resolve the base fixed depth using the existing `target_leaf_particles`.
4. Build initial fixed-depth partitions.
5. Drop empty base leaves for the capacity mode.
6. For each occupied leaf whose count exceeds `leaf_size`, recursively split by child Morton octant.
7. Continue until every occupied leaf satisfies `count <= leaf_size`.
8. If a leaf still exceeds capacity at `max_depth`, raise a deterministic `ValueError`.
9. Build the usual radix topology from the refined leaf partitions.
10. Store `leaf_codes`, `leaf_depths`, Morton geometry flag, fixed bounds, and `leaf_size`.

Key invariant:

```text
max(node_ranges[num_internal:, 1] - node_ranges[num_internal:, 0] + 1) <= leaf_size
```

### Template refresh helper

Expose a helper:

```python
rebuild_capacity_fixed_depth_tree_from_template(
    positions,
    masses,
    template,
    *,
    return_reordered=False,
)
```

Behavior:

- Re-encode new positions using the template bounds.
- Reject positions outside template bounds.
- Re-sort particles.
- For each template leaf code/depth interval, compute the new contiguous particle range.
- Reject any leaf whose new count exceeds template `leaf_size`.
- Reject if template leaves do not cover all particles.
- Preserve structural arrays exactly:
  `parent`, `left_child`, `right_child`, leaf codes/depths, node levels, level offsets.
- Update only:
  `particle_indices`, `morton_codes`, `node_ranges`, and sorted payload arrays.

### yggdrax tests

Add focused tests in `tests/unit/test_tree.py`:

- clustered capacity test,
- topology-preserving template refresh test.

Run:

```bash
micromamba run -n odisseo pytest /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k 'capacity_fixed_depth'
```

## jaccpot Design

### Mode plumbing

Accept `capacity_fixed_depth` wherever tree modes are validated.

Route:

```text
jaccpot TreeBuilderConfig.mode == "capacity_fixed_depth"
  -> yggdrax Tree.from_particles(... build_mode="capacity_fixed_depth")
```

For this mode, `leaf_size` is a hard contract. If built/refreshed tree reports `max_leaf_size > leaf_size`, raise immediately.

### Prepared-state refresh

Extend large-N `refresh_prepared_state` behavior:

1. If mode is not `capacity_fixed_depth`, keep existing same-topology logic.
2. If mode is `capacity_fixed_depth`, refresh through the yggdrax template helper.
3. Rebuild sorted positions/masses and inverse permutation from the refreshed template.
4. Re-run solid-FMM upward payload updates using the same structural topology.
5. Rebuild downstream local/nearfield payloads.
6. Count this as a same-topology refresh hit if the refresh succeeds.
7. If template refresh fails due capacity/bounds, return `None` so existing fallback prepare path can take over.

Do not switch away from solid-FMM. Spherical/solid-harmonic accuracy is required.

### Diagnostics

Ensure runtime diagnostics expose enough to debug:

- `max_leaf_size`
- `max_leaves`
- same-topology refresh attempts/hits/misses
- miss topology / neighbor / no-key / traced counters
- refresh timing buckets

ODISSEO already surfaces many of these; add missing ones there as needed.

### jaccpot tests

Add focused integration test in `tests/integration/test_fmm.py`:

- construct clustered data,
- prepare with `tree_build_mode="capacity_fixed_depth"`,
- refresh slightly moved positions,
- assert max leaf size remains within capacity,
- assert tree mode and leaf count remain stable,
- assert same-topology refresh hit counter increments.

Run when GPU available:

```bash
CUDA_VISIBLE_DEVICES=9 micromamba run -n odisseo pytest /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k 'capacity_fixed_depth'
```

## ODISSEO Design

### User-facing CLI/config

Expose:

```bash
--fmm-tree-build-mode capacity_fixed_depth
```

Keep:

```bash
--fmm-leaf-size 256
```

as the hard capacity contract.

`--fmm-tree-leaf-target` remains a base-depth hint, not the memory-safety cap.

### Reporting

Timing/profile reports should include:

- requested tree mode,
- requested leaf size and leaf target,
- runtime max leaf size,
- runtime max leaves,
- same-topology refresh counters,
- shape signature stability fields,
- prepare event breakdown.

This lets us distinguish:

- topology-template refresh succeeding but payload refresh still slow,
- topology misses,
- neighbor-list shape changes,
- leaf-cap overflow.

## Current Implementation State

See also:

```text
docs/CAPACITY_FIXED_RADIX_STATUS_2026-04-27.md
```

As of the stop point:

### Done

- Branches created in all three repos.
- yggdrax new build mode implemented.
- yggdrax template refresh helper implemented.
- jaccpot mode validation/plumbing implemented.
- jaccpot capacity-template refresh path implemented.
- ODISSEO CLI choice added.
- ODISSEO timing report fields added.
- Focused yggdrax and jaccpot tests added.

### Already passed

```bash
python3 -m py_compile /export/home/tbuck/yggdrax/yggdrax/_tree_impl.py /export/home/tbuck/yggdrax/yggdrax/tree.py
python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py
python3 -m py_compile /export/home/tbuck/Odisseo/odisseo/jaccpot_coupling.py /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py
python3 -m py_compile /export/home/tbuck/yggdrax/tests/unit/test_tree.py
python3 -m py_compile /export/home/tbuck/jaccpot/tests/integration/test_fmm.py
```

Manual smoke tests already observed:

```text
yggdrax:
capacity_fixed_depth 142 60 64
capacity_fixed_depth 142 60 True

jaccpot on GPU 9:
prepare capacity_fixed_depth 128 129
refresh capacity_fixed_depth 128 129 1
```

### Not yet run

The following were not run because no GPU was free:

```bash
micromamba run -n odisseo pytest /export/home/tbuck/yggdrax/tests/unit/test_tree.py -k 'capacity_fixed_depth'

CUDA_VISIBLE_DEVICES=9 micromamba run -n odisseo pytest /export/home/tbuck/jaccpot/tests/integration/test_fmm.py -k 'capacity_fixed_depth'

micromamba run -n odisseo pytest /export/home/tbuck/Odisseo/tests/test_integration_api.py

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

## Tomorrow Continuation Checklist

1. Confirm branch state:

```bash
git -C /export/home/tbuck/yggdrax status --short --branch
git -C /export/home/tbuck/jaccpot status --short --branch
git -C /export/home/tbuck/Odisseo status --short --branch
```

2. Run targeted yggdrax test.
3. Run targeted jaccpot GPU 9 test when GPU is available.
4. Run ODISSEO API regression.
5. Run ODISSEO 200k/20 benchmark on GPU 9.
6. Inspect the benchmark JSON in `/tmp/radix_capacity_fixed_gpu9_200k_20`.
7. If successful, decide whether to commit in each repo.

## Debugging Guide

### If yggdrax capacity test fails

Inspect:

- `_refine_leaf_partitions_by_capacity`
- `build_capacity_fixed_depth_tree`
- `rebuild_capacity_fixed_depth_tree_from_template`

Likely issues:

- mixed-depth Morton interval boundaries,
- missing particles during template refresh,
- empty range handling in internal `node_ranges`.

### If jaccpot refresh falls back

Inspect:

- `_refresh_large_n_same_topology`
- `_rebuild_capacity_fixed_depth_tree_artifacts_from_template`
- `_large_n_neighbor_list_matches`

Likely issues:

- template refresh succeeds but neighbor list changes,
- capacity refresh key handling,
- prepared-state shape drift from nearfield payloads.

### If ODISSEO benchmark runs but remains slow

Inspect profile fields:

- `profiled_prepare_events`
- `runtime_max_leaf_size`
- `runtime_large_n_same_topology_refresh_hits`
- `runtime_large_n_same_topology_refresh_misses`
- `runtime_large_n_same_topology_refresh_miss_topology`
- `runtime_large_n_same_topology_refresh_miss_neighbor`
- `runtime_refresh_tree_upward_seconds`
- `runtime_refresh_dual_downward_seconds`
- `runtime_refresh_nearfield_seconds`

Interpretation:

- Many topology misses: template refresh is not stable enough.
- Many neighbor misses: topology is stable, but neighbor topology changes.
- Hits high but runtime high: bottleneck moved to numerical payload rebuild rather than topology compilation.
- `runtime_max_leaf_size > 256`: capacity contract broken; return to yggdrax.

## Success Criteria For First Merge

Before merging:

- targeted yggdrax tests pass,
- targeted jaccpot test passes,
- ODISSEO API regression passes,
- ODISSEO GPU 9 200k/20 benchmark completes without OOM,
- report shows `runtime_max_leaf_size <= 256`,
- report shows stable post-warmup shapes,
- report shows capacity same-topology refresh hits,
- runtime improves materially versus the ~135s LBVH/topology-rebuild baseline.
