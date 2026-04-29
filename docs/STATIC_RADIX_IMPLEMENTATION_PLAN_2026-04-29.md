# Static Radix Implementation Plan - 2026-04-29

## Summary

Create a new matched branch set named `feat/static-radix` in:

- `/export/home/tbuck/yggdrax`
- `/export/home/tbuck/jaccpot`
- `/export/home/tbuck/Odisseo`

Replace the spatial-template `capacity_fixed_depth` path with a new
`static_radix` path.

`static_radix` means:

- recompute Morton codes from current particle positions every prepare/refresh,
- sort particles by Morton code,
- split the sorted order into fixed count chunks of size `leaf_size`,
- use a fixed balanced tree over leaf bucket indices,
- keep array shapes stable while allowing particle order, node ranges, geometry,
  and interaction contents to change.

Remove `capacity_fixed_depth` from public mode choices, implementation branches,
tests, and docs.

## yggdrax Plan

Add `static_radix` as a tree build mode.

Implementation semantics:

1. Compute Morton codes from current positions and bounds.
2. Sort particles by `(morton_code, original_index)`.
3. Create count buckets:

   ```text
   num_leaves = ceil(num_particles / leaf_size)
   leaf_i = sorted[i * leaf_size : min((i + 1) * leaf_size, N)]
   ```

4. Build a deterministic balanced binary tree over leaf bucket indices.
5. Set `node_ranges` to contiguous sorted-order ranges.
6. Set `use_morton_geometry=False`, so geometry comes from particle ranges, not
   Morton prefix cells.
7. Store `leaf_size` as the static bucket slot count and hard maximum leaf
   occupancy.

Add refresh helper:

```python
rebuild_static_radix_tree_from_template(
    positions,
    masses,
    template,
    *,
    bounds=None,
    return_reordered=False,
)
```

Refresh behavior:

- fixed `N` only,
- recompute bounds if not explicitly supplied,
- recompute Morton codes and sorted indices,
- rebuild sorted payload arrays and `node_ranges`,
- preserve parent/child arrays and total shape exactly,
- reject only particle-count or leaf-count/template incompatibility.

Remove:

- `build_capacity_fixed_depth_tree(...)`
- `rebuild_capacity_fixed_depth_tree_from_template(...)`
- `capacity_fixed_depth` build-mode literals/tests/exports.

Keep existing `fixed_depth` behavior unchanged.

## jaccpot Plan

Accept:

```text
tree_build_mode="static_radix"
```

Reject:

```text
tree_build_mode="capacity_fixed_depth"
```

Prepare path:

- route `static_radix` to yggdrax `Tree.from_particles(..., build_mode="static_radix")`,
- keep `leaf_size` as the static bucket size,
- do not use `target_leaf_particles` for `static_radix`.

Refresh path:

1. Rebuild sorted particles and static-radix tree from current positions.
2. Recompute upward solid-FMM payloads.
3. Rebuild dual/nearfield artifacts into padded profile buffers.
4. Count refresh as successful when the refreshed prepared-state profile is
   same-shape or capacity-compatible with the previous profile.
5. Do not require neighbor-list contents to match the previous state.

For `static_radix`, require:

- same number of particles,
- same leaf count/node count,
- same max leaf slots,
- prepared-state array shapes remain profile-compatible.

Diagnostics to expose:

- `static_radix_refresh_hits`
- `static_radix_refresh_misses`
- `static_radix_profile_overflows`
- existing refresh/fallback/profile transition counters.

Fallback to full prepare should be treated as an acceptance failure unless a
test intentionally permits it.

## ODISSEO Plan

Expose:

```bash
--fmm-tree-build-mode static_radix
```

Remove:

```bash
--fmm-tree-build-mode capacity_fixed_depth
```

For `static_radix`:

- `--fmm-leaf-size` controls the static chunk size and hard leaf slot count.
- `--fmm-tree-leaf-target` is ignored or normalized to `--fmm-leaf-size`.
- keep `--fmm-prepare-stage-memory-split`.

Reports should include:

- requested/runtime tree mode,
- max leaf slots,
- leaf/node count,
- refresh hits/misses,
- fallback full prepares,
- compiled-profile transitions,
- shape stability,
- interaction profile reprofile counts,
- refresh timing buckets,
- static-radix diagnostics.

## Tests

### yggdrax

- `static_radix` builds `ceil(N / leaf_size)` leaves.
- Every leaf has `count <= leaf_size`.
- Moving particles and refreshing preserves parent/child arrays, node count,
  leaf count, and array shapes.
- Refreshed Morton order changes when positions move.
- Geometry uses particle ranges, not Morton cell bounds.

### jaccpot

- `static_radix` mode validates and prepares.
- Refresh after particle motion succeeds without neighbor-list equality.
- Prepared-state shapes remain stable across refresh.
- `capacity_fixed_depth` is rejected.

### ODISSEO

- CLI accepts `static_radix`.
- CLI no longer accepts `capacity_fixed_depth`.
- API regression tests pass.

## Performance Gates

### GPU Smoke

```text
N = 200000
num_steps = 2
t_end_gyr = 0.2
leaf_size = 256
tree_build_mode = static_radix
```

Required:

- refresh hit,
- no fallback full prepare,
- no profile overflow,
- shape stable post-warmup.

### Acceptance

```text
N = 200000
num_steps = 20
t_end_gyr = 2.0
leaf_size = 256
tree_build_mode = static_radix
```

Required:

- no OOM,
- refresh hits close to 19,
- fallback full prepares = 0,
- compiled-profile transitions = 0 after warmup,
- shape stable post-warmup,
- no performance regression versus the best working capacity-fixed smoke path.

## Handoff Documentation

Maintain a rolling status doc:

```text
docs/STATIC_RADIX_STATUS_2026-04-29.md
```

Append after every meaningful session:

- branch state,
- commands run,
- test results,
- benchmark report paths,
- current blocker,
- exact next command/action.

## Assumptions

- `static_radix` leaf chunks are sized by `--fmm-leaf-size`.
- Existing `fixed_depth` remains supported and unchanged.
- `capacity_fixed_depth` is removed, not aliased.
- First implementation target is fixed `N` across refreshes, matching ODISSEO
  galaxy runs.

