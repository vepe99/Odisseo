# Static-Radix Galaxy-Disk Memory Fit Results - 2026-05-06

## Summary

The optimized static-radix large-N path fits the ODISSEO galaxy-disk distribution
at 1M particles on the local 11 GB RTX when the full dual-tree traversal capacity
seed from the isolated jaccpot runs is applied:

```text
max_pair_queue=524288
process_block=256
max_interactions_per_node=16384
max_neighbors_per_leaf=8192
leaf_size=256
```

The earlier failure was not a generic 500k memory limit. Uniform 500k particles
fit with the smaller minimum-memory seed, while the flattened galaxy disk
overflowed that seed's traversal pair queue immediately.

## Key Runs

### 500k Galaxy Disk, jaccpot Profiler

Input: `/tmp/odisseo_galaxy_runs/galaxy_disk_500k_profile_input.npz`

Output: `/tmp/odisseo_galaxy_runs/profile_500k_galaxy_static_radix_lean1m_seed.json`

- Result: fit
- Warm prepare retained bytes: `68784759` (~69 MB)
- Warm prepare wall time: `47.18 s`
- Warm evaluate wall time: `0.714 s`
- Warm evaluate observed peak GPU used: `3036 MB`

### 1M Galaxy Disk, jaccpot Profiler

Input: `/tmp/odisseo_galaxy_runs/galaxy_disk_1m_profile_input.npz`

Output: `/tmp/odisseo_galaxy_runs/profile_1m_galaxy_static_radix_lean1m_seed.json`

- Result: fit
- Warm prepare retained bytes: `154683297` (~155 MB)
- Warm prepare wall time: `53.90 s`
- Warm evaluate wall time: `1.345 s`
- Warm evaluate observed peak GPU used: `4620 MB`

### 200k Galaxy Disk, ODISSEO End-to-End

Output: `/tmp/odisseo_galaxy_runs/galaxy_disk_200k_one_step.npz`

Timing report:
`/tmp/odisseo_galaxy_runs/reports/galaxy_disk_profile_20260506_105505.json`

- Result: fit through ODISSEO script/API plumbing
- Total wall time: `71.43 s`
- Prepare time: `61.94 s`
- Evaluate time: `7.01 s`
- Update time: `1.55 s`
- Static prepared-state shape signature: stable
- Overflow reprofiles: `0`
- Neighbor-edge reprofiles: `0`

The ODISSEO timing includes first-call compile/staging effects. The profiler
warm evaluation timings above are the better measure of the steady evaluation
path.

## Code Changes

- ODISSEO now exposes full jaccpot traversal capacity overrides:
  - `fmm_max_pair_queue`
  - `fmm_pair_process_block`
  - `fmm_max_interactions_per_node`
  - `fmm_max_neighbors_per_leaf`
- The galaxy-disk scalability script accepts and records the two new caps.
- jaccpot profiling utilities can now load ODISSEO-generated NPZ inputs and
  persist profiler JSON output.
- jaccpot benchmark config canonicalization now preserves explicit overrides
  such as `tree_build_mode=static_radix`.

## Next

Use the verified seed for the 200k and 1M movie/timing runs, then compare direct
sum versus FMM notebooks against this explicit static-radix configuration.
