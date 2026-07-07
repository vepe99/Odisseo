# Static-Radix Galaxy-Disk Handoff - 2026-05-06

## Current Branches

Feature branch in all three local repos:

```text
feat/static-radix-memory-fit
```

Touched repos in this handoff:

- `/export/home/tbuck/Odisseo`
- `/export/home/tbuck/jaccpot`

`/export/home/tbuck/yggdrax` is on the branch but has no new edits from this
round.

## Important Correction

The physically correct optimized static-radix production path is:

```text
--fmm-refresh-every 1
```

That means:

- build one persistent jaccpot solver,
- perform one cold full static-radix prepare,
- refresh the static-radix FMM prepared state at every integration step,
- evaluate self gravity after every refresh,
- advance exactly one step from the current FMM field.

Do not use larger `fmm_refresh_every` values for final physical galaxy-disk
timings unless explicitly benchmarking stale-field approximations. The earlier
`refresh20` movie was fast but physically wrong because it reused one FMM
self-gravity evaluation for 20 integration steps.

## ODISSEO Driver Fix

`notebooks/scalability/galaxy_disk_fmm_large_n.py` render mode now uses a
persistent static-radix solver instead of recreating the integrator/solver per
snapshot chunk.

The corrected render loop records:

- `used_persistent_static_solver`
- `full_prepare_calls`
- `refresh_prepare_successes`
- `runtime_static_radix_refresh_hits`
- `runtime_static_radix_refresh_misses`
- reprofile/compiled-profile counters

The script also now defaults to the intended optimized large-N static path:

```text
fmm_preset=large_n_gpu
fmm_runtime_path=large_n
fmm_tree_build_mode=static_radix
fmm_theta=0.8
fmm_leaf_size=256
```

It also exposes full explicit traversal capacities:

```text
--fmm-max-pair-queue
--fmm-pair-process-block
--fmm-max-interactions-per-node
--fmm-max-neighbors-per-leaf
```

## Rendering Update

The movie writer now has a lower-overhead density backend inspired by the
astronomix frame-streaming style:

```text
--render-backend density
--render-resolution 900
--render-cmap magma
```

The simulation evolves all particles. Rendering samples up to
`--snapshot-max-particles` by deterministic index stride, bins sampled projected
positions into a 2D density image, log-scales the density with `log1p`, and
streams frames to GIF/MP4 with `imageio`.

## Validated Physically Correct Run

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
JACCPOT_NVIDIA_SMI_GPU_INDEX=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode render \
  --n-particles 200000 \
  --num-steps 200 \
  --fmm-refresh-every 1 \
  --snapshot-stride 1 \
  --snapshot-chunk-steps 20 \
  --snapshot-max-particles 50000 \
  --movie-path /tmp/odisseo_galaxy_runs/movies/galaxy_disk_200k_static_persistent_refresh1_200steps_density.gif \
  --movie-fps 24 \
  --render-backend density \
  --render-resolution 900 \
  --output /tmp/odisseo_galaxy_runs/galaxy_disk_200k_static_persistent_refresh1_200steps.npz \
  --save-snapshots \
  --snapshot-output /tmp/odisseo_galaxy_runs/snapshots/galaxy_disk_200k_static_persistent_refresh1_200steps_snapshots.npz \
  --report-dir /tmp/odisseo_galaxy_runs/reports \
  --fmm-max-pair-queue 524288 \
  --fmm-pair-process-block 256 \
  --fmm-max-interactions-per-node 16384 \
  --fmm-max-neighbors-per-leaf 8192
```

Output:

```text
Movie:
/tmp/odisseo_galaxy_runs/movies/galaxy_disk_200k_static_persistent_refresh1_200steps_density.gif

Timing JSON:
/tmp/odisseo_galaxy_runs/reports/galaxy_disk_profile_20260506_171718.json

NPZ:
/tmp/odisseo_galaxy_runs/galaxy_disk_200k_static_persistent_refresh1_200steps.npz

Snapshots:
/tmp/odisseo_galaxy_runs/snapshots/galaxy_disk_200k_static_persistent_refresh1_200steps_snapshots.npz
```

Key result:

```text
total_seconds: 120.013
prepare_seconds: 85.741
evaluate_seconds: 29.344
update_seconds: 2.212

prepare_calls: 200
evaluate_calls: 200
update_calls: 200
full_prepare_calls: 1
refresh_prepare_successes: 199

runtime_static_radix_refresh_hits: 199
runtime_static_radix_refresh_misses: 0
runtime_large_n_same_topology_refresh_hits: 199
runtime_large_n_same_topology_refresh_misses: 0
runtime_compiled_profile_transitions: 0
runtime_large_n_overflow_profile_reprofiles: 0
runtime_large_n_neighbor_edges_profile_reprofiles: 0

effective large-N env:
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16
```

The result is slower than the earlier pure perf document because this run also
collects 200 sampled frames and writes a movie/snapshot payload. It is the
correct refresh-every-step static-radix path.

## Memory-Fit Findings

For the flattened galaxy disk, the 1M-particle path needs the explicit traversal
seed:

```text
max_pair_queue=524288
process_block=256
max_interactions_per_node=16384
max_neighbors_per_leaf=8192
leaf_size=256
theta=0.8
```

See:

```text
docs/STATIC_RADIX_MEMORY_FIT_RESULTS_2026-05-06.md
```

Attempts at 2M particles with that seed and with the larger historical seed

```text
max_pair_queue=1048576
process_block=256
max_interactions_per_node=32768
max_neighbors_per_leaf=16384
```

still overflowed the pair queue at `theta=0.8`. A later 2M attempt with
`theta=1.0` was started but stopped when we noticed the physical movie issue.
Do not treat 2M/4M as validated yet.

## Open Issue: Galaxy Disk Explodes

The current 200k refresh-every-step movie still shows the galaxy disk exploding.
This should not happen and reportedly did not happen before the static-radix
optimization work.

Next session should prioritize physical correctness before more scaling runs.

Suggested investigation order:

1. Reproduce a small/medium stable baseline with the older trusted settings or
   direct-sum/FMM comparison notebook.
2. Compare `theta`, softening, timestep, mass normalization, external NFW setup,
   and the quasi-circular velocity initialization against pre-optimization runs.
3. Run a short 200k static-radix `--mode perf --profile-breakdown` with
   conservation reporting if feasible.
4. Compare accelerations on the initial disk between direct sum or an older
   trusted jaccpot path and the optimized static-radix path.
5. Only resume 1M/2M/4M timing after the 200k movie is physically stable.

Known-important distinction:

- `refresh_every=1` is required for physically correct dynamic self-gravity.
- Larger refresh cadences are stale-field approximations and should not be used
  for final physical claims.

## Verification Commands Run

```bash
python3 -m py_compile \
  /export/home/tbuck/Odisseo/odisseo/option_classes.py \
  /export/home/tbuck/Odisseo/odisseo/jaccpot_coupling.py \
  /export/home/tbuck/Odisseo/odisseo/integration_api.py \
  /export/home/tbuck/Odisseo/notebooks/scalability/galaxy_disk_fmm_large_n.py \
  /export/home/tbuck/jaccpot/examples/benchmark_utils.py \
  /export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py \
  /export/home/tbuck/jaccpot/examples/profile_prepare_memory_split.py
```

All listed files compiled successfully.
