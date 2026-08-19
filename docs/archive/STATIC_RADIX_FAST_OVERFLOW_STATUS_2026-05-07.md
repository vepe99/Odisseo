# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Static-Radix Fast Overflow Status - 2026-05-07

## Context

Continuation from:

```text
docs/STATIC_RADIX_TARGET_BLOCK_OVERFLOW_HANDOFF_2026-05-06.md
```

Primary jaccpot worktree:

```text
/export/home/tbuck/jaccpot
branch: fix/radix-fast-lane-overflow
```

Goal: fix the static/radix large-N nearfield bug where radix fast-lane
acceleration omitted target-block overflow nearfield contributions.

## Current Diagnosis

The original severe galaxy-disk failure is still understood as:

- radix fast-lane nearfield includes self work plus the padded fast target-block
  prefix,
- overflow target-block pair work was omitted,
- runs were accurate whenever overflow was zero,
- forcing small target blocks created many overflow blocks and large
  acceleration undercount.

## Implemented Today

Modified jaccpot files:

```text
jaccpot/nearfield/near_field.py
jaccpot/runtime/_large_n_nearfield.py
jaccpot/runtime/_large_n_pipeline.py
jaccpot/runtime/_large_n_types.py
jaccpot/runtime/_fmm_impl.py
examples/benchmark_gpu_radix_worker.py
tests/conftest.py
tests/integration/test_fmm.py
```

### Correctness Fallback

Added:

```text
compute_leaf_p2p_accelerations_target_block_pairs_only
```

in:

```text
jaccpot/nearfield/near_field.py
```

`evaluate_large_n_nearfield_fast_lane` now computes:

```text
radix fast-lane nearfield + overflow contribution
```

when overflow exists.

This fallback is correct but too slow for the worst 200k galaxy overflow case if
used directly.

### Fast Overflow Payload Attempt

Added:

```text
compute_leaf_p2p_accelerations_radix_payload_pairs_only
LargeNPreparedState.radix_overflow_payload
```

`prepare_large_n_state` can build a separate radix-style overflow source-particle
payload when overflow exists and the payload fits:

```text
JACCPOT_LARGE_N_RADIX_OVERFLOW_PAYLOAD_MAX_MB
default: 1024
```

This is correct on small forced-overflow cases, but the 200k forced-overflow
case still ran too long. Treat this as a useful fallback/diagnostic path, not
the final production-speed answer.

### Auto-Full Fast Prefix

Added an automatic layout expansion in:

```text
jaccpot/runtime/_large_n_pipeline.py
```

The prepared fast prefix can now expand from the requested
`JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS` value to the full per-leaf block
count when both memory caps allow it:

```text
JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS=1
JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB=256
JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB=1024
```

This is now the preferred production direction: avoid overflow entirely when
the full radix payload fits. The forced-overflow tests explicitly disable this:

```text
JACCPOT_LARGE_N_SPEED_PREPARED_AUTO_FULL_BLOCKS=0
```

so they continue to exercise the overflow contribution path.

## Test Harness Fix

Fixed `tests/conftest.py`.

Old code used:

```python
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
```

which resolved to:

```text
/export/home/tbuck
```

That made `/export/home/tbuck/yggdrax` appear as a namespace package and caused:

```text
ImportError: cannot import name 'build_tree' from 'yggdrax'
```

New code uses:

```python
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
```

so pytest resolves:

```text
/export/home/tbuck/yggdrax/yggdrax/__init__.py
```

and `from yggdrax import build_tree` works.

## Verified

Syntax/whitespace:

```text
python3 -m py_compile ...
git diff --check
```

passed.

Targeted overflow tests passed:

```bash
cd /export/home/tbuck/jaccpot
env JAX_ENABLE_X64=True micromamba run -n odisseo python -m pytest \
  tests/integration/test_fmm.py \
  -k "radix_fast_lane_includes_overflow_target_blocks or large_n_prepacked_overflow_fallback_matches_tiled_overflow" \
  -q
```

Result:

```text
.. [100%]
```

Small galaxy forced-overflow diagnostic with auto-full not yet added at that
time passed accurately:

```text
n_particles: 20000
n_targets: 128
TARGET_BLOCK_SIZE=4
SPEED_PREPARED_FAST_BLOCKS=8
STATIC_TARGET_BLOCKS=0
overflow_active_blocks: 948
static_vs_direct_rel_err p50: 1.5e-7
static_vs_direct_rel_err p90: 8.6e-7
```

This proved the missing overflow contribution can be recovered correctly.

## Stopped / Not Finished

The 200k galaxy diagnostic with the separate overflow payload still did not
finish quickly enough and was stopped:

```text
n_particles: 200000
n_targets: 512
TARGET_BLOCK_SIZE=4
SPEED_PREPARED_FAST_BLOCKS=8
STATIC_TARGET_BLOCKS=0
```

A timing-only radix worker at 200k was also stopped after several minutes.

Interpretation: the separate overflow payload/fallback path is still too slow
for the worst 200k overflow layout. Production fix should prefer auto-full
prefix/no-overflow when memory allows.

## 2026-05-07 Continuation Results

After this status file was first written, work continued and the auto-full path
was validated.

Important implementation update:

- auto-full prefix expansion now depends on the compact source-leaf block layout
  cap only,
- the source-particle payload cap still controls whether `source_particle_ids`
  are materialized,
- if the source-particle payload is too large, the existing prepacked
  source-leaf fallback is used with overflow still zero.

This matters because the 200k disk with x64 indices needs roughly 1.37 GiB for
the fully expanded source-particle payload, above the default:

```text
JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB=1024
```

but the compact source-leaf layout is tiny and fits easily:

```text
block_padded_shape: (782, 200, 4)
radix_source_particle_shape: (0, 0, 0)
overflow_active_blocks: 0
```

### Additional Tests Passed

Added and passed:

```text
test_radix_fast_lane_auto_full_prefix_eliminates_overflow
```

Targeted test command:

```bash
cd /export/home/tbuck/jaccpot
env JAX_ENABLE_X64=True micromamba run -n odisseo python -m pytest \
  tests/integration/test_fmm.py \
  -k "radix_fast_lane_includes_overflow_target_blocks or radix_fast_lane_auto_full_prefix_eliminates_overflow or large_n_prepacked_overflow_fallback_matches_tiled_overflow" \
  -q
```

Result:

```text
... [100%]
```

### Small Auto-Full Diagnostic

Command shape:

```text
n_particles: 20000
n_targets: 128
TARGET_BLOCK_SIZE=4
SPEED_PREPARED_FAST_BLOCKS=8
STATIC_TARGET_BLOCKS=0
```

Result:

```text
overflow_active_blocks: 0
block_padded_shape: (79, 24, 4)
radix_source_particle_shape: (79, 96, 256)
static_vs_direct_rel_err p50: 1.47e-7
static_vs_direct_rel_err p90: 9.03e-7
```

### 200k Auto-Full Diagnostic

Default cap, 128 target reference:

```text
overflow_active_blocks: 0
block_padded_shape: (782, 200, 4)
radix_source_particle_shape: (0, 0, 0)
static_vs_direct_rel_err p50: 0.003268
static_vs_direct_rel_err p90: 0.042762
```

This matches the documented accurate baseline.

Default cap, 512 target reference:

```text
overflow_active_blocks: 0
block_padded_shape: (782, 200, 4)
radix_source_particle_shape: (0, 0, 0)
static_vs_direct_rel_err p50: 0.002651
static_vs_direct_rel_err p90: 0.034574
```

This is also in the accurate baseline band.

### Timing-Only 200k Worker

Command used the random-distribution radix worker with:

```text
TARGET_BLOCK_SIZE=4
SPEED_PREPARED_FAST_BLOCKS=8
STATIC_TARGET_BLOCKS=0
```

Result:

```text
prepare_mean_seconds: 0.279
evaluate_mean_seconds: 0.753
large_n_overflow_active_blocks: 0
large_n_block_padded_shape: [782, 200, 4]
large_n_radix_source_particle_shape: [0, 0, 0]
```

Interpretation: with auto-full eliminating overflow, the 200k large-N path is
back in a plausible production-speed regime.

## 2026-05-07 Movie Failure Follow-Up

After committing the overflow fix in jaccpot:

```text
48fa71c Fix radix fast-lane overflow nearfield
```

we reran the 200-step 200k disk movie:

```text
notebooks/scalability/galaxy_disk_static_radix_fix_200step.gif
report: notebooks/scalability/reports/galaxy_disk_profile_20260507_114926.json
runtime: 289.199 s
refresh hits: 199
profile transitions/reprofiles: 0
```

The movie still failed: stars moved outward in coherent groups and the final
state became all-NaN. That groupwise behavior pointed away from random
roundoff/noise and toward stale leaf/block/group payloads.

### Force Audit Result

Added an ODISSEO diagnostic script:

```text
tools/galaxy_force_audit.py
```

It compares FMM self-acceleration against direct summation for deterministic
sample targets plus the visually ejected particles.

At t=0, the static-radix large-N FMM result was in the expected approximation
band:

```text
prepared0:
  overflow_active_blocks: 0
  block_padded_shape: (782, 200, 4)
  radix_source_particle_shape: (0, 0, 0)

t0_self relative error:
  p50: 0.0025739413
  p90: 0.0343662742
  p99: 0.0957323751
  max: 0.2216183556
```

After one integration step, using the same solver object before the new guard,
the FMM result was catastrophically wrong:

```text
t1_self, same solver before guard:
  p50: 0.0184
  p90: 0.4949
  p99: 45.4
  max: 126217

particle 94916:
  direct acceleration norm: ~6.97
  FMM acceleration norm: ~879394
```

But a fresh solver on the exact same t=1 state was again healthy:

```text
t1_self, fresh solver:
  p50: 0.00270
  p90: 0.02933
  p99: 0.154
  max: 0.267
```

This proves the FMM mathematics and the current radix payload construction are
not inherently broken for the evolved state. The failure is in stateful
static-radix reuse.

### Second Root Cause

The same-topology refresh path in:

```text
jaccpot/runtime/_fmm_impl.py
```

rebuilt tree/upward/local numeric payloads, then copied previous large-N
nearfield data into the refreshed state:

```text
nearfield_leaf_particle_indices
nearfield_target_block_* arrays
radix_fast_payload
radix_overflow_payload
```

For `static_radix`, the topology shape is fixed while particle membership,
sorted order, geometry, and MAC/neighbor decisions can change every step. Those
copied arrays are therefore stale. The interaction cache had the same problem:
the static-radix cache key described the capacity-fixed tree shape, not the
current leaf membership or geometry, so it could reuse old dual-tree far/near
payloads with new particles.

This matches the movie symptom exactly: whole leaves/target blocks received
wrong accelerations together.

### Local Correctness Guard

Applied a local jaccpot guard in:

```text
jaccpot/runtime/_fmm_impl.py
```

Current local patch:

- static-radix large-N no longer uses the interaction cache,
- `_refresh_large_n_same_topology` returns a miss for `static_radix`, forcing
  `refresh_prepared_state` to call `prepare_state`,
- `prepare_state` rebuilds the current nearfield target-block/radix payloads.

After this patch, the same one-step audit now matches the fresh-solver result:

```text
t1_self, same solver after guard:
  p50: 0.0026998834
  p90: 0.0293329210
  p99: 0.1542345800
  max: 0.2670689148

t1_self, fresh solver after guard:
  p50: 0.0026999494
  p90: 0.0293329210
  p99: 0.1542345017
  max: 0.2670688717
```

The previous 879k acceleration outlier disappeared.

Focused jaccpot tests still pass:

```bash
cd /export/home/tbuck/jaccpot
env JAX_ENABLE_X64=True micromamba run -n odisseo python -m pytest \
  tests/integration/test_fmm.py \
  -k "radix_fast_lane_includes_overflow_target_blocks or radix_fast_lane_auto_full_prefix_eliminates_overflow or large_n_prepacked_overflow_fallback_matches_tiled_overflow" \
  -q
```

Result:

```text
... [100%]
```

### 20-Step Smoke Test With Guard

A 20-step 200k smoke movie completed with the guard:

```text
movie: notebooks/scalability/galaxy_disk_smoke_static_radix_cache_fix.gif
snapshots: /tmp/galaxy_smoke_static_radix_cache_fix_snapshots_20.npz
final: /tmp/galaxy_smoke_static_radix_cache_fix_final_20.npz
report: notebooks/scalability/reports/galaxy_disk_profile_20260507_122548.json
```

Result:

```text
script_runtime_seconds: 465.380
prepare_calls: 20
evaluate_calls: 20
prepare_seconds: 441.561
evaluate_seconds: 21.921
refresh_prepare_successes: 19
runtime_large_n_same_topology_refresh_hits: 0
runtime_large_n_same_topology_refresh_misses: 19
runtime_static_radix_refresh_hits: 0
runtime_static_radix_refresh_misses: 19
runtime_compiled_profile_transitions: 0
```

Snapshot validation:

```text
snapshot_positions shape: (21, 50000, 3)
finite_all: True
max_r at t=0.0: 12.7127686
max_r at t=0.2: 12.7128687
p99_r at t=0.0: 5.5813508
p99_r at t=0.2: 5.5818677
p999_r at t=0.0: 8.5807610
p999_r at t=0.2: 8.5809984
```

Before the guard, the equivalent 20-step snapshot probe grew to:

```text
max_r at t=0.2: ~35432
```

Interpretation: the immediate coherent group ejection is gone in the smoke.
The guard is much slower than the unsafe refresh path, so the next production
task is to make refreshed current nearfield/radix payload rebuilds fast.

## Next Steps

1. Review the jaccpot diff carefully and decide whether to keep the separate
   `radix_overflow_payload` fallback in the final patch. It is correct and
   guarded by tests, but production speed for the galaxy case now comes from
   auto-full no-overflow layout.

2. Replace the correctness guard with a genuinely fast static-radix refresh
   path. It must rebuild current large-N nearfield/radix payloads from the new
   sorted particle order and current neighbor list; it must not copy previous
   target-block/radix arrays.

3. Revisit static-radix interaction caching. A safe cache key must include a
   geometry/membership signature strong enough for far/near traversal decisions,
   or the cache must stay disabled for evolving static-radix positions.

4. Run a larger-N memory/performance sweep if needed:

```text
N = 500k, 1M
TARGET_BLOCK_SIZE=4
SPEED_PREPARED_FAST_BLOCKS=8
AUTO_FULL_BLOCKS=1
```

Check whether the compact full prefix still fits under:

```text
JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB
```

5. Re-enable ODISSEO large-N overrides and run the initial acceleration report
   from `galaxy_disk_fmm_large_n.py`.

6. Then run a longer physical movie with:

```text
--fmm-refresh-every 1
--ic-velocity-potential nfw_analytic_disk
--ic-analytic-disk-mass-factor 1.0
--ic-thick-disk-mass-fraction 0.0
```

7. If full prefix is too memory-heavy at larger N, then continue optimizing a
real overflow path.

Current likely direction:

- avoid a second full source-particle payload,
- either fold overflow into a single radix payload layout,
- or implement a compressed ragged/segmented overflow kernel that scans only
  actual per-leaf overflow slots without padding to a costly global max.

8. Once movie behavior is stable, commit jaccpot changes and update ODISSEO
   docs/runbook with the new default-safe large-N override behavior.

## Worktree Reminder

Before committing, review all uncommitted jaccpot changes carefully:

```bash
cd /export/home/tbuck/jaccpot
git status --short
git diff --stat
```

Current changes include both code and benchmark/test harness edits.
