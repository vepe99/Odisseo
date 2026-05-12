# Static-Radix Target-Block Overflow Handoff - 2026-05-06

## Context

Continue from:

```text
docs/STATIC_RADIX_GALAXY_HANDOFF_2026-05-06.md
```

The physical galaxy-disk run must use:

```text
--fmm-refresh-every 1
```

The old visually stable movie is:

```text
notebooks/scalability/galaxy_disk_gpu9.gif
```

The current exploding movie symptom is patch-wise ejection: groups of particles
are launched together early in the run. That pattern now looks consistent with
nearfield contributions missing for target-leaf/block groups.

## Key Finding

The large initial FMM acceleration error is caused by target-block overflow
nearfield contributions being omitted on the radix fast-lane evaluation path.

This is not an IC close-pair singularity and not primarily the static
target-block builder.

Evidence from `/tmp/galaxy_accel_diag.py` on the 200k disk, `theta=0.6`,
`leaf_size=256`, `state_dtype=float64`, softening `0.002` code units:

### Accurate Baseline

No ODISSEO large-N environment override:

```text
target_block_size: 32
block_padded_shape: (782, 32, 32)
overflow_active_blocks: 0
static_vs_direct_rel_err p50: 0.00327
static_vs_direct_rel_err p90: 0.04276
```

Important subtlety: this accurate baseline still uses radix fast-lane target
blocks. It is not avoiding target blocks. It is accurate because the internal
default block size 32 fits all nearfield source-leaf blocks in the padded fast
payload, so there is no overflow work.

### Minimal Bad Reproducer

Setting only:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
```

produces:

```text
target_block_size: 4
block_padded_shape: (782, 8, 4)
overflow_active_blocks: 108720
overflow_capacity: 217440
radix_source_particle_shape: (782, 32, 256)
static_vs_direct_rel_err p50: 0.880
static_vs_direct_rel_err p90: 0.971
```

The FMM self-acceleration norm is strongly undercounted relative to direct
summation, matching the patch-wise ejection failure mode.

### Static Blocks Are Not the Root Cause

Setting:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=0
```

gives the same bad result:

```text
overflow_active_blocks: 108720
static_vs_direct_rel_err p50: 0.880
```

So static prepacked target blocks are not the essential bug. The failure follows
overflow work.

### Overflow Controls the Error

Increasing the fast prefix reduces overflow and improves accuracy:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=128
```

Result:

```text
block_padded_shape: (782, 128, 4)
overflow_active_blocks: 21597
static_vs_direct_rel_err p50: 0.202
```

Eliminating overflow restores baseline accuracy:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=256
```

Result:

```text
block_padded_shape: (782, 200, 4)
overflow_active_blocks: 0
static_vs_direct_rel_err p50: 0.00327
static_vs_direct_rel_err p90: 0.04276
```

## Suspected Code Path

In jaccpot:

```text
/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py
/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py
/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py
```

`prepare_large_n_state` builds:

- padded fast target-block payload,
- overflow target-block arrays,
- `LargeNPreparedState.nearfield_target_block_*`,
- `LargeNPreparedState.radix_fast_payload`.

The generic compiled evaluator can add padded + overflow work through
`compute_leaf_p2p_accelerations_large_n_accel_only`.

But the large-N acceleration route is locked to the radix fast lane in
`evaluate_large_n_state`, which calls:

```python
evaluate_large_n_nearfield_fast_lane(...)
```

That function currently returns only:

```python
compute_leaf_p2p_accelerations_radix_fast_lane(...)
```

This path includes:

- self leaf contribution,
- padded fast target-block source slots.

It does not add:

- `state.nearfield_target_block_source_leaf_ids`,
- `state.nearfield_target_block_valid_mask`,
- `state.nearfield_target_block_offsets`,
- the overflow target-block pair contribution.

That explains why the result is correct whenever overflow is zero and wrong in
proportion to the amount of overflow.

## 2026-05-07 Implementation Attempt

Started a focused jaccpot branch:

```text
/export/home/tbuck/jaccpot
branch: fix/radix-fast-lane-overflow
```

Current uncommitted files on that branch:

```text
jaccpot/nearfield/near_field.py
jaccpot/runtime/_large_n_nearfield.py
tests/integration/test_fmm.py
```

The attempted patch added a small wrapper for target-block pair contributions
only, then changed `evaluate_large_n_nearfield_fast_lane` to return:

```text
radix_fast_payload_nearfield + overflow_target_block_pair_acc
```

when overflow target blocks are present.

This is the straightforward correctness fix and it is useful as a guardrail:
it should make the block-size-4 overflow case physically correct again. However,
the 200k galaxy diagnostic became extremely slow once this missing work was
actually evaluated. The long diagnostic was stopped after many minutes.

Conclusion: the fast path was fast partly because it skipped a huge overflow
nearfield workload. Adding that work back through the existing generic
target-block overflow kernels is not the performance fix we want to ship.

Do not treat the current uncommitted jaccpot patch as done. It is a useful
starting point for a regression test and for proving the missing-work diagnosis,
but likely not the final implementation.

## Correctness Fix Shape

A correctness-only patch would make `evaluate_large_n_nearfield_fast_lane`
return:

```text
radix_fast_payload_nearfield + overflow_target_block_pair_acc
```

where the overflow addition uses the existing target-block pair kernels:

```text
_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl
_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl
```

or a small public wrapper if needed to keep module boundaries clean.

Be careful not to double-count self interactions. The radix fast-lane payload
already includes self leaf contribution internally. The overflow addition should
add pair-block work only.

That patch is correct in spirit, but on the 200k flattened disk it appears too
slow when `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4` creates roughly 108k overflow
blocks. The performance solution should avoid or accelerate that overflow work,
not merely append it through the slow fallback path.

## Performance Direction

The next implementation should preserve correctness while restoring the speed
target. Promising directions:

1. Prefer a correct no-overflow layout when memory allows it.

   The default radix fast-lane block size 32 produced:

   ```text
   block_padded_shape: (782, 32, 32)
   overflow_active_blocks: 0
   static_vs_direct_rel_err p50: 0.00327
   ```

   This was accurate and avoids the slow overflow path entirely.

2. Tune the fast prefix rather than forcing block size 4 with tiny prefix.

   For 200k, this also eliminated overflow:

   ```text
   JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
   JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=256
   block_padded_shape: (782, 200, 4)
   overflow_active_blocks: 0
   ```

   This may cost more memory and must be tested at larger N.

3. If overflow is unavoidable, implement a genuinely fast overflow path.

   The existing target-block overflow kernels are correct infrastructure but
   appear too slow for the large overflow count seen in the galaxy disk. A final
   optimized implementation may need an overflow payload analogous to
   `radix_fast_payload`, or a fused/padded layout that keeps source-particle
   gathers efficient without silently dropping tail blocks.

4. In ODISSEO, reconsider the automatic block-size-4 override.

   The override:

   ```text
   JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
   JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS=1
   JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=16
   ```

   is unsafe unless the resulting overflow work is either zero or evaluated by a
   fast correct path.

## Regression Test To Add

Add a jaccpot test that forces overflow:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=8
```

Then compare acceleration against either:

1. direct summation for a small deterministic distribution, or
2. an equivalent no-overflow configuration such as:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=256
```

The test should fail before the fix because the overflow contribution is
missing, and pass after the fix.

The current uncommitted jaccpot branch already contains an initial version of
such a test:

```text
test_radix_fast_lane_includes_overflow_target_blocks
```

Pytest collection was blocked in this local environment by the known yggdrax
import/x64 setup issue:

```text
OverflowError without JAX_ENABLE_X64=True
ImportError: cannot import name 'build_tree' from 'yggdrax'
```

So tomorrow's test work should first fix or document the local test invocation,
then verify that the regression fails before the final fix and passes after it.

## Temporary Workarounds

For physical galaxy-disk runs, avoid overflow until the code is fixed.

Best immediate workaround:

```text
--no-fmm-large-n-environment-overrides
```

This avoids ODISSEO's block-size-4 override and lets jaccpot use the radix
fast-lane default block size 32, which had zero overflow and accurate initial
accelerations in the 200k diagnostic.

Alternative workaround:

```text
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=4
JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS=256
```

This eliminated overflow for the 200k diagnostic, but may use substantially
more memory and must be rechecked for larger N.

## IC Generation Note

The IC velocity field and runtime external potential were previously mismatched.
The driver now has controls to generate quasi-circular velocities using the same
halo plus an analytic disk proxy matching the sampled disk mass.

Relevant file:

```text
notebooks/scalability/galaxy_disk_fmm_large_n.py
```

New/important IC controls:

```text
--ic-velocity-potential {nfw,nfw_analytic_disk}
--ic-analytic-disk-mass-factor
--ic-thick-disk-mass-fraction
--ic-thin-disk-radius-kpc
--ic-thin-disk-height-kpc
--ic-thick-disk-radius-kpc
--ic-thick-disk-height-kpc
```

Use:

```text
--ic-velocity-potential nfw_analytic_disk
--ic-analytic-disk-mass-factor 1.0
--ic-thick-disk-mass-fraction 0.0
```

for the current single sampled disk. This makes the IC circular velocities use:

- the same NFW halo parameters as the simulation,
- an analytic disk potential with mass matching the sampled live disk.

Runtime integration still uses:

```text
external_accelerations=(NFW_POTENTIAL,)
live self-gravity from FMM
```

The analytic disk is only used for IC velocity generation. It should not be
added again as a runtime external disk unless the live disk self-gravity is
changed accordingly, or the disk would be double-counted.

The diagnostic reports now record IC metadata, disk/halo parameters, state
dtype, and effective large-N environment overrides. Use these fields to verify
that IC generation and runtime parameters match before interpreting a movie.

## Other Useful Facts

Old stable movie likely used:

```text
leaf_size=64
theta=0.6
softening=0.02 kpc = 0.002 code units
expansion order=4
state/mass float64, FMM working dtype float32
external disk components in the older runtime path
```

Current accurate initial-acceleration diagnostic with the no-overflow/default
block-size path:

```text
nearest-neighbor distance p50: 0.01126 code units
nearest-neighbor / softening p50: 5.63
```

So the ICs do not show a close-particle singularity problem.

## Next Session Checklist

1. Decide what to do with the current uncommitted jaccpot patch on
   `fix/radix-fast-lane-overflow`:

   ```text
   keep as a correctness/reference patch, or replace with a faster design
   before committing
   ```

2. Establish a fast correctness target before more code:

   ```text
   accurate initial acceleration, no skipped overflow, and near production speed
   ```

3. Benchmark correct layouts first:

   ```text
   default block size 32 / overflow 0
   block size 4 + fast blocks 256 / overflow 0
   block size 4 + small prefix / overflow nonzero
   ```

   Record both accuracy and prepare/evaluate timing.

4. If using the correctness-only overflow addition, test on a smaller reproducer
   first. Do not use the full 200k galaxy diagnostic as the first feedback loop;
   it runs too long once overflow is actually computed.

5. Add and run the forced-overflow regression test once the local pytest
   `yggdrax` import issue is resolved.

6. Re-run `/tmp/galaxy_accel_diag.py` with `TARGET_BLOCK_SIZE=4` and confirm:

```text
static_vs_direct_rel_err p50 returns to roughly 0.003
overflow_active_blocks remains nonzero
```

7. Re-enable ODISSEO large-N environment overrides and rerun the initial
   acceleration report from `galaxy_disk_fmm_large_n.py`.

8. Only after the initial acceleration is accurate and the timing remains
   acceptable, rerun the short physical
   movie with matched IC generation.

## 2026-05-12 Session Update

### What We Implemented

In `jaccpot`, we added a first medium-redesign scaffold for refresh dual-artifact
planning in the static-radix production lane:

- Added refresh planner hint plumbing into
  `jaccpot/runtime/_interaction_cache.py` so dual-artifact split planning can be
  decided once and reused.
- Added runtime planner cache + diagnostics counters in
  `jaccpot/runtime/_fmm_impl.py`:
  - `refresh_dual_planner_cache_hits`
  - `refresh_dual_planner_cache_misses`
  - `refresh_dual_planner_compile_count`
  - `refresh_dual_planner_execute_count`
- Added mode flag:
  - `JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE={off,auto,on}`
  - `auto` activates only for `large_n_gpu + static_radix`.
- Added parity/instrumentation integration test:
  - `test_static_radix_refresh_dual_planner_mode_parity_and_diagnostics`

### Validation Completed

Executed in `jaccpot` test suite:

- `test_static_radix_refresh_dual_planner_mode_parity_and_diagnostics` (pass)
- key static-radix refresh and overflow regression subset (pass)

Current interpretation:

- No evidence of new correctness regression from this planner scaffold.
- "Galaxy explosion" class regression remains addressed by existing overflow
  correctness path and related tests.

### Bottleneck-Focused Continuation Plan

Dominant bottleneck remains refresh dual-artifact build
(`runtime_refresh_dual_artifact_build_seconds` and
`runtime_refresh_dual_split_shared_far_near_seconds`).

Next concrete implementation steps:

1. Replace Python-stage split-shared routing in steady refresh hits with a
   compiled planner execution function (JAX control flow).
2. Move far-pair/neighbor counting + compact index planning into the compiled
   planner body for profile-stable executions.
3. Keep fallback path unchanged for profile miss and non-static-radix modes.
4. Add A/B runtime counters for old vs new planner sections without introducing
   extra host sync on hot path.
5. Re-run single-GPU (`autocvd`, one GPU) performance lane at 200k and accept
   only if:
   - `runtime_refresh_dual_artifact_build_seconds` improves by >= 40%
   - `prepare_seconds` improves by >= 25%
   - refresh misses do not increase
   - conservation drift stays within existing run-to-run noise.

### 2026-05-12 Incremental Optimization (Steady Refresh)

Added a low-risk hot-path overhead reduction in `jaccpot` for planner cache-hit
refreshes:

- New planner hint field:
  - `suppress_substage_timing`
- New env toggle (default enabled):
  - `JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING=1`
- Behavior:
  - On planner cache-hit steady refreshes, disable dual-artifact substage timing
    callbacks to avoid repeated host-side timing callback overhead/sync in the
    hot loop.
  - Keep top-level stage timings and all correctness behavior unchanged.
- New diagnostic counter:
  - `refresh_dual_planner_steady_timing_bypass_count`

Validation:

- Planner parity/diagnostics integration test remains green and now asserts the
  new bypass counter is exercised.

## 2026-05-12 Plan Revision: Device-Resident Planning + Toward Single-JIT Step

### Goal Adjustment

We now explicitly target a mostly device-resident execution path:

- Near-term: refresh dual-artifact planning/count/fill runs in compiled JAX
  control flow for stable static-radix profiles.
- Mid-term: move the production time-integration step toward one jitted step
  function (`lax.scan`/`lax.while_loop` style), minimizing host orchestration.

Reason:

- Current kernels are often jitted individually, but stage orchestration and
  routing still spend host time and create launch/sync overhead that can keep
  GPU utilization low.

### Why Planning Is Still Required With Fixed Data Structure

`static_radix` stabilizes tree shape/capacity and avoids recompilation churn,
but active interaction sets still depend on current geometry:

- MAC acceptance changes with positions.
- Nearfield neighbor relations change as particles move.
- Compact artifact sizes/offsets must be recomputed unless we accept very large
  overprovisioned dense work.

So fixed structure removes shape instability, not per-refresh interaction
selection.

### Smarter Planning Strategy (Concrete)

1. **Two-tier planner representation**
- Keep static topology metadata cached once (node/leaf ancestry, capacity
  envelopes, static indexing tables).
- Recompute only dynamic masks/counts each refresh.

2. **Count-then-fill fused planner kernel**
- Use one compiled planner pipeline with JAX control flow:
  - pass A: compute far/near validity masks + per-block counts
  - prefix-sum/scans for offsets
  - pass B: fill compact buffers directly
- Avoid Python-stage branch routing in steady mode.

3. **Bucketed occupancy planning**
- Pre-bucket leaves/nodes by static depth and capacity class.
- Run class-wise vectorized planning kernels (`vmap`) to improve coherence and
  reduce divergent control paths.

4. **Adaptive sparse-vs-dense switch inside compiled planner**
- If occupancy exceeds threshold, select dense evaluation path for that class.
- Otherwise use compact sparse buffers.
- Keep this decision in compiled control flow, not Python.

5. **Refresh delta optimization (optional after parity)**
- Detect low-displacement refresh windows.
- Reuse prior interaction membership where validity is provably unchanged and
  recompute only invalidated regions.
- Guard behind strict correctness checks first.

### Implementation Phases

1. Phase A: compiled routing/count scaffold
- Introduce compiled planner entrypoint for split/shared routing + count arrays.
- Maintain existing fill kernels initially.

2. Phase B: compiled compact fill
- Move compact far-pair and neighbor fill into same compiled planner pipeline.
- Preserve existing fallback and overflow correctness behavior.

3. Phase C: jitted integration-step prototype
- Create optional production lane where one step loop executes in one jitted
  function with refresh + evaluate + integrate.
- Keep non-jitted orchestration path as fallback.

### Acceptance Gates

1. Correctness
- Old vs new planner parity for refresh-hit and refresh-miss paths.
- Overflow-focused tests remain green.
- Conservation and acceleration drift within current noise envelope.

2. Performance (single `autocvd` GPU)
- `runtime_refresh_dual_artifact_build_seconds` improvement target: >= 40%
- `prepare_seconds` improvement target: >= 25%
- GPU utilization: sustained increase and reduced idle fraction in refresh-heavy
  windows.

3. Stability
- No increase in `runtime_large_n_same_topology_refresh_misses`
- No planner cache churn for stable-profile production runs.

## 2026-05-12 Phase A Implementation Progress (Compiled Planner Route)

Implemented in `jaccpot`:

- Added compiled refresh planner routing entrypoint in
  `jaccpot/runtime/_interaction_cache.py`:
  - `_compiled_refresh_dual_planner_route(...)` (`jax.jit`)
  - Computes route booleans in compiled JAX control flow:
    - split-build eligibility
    - compact shared far+near eligibility
    - steady timing suppression eligibility
- Wired compiled route output into static-radix refresh planner setup in
  `jaccpot/runtime/_fmm_impl.py` (planner mode `on/auto` path).
- Added runtime diagnostic counter:
  - `refresh_dual_planner_compiled_route_count`
- Extended planner parity test to assert compiled route usage.

Validation:

- planner parity/diagnostics test: pass
- focused static-radix refresh + overflow regression subset: pass

Notes:

- This Phase A slice is a scaffold: it moves routing decisions into compiled
  execution while preserving existing fill semantics and fallback behavior.
- Next step remains Phase B: compiled compact fill/count-to-fill pipeline.
