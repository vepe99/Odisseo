# Static-Radix Fused Compact-Pack Handoff (2026-05-26)

## Context

We are optimizing the large-N strict fused path to reduce host/Python orchestration before running expensive 200k/20 timing gates.

Current branch already includes:
- compiled-state compatibility (`LargeNCompiledState`) for large-N runtime,
- dedicated fused refresh route,
- frozen fused config object at fused entry,
- fused route diagnostics counters exported into Odisseo reports.

## What Was Attempted

A new "compact fused nearfield pack" path was added inside fused refresh to avoid rebuilding branch-heavy nearfield payload scaffolding each step.

Intent:
- reduce per-step host staging,
- rely on static topology + carry reuse to keep payload stable,
- increase sustained GPU occupancy.

## Current Blocker

The compact path is currently unstable under `jax.lax.scan` carry constraints.

Observed failure mode:
- strict fused 2-step runs fall back,
- reason is scan carry pytree mismatch across steps.

Most recent mismatch source:
- neighbor-list/carry subfields differ across scan iterations in the compact path,
- this violates scan requirement that carry structure/metadata be identical each step.

Representative report signal:
- `runtime_strict_fused_mode_active=False`
- `runtime_strict_fused_fallback_count=1`
- `runtime_strict_fused_last_fallback_reason` includes scan carry mismatch

## Stable Baseline Status (Important)

The non-compact fused route remains stable and validated:
- fused mode active on tiny/prod-shape checks,
- zero fallback on 200k 2-step gate,
- dedicated fused route markers incrementing as expected.

This means we already have a viable fused baseline path for further optimization and testing.

## Immediate Decision

Disable compact fused path by default **right now** and keep it behind an explicit opt-in flag.

Recommended default policy:
- `JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK=0` (default off)
- keep stable fused path as active production candidate for ongoing profiling

Rationale:
- restores deterministic fused execution quickly,
- avoids losing time to scan-structure regressions during walltime/utilization gates,
- allows compact path iteration later without destabilizing main path.

## Concrete Next Steps (when resuming)

1. Finalize compact-path gating
- Ensure compact path executes only when explicit env flag is enabled.
- Default-off in all regular fused runs.

2. Re-validate stable fused baseline (default compact off)
- tiny 2-step: fused active, fallback 0
- 20k 2-step (profile set includes 20k): fused active, fallback 0
- 200k 2-step: fused active, fallback 0

3. Run canonical throughput oracle on stable fused baseline
- 200k/20 `walltime_ab_compare.py` (fused on vs fused off)
- with `JACCPOT_LARGE_N_COMPILED_STATE_MODE=on`

4. Run utilization gate on the same lane
- single active GPU utilization sampling during run,
- confirm occupancy trend before enabling any compact path experiments.

5. Compact path v2 work (separate guarded track)
- enforce exact carry field/metadata invariance (including neighbor-list shapes and aux metadata),
- only then retest with compact flag on.

## Files Touched in This Session

Primary runtime files:
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py`
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py`

Odisseo diagnostics mapping:
- `/export/home/tbuck/Odisseo/odisseo/jaccpot_coupling.py`

## Key Reminder

Do not run large timing tests with compact path enabled until scan-carry invariance is proven.
Use stable fused baseline (compact off) for trustworthy walltime/utilization decisions.
