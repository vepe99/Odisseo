# Static-Radix Large-N Compiled-State Handoff (2026-05-26)

## Objective

Remove Python/object orchestration from large-N prepare/refresh hot paths and move execution toward JAX-carryable compiled state with stable per-step kernels.

Primary success gate:
- `>10%` walltime improvement on canonical 200k / 20-step strict lane.

## Scope (Current Program)

- Scope includes all large-N modes (not strict-only).
- API/behavior changes are allowed when needed for throughput.
- Walltime oracle remains the canonical decision signal.

## Current Implementation Status

### Landed in jaccpot (today)

A first compatibility/foundation slice is implemented in local `jaccpot` runtime:

- New compiled-state boundary:
  - `LargeNCompiledState` added in `jaccpot/runtime/_large_n_types.py`.
  - Conversion helpers added:
    - `large_n_as_prepared_state(...)`
    - `large_n_to_compiled_state(...)`

- Pipeline compatibility:
  - `prepare_large_n_state(...)` can return compiled-state via `return_compiled_state`.
  - `evaluate_large_n_state(...)` accepts both `LargeNPreparedState` and `LargeNCompiledState` through normalization.

- Runtime routing updates:
  - `PreparedStateLike` widened to include `LargeNCompiledState`.
  - Large-N prepare/refresh/evaluate strict path accepts compiled-state shim.
  - mode flag added: `JACCPOT_LARGE_N_COMPILED_STATE_MODE` (default `on`).

### Validation completed

- Syntax:
  - `python3 -m py_compile` passed for:
    - `jaccpot/runtime/_large_n_types.py`
    - `jaccpot/runtime/_large_n_pipeline.py`
    - `jaccpot/runtime/_fmm_impl.py`

- Runtime smoke:
  - tiny strict lane (`n=256`, `steps=1`, compiled-state on): pass
  - 20k smoke (`n=20000`, `steps=2`, compiled-state on): pass
  - 20k smoke (`n=20000`, `steps=2`, compiled-state off): pass

- Quick timing sanity (non-canonical):
  - `n=20000`, `steps=2`
  - compiled-state `on`: `58.215 s`
  - compiled-state `off`: `58.453 s`
  - interpretation: neutral/slightly positive; not a meaningful throughput win yet.

## What Is Not Done Yet

The major bottlenecks remain:

- `_refresh_large_n_same_topology(...)` still drives substantial generic orchestration.
- `prepare_large_n_state(...)` still performs branch-heavy staging/object packing in hot path.
- Nearfield overflow/neighbor profiling and related host decisions are not fully removed from runtime-critical loops.
- Full canonical 200k walltime oracle has not yet shown target improvement from this slice.

## Active Rewrite Plan (Next Slices)

1. Add dedicated fused large-N refresh entrypoint
- Create a fused-only refresh path that bypasses generic `_refresh_large_n_same_topology` branches.
- Restrict fallback to outer boundary only.

2. Freeze config once per run (host planning phase)
- Resolve traversal/nearfield/chunk/tile/cap config once.
- Pass only frozen/static config into per-step execution.

3. Replace generic nearfield prep with stable-shape JAX flow
- Remove in-loop overflow/neighbor profile churn.
- Use fixed-capacity/padded tensors selected once at initialization.

4. Tighten dual/downward and tree/upward pipeline
- Reduce host-side artifact assembly and Python bookkeeping inside step loop.
- keep retries/diagnostics outside the timing-critical path.

5. Validate and gate
- Correctness parity checks across tiny, medium, and 200k lanes.
- Execution-path checks (compiled path active, no unintended fallback).
- Canonical 200k/20 walltime A/B gate (require `>10%` improvement).

## Canonical Throughput Oracle

Use only:

- `tools/walltime_ab_compare.py`

Lane:
- `fmm_preset=large_n_gpu`
- `fmm_runtime_path=large_n`
- `fmm_tree_build_mode=static_radix`
- `fmm_leaf_size=256`
- `fmm_refresh_every=1`
- fixed IC file (shared for baseline/variant)

## Current Risks

- Foundation slice expands type/API surface; strict regression checks are needed at each step.
- Large-N generic path remains partially active; performance gains may be delayed until fused refresh path is fully isolated.
- Existing local workspace is already dirty; care is required to avoid conflating unrelated edits.

## Immediate Next Action

Implement the next isolated slice:
- dedicated fused large-N refresh route with frozen config object and no generic prepare orchestration in per-step loop,
- then run a short correctness gate and one canonical walltime A/B.
