# Static-Radix Near-Single-JIT Execution Plan (2026-05-13)

## Summary
This plan drives the strict static-radix production lane (`large_n_gpu + static_radix`) toward a near-single-JIT execution shape with fail-fast correctness contracts and minimal host orchestration.

Primary outcome targets:
- strict static lane is the production default path
- strict lane fails fast on cap/profile mismatch (no retry growth)
- refresh-heavy execution removes avoidable Python/host control flow
- physics/correctness behavior remains stable

## Implementation Plan

### 1) Jaccpot strict execution API (additive)
- Keep existing APIs unchanged:
  - `prepare_state`
  - `refresh_prepared_state`
  - `evaluate_prepared_state`
- Extend with strict additive APIs:
  - completed: `strict_prepare_refresh_and_evaluate(...)`
  - next: compiled multi-step strict segment runner API that executes:
    - strict refresh/prepare
    - strict evaluate
    - `refresh_every` inner integration steps
    - repeat for full segments + tail

### 2) Strict routing decisions frozen at construction
- Resolve strict/planner/split-mode routing knobs once at solver init.
- Reuse cached booleans/predicates in hot paths.
- Keep non-strict behavior intact but avoid it in strict hot path.

### 3) Profile-keyed strict runtime contracts
- Key by topology + leaf parameter + N (+ traversal/mode-relevant dimensions).
- Reuse compiled/runtime artifacts per key.
- Strict lane behavior:
  - key miss or undersized profile -> fail-fast error
  - no fallback/retry growth in steady production
- Non-strict lane keeps existing fallback behavior.

### 4) Strict runner diagnostics (low-overhead)
- Keep timing-heavy refresh substage instrumentation optional and off by default in strict steady mode.
- Track counters:
  - strict runner compile count
  - strict runner execute count
  - strict profile-key hits/misses
  - strict fail-fast reject count

### 5) Odisseo strict-lane thinning
- Use jaccpot strict one-call APIs in strict non-profile lane.
- Keep general lanes intact.
- Preserve strict guards:
  - no `active_indices_schedule`
  - no `active_indices_fn`
  - `refresh_after_position_update=False`

### 6) Final strict compiled segment migration
- Move remaining strict segment orchestration ownership into jaccpot API boundary.
- Reduce Odisseo strict lane to minimal call + state pass-through.
- Keep external acceleration handling behavior unchanged and explicit.

## Current Status (already implemented)
- Strict one-call API added in jaccpot:
  - `strict_prepare_refresh_and_evaluate(...)`
- Strict exact-profile fail-fast contract added (configurable gate, default enabled).
- Strict diagnostics counters added and exported.
- Odisseo strict non-profile lane now routes refresh+evaluate through the new strict API.
- Targeted integration tests added/updated and passing.

## Remaining Work Items
1. Add compiled multi-step strict segment runner API in jaccpot.
2. Route strict integration cadence through that API.
3. Ensure strict lane avoids remaining host fallback branches in steady mode.
4. Extend parity tests for segment-runner path (state trajectory and invariants).
5. Run production 200k validation with `autocvd` single GPU and capture utilization/perf deltas.

## Test/Validation Plan
- Correctness parity:
  - old strict path vs new strict runner
  - acceleration parity and state trajectory parity over fixed windows
- Refresh behavior:
  - strict refresh-hit parity on artifacts/downstream invariants
  - strict key mismatch fail-fast verified
- Regression:
  - existing static-radix refresh/planner tests remain green
  - strict no-history/no-profile Odisseo lane remains green
- Performance acceptance (single autocvd GPU):
  - improved sustained utilization
  - reduced idle fraction
  - reduced refresh orchestration overhead metrics
  - no increase in strict refresh misses

## Structured Commit Sequence
1. strict API scaffolding + diagnostics + base tests
2. strict fail-fast cap/profile hardening
3. Odisseo strict lane wiring to one-call strict API
4. compiled multi-step strict segment runner API
5. Odisseo strict lane migration to segment runner API
6. final parity/perf validation + handoff documentation

## Assumptions and Defaults
- Strategy: Jaccpot API first.
- Production path:
  - `preset=large_n_gpu`
  - `runtime_path=large_n`
  - `tree_build_mode=static_radix`
  - `leaf_size=256`
- Strict mode on for production validation.
- Timing-heavy profiling off in strict steady runs unless explicitly enabled.
