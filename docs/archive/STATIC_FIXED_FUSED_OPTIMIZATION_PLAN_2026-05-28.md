# Static-Fixed Fused Optimization Plan (S3-First, GPU Utilization Gate)

## Summary

Optimize strict fused runtime for long-horizon S3 first, while keeping fixed/static production policy and user-chosen global caps.

Success criteria for this cycle:
1. `fused-on <= fused-off` walltime in S1/S2/S3.
2. Baseline fused Nsight utilization gate passes (`gpu_active_percent >= 35%`) with lower host idle p95 vs cycle-start baseline.
3. Fused path stays active with `fallback_count = 0`.

## Implementation Changes

- Policy + tooling lock (no autotune/adaptive sizing):
  - Keep fixed policy as the only benchmark lane.
  - Change fixed-policy defaults in benchmark tools to enable compiled fused execution by default:
    - `JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP=1`
    - `JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL=1`
  - Require explicit fixed cap input for neighbor edges in fixed-policy runs (CLI/env), and persist that value in run metadata so cap choice is always auditable.

- Strict fused cadence: eliminate remaining host-driven control-flow in hot loop:
  - Make compiled segment batching the canonical strict-fused path (not optional fallback-first).
  - Keep one guarded fallback path for debug only, but disable it in production gate runs.
  - Move per-segment Python-side bookkeeping out of the cadence-critical path (profile counters, side-effect writes, repeated resolver logic).

- Refresh path decomposition for device-first execution:
  - Split `_refresh_large_n_same_topology` into:
    - a pure device-refresh core used by strict fused compiled loops,
    - a host-orchestration wrapper for non-fused/debug flows.
  - Remove/avoid host synchronization checks on strict fused static-radix path during cadence (topology already fixed by policy + cap gate).

- Nearfield payload/carry stabilization for S3:
  - Enforce fixed-size overflow and neighbor capacities in fused mode (no per-step active-size carry shaping).
  - Pad-or-fail-fast only against user-selected global caps.
  - Minimize prepared-state carry to only tensors required by fused fast-lane evaluation.

## Test Plan

- Correctness/shape invariance checks:
  - Run strict fused S1 smoke with fixed policy and explicit cap; verify:
    - fused active `true`,
    - fallback `0`,
    - no cap-overflow error at chosen cap,
    - no shape-mismatch recompilation failures.

- Performance gate runs:
  - Run S1/S2/S3 A/B with identical fixed policy + identical cap.
  - Pass if all deltas satisfy `variant_minus_baseline >= 0` (fused-on not slower).

- GPU utilization gate:
  - Run Nsight-enabled fused audit at least for S3 baseline (and S2 if needed for diagnosis).
  - Pass if `gpu_active_percent >= 35%` and host idle p95 improves vs cycle-start baseline run.

- Regression guard:
  - Keep canonical IC + autocvd discipline unchanged.
  - Re-run checklist/status update after each phase with recorded run roots and gate outcomes.

## Interface/Public-Behavior Changes

- Fixed-policy benchmark behavior changes:
  - Compiled strict fused loop and compiled refresh/eval become default-on in fixed policy.
  - Fixed-policy runs require an explicit user-selected neighbor cap (per particle-count scenario), recorded in summaries.
- No solver API redesign required; changes are runtime-path and tooling-policy behavior.

## Assumptions and Defaults

- Primary objective is S3 throughput first; S1/S2 remain required gate checks.
- Autotune/adaptive sizing remains disabled for production policy.
- Neighbor cap is user-chosen and frozen per scenario; for current 200k lane, continue with `262144` unless user supplies a different value.
- Existing canonical IC path and run discipline (`micromamba -n odisseo`, `--require-autocvd`) stay unchanged.
