# Fused Path Recovery Handoff (2026-05-27)

## Scope
Recover strict fused runtime path in local `jaccpot`, then resume S1 forensic checks and fused-path bottleneck optimization.

## Current State (as of 2026-05-27)

- Nsight Systems user-space install succeeded:
  - `nsys` path: `/export/home/tbuck/micromamba/envs/nsight-2026/nsight-compute-2026.1.1/host/target-linux-x64/nsys`
  - version: `2026.1.1.0`
- ODISSEO audit tooling exists and works:
  - `tools/fused_audit_runner.py`
  - `tools/fused_audit_report.py`
- Nsight CSV parser in `tools/fused_audit_runner.py` was updated to handle newer column names and now produces non-null timeline metrics.

## Critical Finding

Latest fused reactivation check confirms strict fused is currently **not active**:

- Artifact:
  - `/tmp/odisseo_fused_audit/20260527_123434/fused_reactivation_check_S1/audit_summary.json`
- Key fields (baseline and variant):
  - `runtime_strict_fused_mode_active = false`
  - `runtime_strict_fused_execute_count = 0`
  - `runtime_strict_v2_execute_count = 1`
  - `runtime_strict_fused_last_fallback_reason = ""`

## Root Cause Diagnosis

In local `jaccpot` checkout, current `jaccpot/runtime/_fmm_impl.py` has a strict v2 segmented runner but **does not contain fused strict runtime implementation** (no `_strict_run_v2_fused_profile`, no strict fused dispatch/diagnostics path).

Result: fused env flags are set, but runtime cannot enter fused path because fused branch code is missing.

## Additional Runtime Integrity Note

During recovery attempts, local `jaccpot` state showed internal drift among runtime files:

- Modified files observed in local `jaccpot`:
  - `jaccpot/runtime/_large_n_nearfield.py`
  - `jaccpot/runtime/_large_n_pipeline.py`
  - `jaccpot/runtime/_large_n_types.py`
  - `jaccpot/solver.py`

An AttributeError occurred at one point (`_prepare_state_tree_upward_and_dual_downward` missing) indicating partial API mismatch across local runtime files. A compatibility shim was added in `_fmm_impl.py` to restore immediate compatibility with the current pipeline callsite.

## Performance Baseline Snapshot (non-fused strict v2 lane)

Recent strict run (not fused) example:

- Report: `/tmp/odisseo_recovery_s1/galaxy_disk_profile_20260527_120201.json`
- `script_runtime_seconds`: `73.77`
- `runtime_refresh_tree_upward_seconds`: `58.53`
- `runtime_refresh_dual_artifact_build_seconds`: `34.67`
- `runtime_refresh_nearfield_seconds`: `1.39`

This is not the fused lane we need to optimize.

## Plan For Next Session

### Phase 1: Recover fused-capable runtime state

1. Recover/reconstruct strict fused runtime code in local `jaccpot/runtime/_fmm_impl.py`.
   - Required capabilities to restore:
     - strict v2 eligibility routing into fused path
     - fused scan profile function
     - fused diagnostics counters + reason fields
2. Keep local runtime API consistency with currently modified files:
   - `_large_n_pipeline.py`
   - `_large_n_nearfield.py`
   - `_large_n_types.py`
   - `solver.py`
3. Syntax check:
   - `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

### Phase 2: Reactivation gate (must pass)

Run:

```bash
/export/home/tbuck/micromamba/envs/odisseo/bin/python tools/fused_audit_runner.py \
  --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz \
  --n-particles 200000 --num-steps 2 --state-dtype float32 \
  --leaf-size 256 --refresh-every 1 \
  --audit-mode --audit-tag fused_reactivation_check --run-class S1 \
  --emit-metadata --audit-root /tmp/odisseo_fused_audit \
  --require-autocvd --status-interval-seconds 30
```

Pass criteria:

- `runtime_strict_fused_mode_active = true`
- `runtime_strict_fused_execute_count > 0`
- `runtime_strict_fused_fallback_count = 0`

### Phase 3: Resume original optimization objective

After fused reactivation, continue bottleneck plan focused on:

1. reducing GPU idle gaps,
2. reducing `runtime_refresh_dual_artifact_build_seconds` toward warm-path millisecond-scale,
3. improving compute density via strict fused JIT/batching/vectorization behavior.

## Notes On Prior Experiments

- Attempted strict fast-path interaction-cache wiring did not improve S1 and was reverted.
- `rematerialize_between_refresh` OFF was slower in controlled S1 A/B; keep ON unless later evidence changes.
- `--fmm-prepare-stage-memory-split` probe triggered pair-queue-capacity overflow in this local runtime state.

## Key Artifacts Created Today

- Fused reactivation check (non-NSYS):
  - `/tmp/odisseo_fused_audit/20260527_123434/fused_reactivation_check_S1/audit_summary.json`
  - `/tmp/odisseo_fused_audit/20260527_123434/fused_reactivation_check_S1/audit_report.md`
- Recovery strict run report:
  - `/tmp/odisseo_recovery_s1/galaxy_disk_profile_20260527_120201.json`
- Nsight-enabled parser-fix run (when fused had previously been active in earlier state):
  - `/tmp/odisseo_fused_audit/20260527_112213/fused_forensic_newnsys_parserfix_S1/audit_summary.json`

## Recommended First Command On Return

Start by re-checking local runtime state and recent `jaccpot` modifications:

```bash
git -C /export/home/tbuck/jaccpot status --short
```

Then proceed with Phase 1 recovery above.

