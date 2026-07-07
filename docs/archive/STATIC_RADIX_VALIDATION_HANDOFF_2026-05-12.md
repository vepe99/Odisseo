# Static-Radix Validation Handoff - 2026-05-12

## Purpose

Document the current confidence level after the radix fast-lane overflow fix work and define the next continuation steps for production readiness.

## What Was Verified Today

### 1. Cross-repo state was reviewed

Read in full:

- `docs/STATIC_RADIX_TARGET_BLOCK_OVERFLOW_HANDOFF_2026-05-06.md`

Scanned recent status/history in:

- `Odisseo`
- `jaccpot`
- `yggdrax`

Confirmed `jaccpot` fast-lane nearfield path currently includes overflow contributions (payload fast path and fallback path).

### 2. jaccpot regression-focused tests passed

Command:

```bash
micromamba run -n odisseo python -m pytest -q tests/integration/test_fmm.py -k "radix_fast_lane_includes_overflow_target_blocks or static_radix_refresh_rebuilds_current_large_n_payloads"
```

Result:

```text
2 passed
```

### 3. Physical sanity simulation (short) passed

Command (executed from `Odisseo`):

```bash
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf \
  --n-particles 12000 \
  --num-steps 60 \
  --t-end-gyr 0.03 \
  --fmm-refresh-every 1 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-theta 0.6 \
  --fmm-leaf-size 256 \
  --fmm-max-order 4 \
  --state-dtype float64 \
  --initial-accel-report \
  --initial-accel-sample-targets 192 \
  --conservation-report \
  --conservation-stride 5 \
  --report-dir notebooks/scalability/reports \
  --output /tmp/galaxy_test_default_overrides_12k.npz
```

Generated artifacts:

- `notebooks/scalability/reports/galaxy_disk_initial_acceleration_20260512_143030.json`
- `notebooks/scalability/reports/galaxy_disk_profile_20260512_143115.json`
- `notebooks/scalability/reports/galaxy_disk_conservation_20260512_143134.json`
- `/tmp/galaxy_test_default_overrides_12k.npz`

Key metrics from report parsing:

- `fmm_vs_direct_rel_err.p50 = 1.512e-07`
- `fmm_vs_direct_rel_err.p90 = 9.056e-07`
- `fmm_vs_direct_rel_err.max = 4.930e-06`
- `max_abs_dE_over_E0 = 1.080e-03`
- `max_abs_dL_over_L0 = 4.949e-05`

Interpretation:

- No sign of the previous catastrophic "exploding galaxy" behavior in this short-run sanity test.
- Initial self-gravity acceleration agreement is now excellent on sampled targets.

## Confidence Statement

Current status is **encouraging but not production-final**:

- We have regained correctness on targeted overflow bug behavior and a short physical sanity run.
- We still need larger-scale and longer-horizon validation to claim production readiness.

## Known Remaining Risk

Overflow-heavy layouts can still be expensive. Correctness appears restored; performance under severe overflow conditions is still the main optimization risk.

## Continuation Plan (Next Session)

1. Run medium scale stability check:
   - `n=50k`, short horizon, same physical settings, collect acceleration + conservation reports.

2. Run target-scale initial acceleration check:
   - `n=200k`, sampled direct-vs-FMM initial acceleration report only first (before long integration).

3. Run short render/movie smoke at target-scale settings:
   - verify no patch-wise ejection signatures in early frames.

4. Record overflow/performance envelope across three layouts:
   - default block-size path (overflow ~0 expected),
   - block size 4 + large fast prefix (overflow ~0 expected),
   - block size 4 + small prefix (overflow > 0 stress case).

5. Decide production profile defaults in Odisseo:
   - keep safe no-overflow configuration as default until overflow-heavy performance is benchmarked acceptable.

## Suggested Command Templates for Next Session

### Medium-scale stability

```bash
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 50000 --num-steps 120 --t-end-gyr 0.06 \
  --fmm-refresh-every 1 --fmm-preset large_n_gpu --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix --fmm-theta 0.6 --fmm-leaf-size 256 \
  --fmm-max-order 4 --state-dtype float64 --initial-accel-report \
  --conservation-report --report-dir notebooks/scalability/reports \
  --output /tmp/galaxy_test_50k.npz
```

### Target-scale acceleration-only focus

```bash
micromamba run -n odisseo python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode perf --n-particles 200000 --num-steps 1 --t-end-gyr 0.001 \
  --fmm-refresh-every 1 --fmm-preset large_n_gpu --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix --fmm-theta 0.6 --fmm-leaf-size 256 \
  --fmm-max-order 4 --state-dtype float64 --initial-accel-report \
  --initial-accel-sample-targets 512 --report-dir notebooks/scalability/reports \
  --output /tmp/galaxy_test_200k_accel_only.npz
```

## Commit and Traceability Notes

Related commits created during this session:

- `jaccpot`: `ed18902` - `Refresh static-radix large-N payloads on topology reuse`
- `Odisseo`: `d447a0b` - `Document static-radix production status checkpoint`

This document captures the additional validation checkpoint performed after those commits.
