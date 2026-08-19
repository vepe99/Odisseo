# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

# Static-Radix Production Test Result - 2026-05-12

## Run Objective

Execute the currently recommended safe production test path for galaxy-disk:

- `preset=large_n_gpu`
- `runtime_path=large_n`
- `tree_build_mode=static_radix`
- `refresh_every=1`
- one free GPU selected via `autocvd`
- disable ODISSEO large-N env overrides (avoid forced block-size-4 path)

## Command

```bash
micromamba run -n odisseo python /export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd --autocvd-num-gpus 1 -- \
  python notebooks/scalability/galaxy_disk_fmm_large_n.py \
    --mode perf \
    --n-particles 200000 \
    --num-steps 120 \
    --t-end-gyr 0.06 \
    --fmm-refresh-every 1 \
    --fmm-preset large_n_gpu \
    --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix \
    --fmm-theta 0.6 \
    --fmm-leaf-size 256 \
    --fmm-max-order 4 \
    --state-dtype float64 \
    --no-fmm-large-n-environment-overrides \
    --initial-accel-report \
    --conservation-report \
    --profile-breakdown \
    --report-dir notebooks/scalability/reports \
    --output /tmp/galaxy_prodtest_200k_safepath.npz
```

## Artifacts

- Initial acceleration report:
  - `notebooks/scalability/reports/galaxy_disk_initial_acceleration_20260512_180850.json`
- Timing profile report:
  - `notebooks/scalability/reports/galaxy_disk_profile_20260512_180758.json`
  - `notebooks/scalability/reports/galaxy_disk_profile_20260512_180758.csv`
- Conservation report:
  - `notebooks/scalability/reports/galaxy_disk_conservation_20260512_181332.json`
  - `notebooks/scalability/reports/galaxy_disk_conservation_20260512_181332.csv`

## Results

### Initial-acceleration agreement (sampled direct vs FMM)

- `fmm_vs_direct_rel_err.p50 = 2.6507e-03`
- `fmm_vs_direct_rel_err.p90 = 3.4574e-02`
- `fmm_vs_direct_rel_err.max = 2.2162e-01`

Interpretation:

- Large-N sampled initial acceleration error is in the expected FMM approximation range for this setup (not the catastrophic undercount signature seen in the earlier overflow-omission bug).

### Conservation

- `max_abs_dE_over_E0 = 3.7019e-03`
- `max_abs_dL_over_L0 = 1.4397e-04`
- `max_com_drift = 6.7450e-04`

Interpretation:

- No obvious catastrophic blow-up indicator from this report alone.
- Conservation drift is nonzero and should be judged against acceptance thresholds for production science use.

### Runtime / Bottleneck split

- `script_runtime_seconds = 2183.01 s` (~36.4 min)
- `prepare_seconds = 2061.62 s`
- `evaluate_seconds = 119.12 s`
- `update_seconds = 1.45 s`

Dominant internal refresh contributors:

- `runtime_refresh_dual_artifact_build_seconds = 1587.58 s`
- `runtime_refresh_dual_split_shared_far_near_seconds = 1587.57 s`
- `runtime_refresh_tree_upward_seconds = 75.74 s`
- `runtime_refresh_nearfield_seconds = 27.01 s`

Refresh correctness counters:

- `runtime_large_n_same_topology_refresh_hits = 119`
- `runtime_large_n_same_topology_refresh_misses = 0`
- `runtime_large_n_overflow_profile_reprofiles = 0`

Interpretation:

- The path is functionally stable (refresh hits, no overflow reprofiles), but still heavily prepare/dual-planning bound and far from desired production throughput.

## Conclusion

This safe production-path test completed and did not reproduce the previous catastrophic overflow bug signature, but runtime is still dominated by expensive refresh preparation/planning work.

For next optimization iteration, prioritize reducing refresh dual-artifact planning overhead while preserving the same stability configuration.
