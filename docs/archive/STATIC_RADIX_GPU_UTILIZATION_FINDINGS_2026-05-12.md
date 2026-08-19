# Static-Radix GPU Utilization Findings - 2026-05-12

## Objective

Verify medium-scale physical stability while measuring where runtime is spent and whether the selected single GPU is compute-saturated.

## Run Configuration

Single-GPU selection:

- `autocvd(num_gpus=1)` selected physical GPU `8`
- run pinned to `CUDA_VISIBLE_DEVICES=8`

Simulation command:

```bash
micromamba run -n odisseo python /export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py \
  --cuda-visible-devices 8 -- \
  python notebooks/scalability/galaxy_disk_fmm_large_n.py \
    --mode perf \
    --n-particles 20000 \
    --num-steps 80 \
    --t-end-gyr 0.04 \
    --fmm-refresh-every 1 \
    --fmm-preset large_n_gpu \
    --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix \
    --fmm-theta 0.6 \
    --fmm-leaf-size 256 \
    --fmm-max-order 4 \
    --state-dtype float64 \
    --initial-accel-report \
    --initial-accel-sample-targets 256 \
    --conservation-report \
    --conservation-stride 5 \
    --profile-breakdown \
    --report-dir notebooks/scalability/reports \
    --output /tmp/galaxy_test_20k_gpu8_profile.npz
```

GPU utilization sampling (1 Hz):

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -i 8 -l 1
```

## Artifacts

- `notebooks/scalability/reports/galaxy_disk_initial_acceleration_20260512_164232.json`
- `notebooks/scalability/reports/galaxy_disk_profile_20260512_164350.json`
- `notebooks/scalability/reports/galaxy_disk_conservation_20260512_164411.json`
- `/tmp/galaxy_test_20k_gpu8_profile.npz`
- `/tmp/gpu_util_50k_profile_20260512_164145.csv`

## Stability/Correctness Result

No blow-up signature in this medium run.

Initial acceleration accuracy (sampled direct vs FMM):

- `rel_err p50 = 1.520852e-07`
- `rel_err p90 = 9.281299e-07`
- `rel_err max = 9.451030e-06`

Conservation (80-step run):

- `max_abs_dE_over_E0 = 1.440531e-03`
- `max_abs_dL_over_L0 = 7.365395e-05`
- `max_com_drift = 7.970634e-04`

## Performance Result

Wall-clock:

- `script_runtime_seconds = 78.38 s`

Top-level timing split:

- `prepare_seconds = 67.29 s`
- `evaluate_seconds = 9.79 s`
- `update_seconds = 0.86 s`

Interpretation: runtime is strongly prepare-bound.

Refresh internals (dominant contributors):

- `runtime_refresh_tree_upward_seconds = 45.14 s`
- `runtime_refresh_dual_split_shared_far_near_seconds = 9.04 s`
- `runtime_refresh_dual_artifact_build_seconds = 9.04 s`
- `runtime_refresh_dual_downward_compute_seconds = 9.57 s`
- `runtime_refresh_nearfield_seconds = 2.94 s`

## GPU Utilization Result

From `/tmp/gpu_util_50k_profile_20260512_164145.csv`:

- samples: `148`
- `gpu_util_mean = 8.52%`
- `gpu_util_p50 = 5%`
- `gpu_util_p90 = 23%`
- `gpu_util_max = 28%`
- fraction with `util < 1%`: `40.5%`

Interpretation: GPU is idle much of the time; bottleneck is dominated by host-side orchestration/dispatch and/or many small kernels in refresh-preparation stages rather than sustained dense compute.

## Immediate Optimization Plan

1. Minimize refresh-stage host overhead on static-radix path.
   - Focus first on the `runtime_refresh_tree_upward_seconds` hotspot.
   - Audit whether per-refresh work can stay device-resident and fused.

2. Reduce dual-traversal planning overhead in refresh path.
   - Target `runtime_refresh_dual_split_shared_far_near_seconds` and dual artifact build stages.

3. Confirm kernel granularity issue.
   - Capture a JAX/XLA trace or Nsight Systems pass for one short run to count kernels and identify launch-latency dominated sections.

4. Keep current correctness-safe configuration while optimizing.
   - Overflow profile remained zero in this run, and correctness looked good.

5. Re-test utilization after each optimization commit.
   - Primary KPI: sustained GPU utilization increase and lower `prepare_seconds` at fixed physics settings.

## Next Suggested Experiment

Run an A/B with identical physics and only refresh cadence changed (`--fmm-refresh-every 1` vs `2`) to estimate how much runtime is tied to refresh frequency versus unavoidable per-step work. This is a measurement step only; production physics policy can remain at refresh=1 if needed.
