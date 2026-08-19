#!/usr/bin/env bash
# =============================================================================
# ODISSEO jaccpot-FMM perf timing on the RTX 2080 Ti (sm_75, Pallas OFF).
# Same IC / softening / fused device-only fast-lane as the A100 bridge run, so
# the per-step number is directly comparable to the A100 perf (77.1 ms/step) and,
# scaled x4000, to the Bonsai full-run time on the same 2080 Ti.
# =============================================================================
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"
OUT="$HERE"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
cd "$REPO"
source "$HERE/env_rtx2080.sh"

GPU=$(bash "$HERE/pick_gpu.sh") || { echo "[perf] no free GPU"; exit 1; }
echo "[perf] GPU=$GPU  Pallas=$ODISSEO_FMM_USE_PALLAS  x64=$JAX_ENABLE_X64"

CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
  --mode perf \
  --n-particles 200000 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
  --no-fmm-large-n-environment-overrides \
  --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
  --num-steps 200 --t-end-gyr 0.1 \
  --profile-breakdown --perf-warmup-runs 1 --perf-measure-runs 3 \
  --report-dir "$OUT/reports_odisseo_perf" \
  --output "$OUT/perf_odisseo.npz"
echo "[perf] rc=$? DONE"
