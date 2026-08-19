#!/usr/bin/env bash
# nsys kernel-time profile of the fast-lane step (compute-bound on 2080 Ti).
# Captures a steady-state window (delay past compile) and prints the top kernels
# by total GPU time -> names the rebuild/near-field compute hotspot to optimize.
set -uo pipefail
NSYS=/usr/local/cuda-12.4/bin/nsys
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/nsys"; mkdir -p "$OUT"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
cd "$REPO"; source "$HERE/env_rtx2080.sh"
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto
GPU=$(PICK_GPU_MIN_FREE_MB=9000 bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
echo "nsys profiling on GPU $GPU (delay 55s past compile, 35s capture)..."
CUDA_VISIBLE_DEVICES="$GPU" "$NSYS" profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --delay=55 --duration=35 \
  --force-overwrite=true -o "$OUT/faslane_step" \
  micromamba run -n odisseo python "$SIM" \
  --mode perf --n-particles 200000 \
  --fmm-preset large_n_gpu --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
  --no-fmm-large-n-environment-overrides \
  --fmm-basis complex --fmm-max-order 4 --fmm-theta 0.8 \
  --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
  --num-steps 60 --perf-warmup-runs 1 --perf-measure-runs 3 \
  --report-dir "$OUT/reports" --output "$OUT/perf.npz" \
  > "$OUT/run.log" 2>&1
echo "profile rc=$?"
echo "=== top kernels by total GPU time ==="
"$NSYS" stats --report cuda_gpu_kern_sum --format table "$OUT/faslane_step.nsys-rep" 2>/dev/null | head -35
