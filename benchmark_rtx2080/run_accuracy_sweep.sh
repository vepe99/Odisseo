#!/usr/bin/env bash
# =============================================================================
# FMM accuracy <-> cost curve on the RTX 2080 Ti, for the equal-accuracy-budget
# comparison against Bonsai (a Barnes-Hut monopole+quadrupole tree, theta=0.5).
# For each (order, theta) it records the sampled relative force error vs a
# direct-sum reference (--initial-accel-report) AND a rough per-step time.
# We then match the FMM order whose force error ~ Bonsai's, and do precise
# timing there. Complex basis here (production baseline); real basis timed
# separately at the matched order.
# =============================================================================
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"
OUT="$HERE/accuracy_sweep"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"
cd "$REPO"
source "$HERE/env_rtx2080.sh"

ORDERS=${ORDERS:-"1 2 3 4"}
THETAS=${THETAS:-"0.5 0.8"}
BASIS=${BASIS:-complex}

for th in $THETAS; do
  for p in $ORDERS; do
    GPU=$(bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
    tag="p${p}_th${th}_${BASIS}"
    echo "=== [$tag] GPU=$GPU ==="
    CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
      --mode perf \
      --n-particles 200000 \
      --fmm-preset large_n_gpu --fmm-runtime-path large_n \
      --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
      --no-fmm-large-n-environment-overrides \
      --fmm-basis "$BASIS" --fmm-max-order "$p" --fmm-theta "$th" \
      --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
      --initial-accel-report --initial-accel-sample-targets 4096 \
      --num-steps 50 --t-end-gyr 0.025 \
      --profile-breakdown --perf-warmup-runs 1 --perf-measure-runs 2 \
      --report-dir "$OUT/reports_$tag" \
      --output "$OUT/perf_$tag.npz" \
      > "$OUT/$tag.log" 2>&1
    echo "    rc=$? -> $OUT/$tag.log"
  done
done
echo "ALL ACCURACY-SWEEP RUNS DONE -> $OUT"
