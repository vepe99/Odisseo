#!/usr/bin/env bash
# Near-field kernel TILING sweep (accuracy-preserving: pure XLA codegen, identical
# math/results). Tunes how the target-block P2P is tiled/unrolled -> ILP, register
# reuse, SM occupancy on the compute-bound 2080 Ti. Timing-only (no accel report;
# results are bit-identical across these). Baseline leaf_batch32/tile8/unroll1/1 = 357ms.
#   env: LEAF_BATCH / TILE / TU (tile_scan_unroll) / BU (batch_scan_unroll)
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/tiling_sweep"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto

run() { # tag leafbatch tile tu bu
  local tag="$1" lb="$2" tile="$3" tu="$4" bu="$5"
  local GPU; GPU=$(PICK_GPU_MIN_FREE_MB=9000 bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [$tag] leaf_batch=$lb tile=$tile tu=$tu bu=$bu GPU=$GPU ==="
  JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE="$lb" \
  JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE="$tile" \
  JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL="$tu" \
  JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL="$bu" \
  CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order 4 --fmm-theta 0.8 \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --num-steps 60 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_$tag" --output "$OUT/perf_$tag.npz" \
    > "$OUT/$tag.log" 2>&1
  echo "    rc=$?"
}
# baseline (lb32/t8/1/1) already = 357ms; probe variations:
run t16_11_lb32   32 16 1 1
run t8_21_lb32    32 8  2 1
run t8_41_lb32    32 8  4 1
run t8_22_lb32    32 8  2 2
run t8_11_lb64    64 8  1 1
run t16_22_lb64   64 16 2 2
echo "TILING SWEEP DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
print("\n  tag            ms/step   (baseline lb32/t8/u1/1 = 357ms)")
for d in sorted(glob.glob("benchmark_rtx2080/tiling_sweep/reports_*")):
    tag=d.split("reports_")[-1]
    pf=glob.glob(f"{d}/*profile*.json")
    ms=json.load(open(sorted(pf)[-1])).get("perf_measured_median_step_seconds") if pf else None
    print(f"  {tag:14} {(f'{ms*1e3:7.1f}' if ms else '  ---')}")
PY
