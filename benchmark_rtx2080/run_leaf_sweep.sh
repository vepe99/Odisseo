#!/usr/bin/env bash
# Leaf-size sweep (compute-bound regime on the 2080 Ti): leaf size trades
# near-field P2P (grows with leaf size) against tree depth / far-field / payload
# build (shrink with leaf size). Find the per-step optimum. order4/theta0.8,
# auto near-field cap, clean per-step medians (--profile-breakdown).
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/leaf_sweep"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto   # right-sized cap
LEAVES=${LEAVES:-"64 128 256 512"}
for leaf in $LEAVES; do
  GPU=$(PICK_GPU_MIN_FREE_MB=9000 bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [leaf=$leaf] GPU=$GPU ==="
  CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size "$leaf" --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order 4 --fmm-theta 0.8 \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --initial-accel-report --initial-accel-sample-targets 4096 \
    --num-steps 100 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_leaf$leaf" --output "$OUT/perf_leaf$leaf.npz" \
    > "$OUT/leaf$leaf.log" 2>&1
  echo "    rc=$?"
done
echo "LEAF SWEEP DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
print("\n  leaf   ms/step  relerr_p50  active_leaves")
for leaf in [64,128,256,512]:
    fs=glob.glob(f"benchmark_rtx2080/leaf_sweep/reports_leaf{leaf}/*profile*.json")
    af=glob.glob(f"benchmark_rtx2080/leaf_sweep/reports_leaf{leaf}/*initial_acceleration*.json")
    ms=nl=e=None
    if fs:
        d=json.load(open(sorted(fs)[-1])); s=d.get("perf_measured_median_step_seconds"); ms=s*1e3 if s else None
        nl=d.get("large_n_eval_active_leaf_count")
    if af: e=json.load(open(sorted(af)[-1])).get("fmm_vs_direct_rel_err",{}).get("p50")
    print(f"  {leaf:5} {(f'{ms:7.1f}' if ms else '   ---')}  {(f'{e*100:.3f}%' if e else '-'):>10}  {nl}")
PY
