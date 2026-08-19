#!/usr/bin/env bash
# Smaller-leaf test (user's idea): leaf 256 -> 128/192 deepens the tree, shrinks
# the near-field region (fewer expensive P2P interactions), shifts work to cheap
# far-field. Earlier leaf<256 failed ONLY on fixed-shape buffer caps (far-pair
# cap 131072), not physics -> raise the caps so the smaller leaf can run.
# order4/theta0.8 (baseline-accuracy config). Force error + clean per-step median.
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/leaf128_test"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto
# raise the fixed-shape caps that smaller leaves overflow
export JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=1048576       # was 131072
export JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=8388608        # was 2097152
for leaf in ${LEAVES:-96 128 160 192}; do
  GPU=$(PICK_GPU_MIN_FREE_MB=9000 bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [leaf=$leaf] GPU=$GPU far_pair_cap=$JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP ==="
  CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size "$leaf" --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order 4 --fmm-theta 0.8 \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --initial-accel-report --initial-accel-sample-targets 4096 \
    --num-steps 80 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_leaf$leaf" --output "$OUT/perf_leaf$leaf.npz" \
    > "$OUT/leaf$leaf.log" 2>&1
  echo "    rc=$?"
done
echo "LEAF128 TEST DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
print("\n  leaf   relerr_p50   ms/step  active_leaves  far_pairs  (baseline leaf256: 0.283% / 357ms)")
for leaf in [96,128,160,192,256]:
    pf=glob.glob(f"benchmark_rtx2080/leaf128_test/reports_leaf{leaf}/*profile*.json")
    af=glob.glob(f"benchmark_rtx2080/leaf128_test/reports_leaf{leaf}/*initial_acceleration*.json")
    if not pf: continue
    d=json.load(open(sorted(pf)[-1])); ms=d.get("perf_measured_median_step_seconds")
    e=json.load(open(sorted(af)[-1])).get("fmm_vs_direct_rel_err",{}).get("p50") if af else None
    print(f"  {leaf:5}  {(f'{e*100:.3f}%' if e else '-'):>10}  {(f'{ms*1e3:7.1f}' if ms else '---')}  {d.get('large_n_eval_active_leaf_count'):>12}  {d.get('runtime_recent_dual_far_pair_count')}")
PY
