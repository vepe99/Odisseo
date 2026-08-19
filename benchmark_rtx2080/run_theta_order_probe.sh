#!/usr/bin/env bash
# Near/far-boundary retune probe: near-field P2P is 78% of the step and ~4x
# costlier per converted pair than far-field. Push theta outward (fewer near
# pairs) with compensating order (keep accuracy) and see if total time drops at
# fixed ~0.3% force error. Records force error (--initial-accel-report) + clean
# per-step median (--profile-breakdown).
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/theta_order_probe"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto
# (theta order) pairs to probe; baseline th0.8/o4 = 357ms/0.28% already measured
PAIRS=${PAIRS:-"0.8:6 1.0:6 1.2:8 1.5:8"}
for pair in $PAIRS; do
  th="${pair%%:*}"; p="${pair##*:}"
  GPU=$(PICK_GPU_MIN_FREE_MB=9000 bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  tag="th${th}_o${p}"
  echo "=== [$tag] GPU=$GPU ==="
  CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order "$p" --fmm-theta "$th" \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --initial-accel-report --initial-accel-sample-targets 4096 \
    --num-steps 80 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_$tag" --output "$OUT/perf_$tag.npz" \
    > "$OUT/$tag.log" 2>&1
  echo "    rc=$?"
done
echo "THETA-ORDER PROBE DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
print("\n  theta  order  relerr_p50  ms/step   (baseline th0.8/o4 = 0.283% / 357ms)")
for d in sorted(glob.glob("benchmark_rtx2080/theta_order_probe/reports_th*")):
    tag=d.split("reports_")[-1]
    pf=glob.glob(f"{d}/*profile*.json"); af=glob.glob(f"{d}/*initial_acceleration*.json")
    ms=json.load(open(sorted(pf)[-1])).get("perf_measured_median_step_seconds") if pf else None
    e=json.load(open(sorted(af)[-1])).get("fmm_vs_direct_rel_err",{}).get("p50") if af else None
    th=tag.split("_")[0][2:]; o=tag.split("_o")[-1]
    print(f"  {th:5}  {o:5}  {(f'{e*100:.3f}%' if e else '-'):>10}  {(f'{ms*1e3:7.1f}' if ms else '  ---')}")
PY
