#!/usr/bin/env bash
# Clean per-step stage decomposition via JACCPOT_LARGE_N_EVAL_DIAG_MODE (runs in
# the normal perf harness -> compile-free per-step medians, unlike the refresh
# diag-mode truncation). Modes:
#   full      = near-field P2P + far L2P + everything
#   near_zero = full minus near-field P2P  -> near-field cost = full - near_zero
#   far_zero  = full minus far L2P         -> L2P cost        = full - far_zero
#   zero      = neither eval               -> prepare/M2L/upward/tree/overhead floor
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/eval_decomp"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
ORDER=${ORDER:-4}; THETA=${THETA:-0.8}
for mode in ${MODES:-full near_zero far_zero zero}; do
  GPU=$(bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [eval=$mode] order=$ORDER theta=$THETA GPU=$GPU ==="
  JACCPOT_LARGE_N_EVAL_DIAG_MODE="$mode" CUDA_VISIBLE_DEVICES="$GPU" \
    micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order "$ORDER" --fmm-theta "$THETA" \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --num-steps 80 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_$mode" --output "$OUT/perf_$mode.npz" \
    > "$OUT/$mode.log" 2>&1
  echo "    rc=$?"
done
echo "EVAL DECOMP DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob,os
HERE=os.path.dirname(os.path.abspath("benchmark_rtx2080/x"))
base="benchmark_rtx2080/eval_decomp"
def step_ms(m):
    fs=glob.glob(f"{base}/reports_{m}/*profile*.json")
    if not fs: return None
    d=json.load(open(sorted(fs)[-1]))
    s=d.get("perf_measured_median_step_seconds")
    return s*1e3 if s else None
t={m:step_ms(m) for m in ["full","near_zero","far_zero","zero"]}
for m,v in t.items(): print(f"  {m:10} {v:8.1f} ms/step" if v else f"  {m:10}   (none)")
if t["full"] and t["near_zero"]:
    print(f"\n  near-field P2P = full - near_zero = {t['full']-t['near_zero']:.1f} ms  ({100*(t['full']-t['near_zero'])/t['full']:.0f}%)")
if t["full"] and t["far_zero"]:
    print(f"  far L2P        = full - far_zero  = {t['full']-t['far_zero']:.1f} ms  ({100*(t['full']-t['far_zero'])/t['full']:.0f}%)")
if t["full"] and t["zero"]:
    print(f"  prepare/M2L/upward/tree floor (zero) = {t['zero']:.1f} ms  ({100*t['zero']/t['full']:.0f}%)")
PY
