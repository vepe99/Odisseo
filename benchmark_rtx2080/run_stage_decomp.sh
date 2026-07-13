#!/usr/bin/env bash
# Stage decomposition of the fused fast-lane per-step time on the RTX 2080 Ti,
# via JACCPOT_STRICT_REFRESH_DIAG_MODE (truncates the refresh pipeline):
#   upward_only   = tree build + upward (P2M/M2M)
#   downward_only = + M2L + L2L         (far field)
#   full          = + eval (L2P) + near-field (P2P)
# => far-field  ~ downward_only - upward_only
#    near+L2P    ~ full          - downward_only
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"
OUT="$HERE/stage_decomp"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"

ORDER=${ORDER:-4}; THETA=${THETA:-0.8}; BASIS=${BASIS:-complex}
for mode in upward_only downward_only full; do
  GPU=$(bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [$mode] order=$ORDER theta=$THETA GPU=$GPU ==="
  JACCPOT_STRICT_REFRESH_DIAG_MODE="$mode" CUDA_VISIBLE_DEVICES="$GPU" \
    micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis "$BASIS" --fmm-max-order "$ORDER" --fmm-theta "$THETA" \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --num-steps 80 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_$mode" --output "$OUT/perf_$mode.npz" \
    > "$OUT/$mode.log" 2>&1
  echo "    rc=$?"
done
echo "STAGE DECOMP DONE -> $OUT"
micromamba run -n odisseo python - <<'PY'
import json,glob
def ms(m):
    fs=glob.glob(f"benchmark_rtx2080/stage_decomp/reports_{m}/*profile*.json")
    if not fs: return None
    d=json.load(open(sorted(fs)[-1])); s=d.get("perf_measured_median_step_seconds")
    return s*1e3 if s else None
t={m:ms(m) for m in ["upward_only","downward_only","full"]}
for m,v in t.items(): print(f"  {m:15} {(f'{v:7.1f} ms/step' if v else '  (none)')}")
if all(t.values()):
    up,dn,fu=t["upward_only"],t["downward_only"],t["full"]
    print(f"\n  tree+upward        {up:7.1f} ms  ({100*up/fu:.0f}%)")
    print(f"  far-field M2L+L2L  {dn-up:7.1f} ms  ({100*(dn-up)/fu:.0f}%)")
    print(f"  near-field+L2P     {fu-dn:7.1f} ms  ({100*(fu-dn)/fu:.0f}%)")
    print(f"  TOTAL              {fu:7.1f} ms")
PY
