#!/usr/bin/env bash
# Decisive auto-cap test WITH --profile-breakdown (emits clean per-step medians).
# Compares the pinned near-field target-block cap (64, from the A100 env) against
# data-driven "auto" right-sizing, at the production config (order4/theta0.8) and
# at the Bonsai-accuracy-matched config (order2/theta0.5).
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/autocap_test"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"

run() { # tag order theta cap
  local tag="$1" order="$2" theta="$3" cap="$4"
  local GPU; GPU=$(bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [$tag] order=$order theta=$theta cap=$cap GPU=$GPU ==="
  JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF="$cap" CUDA_VISIBLE_DEVICES="$GPU" \
    micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every 1 \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order "$order" --fmm-theta "$theta" \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --initial-accel-report --initial-accel-sample-targets 4096 \
    --num-steps 100 --perf-warmup-runs 1 --perf-measure-runs 3 --profile-breakdown \
    --report-dir "$OUT/reports_$tag" --output "$OUT/perf_$tag.npz" \
    > "$OUT/$tag.log" 2>&1
  echo "    rc=$?"
}

run p4th8_cap64   4 0.8 64
run p4th8_capauto 4 0.8 auto
run p2th5_cap64   2 0.5 64
run p2th5_capauto 2 0.5 auto
echo "AUTOCAP TEST DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
def get(tag):
    fs=glob.glob(f"benchmark_rtx2080/autocap_test/reports_{tag}/*profile*.json")
    if not fs: return (None,None)
    d=json.load(open(sorted(fs)[-1]))
    s=d.get("perf_measured_median_step_seconds")
    return (s*1e3 if s else None, d.get("large_n_radix_payload_source_leaf_shape"))
def err(tag):
    fs=glob.glob(f"benchmark_rtx2080/autocap_test/reports_{tag}/*initial_acceleration*.json")
    if not fs: return None
    return json.load(open(sorted(fs)[-1])).get("fmm_vs_direct_rel_err",{}).get("p50")
print("\n  config              cap    ms/step  payload_shape        relerr_p50")
for tag in ["p4th8_cap64","p4th8_capauto","p2th5_cap64","p2th5_capauto"]:
    ms,sh=get(tag); e=err(tag)
    cap="auto" if "auto" in tag else "64"
    cfg=tag.replace("_cap64","").replace("_capauto","")
    print(f"  {cfg:12} {cap:>8} {(f'{ms:7.1f}' if ms else '  none')}   {str(sh):18}  {(f'{e*100:.3f}%' if e else '-')}")
PY
