#!/usr/bin/env bash
# Decisive clean-per-step probe of where the fast-lane step time goes.
#  (A) refresh_every sweep: prepare (tree+upward+M2L+near-field payload build) runs
#      only on refresh steps; eval (near P2P + L2P) runs every step. If per-step
#      drops as refresh_every grows, the per-step PREPARE/rebuild dominates.
#  (B) auto-cap: JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto vs pinned 64
#      -> does right-sizing the near-field target-block padding help?
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"; OUT="$HERE/prepare_probe"
IC="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
SIM="$REPO/notebooks/scalability/galaxy_disk_fmm_large_n.py"
mkdir -p "$OUT"; cd "$REPO"; source "$HERE/env_rtx2080.sh"
ORDER=4; THETA=0.8

run() { # tag  refresh_every  cap
  local tag="$1" re="$2" cap="$3"
  local GPU; GPU=$(bash "$HERE/pick_gpu.sh") || { echo "no free GPU"; exit 1; }
  echo "=== [$tag] refresh_every=$re cap=$cap GPU=$GPU ==="
  JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF="$cap" CUDA_VISIBLE_DEVICES="$GPU" \
    micromamba run -n odisseo python "$SIM" \
    --mode perf --n-particles 200000 \
    --fmm-preset large_n_gpu --fmm-runtime-path large_n \
    --fmm-tree-build-mode static_radix --fmm-leaf-size 256 --fmm-refresh-every "$re" \
    --no-fmm-large-n-environment-overrides \
    --fmm-basis complex --fmm-max-order "$ORDER" --fmm-theta "$THETA" \
    --ic-source load --ic-input-path "$IC" --no-ic-require-runtime-potential-match \
    --num-steps 100 --perf-warmup-runs 1 --perf-measure-runs 3 \
    --report-dir "$OUT/reports_$tag" --output "$OUT/perf_$tag.npz" \
    > "$OUT/$tag.log" 2>&1
  echo "    rc=$?"
}

run re1_cap64   1 64
run re2_cap64   2 64
run re4_cap64   4 64
run re8_cap64   8 64
run re1_capauto 1 auto
echo "PREPARE PROBE DONE"
micromamba run -n odisseo python - <<'PY'
import json,glob
def step_ms(tag):
    fs=glob.glob(f"benchmark_rtx2080/prepare_probe/reports_{tag}/*profile*.json")
    if not fs: return None
    d=json.load(open(sorted(fs)[-1]))
    s=d.get("perf_measured_median_step_seconds")
    if s is None:
        med=d.get("perf_measured_median_seconds"); n=d.get("num_steps"); s=med/n if med and n else None
    return s*1e3 if s else None
def shape(tag):
    fs=glob.glob(f"benchmark_rtx2080/prepare_probe/reports_{tag}/*profile*.json")
    if not fs: return None
    d=json.load(open(sorted(fs)[-1]))
    return d.get("large_n_radix_payload_source_leaf_shape")
print("\n  tag           ms/step   payload_shape")
for tag in ["re1_cap64","re2_cap64","re4_cap64","re8_cap64","re1_capauto"]:
    v=step_ms(tag); sh=shape(tag)
    print(f"  {tag:13} {(f'{v:7.1f}' if v else '   none')}   {sh}")
PY
