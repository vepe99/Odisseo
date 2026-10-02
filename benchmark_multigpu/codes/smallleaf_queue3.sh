#!/bin/bash
# After queue2: the CSR M2L lane ON (dev worktree), per-step timings at every leaf, then a CSR-on U-curve.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue2.sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt
cd $B
DEV=/export/home/tbuck/jaccpot-smallleaf-dev-wt
run() {
  local tag=$1; shift
  local attempt=0
  while [ $attempt -lt 60 ]; do
    attempt=$((attempt+1))
    echo "[q3] $(date +%m-%d\ %H:%M:%S) waiting for an idle card for $tag (attempt $attempt)"
    unset CUDA_VISIBLE_DEVICES
    JACCPOT_WORKTREE=$DEV ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 "$@" > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
    rc=$?
    if grep -q "only 0 idle GPU" artifacts/smallleaf/probes/baseline_${tag}.log; then sleep 120; continue; fi
    echo "[q3] $(date +%m-%d\ %H:%M:%S) done $tag rc=$rc"
    grep -h "ms/step\|eval-only\|attribution\|FAILED\|flags=\[.*foreign" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-220
    return
  done
}
run p1p2b_leaf64_th0.6  --leaf 64  --theta 0.6 --modes refresh,detail
run p1p2b_leaf32_th0.6  --leaf 32  --theta 0.6 --modes refresh,detail --no-trace
run p1p2b_leaf128_th0.6 --leaf 128 --theta 0.6 --modes refresh,detail
run p1p2b_leaf256_th0.6 --leaf 256 --theta 0.6 --modes refresh,detail
run p1p2b_leaf64_th0.8  --leaf 64  --theta 0.8 --modes refresh
run p1p2b_leaf32_th0.8  --leaf 32  --theta 0.8 --modes refresh --no-trace
echo "[q3] $(date +%m-%d\ %H:%M:%S) ucurve (CSR on) start"
unset CUDA_VISIBLE_DEVICES
JACCPOT_WORKTREE=$DEV ./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 \
   --jac-thetas 0.4 0.6 0.8 --orders 4 6 --jac-leaf-single 32 64 128 256 --ref-targets 4096 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 \
   --out artifacts/smallleaf/ucurve_plummer200k_p1p2b.json > artifacts/smallleaf/probes/ucurve_p1p2b.log 2>&1
echo "[q3] $(date +%m-%d\ %H:%M:%S) ucurve done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/ucurve_p1p2b.log | cut -c1-200
