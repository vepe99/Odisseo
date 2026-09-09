#!/bin/bash
# Ordered re-queue (2026-09-09): void/contaminated configs first, then leaf 32 without the profiler trace
# (the traced leaf-32 run hung for 44 h after its full timing), then the chunk sweep, then the U-curve.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt
cd $B
run() {  # run <worktree> <tag> <args...>  -- retried until the run gets a card (the idle window
  # between run_when_idle's check and the script's own guard closes often on this box)
  local wt=$1 tag=$2; shift 2
  local attempt=0
  while [ $attempt -lt 60 ]; do
    attempt=$((attempt+1))
    echo "[q2] $(date +%m-%d\ %H:%M:%S) waiting for an idle card for $tag (attempt $attempt)"
    unset CUDA_VISIBLE_DEVICES
    JACCPOT_WORKTREE=$wt ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag "$@" > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
    rc=$?
    if grep -q "only 0 idle GPU" artifacts/smallleaf/probes/baseline_${tag}.log; then sleep 120; continue; fi
    echo "[q2] $(date +%m-%d\ %H:%M:%S) done $tag rc=$rc"
    grep -h "ms/step\|eval-only\|attribution\|FAILED\|no idle\|flags=\[.*foreign" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-220
    return
  done
  echo "[q2] gave up on $tag after $attempt attempts"
}
DEV=/export/home/tbuck/jaccpot-smallleaf-dev-wt; BASE=/export/home/tbuck/jaccpot-smallleaf-wt
# G2a first: the CSR M2L kernel microbench (2-3 min), retried until it gets a card
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "[q2] $(date +%m-%d\ %H:%M:%S) microbench attempt $attempt"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 172800 bash -c 'export CUDA_VISIBLE_DEVICES=$('"$PY"' -c "import sys; sys.path.insert(0,\"'"$B"'\"); from common.gpu_guard import pick_idle_gpus; print(pick_idle_gpus(1)[0])") && JACCPOT_WORKTREE='"$DEV"' '"$PY"' '"$DEV"'/bench/m2l_csr_microbench.py --out artifacts/smallleaf/m2l_csr_microbench.json' > artifacts/smallleaf/probes/m2l_csr_microbench.log 2>&1
  if grep -q "wrote" artifacts/smallleaf/probes/m2l_csr_microbench.log; then break; fi
  sleep 120
done
grep -h "ns/pair\|Error\|wrote\|only 0 idle" artifacts/smallleaf/probes/m2l_csr_microbench.log | cut -c1-240
run $DEV  p1p2a_leaf64_th0.6_r2   --leaf 64  --theta 0.6 --modes refresh,nearfield,detail
run $DEV  p1p2a_leaf256_th0.6_r2  --leaf 256 --theta 0.6 --modes refresh,nearfield,detail
run $BASE base_leaf32_th0.6       --leaf 32  --theta 0.6 --modes refresh --no-trace
run $DEV  p1p2a_leaf32_th0.6      --leaf 32  --theta 0.6 --modes refresh --no-trace
run $DEV  p1p2adb_leaf32_th0.6    --leaf 32  --theta 0.6 --modes refresh --no-trace --env JACCPOT_M2L_DEGREE_BATCHED=1
run $DEV  p1p2a_c16384_leaf64_th0.6 --leaf 64 --theta 0.6 --modes refresh --m2l-chunk 16384
run $DEV  p1p2a_c65536_leaf64_th0.6 --leaf 64 --theta 0.6 --modes refresh --m2l-chunk 65536
echo "[q2] $(date +%m-%d\ %H:%M:%S) ucurve start"
JACCPOT_WORKTREE=$DEV ./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 \
   --jac-thetas 0.4 0.6 0.8 --orders 4 6 --jac-leaf-single 32 64 128 256 --ref-targets 4096 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 \
   --out artifacts/smallleaf/ucurve_plummer200k_p1p2a.json > artifacts/smallleaf/probes/ucurve_p1p2a.log 2>&1
echo "[q2] $(date +%m-%d\ %H:%M:%S) ucurve done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/ucurve_p1p2a.log | cut -c1-200
