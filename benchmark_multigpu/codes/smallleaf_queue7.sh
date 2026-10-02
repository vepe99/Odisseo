#!/bin/bash
# After queue6: the 1M leaf-256 row with the CSR lane (its fan-out child hit the stale-utilisation guard).
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue6.sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
cd $B; unset CUDA_VISIBLE_DEVICES
echo "[q7] $(date +%m-%d\ %H:%M:%S) 1M leaf 256 (CSR on) start"
./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 1000000 --skip-pkdgrav3 --ref-targets 4096 \
   --jac-thetas 0.5 0.6 --orders 4 --jac-leaf-single 256 --jac-max-interactions 8192 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 \
   --out artifacts/smallleaf/compare_force_plummer1M_p1p2b_leaf256.json > artifacts/smallleaf/probes/compare_1M_p1p2b_leaf256.log 2>&1
echo "[q7] $(date +%m-%d\ %H:%M:%S) done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/compare_1M_p1p2b_leaf256.log | cut -c1-200
