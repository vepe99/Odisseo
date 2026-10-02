#!/bin/bash
# After queue5: the two U-curve rows that overflowed the per-node far cap (leaf 64, theta 0.4), CSR on.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue[45].sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
cd $B; unset CUDA_VISIBLE_DEVICES
echo "[q6] $(date +%m-%d\ %H:%M:%S) leaf 64 theta 0.4 rows start"
./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 --ref-targets 4096 \
   --jac-thetas 0.4 --orders 4 6 --jac-leaf-single 64 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 \
   --out artifacts/smallleaf/ucurve_leaf64_th0.4_p1p2b.json > artifacts/smallleaf/probes/ucurve_leaf64_th0.4.log 2>&1
echo "[q6] $(date +%m-%d\ %H:%M:%S) done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/ucurve_leaf64_th0.4.log | cut -c1-200
