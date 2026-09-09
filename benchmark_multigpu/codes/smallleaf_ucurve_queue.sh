#!/bin/bash
# Phase 3.1: the leaf U-curve on the dev worktree (fold + scatter fix), eval-only + walk, aggL2 vs fp64 direct.
# One process per leaf (compare_force fans out); far-pair cap raised globally because theta 0.4 has ~3x the far pairs.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "smallleaf_rerun_missing.sh|smallleaf_ab_queue.sh|smallleaf_db_queue.sh" > /dev/null; do sleep 60; done
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
unset CUDA_VISIBLE_DEVICES
echo "[ucurve] $(date +%H:%M:%S) start"
./codes/run_when_idle.sh 28800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 \
   --jac-thetas 0.4 0.6 0.8 --orders 4 6 --jac-leaf-single 32 64 128 256 --ref-targets 4096 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 \
   --out artifacts/smallleaf/ucurve_plummer200k_p1p2a.json > artifacts/smallleaf/probes/ucurve_p1p2a.log 2>&1
echo "[ucurve] $(date +%H:%M:%S) done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/ucurve_p1p2a.log | cut -c1-200
