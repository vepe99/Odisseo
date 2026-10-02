#!/bin/bash
# After the other queues: M2L chunk-size sweep on the dev worktree at leaf 64 theta 0.6 (per-chunk fixed cost x 247 chunks).
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "smallleaf_rerun_missing.sh|smallleaf_ab_queue.sh|smallleaf_db_queue.sh|smallleaf_ucurve_queue.sh" > /dev/null; do sleep 60; done
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
for chunk in 16384 65536; do
  tag="p1p2a_c${chunk}_leaf64_th0.6"
  echo "[chunk] $(date +%H:%M:%S) waiting for an idle card for $tag"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 28800 $PY codes/smallleaf_baseline.py --leaf 64 --theta 0.6 --tag $tag --modes refresh --m2l-chunk $chunk > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
  echo "[chunk] $(date +%H:%M:%S) done $tag rc=$?"
  grep -h "ms/step\|eval-only\|attribution\|FAILED\|no idle" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-200
done
