#!/bin/bash
# After the A/B queue: the plan's 2.0 candidate (degree-batched M2L) on top of the dev worktree, leaf 64 and 32.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "smallleaf_rerun_missing.sh|smallleaf_ab_queue.sh" > /dev/null; do sleep 60; done
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
for cfg in 64:0.6 32:0.6; do
  leaf=${cfg%%:*}; theta=${cfg##*:}; tag="p1p2adb_leaf${leaf}_th${theta}"
  echo "[db] $(date +%H:%M:%S) waiting for an idle card for $tag"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 28800 $PY codes/smallleaf_baseline.py --leaf $leaf --theta $theta --tag $tag --modes refresh,detail --env JACCPOT_M2L_DEGREE_BATCHED=1 > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
  echo "[db] $(date +%H:%M:%S) done $tag rc=$?"
  grep -h "ms/step\|eval-only\|attribution\|FAILED\|no idle" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-200
done
