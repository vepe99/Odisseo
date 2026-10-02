#!/bin/bash
# Wait for the baseline re-run queue, then time the dev worktree (fold + scatter fix) on an idle card.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f smallleaf_rerun_missing.sh > /dev/null; do sleep 60; done
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
for cfg in 64:0.6 256:0.6 32:0.6 128:0.6; do
  leaf=${cfg%%:*}; theta=${cfg##*:}; tag="p1p2a_leaf${leaf}_th${theta}"
  echo "[ab] $(date +%H:%M:%S) waiting for an idle card for $tag"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 28800 $PY codes/smallleaf_baseline.py --leaf $leaf --theta $theta --tag $tag --modes refresh,nearfield,detail > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
  echo "[ab] $(date +%H:%M:%S) done $tag rc=$?"
  grep -h "ms/step\|eval-only\|attribution\|FAILED\|no idle" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-200
done
