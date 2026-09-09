#!/bin/bash
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
for cfg in 128:0.6 128:0.8 32:0.6 32:0.8; do
  leaf=${cfg%%:*}; theta=${cfg##*:}; tag="base_leaf${leaf}_th${theta}"
  echo "[rerun] $(date +%H:%M:%S) waiting for an idle card for $tag"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 28800 $PY codes/smallleaf_baseline.py --leaf $leaf --theta $theta --tag $tag > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
  echo "[rerun] $(date +%H:%M:%S) done $tag rc=$?"
  grep -h "ms/step\|eval-only\|attribution\|FAILED\|busy card\|no idle" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-200
done
