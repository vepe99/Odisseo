#!/bin/bash
# Run codes/smallleaf_baseline.py for a list of "leaf:theta" configs sequentially on ONE GPU.
# Usage: smallleaf_matrix.sh <gpu> <tagprefix> <leaf:theta> [<leaf:theta> ...]   (extra env via BASELINE_ARGS)
gpu=$1; shift; prefix=$1; shift
export PYTHONPATH=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt
export JACCPOT_WORKTREE=${JACCPOT_WORKTREE:-/export/home/tbuck/jaccpot-smallleaf-wt}
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd /export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
mkdir -p artifacts/smallleaf/probes
for cfg in "$@"; do
  leaf=${cfg%%:*}; theta=${cfg##*:}
  tag="${prefix}_leaf${leaf}_th${theta}"
  echo "[matrix gpu$gpu] $(date +%H:%M:%S) start $tag"
  CUDA_VISIBLE_DEVICES=$gpu $PY codes/smallleaf_baseline.py --leaf "$leaf" --theta "$theta" --tag "$tag" $BASELINE_ARGS \
      > "artifacts/smallleaf/probes/baseline_${tag}.log" 2>&1
  echo "[matrix gpu$gpu] $(date +%H:%M:%S) done $tag rc=$?"
  grep -h "ms/step\|eval-only\|attribution\|FAILED" "artifacts/smallleaf/probes/baseline_${tag}.log" | cut -c1-260
done
