#!/bin/bash
# Run codes/smallleaf_baseline.py for a list of "leaf:theta" configs sequentially on ONE GPU,
# waiting (up to 10 min) for that card to read 0 % utilisation before each config -- the
# guard otherwise rejects the card on the stale utilisation reading left by the previous process.
# Usage: smallleaf_matrix2.sh <gpu> <tagprefix> <leaf:theta> [<leaf:theta> ...]   (extra args via BASELINE_ARGS)
gpu=$1; shift; prefix=$1; shift
export PYTHONPATH=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt
export JACCPOT_WORKTREE=${JACCPOT_WORKTREE:-/export/home/tbuck/jaccpot-smallleaf-wt}
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd /export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
mkdir -p artifacts/smallleaf/probes
wait_idle() {
  local waited=0
  while [ $waited -lt 600 ]; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" | wc -l)
    if [ "$util" = "0" ] && [ "$procs" = "0" ]; then sleep 5; return 0; fi
    sleep 10; waited=$((waited+10))
  done
  echo "[matrix gpu$gpu] card never went idle (util=$util procs=$procs)"; return 1
}
for cfg in "$@"; do
  leaf=${cfg%%:*}; theta=${cfg##*:}
  tag="${prefix}_leaf${leaf}_th${theta}"
  wait_idle
  echo "[matrix gpu$gpu] $(date +%H:%M:%S) start $tag"
  CUDA_VISIBLE_DEVICES=$gpu $PY codes/smallleaf_baseline.py --leaf "$leaf" --theta "$theta" --tag "$tag" $BASELINE_ARGS \
      > "artifacts/smallleaf/probes/baseline_${tag}.log" 2>&1
  rc=$?
  echo "[matrix gpu$gpu] $(date +%H:%M:%S) done $tag rc=$rc"
  grep -h "ms/step\|eval-only\|attribution\|FAILED\|busy card" "artifacts/smallleaf/probes/baseline_${tag}.log" | cut -c1-260
done
