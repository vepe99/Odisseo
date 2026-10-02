#!/bin/bash
# Phase 3 of the tree-walk plan: the flat-emission walk + CSR M2L + int32 in the step, per leaf, on an idle card.
# Each config retries until it gets a card AND its timed rows carry no foreign process.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-walk-wt
cd $B
run() {  # run <tag> <edge_cap> <args...>
  local tag=$1 edge=$2; shift 2
  for attempt in $(seq 1 40); do
    echo "[q8] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes refresh \
       --env JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=1 JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=$edge "$@" \
       > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
    if grep -q "only 0 idle GPU" artifacts/smallleaf/probes/baseline_${tag}.log; then sleep 120; continue; fi
    if grep "full:" artifacts/smallleaf/probes/baseline_${tag}.log | grep -q "foreign-process"; then echo "[q8] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q8] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "eval-only\|full:\|step trace\|attribution\|FAILED" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-260
    return
  done
}
run flat_leaf64_th0.6  8388608  --leaf 64  --theta 0.6
run flat_leaf32_th0.6  16777216 --leaf 32  --theta 0.6 --no-trace
run flat_leaf128_th0.6 4194304  --leaf 128 --theta 0.6
run flat_leaf256_th0.6 2097152  --leaf 256 --theta 0.6
run flat_leaf64_th0.8  8388608  --leaf 64  --theta 0.8
run flat_leaf32_th0.8  16777216 --leaf 32  --theta 0.8 --no-trace
