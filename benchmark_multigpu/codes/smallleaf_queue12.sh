#!/bin/bash
# Tier 3 of the tree-walk plan (2026-09-11): the wavefront width ladder in yggdrax dual_tree_walk_mutual.
# Per config: flags at their new defaults (flat walk + CSR M2L), ladder ON (default) vs OFF
# (YGGDRAX_MUTUAL_WALK_LADDER=0), in the step (smallleaf_baseline) and isolated (traversal_walk_bench).
# Each row retries until it gets an idle card AND its timed rows carry no foreign process.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
Y=/export/home/tbuck/yggdrax-walk-wt
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt YGGDRAX_WORKTREE=$Y
cd $B
step() {  # step <tag> <args...>
  local tag=$1; shift 1
  for attempt in $(seq 1 60); do
    echo "[q12] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes refresh "$@" \
       > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/smallleaf/probes/baseline_${tag}.log; then sleep 120; continue; fi
    if grep "full:" artifacts/smallleaf/probes/baseline_${tag}.log | grep -q "foreign-process"; then echo "[q12] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q12] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "eval-only\|full:\|step trace\|attribution\|FAILED\|Error" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-260
    return
  done
}
walk() {  # walk <tag> <args...>  (isolated yggdrax bench on an idle card; contention flags land in the JSON)
  local tag=$1; shift 1
  for attempt in $(seq 1 60); do
    echo "[q12] $(date +%m-%d\ %H:%M:%S) walk $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    (cd $Y && $B/codes/run_when_idle.sh 172800 $PY bench/traversal_walk_bench.py --walks mutual --index int32 --repeats 7 "$@" --out bench/results/walk_${tag}.json) \
       > artifacts/smallleaf/probes/walk_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/smallleaf/probes/walk_${tag}.log; then sleep 120; continue; fi
    echo "[q12] $(date +%m-%d\ %H:%M:%S) done walk $tag"
    grep -h "mutual\|util\|contend\|Error" artifacts/smallleaf/probes/walk_${tag}.log | cut -c1-220
    return
  done
}
step ladder_leaf32_th0.6   --leaf 32 --theta 0.6 --no-trace
step noladder_leaf32_th0.6 --leaf 32 --theta 0.6 --no-trace --env YGGDRAX_MUTUAL_WALK_LADDER=0
step ladder_leaf64_th0.6   --leaf 64 --theta 0.6
step noladder_leaf64_th0.6 --leaf 64 --theta 0.6 --env YGGDRAX_MUTUAL_WALK_LADDER=0
walk ladder_leaf64   --leaf-size 64 --max-pair-queue 524288
walk noladder_leaf64 --leaf-size 64 --max-pair-queue 524288 --no-ladder
walk ladder_leaf32   --leaf-size 32 --max-pair-queue 1048576
walk noladder_leaf32 --leaf-size 32 --max-pair-queue 1048576 --no-ladder
step ladder_leaf32_th0.8   --leaf 32 --theta 0.8 --no-trace
step ladder_leaf128_th0.6  --leaf 128 --theta 0.6
