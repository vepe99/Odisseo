#!/bin/bash
# Phase 1.2 of the sub-10 ms plan: the COM-consistent MAC geometry (JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com)
# on the fused step, leaf 64 and 32, theta 0.6/0.8/1.0, p 4/5/6 -- same protocol as queue0's jaccpot half.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
step() {  # step <tag> <args...>
  local tag=$1; shift 1
  for attempt in $(seq 1 60); do
    echo "[q1] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes "" "$@" \
       --env JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS=exact JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=4194304 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=4194304 \
       --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jaccpot_${tag}.log; then sleep 120; continue; fi
    if grep "full:\|eval-only" artifacts/sub10ms/probes/jaccpot_${tag}.log | grep -q "foreign-process"; then echo "[q1] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q1] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "eval-only\|full:\|step trace\|FAILED\|Error" artifacts/sub10ms/probes/jaccpot_${tag}.log | cut -c1-260
    return
  done
}
for th in 0.8 1.0 0.6; do for p in 4 5 6; do
  step com_leaf64_th${th}_p${p} --leaf 64 --theta $th --order $p
done; done
for th in 0.8 1.0 0.6; do for p in 4 5 6; do
  step com_leaf32_th${th}_p${p} --leaf 32 --theta $th --order $p --no-trace
done; done
