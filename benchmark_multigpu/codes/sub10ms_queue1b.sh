#!/bin/bash
# COM exact rows that overflowed the 2^22 caps in queue1: leaf 32 theta 0.6 (near edges 4.7M).
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
for p in 4 5 6; do
  tag=com_leaf32_th0.6_p${p}
  for attempt in $(seq 1 30); do
    echo "[q1b] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes "" --no-trace --leaf 32 --theta 0.6 --order $p \
       --env JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS=exact JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=8388608 \
       --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jaccpot_${tag}.log; then sleep 120; continue; fi
    if grep "full:\|eval-only" artifacts/sub10ms/probes/jaccpot_${tag}.log | grep -q "foreign-process"; then sleep 120; continue; fi
    grep -h "eval-only\|full:\|FAILED\|Error" artifacts/sub10ms/probes/jaccpot_${tag}.log | cut -c1-200
    break
  done
done
