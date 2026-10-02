#!/bin/bash
# Front rerun after Phases 2-5 of the sub-10 ms plan: the fused step on CELL leaves (TreeConfig leaf_partition=cells) with the
# COM-consistent MAC (exact radii), leaf 64 and 32, theta 0.8/1.0/0.6, p 4/5/6 -- queue0/1 protocol.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"  # WHILE capture crashes (bisect 2026-09-11)
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
step() {  # step <tag> <args...>
  local tag=$1; shift 1
  for attempt in $(seq 1 60); do
    echo "[q3] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag q3_$tag --modes "" --leaf-partition cells "$@" \
       --env JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS=exact JACCPOT_NEARFIELD_LEAFPAIR_CSR=1 JACCPOT_CASCADE_PALLAS=1 JACCPOT_STATIC_STRICT_FUSED_WALK=pallas JACCPOT_M2L_CSR_KERNEL=lanes \
       --out artifacts/sub10ms/jaccpot_q3_${tag}.json > artifacts/sub10ms/probes/jaccpot_q3_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jaccpot_q3_${tag}.log; then sleep 120; continue; fi
    if grep "full:\|eval-only" artifacts/sub10ms/probes/jaccpot_q3_${tag}.log | grep -q "foreign-process"; then echo "[q3] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q3] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "cell leaves\|eval-only\|full:\|step trace\|FAILED\|Error" artifacts/sub10ms/probes/jaccpot_q3_${tag}.log | cut -c1-260
    return
  done
}
for th in 0.8 1.0 0.6; do for p in 5 4 6; do
  step cells64_th${th}_p${p} --leaf 64 --theta $th --order $p
done; done
for th in 0.8 1.0 0.6; do for p in 5 4 6; do
  step cells32_th${th}_p${p} --leaf 32 --theta $th --order $p --env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=4194304
done; done
