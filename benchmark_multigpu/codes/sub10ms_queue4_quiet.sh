#!/bin/bash
# QUIET variant of queue3: waits for host loadavg < 12 as well as an idle card (see run_when_quiet.sh).
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
    echo "[q4] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_quiet.sh 172800 16 $PY codes/smallleaf_baseline.py --tag q4_$tag --modes "" --leaf-partition cells "$@" \
       --env JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS=exact JACCPOT_NEARFIELD_LEAFPAIR_CSR=1 JACCPOT_CASCADE_PALLAS=1 JACCPOT_STATIC_STRICT_FUSED_WALK=pallas JACCPOT_M2L_CSR_KERNEL=lanes \
       --out artifacts/sub10ms/jaccpot_q4_${tag}.json > artifacts/sub10ms/probes/jaccpot_q4_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jaccpot_q4_${tag}.log; then sleep 120; continue; fi
    if grep "full:\|eval-only" artifacts/sub10ms/probes/jaccpot_q4_${tag}.log | grep -q "foreign-process"; then echo "[q4] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q4] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "cell leaves\|eval-only\|full:\|step trace\|FAILED\|Error" artifacts/sub10ms/probes/jaccpot_q4_${tag}.log | cut -c1-260
    return
  done
}
for th in 0.8 1.0 0.6; do for p in 5 4 6; do
  step cells64_th${th}_p${p} --leaf 64 --theta $th --order $p
done; done
# leaf 32 is uniformly worse than 64 in the step (queue3: 16.5-18.7 vs 11.4-15.7 ms), so the quiet
# record measures leaf 64 only, then reruns jz-fmm's front rows under the SAME quiet conditions --
# a ratio is only meaningful when both sides saw the same host.
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
for row in "64 0.8 5" "64 1.0 4" "32 0.8 5" "64 0.6 6"; do
  set -- $row
  tag=q4jz_leaf$1_th$2_p$3
  unset CUDA_VISIBLE_DEVICES
  echo "[q4] $(date +%m-%d\ %H:%M:%S) $tag"
  ./codes/run_when_quiet.sh 172800 16 $JZPY codes/jzfmm_force_eval.py --tag $tag --leaf $1 --theta $2 --p $3 \
     --out artifacts/jzfmm/jzfmm_front_${tag}.json > artifacts/sub10ms/probes/jzfmm_${tag}.log 2>&1
  grep -h "ms\b" artifacts/sub10ms/probes/jzfmm_${tag}.log | tail -1 | cut -c1-200
done
