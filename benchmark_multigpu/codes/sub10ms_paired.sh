#!/bin/bash
# Interleaved head-to-head at the matched operating point. Absolute ms are poisoned by host load
# (the step is launch-bound: load 23-31 gave 12.65 ms, load 37-46 gave 14.4-14.8 for the SAME config),
# but a RATIO drawn from runs alternated within minutes of each other survives it. Each row records
# the load it saw, so a pair taken under very different loads can be thrown away afterwards.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"

for rep in 1 2 3; do
  for side in jac jz; do
    tag=paired${rep}_${side}
    unset CUDA_VISIBLE_DEVICES
    echo "[paired] $(date +%H:%M:%S) $tag load=$(cut -d' ' -f1 /proc/loadavg)"
    if [ $side = jac ]; then
      ./codes/run_when_idle.sh 7200 $PY codes/smallleaf_baseline.py --tag $tag --modes "" \
         --leaf-partition cells --leaf 64 --theta 0.8 --order 6 \
         --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
      grep -h "eval-only\|full:" artifacts/sub10ms/probes/jaccpot_${tag}.log | sed 's/ (spread/ (spread/' | cut -c1-150
    else
      ./codes/run_when_idle.sh 7200 $JZPY codes/jzfmm_force_eval.py --tag $tag \
         --leaf 64 --theta 0.8 --p 5 --no-trace \
         --out artifacts/jzfmm/jzfmm_front_${tag}.json > artifacts/sub10ms/probes/jzfmm_${tag}.log 2>&1
      grep -h "ms\b\|aggL2" artifacts/sub10ms/probes/jzfmm_${tag}.log | tail -2 | cut -c1-150
    fi
  done
done
