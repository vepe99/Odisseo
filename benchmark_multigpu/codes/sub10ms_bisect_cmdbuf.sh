#!/bin/bash
# Bisect the CUDA_ERROR_ILLEGAL_ADDRESS seen in queue3 (XLA command buffers incl. WHILE/CONDITIONAL):
# the crashing row (cells64 th0.8 p4) with (1) default XLA flags, (2) graphs without WHILE/CONDITIONAL, (3) full set again.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
run() {  # run <tag> <xla_flags>
  local tag=$1; local flags=$2
  unset CUDA_VISIBLE_DEVICES
  echo "[bisect] $(date +%H:%M:%S) $tag XLA_FLAGS='$flags'"
  XLA_FLAGS="$flags" ./codes/run_when_idle.sh 14400 $PY codes/smallleaf_baseline.py --tag $tag --modes "" --leaf-partition cells --leaf 64 --theta 0.8 --order 4 \
     --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
  grep -h "eval-only\|full:\|ILLEGAL\|Error" artifacts/sub10ms/probes/jaccpot_${tag}.log | sort | uniq -c | sort -rn | head -4 | cut -c1-200
}
run bisect_p4_noflags ""
run bisect_p4_graphs_nowhile "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"
run bisect_p4_graphs_full "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=2"
