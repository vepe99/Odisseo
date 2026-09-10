#!/bin/bash
# Default-path check after the 2026-09-10 default switch (jaccpot PR #341): NO walk / M2L flags in the env --
# the flat walk and the CSR M2L kernel are jaccpot's defaults now, and the harness picks the flat-lane edge cap
# (pow2(1.5 x directed near pairs): 2^22 at leaf 64, 2^23 at leaf 32). Retries until a card is idle AND the
# timed rows carry no foreign process.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-walk-wt
cd $B
run() {  # run <tag> <args...>
  local tag=$1; shift 1
  for attempt in $(seq 1 40); do
    echo "[q11] $(date +%m-%d\ %H:%M:%S) $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes refresh "$@" \
       > artifacts/smallleaf/probes/baseline_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/smallleaf/probes/baseline_${tag}.log; then sleep 120; continue; fi
    if grep "full:" artifacts/smallleaf/probes/baseline_${tag}.log | grep -q "foreign-process"; then echo "[q11] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q11] $(date +%m-%d\ %H:%M:%S) done $tag"
    grep -h "eval-only\|full:\|step trace\|attribution\|FAILED\|Error\|validated_caps\|near_edge" artifacts/smallleaf/probes/baseline_${tag}.log | cut -c1-260
    return
  done
}
run default_leaf64_th0.6 --leaf 64 --theta 0.6
run default_leaf32_th0.6 --leaf 32 --theta 0.6 --no-trace
