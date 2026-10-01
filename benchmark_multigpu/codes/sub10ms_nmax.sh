#!/bin/bash
# What N still fits ONE A100 on the sub-10 ms lane (cell leaves, Pallas walk/cascades/M2L, CSR near field)?
# PURE FMM: --no-accuracy, because the direct-sum reference is an O(block x N) fp64 sum whose buffers are the
# largest allocation in the run -- it decided the ceiling at N=4e6 (a 30.5 GiB block) and inflated every peak.
# Caps are left UNNAMED so they auto-grow -- this measures the hardware ceiling, not a cap setting, and the
# realized occupancies it prints (far pairs, near edges, leaf count) are what tight caps get sized from.
# One FRESH PROCESS per rung: a ladder inside one process reuses cached state and mislead once before
# (memory `distributed-per-device-ceiling-lifted`). Peak memory comes from the allocator's own high-water
# mark, not from grepping the log for OOM, which misses the retries XLA makes before giving up.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LEAF=${LEAF:-64}
PART=${PART:-cells}
# Rungs run to 2.56e8. One 40 GiB card cannot reach 1e9: the fp32 particle state a KDK step keeps live
# (pos+mass+vel+acc, the Morton-sorted copy, codes and permutation) is 68 B/particle = 63 GiB at 1e9,
# before a single tree node exists; and int32 indexing overflows a 3N float array at N = 7.2e8.
for n in ${NS:-200000 500000 1000000 2000000 4000000 8000000 16000000 32000000 64000000 128000000 256000000}; do
  tag=nmax_${PART}${LEAF}_n${n}
  unset CUDA_VISIBLE_DEVICES
  echo "[nmax] $(date +%H:%M:%S) N=$n load=$(cut -d' ' -f1 /proc/loadavg)"
  ./codes/run_when_idle.sh 14400 $PY codes/smallleaf_baseline.py --tag $tag --modes "" --no-trace \
     --n $n --ic plummer --leaf-partition $PART --leaf $LEAF --theta 0.8 --order 5 \
     --steps 2 --reps 2 --no-accuracy --out artifacts/sub10ms/jaccpot_${tag}.json \
     > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
  rc=$?
  line=$(grep -h "full:" artifacts/sub10ms/probes/jaccpot_${tag}.log | sed 's/ flags=.*//' | cut -c1-80)
  peak=$($PY -c "
import json
try:
    d = json.load(open('artifacts/sub10ms/jaccpot_${tag}.json'))
    v = (d.get('scan_full') or {}).get('peak_gib')
    if v is None:
        v = (d.get('eval_only') or {}).get('peak_gib')
    print('n/a' if v is None else f'{v:.1f} GiB')
except Exception:
    print('-')" 2>/dev/null)
  if [ -n "$line" ]; then
    echo "[nmax] N=$n OK  peak $peak  $line"
  else
    why=$(grep -hoE "RESOURCE_EXHAUSTED|Out of memory|overflowed: capacity [0-9]+|could not fit|leaf_capacity|RuntimeError.*" artifacts/sub10ms/probes/jaccpot_${tag}.log | head -1 | cut -c1-110)
    echo "[nmax] N=$n FAILED rc=$rc peak $peak  ${why:-see log}"
    break
  fi
done
