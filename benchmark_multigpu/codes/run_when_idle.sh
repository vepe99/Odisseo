#!/bin/bash
# Wait until the GPU guard reports at least one idle card, then run the command.
# Usage: run_when_idle.sh <max_wait_seconds> <command...>
maxwait=$1; shift
waited=0
while true; do
  idle=$(/export/home/tbuck/jaccpot/.venv/bin/python -c "
import sys; sys.path.insert(0,'/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu')
from common.gpu_guard import idle_gpus
i,_=idle_gpus(settle_s=1.5, samples=3); print(','.join(map(str,i)))" 2>/dev/null)
  if [ -n "$idle" ]; then echo "[run_when_idle] idle GPUs: $idle after ${waited}s -> running: $*"; exec "$@"; fi
  if [ "$waited" -ge "$maxwait" ]; then echo "[run_when_idle] no idle GPU after ${waited}s; giving up"; exit 3; fi
  sleep 60; waited=$((waited+60))
done
