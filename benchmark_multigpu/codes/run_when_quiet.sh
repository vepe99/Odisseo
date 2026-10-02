#!/bin/bash
# Like run_when_idle.sh, but also waits for a QUIET HOST. The fused step issues ~1700 launches and is
# launch-bound, so host load inflates it directly: the same config measured 12.65 ms at load 23-31 and
# 14.4-14.8 ms at load 37-46. An idle card is not enough for a record row.
# Usage: run_when_quiet.sh <max_wait_seconds> <max_loadavg> <command...>
maxwait=$1; maxload=$2; shift 2
waited=0
while true; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  idle=$(/export/home/tbuck/jaccpot/.venv/bin/python -c "
import sys; sys.path.insert(0,'/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu')
from common.gpu_guard import idle_gpus
i,_=idle_gpus(settle_s=1.5, samples=3); print(','.join(map(str,i)))" 2>/dev/null)
  if [ -n "$idle" ] && awk "BEGIN{exit !($load < $maxload)}"; then
    echo "[run_when_quiet] load $load < $maxload, idle GPUs: $idle after ${waited}s -> running: $*"
    exec "$@"
  fi
  if [ "$waited" -ge "$maxwait" ]; then echo "[run_when_quiet] no quiet slot (load $load, idle '$idle') after ${waited}s; giving up"; exit 3; fi
  sleep 120; waited=$((waited+120))
done
