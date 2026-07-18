#!/usr/bin/env bash
# =============================================================================
# pick_gpu.sh -- autocvd-equivalent free-GPU picker that TOLERATES a faulted GPU.
#
# On compgpu5 one card (GPU2, PCI 0000:1C:00.0) is hardware-faulted, so
# `nvidia-smi -L` exits 255 and `autocvd` (which calls `nvidia-smi -L`) crashes.
# The `--query-gpu` interface still works -- it simply omits the broken index.
# This picker replicates autocvd's behaviour (least-used / most-free card, wait
# for a free one) using only `--query-gpu`, so the broken GPU is silently skipped.
#
# Usage:  GPU=$(bash pick_gpu.sh);  export CUDA_VISIBLE_DEVICES="$GPU"
# Env knobs:
#   PICK_GPU_MIN_FREE_MB   minimum free MiB to consider a GPU usable (default 8000)
#   PICK_GPU_MAX_UTIL      maximum utilization%% to consider "free"       (default 10)
#   PICK_GPU_EXCLUDE       extra comma-sep indices to skip                (default "")
#   PICK_GPU_WAIT_SECS     if no free GPU, wait up to this many secs      (default 0)
#   PICK_GPU_INTERVAL      poll interval while waiting                    (default 15)
# NOTE: this does NOT fix the cuInit(0) CUDA_ERROR_UNKNOWN caused by the faulted
# GPU2 -- that is a node-level driver fault needing an admin reset/reboot. This
# only picks a sane index once the driver is healthy again.
# =============================================================================
set -uo pipefail
MIN_FREE_MB="${PICK_GPU_MIN_FREE_MB:-8000}"
MAX_UTIL="${PICK_GPU_MAX_UTIL:-10}"
EXCLUDE="${PICK_GPU_EXCLUDE:-}"
WAIT_SECS="${PICK_GPU_WAIT_SECS:-0}"
INTERVAL="${PICK_GPU_INTERVAL:-15}"

excluded() { case ",$EXCLUDE," in *",$1,"*) return 0;; esac; return 1; }

pick_once() {
  # index, util%, free MiB  (broken GPUs are simply absent from this output)
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
             --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' -v minf="$MIN_FREE_MB" -v maxu="$MAX_UTIL" '
      { gsub(/ /,""); idx=$1; util=$2; free=$3;
        if (free+0 >= minf && util+0 <= maxu) print util, -free, idx }' \
  | sort -n \
  | while read -r _ _ idx; do echo "$idx"; done
}

deadline=$(( $(date +%s) + WAIT_SECS ))
while :; do
  for idx in $(pick_once); do
    excluded "$idx" && continue
    echo "$idx"
    exit 0
  done
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    echo "[pick_gpu] no free GPU (>= ${MIN_FREE_MB}MiB free, <= ${MAX_UTIL}% util)" >&2
    exit 1
  fi
  echo "[pick_gpu] waiting for a free GPU... ($((deadline-now))s left)" >&2
  sleep "$INTERVAL"
done
