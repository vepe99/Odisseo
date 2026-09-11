#!/bin/bash
# Phase 0 of the sub-10 ms plan (2026-09-11): both fronts on idle A100s, one plot.
#   $1 = jzfmm | jaccpot   (run the two halves as separate processes so each holds its own card;
#                            the guard never books a card with ANY compute process on it)
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
PY=/export/home/tbuck/jaccpot/.venv/bin/python
cd $B
mkdir -p artifacts/sub10ms/probes
jz() {  # jz <tag> <args...>
  local tag=$1; shift 1
  for attempt in $(seq 1 60); do
    echo "[q0] $(date +%m-%d\ %H:%M:%S) jzfmm $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $JZPY codes/jzfmm_force_eval.py --tag $tag "$@" \
       > artifacts/sub10ms/probes/jzfmm_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jzfmm_${tag}.log; then sleep 120; continue; fi
    if grep "min " artifacts/sub10ms/probes/jzfmm_${tag}.log | grep -q "foreign-process"; then echo "[q0] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q0] $(date +%m-%d\ %H:%M:%S) done jzfmm $tag"
    grep -h "min \|FAILED\|Error" artifacts/sub10ms/probes/jzfmm_${tag}.log | cut -c1-240
    return
  done
}
step() {  # step <tag> <args...>   -- jaccpot fused step, flags at their defaults (main = #341)
  local tag=$1; shift 1
  export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-walk-wt
  for attempt in $(seq 1 60); do
    echo "[q0] $(date +%m-%d\ %H:%M:%S) jaccpot $tag attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_baseline.py --tag $tag --modes "" "$@" \
       --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
    if grep -q "only 0 idle GPU\|no idle GPU" artifacts/sub10ms/probes/jaccpot_${tag}.log; then sleep 120; continue; fi
    if grep "full:\|eval-only" artifacts/sub10ms/probes/jaccpot_${tag}.log | grep -q "foreign-process"; then echo "[q0] $tag contaminated, retrying"; sleep 120; continue; fi
    echo "[q0] $(date +%m-%d\ %H:%M:%S) done jaccpot $tag"
    grep -h "eval-only\|full:\|step trace\|FAILED\|Error" artifacts/sub10ms/probes/jaccpot_${tag}.log | cut -c1-260
    return
  done
}
case "$1" in
  jzfmm)
    jz plummer200000 --n 200000 --p 3 4 5 6 --theta 0.5 0.6 0.8 1.0 --leaf 32 64
    jz plummer1000000 --n 1000000 --p 4 5 6 --theta 0.6 0.8 1.0 --leaf 32 64
    ;;
  jaccpot)
    for th in 0.6 0.8 1.0; do for p in 4 5 6; do
      step leaf64_th${th}_p${p} --leaf 64 --theta $th --order $p
    done; done
    for th in 0.6 0.8 1.0; do for p in 4 5 6; do
      step leaf32_th${th}_p${p} --leaf 32 --theta $th --order $p --no-trace
    done; done
    ;;
esac
