#!/bin/bash
# T1.2 quiet timing (perf worktree), then T1.3 N=1M both codes (main jaccpot).
cd /export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
export JAX_ENABLE_X64=1
echo "=== T1.2 timing (perf worktree) ==="
JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-perf-wt PYTHONPATH=$PWD/sitecustom_wt PROBE_NO_REF=1 \
  /export/home/tbuck/jaccpot/.venv/bin/python codes/probe_t12.py 2>&1 | grep -v "Deprecation\|s = FastMultipole" | tee artifacts/probes/probe_t12_timing_idle.log
echo "=== T1.2 timing at leaf 64 (perf worktree) ==="
JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-perf-wt PYTHONPATH=$PWD/sitecustom_wt PROBE_NO_REF=1 PROBE_LEAF=64 PROBE_THETAS="0.6 0.8 1.0" PROBE_SETTINGS="baseline both" \
  /export/home/tbuck/jaccpot/.venv/bin/python codes/probe_t12.py 2>&1 | grep -v "Deprecation\|s = FastMultipole" | tee artifacts/probes/probe_t12_timing_idle_leaf64.log
echo "=== T1.3 N=1M, both codes (main jaccpot, 4096-target reference) ==="
/export/home/tbuck/jaccpot/.venv/bin/python codes/compare_force.py --n 1000000 --ref-targets 4096 --orders 4 \
  --jac-thetas 0.5 0.6 0.8 1.0 --pkd-thetas 0.5 0.6 0.7 0.8 --pkd-nbucket 16 --repeats 5 --warmup 2 \
  --out artifacts/compare_force_plummer1M_2026-09-06.json --workdir artifacts/compare_force_1M 2>&1 \
  | grep -v "Deprecation\|solver = FastMultipole" | grep -E "IC=|jaccpot\[|pkdgrav3\[|!!|Error|error|wrote|epsilon" | tee artifacts/probes/compare_force_1M.log
