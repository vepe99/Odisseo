#!/bin/bash
# Tree-walk Phase 3: 300-step dynamics flat walk on/off at leaf 64 -- preset edge cap (2^25) so the dual-walk arm fits.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue9.sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-walk-wt
cd $B; unset CUDA_VISIBLE_DEVICES
for attempt in 1 2 3 4 5 6; do
  echo "[q10] $(date +%m-%d\ %H:%M:%S) dynamics leaf 64 attempt $attempt"
  ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_dynamics.py --leaf 64 --steps 300 --flag JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK --env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 > artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log 2>&1
  if ! grep -q "only 0 idle GPU" artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log; then break; fi; sleep 120
done
echo "[q10] $(date +%m-%d\ %H:%M:%S) done"
grep -h "=0:\|=1:\|divergence\|wrote\|Error" artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log | cut -c1-240
