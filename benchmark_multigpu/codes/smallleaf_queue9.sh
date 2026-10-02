#!/bin/bash
# Tree-walk Phase 3: 300-step dynamics flat walk on/off (CSR on) at leaf 64, then the per-order sweep with the flag.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue8.sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-walk-wt
cd $B; unset CUDA_VISIBLE_DEVICES
for attempt in 1 2 3 4 5 6; do
  echo "[q9] $(date +%m-%d\ %H:%M:%S) dynamics leaf 64 attempt $attempt"
  ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_dynamics.py --leaf 64 --steps 300 --flag JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK --env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=8388608 > artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log 2>&1
  if ! grep -q "only 0 idle GPU" artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log; then break; fi; sleep 120
done
grep -h "=0:\|=1:\|divergence\|wrote\|Error" artifacts/smallleaf/probes/dynamics_flatwalk_leaf64.log | cut -c1-240
echo "[q9] $(date +%m-%d\ %H:%M:%S) per-order sweep (flat walk) start"
./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 --ref-targets 4096 \
   --jac-thetas 0.6 --orders 2 3 4 5 6 --jac-leaf-single 64 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=1 JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=8388608 \
   --out artifacts/smallleaf/order_sweep_leaf64_th0.6_flatwalk.json > artifacts/smallleaf/probes/order_sweep_flatwalk.log 2>&1
echo "[q9] $(date +%m-%d\ %H:%M:%S) done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/order_sweep_flatwalk.log | cut -c1-200
