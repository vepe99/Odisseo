#!/bin/bash
# After queue4: Phase 3.3 dynamics (300 steps, CSR on/off, leaf 64 and 256), then the 1M re-run with CSR on.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue4.sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
cd $B
for leaf in 64 256; do
  for attempt in 1 2 3 4 5 6; do
    echo "[q5] $(date +%m-%d\ %H:%M:%S) dynamics leaf $leaf attempt $attempt"
    unset CUDA_VISIBLE_DEVICES
    ./codes/run_when_idle.sh 172800 $PY codes/smallleaf_dynamics.py --leaf $leaf --steps 300 > artifacts/smallleaf/probes/dynamics_leaf${leaf}.log 2>&1
    if ! grep -q "only 0 idle GPU" artifacts/smallleaf/probes/dynamics_leaf${leaf}.log; then break; fi; sleep 120
  done
  grep -h "CSR=\|divergence\|wrote\|Error" artifacts/smallleaf/probes/dynamics_leaf${leaf}.log | cut -c1-240
done
echo "[q5] $(date +%m-%d\ %H:%M:%S) 1M compare_force (CSR on) start"
unset CUDA_VISIBLE_DEVICES
./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 1000000 --skip-pkdgrav3 --ref-targets 4096 \
   --jac-thetas 0.5 0.6 --orders 4 --jac-leaf-single 128 256 --jac-max-interactions 8192 \
   --jac-env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1 \
   --out artifacts/smallleaf/compare_force_plummer1M_p1p2b.json > artifacts/smallleaf/probes/compare_1M_p1p2b.log 2>&1
echo "[q5] $(date +%m-%d\ %H:%M:%S) 1M done rc=$?"
grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/compare_1M_p1p2b.log | cut -c1-200
