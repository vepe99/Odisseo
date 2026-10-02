#!/bin/bash
# After queue3: the per-order sweep at leaf 64 theta 0.6 (plan 2.4: force error must be MONOTONE in p with the
# CSR lane on -- the only test that separates an acceptance/coverage bug from truncation), CSR on and off.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
while pgrep -f "^/bin/bash $B/codes/smallleaf_queue[23].sh" > /dev/null; do sleep 60; done
PY=/export/home/tbuck/jaccpot/.venv/bin/python
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-dev-wt
cd $B
for csr in 1 0; do
  echo "[q4] $(date +%m-%d\ %H:%M:%S) per-order sweep CSR=$csr start"
  unset CUDA_VISIBLE_DEVICES
  ./codes/run_when_idle.sh 172800 $PY codes/compare_force.py --n 200000 --skip-pkdgrav3 \
     --jac-thetas 0.6 --orders 2 3 4 5 6 --jac-leaf-single 64 --ref-targets 4096 \
     --jac-env JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=$csr \
     --out artifacts/smallleaf/order_sweep_leaf64_th0.6_csr${csr}.json > artifacts/smallleaf/probes/order_sweep_csr${csr}.log 2>&1
  echo "[q4] $(date +%m-%d\ %H:%M:%S) done CSR=$csr rc=$?"
  grep -h "jaccpot\[1gpu\]\|FAILED\|wrote" artifacts/smallleaf/probes/order_sweep_csr${csr}.log | cut -c1-200
done
