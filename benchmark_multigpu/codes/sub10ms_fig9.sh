#!/bin/bash
# jz-fmm's Fig. 9 (arXiv:2609.09307) on OUR box: force-evaluation time vs the 90th percentile of the
# per-particle relative force error, points labelled by opening angle, one curve per expansion order.
# Their setup is 4x10^7 Hernquist on 4 A100s against gadget4/pkdgrav3; this is the same AXES and the
# same METRIC at our operating point (2x10^5, ONE A100, jaccpot vs jz-fmm). Reference: direct sum, fp64,
# the same 4096 targets (seed 12345) for both codes.
#
# SOFTENING: their paper states zero, but jz-fmm returns NaN forces at exactly eps=0 -- measured on BOTH
# distributions (hernquist at eps=1e-7 gives aggL2 1.027e-3; plummer at eps=0 gives NaN), so it is the softening
# and not the IC. Its PlummerKernel self-pair is 0/0 and `remove_self_interaction` masks AFTER the divide, where
# 0*NaN is still NaN. Both codes and both references therefore run at eps=1e-7, as the rest of the record does;
# at 1e-7 the softening is orders of magnitude below any pair separation here, so it is eps=0 numerically without
# the singular self-pair.
B=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu
PY=/export/home/tbuck/jaccpot/.venv/bin/python
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
IC=${IC:-hernquist}
cd $B
export PYTHONPATH=$B/sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"

jac() {  # jac <theta> <order>
  local tag=fig9_${IC}_jac_th$1_p$2
  unset CUDA_VISIBLE_DEVICES
  echo "[fig9] $(date +%H:%M:%S) $tag load=$(cut -d' ' -f1 /proc/loadavg)"
  ./codes/run_when_quiet.sh 172800 16 $PY codes/smallleaf_baseline.py --tag $tag --modes "" \
     --ic $IC --leaf-partition cells --leaf 64 --theta $1 --order $2 --softening 1e-7 \
     --env JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=8388608 JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=4194304 \
     --out artifacts/sub10ms/jaccpot_${tag}.json > artifacts/sub10ms/probes/jaccpot_${tag}.log 2>&1
  grep -h "eval-only\|full:" artifacts/sub10ms/probes/jaccpot_${tag}.log | sed 's/ flags=.*//' | cut -c1-160
}
jz() {   # jz <theta> <order>
  local tag=fig9_${IC}_jz_th$1_p$2
  unset CUDA_VISIBLE_DEVICES
  echo "[fig9] $(date +%H:%M:%S) $tag load=$(cut -d' ' -f1 /proc/loadavg)"
  ./codes/run_when_quiet.sh 172800 16 $JZPY codes/jzfmm_force_eval.py --tag $tag \
     --ic $IC --leaf 64 --theta $1 --p $2 --softening 1e-7 --no-trace --alloc-fac-ilist ${JZ_ILIST:-64} \
     --out artifacts/jzfmm/jzfmm_front_${tag}.json > artifacts/sub10ms/probes/jzfmm_${tag}.log 2>&1
  grep -h "min .* ms" artifacts/sub10ms/probes/jzfmm_${tag}.log | tail -1 | cut -c1-170
}
# their figure varies theta at fixed p, two orders per code; alternate the codes so both see one host
for th in ${THETAS:-1.0 0.8 0.6 0.5}; do
  for p in 5 7; do jac $th $p; jz $th $p; done
done
