#!/bin/bash
# Production candidate: jax 0.10.2 (standalone venv) with the NATIVE ragged halo exchange, moving
# positions, probed against an fp64 direct sum at EVERY step. Correct at steps 1-3 => the native
# path is safe on 0.10.2 in the full pipeline, not just in the 8-element repro.
set -u
PY=/export/scratch/tbuck/venv_prod_jax0102/bin/python   # standalone jax 0.10.2 venv, editable jaccpot/yggdrax-main-wt/Odisseo
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/a39fa1ce-5bcc-4d98-a2cb-fc5d97be87b5/scratchpad
OUT=/export/scratch/tbuck/odisseo_runs/rollprobe_align
export CUDA_VISIBLE_DEVICES=${CARDS:-1,3,4,6} XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/export/scratch/tbuck/jax_cache_jax0102 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
echo "######## ROLLOUT-PROBE-JAX0102-NATIVE crit $(date -Is) ########"
timeout 7200 $PY -u /export/home/tbuck/Odisseo/tools/mesh_galaxy_run.py \
  --ic /export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz \
  --ndev 4 --leaf 1024 --theta 0.7 --order 6 --dtype float32 --nearfield-accum wide \
  --mac-type dehnen_error --adaptive-eps 1e-5 \
  --dt 5e-4 --steps 4 --probe 256 --probe-seed 20260901 --probe-every 1 \
  --diag-every 1 --overflow-every 1 --checkpoint-every 0 --halo-exchange native --out-prefix "$OUT/crit_jax0102"
echo "######## END rc=$? ########"; echo "ROLLOUT-PROBE-JAX0102-NATIVE DONE"
