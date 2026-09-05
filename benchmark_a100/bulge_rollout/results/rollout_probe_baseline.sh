#!/bin/bash
# THE discriminator. Both MACs are 45-50 % wrong on every call after the first once positions
# MOVE (rollout_probe: geo 0.466/0.503, criterion 0.449 at step 1), with every guard silent
# and the identical-input test blind to it. The compiled program is pure over its inputs, so
# a wrong answer for changed input can only come from memory it did not write on THIS call.
# Both MACs share the Pallas near-field. If the pure-JAX 'baseline' near-field is accurate at
# steps 1-3 where 'pallas' is not, the Pallas kernel owns the defect.
set -u
PY=/export/home/tbuck/micromamba/envs/odisseo/bin/python
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/a39fa1ce-5bcc-4d98-a2cb-fc5d97be87b5/scratchpad
OUT=/export/scratch/tbuck/odisseo_runs/rollprobe_bl
export CUDA_VISIBLE_DEVICES=1,3,4,6 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$SP/ygg_pin
export JAX_COMPILATION_CACHE_DIR=/export/scratch/tbuck/jax_cache_qorbit JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
run () {
  local tag=$1; shift
  echo "######## ROLLOUT-PROBE-BASELINE $tag $(date -Is) ########"
  timeout 7200 $PY -u /export/home/tbuck/Odisseo/tools/mesh_galaxy_run.py \
    --ic /export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz \
    --ndev 4 --leaf 1024 --theta 0.7 --order 6 --dtype float32 --nearfield-accum wide \
    --nearfield-backend baseline \
    --dt 5e-4 --steps 3 --probe 256 --probe-seed 20260901 --probe-every 1 \
    --diag-every 1 --overflow-every 1 --checkpoint-every 0 "$@" --out-prefix "$OUT/$tag"
  echo "######## END $tag rc=$? ########"
}
run crit_baseline --mac-type dehnen_error --adaptive-eps 1e-5
echo "ROLLOUT-PROBE-BASELINE DONE"
