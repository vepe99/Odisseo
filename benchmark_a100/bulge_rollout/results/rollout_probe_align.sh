#!/bin/bash
# Is the FORCE wrong, or only its MAPPING back to input rows?
# Both MACs show good/bad/bad/good over steps 0-3 -- deterministic, step-indexed, identical in
# two processes. Every probed step now (1) re-verifies the on-device aligner against jaccpot's
# host scatter and (2) scores the RAW force in the evaluator's own Morton order against a
# direct sum at the positions in that order. Raw accurate + aligned wrong => the mapping.
# Five steps, to see whether the pattern has a period.
set -u
PY=/export/home/tbuck/micromamba/envs/odisseo/bin/python
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/a39fa1ce-5bcc-4d98-a2cb-fc5d97be87b5/scratchpad
OUT=/export/scratch/tbuck/odisseo_runs/rollprobe_align
export CUDA_VISIBLE_DEVICES=1,3,4,6 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$SP/ygg_pin
export JAX_COMPILATION_CACHE_DIR=/export/scratch/tbuck/jax_cache_qorbit JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
echo "######## ROLLOUT-PROBE-ALIGN crit $(date -Is) ########"
timeout 7200 $PY -u /export/home/tbuck/Odisseo/tools/mesh_galaxy_run.py \
  --ic /export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz \
  --ndev 4 --leaf 1024 --theta 0.7 --order 6 --dtype float32 --nearfield-accum wide \
  --mac-type dehnen_error --adaptive-eps 1e-5 \
  --dt 5e-4 --steps 5 --probe 256 --probe-seed 20260901 --probe-every 1 \
  --diag-every 1 --overflow-every 1 --checkpoint-every 0 --snapshot-steps 1,2,3,4,5 --out-prefix "$OUT/crit"
echo "######## END rc=$? ########"; echo "ROLLOUT-PROBE-ALIGN DONE"
