#!/bin/bash
# 25,165,824-particle disc + bulge, a quarter orbit, on all 8 A100s.
# Auto-restarting: retries ONLY a non-finite abort, stops on anything else.
#
# --probe-every 25: score the force against a fresh fp64 direct sum ~20 times over the run, not
# only at t=0. Every rollout before 2026-09-04 had ~45 % wrong forces from step 1 with every guard
# silent (findings 14: XLA ragged_all_to_all on jax 0.9.0); a t=0 probe alone is what let that pass.
# ~160 s of host numpy per probe at 25 M / 256 targets, ~55 min over 489 steps.
# --halo-exchange native: explicit, so the log records which exchange ran (jax 0.10.2 is clean).
set -u
PY=/export/scratch/tbuck/venv_prod_jax0102/bin/python   # standalone jax 0.10.2: the ragged_all_to_all fix (findings 14)
REPO=/export/home/tbuck/Odisseo
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/a39fa1ce-5bcc-4d98-a2cb-fc5d97be87b5/scratchpad
IC=/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz
OUT=/export/scratch/tbuck/odisseo_runs/qorbit25m
CKPT=$OUT/q_ckpt.npz
mkdir -p "$OUT"
STEPS=489
MAX_ATTEMPTS=12

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# No sitecustomize pin needed: the venv's editable yggdrax points at yggdrax-main-wt directly.
export JAX_COMPILATION_CACHE_DIR=/export/scratch/tbuck/jax_cache_prod_jax0102
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  ALOG=$OUT/attempt_${attempt}.log
  RESTART=""
  [ -s "$CKPT" ] && RESTART="--restart-from $CKPT"
  echo "########## attempt $attempt $(date -Is) ${RESTART:-(from IC)} ##########"
  timeout 82800 $PY -u "$REPO/tools/mesh_galaxy_run.py" \
    --ic "$IC" $RESTART \
    --ndev 8 --leaf 1024 --theta 0.7 --order 6 --dtype float32 \
    --nearfield-accum wide --mac-type dehnen_error --adaptive-eps 1e-5 \
    --dt 5e-4 --steps $STEPS \
    --probe 256 --probe-seed 20260901 --probe-every 25 --halo-exchange native \
    --render-every 10 --projection xy,xz --render-res 800 --render-extent 1.2 \
    --repartition-every 100 --checkpoint-every 20 \
    --diag-every 10 --overflow-every 10 \
    --max-hours 22 \
    --out-prefix "$OUT/q" 2>&1 | tee "$ALOG"
  rc=${PIPESTATUS[0]}
  echo "########## attempt $attempt exited rc=$rc $(date -Is) ##########"
  [ "$rc" -eq 0 ] && { echo "RUN COMPLETE after $attempt attempt(s)"; break; }
  if grep -q "NON-FINITE STATE" "$ALOG"; then
    echo ">>> non-finite abort: resuming from the last checkpoint"; continue
  fi
  echo ">>> rc=$rc with no non-finite abort -- NOT retrying. Last lines:"; tail -20 "$ALOG"; break
done
