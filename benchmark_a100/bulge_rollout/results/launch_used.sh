#!/bin/bash
# A realistic disc + bulge, 21,012,480 particles, 6 x A100, a QUARTER ORBIT.
#
#   T_orb at the baryonic half-mass radius (r = 0.5301 = 5.30 kpc, v_c = 3.4088)
#     = 0.9770 code = 145.7 Myr;  quarter orbit = 0.2443 code = 36.4 Myr
#
# Usage: production.sh <mac> <eps> <leaf> <dt> [steps]
#   steps defaults to ceil(0.2443 / dt), i.e. exactly a quarter orbit.
set -u
MAC=${1:-dehnen}
EPS=${2:-0}
LEAF=${3:-512}
DT=${4:-2.5e-4}
STEPS=${5:-0}

PY=/export/home/tbuck/micromamba/envs/odisseo/bin/python
IC=/export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz
OUT=/export/scratch/tbuck/odisseo_runs/quarter_orbit
DEV=0,2,4,5,6,7

if [ "$STEPS" = "0" ]; then
  STEPS=$($PY -c "import math;print(int(math.ceil(0.2443/$DT)))")
fi

EXTRA=""
if [ "$MAC" = "dehnen_error" ]; then
  EXTRA="--adaptive-eps $EPS"
  # At leaf 512 the criterion's derived cross caps ask for a single 46-64 GiB buffer
  # on a 40 GB card; pin them to the geometric values, whose footprint is known to
  # fit. At leaf 1024 the derived caps already fit (~40.1 GB peak) and are left alone,
  # because pinning caps the calibration says the criterion needs buys a run that
  # truncates. --overflow-every reads the flags back every 10 steps either way.
  if [ "$LEAF" -le 512 ]; then
    EXTRA="$EXTRA --cross-queue 8388608 --cross-interactions 32768 --m2l-chunk 8192 --nearfield-chunk 128"
  fi
fi

echo "# launching: mac=$MAC eps=$EPS leaf=$LEAF dt=$DT steps=$STEPS"
# preallocate=false and pinned devices: see benchmark_a100/bulge_rollout/findings.md.
# The anomaly in section 4 does NOT reach this configuration (forces agree to 7 s.f.),
# but the pinning also keeps the run off the two cards other users are on.
CUDA_VISIBLE_DEVICES=$DEV XLA_PYTHON_CLIENT_PREALLOCATE=false \
exec $PY -u /export/home/tbuck/Odisseo/tools/mesh_galaxy_run.py \
  --ic "$IC" --ndev 6 --leaf "$LEAF" --theta 0.7 --order 6 --dtype float32 \
  --nearfield-accum wide --mac-type "$MAC" $EXTRA \
  --dt "$DT" --steps "$STEPS" \
  --probe 256 --probe-seed 20260901 \
  --render-every 10 --projection xy,xz --render-res 800 --render-extent 1.2 \
  --repartition-every 100 --checkpoint-every 100 \
  --diag-every 10 --overflow-every 10 \
  --max-hours 23 \
  --out-prefix "$OUT/qorbit"
