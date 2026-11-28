#!/bin/bash
# The K ~ 150 bundle: every catalogue globular cluster with a mass and a half-mass radius (K = 165,
# M = 1000 stars each, King models) at TODAY's positions in the live Milky Way (a4 'galaxy' model).
#
#   bundle.sh functional   a few segments (functional: verdicts, membership/ledger/energy columns,
#                          device peak) -- the HH4b gate-3 run
#   bundle.sh long         the long run (default 1 Gyr), resumable: rerun the same command and it
#                          continues from $OUT/restart.npz (written every 10 segments)
#
# Env: PY (python with jax >= 0.11.2 + jaccpot + nornax; default python), GALAXY (the a4 .npz; default
# $ODISSEO_GC_RUNS/ics/mw_10000000.npz), ODISSEO_GC_DATA (fetch_baumgardt.py --out), OUT, T_GYR,
# PREFIX (e.g. "on-free-card 30" or "srun ..."). One card. At 1e7 + 165 x 1e3 set the allocator below;
# a full 1e8 galaxy needs ~40 GB (nornax docs/hybrid_hermite.md sections 13-14).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-functional}
PY=${PY:-python}
RUNS=${ODISSEO_GC_RUNS:-$HERE/runs}
GALAXY=${GALAXY:-$RUNS/ics/mw_10000000.npz}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-cuda_async}
case $MODE in
  functional)
    OUT=${OUT:-$RUNS/bundle_functional}
    exec ${PREFIX:-} "$PY" "$HERE/run_live.py" --galaxy "$GALAXY" --names ALL --M 1000 \
      --t-gyr 1 --segments ${SEGMENTS:-4} --steps 10 --energy-every 1 --snap-every 2 --restart-every 1 \
      --out "$OUT" ;;
  long)
    OUT=${OUT:-$RUNS/bundle_long}
    exec ${PREFIX:-} "$PY" "$HERE/run_live.py" --galaxy "$GALAXY" --names ALL --M 1000 \
      --t-gyr "${T_GYR:-1}" --steps 50 --energy-every 10 --snap-every 20 --restart-every 10 \
      --out "$OUT" ;;
  *) echo "usage: bundle.sh functional|long" >&2; exit 2 ;;
esac
