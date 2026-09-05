#!/bin/bash
# 21,012,480-particle disc + bulge, a quarter orbit, on ONE HoreKa GPU node.
#
# ---------------------------------------------------------------------------
# CONFIRM BEFORE SUBMITTING -- these depend on your allocation, not on the code:
#   --partition   HoreKa's GPU partitions and their per-GPU memory
#   --account     your compute project
#   the module/env block in section 1
#   GPUS_PER_NODE and LEAF in section 2 (see the memory table below)
# ---------------------------------------------------------------------------
#
# WHY ONE NODE: this lane is single-process. It builds one shard_map mesh over
# jax.devices() and every host-side step (RCB partition, the fp64 probe, checkpointing,
# the realignment map) assumes one process holding all 21 M rows. There is no
# jax.distributed.initialize(), so it CANNOT span nodes as written. Requesting two nodes
# would silently use only the first node's GPUs.
#
# MEMORY, which decides LEAF (proxy scaled to a measured point: ndev 6 / leaf 1024 =
# 40,105 MiB coordinator on A100-40GB):
#
#   ndev  leaf   leaves/dev   est. coordinator   40 GB   80/94 GB
#      4  1024        5 130         60 881 MiB      no        YES
#      4  2048        2 565         15 220 MiB     YES        YES   (but ~2x slower/step)
#      6  1024        3 420         40 105 MiB     98 %       YES
#      8  1024        2 565         27 396 MiB     YES        YES   (needs 8 GPUs in ONE node)
#
#   On 4 x 40 GB use LEAF=2048 and expect ~250-270 s/step (~35 h) -- or switch to the
#   geometric MAC (see GEOMETRIC FALLBACK at the bottom), which fits at leaf 512.
#   On 4 x >=80 GB use LEAF=1024, the configuration that was measured.

#SBATCH --job-name=disc_bulge_21m_qorbit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4                  # CONFIRM: GPUs per node on your partition
#SBATCH --partition=accelerated       # CONFIRM
##SBATCH --account=CHANGE_ME          # CONFIRM, then uncomment
#SBATCH --time=24:00:00
#SBATCH --output=qorbit_%j.out
#SBATCH --error=qorbit_%j.err
#SBATCH --exclusive                   # a shared card ruins both the timing and the memory

set -euo pipefail

# ---- 1. environment ------------------------------------------------------------------
# CONFIRM this block. The run needs: python >=3.12, jax with CUDA, numpy, and the
# jaccpot + yggdrax + Odisseo checkouts importable. jaccpot needs jax >=0.10.2,<0.11.
# module load devel/cuda/12.4 || true
# module load devel/python/3.12 || true
# source "$HOME/venvs/odisseo/bin/activate"
PY=${PY:-python}

REPO=${REPO:-$HOME/Odisseo}
export PYTHONPATH="${PYTHONPATH:-}:$HOME/jaccpot:$HOME/yggdrax:$REPO"   # nornax not needed

# ---- 2. configuration ----------------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
LEAF=${LEAF:-1024}                 # 1024 needs >=64 GB/GPU at ndev 4; else 2048
MAC=${MAC:-dehnen_error}
EPS=${EPS:-1e-5}
DT=${DT:-5e-4}

IC=${IC:-$WORK/odisseo_ic/disk_bulge_21m_v2.npz}
OUT=${OUT:-$WORK/odisseo_runs/quarter_orbit}
mkdir -p "$OUT"

N=21012480
# A quarter orbit at the baryonic half-mass radius is 0.2443 code time units.
STEPS=$($PY -c "import math;print(int(math.ceil(0.2443/$DT)))")

# N must be ndev * k * leaf so cap == count and no device is padded; the run hard-errors
# otherwise, but failing here costs seconds instead of a queue slot.
$PY - "$N" "$GPUS_PER_NODE" "$LEAF" <<'PYCHK'
import sys
n, nd, leaf = (int(x) for x in sys.argv[1:4])
if n % (nd * leaf):
    sys.exit(f"N={n} is not divisible by ndev*leaf={nd*leaf}; "
             f"use {(n // (nd*leaf)) * nd * leaf} or a different leaf")
print(f"# N={n:,} / (ndev {nd} x leaf {leaf}) = {n // (nd*leaf):,} leaves per device")
PYCHK

EXTRA=""
[ "$MAC" = "dehnen_error" ] && EXTRA="--adaptive-eps $EPS"
# Deliberately NOT pinning caps: at leaf >=1024 the derived caps fit and are what the
# criterion needs. Pinning them is a leaf<=512 workaround that OOMs anyway.

# ---- 3. run --------------------------------------------------------------------------
# Grow on demand rather than reserving 75 % up front.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "# $(date -Is) node=$(hostname) gpus=$GPUS_PER_NODE leaf=$LEAF mac=$MAC dt=$DT steps=$STEPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun --ntasks=1 $PY -u "$REPO/tools/mesh_galaxy_run.py" \
  --ic "$IC" \
  --ndev "$GPUS_PER_NODE" --leaf "$LEAF" --theta 0.7 --order 6 --dtype float32 \
  --nearfield-accum wide --mac-type "$MAC" $EXTRA \
  --dt "$DT" --steps "$STEPS" \
  --probe 256 --probe-seed 20260901 \
  --render-every 10 --projection xy,xz --render-res 800 --render-extent 1.2 \
  --repartition-every 100 --checkpoint-every 100 \
  --diag-every 10 --overflow-every 10 \
  --max-hours 23 \
  --out-prefix "$OUT/qorbit"

# ---- 4. post-process -----------------------------------------------------------------
$PY -u "$REPO/tools/mesh_frames_to_movie.py" --prefix "$OUT/qorbit" \
    --extent 1.2 --length-unit-kpc 10 --dt "$DT" --render-every 10 --fps 20 || true

$PY -u "$REPO/tools/mesh_rollout_analysis.py" --ic "$IC" \
    --final "$OUT/qorbit_final.npz" --json-out "$OUT/qorbit_analysis.json" || true

echo "# $(date -Is) done"

# ---- GEOMETRIC FALLBACK (fits 4 x 40 GB, 3.77x worse forces) -------------------------
#   LEAF=512 MAC=dehnen DT=2.5e-4 sbatch horeka_quarter_orbit.sbatch
#   measured here: 63.66 s/step at ~31.4 GB/card, rel_l2 1.5200e-02, 978 steps = 17.3 h
#   (that timing is for ndev 6; at ndev 4 expect ~1.5x the per-device work)
