#!/bin/bash
# Runs INSIDE one allocation: validate, then sweep. Usage: fused_sweep.sh <out_dir>
# Needs JACCPOT_DIR, BENCH_DIR (set by the sbatch script) and an activated venv.
set -uo pipefail
O=$1
PROBE="$JACCPOT_DIR/bench/multigpu_c4_cross_force_probe.py"
probe() {  # tag ndev n order reps arms
  local tag=$1 ndev=$2 n=$3 p=$4 reps=$5 arms=$6
  local cards=$(seq -s, 0 $((ndev - 1)))
  echo "=== $tag ndev=$ndev N=$n p=$p $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$cards PROBE_NDEV=$ndev PROBE_N=$n PROBE_LEAF=64 PROBE_ORDER=$p \
    PROBE_DTYPE=float32 PROBE_TIME_REPS=$reps PROBE_TIME_WARMUP=3 PROBE_TIME_ARMS=$arms \
    PROBE_JSON=$O/$tag.json timeout 3600 python "$PROBE" > "$O/$tag.log" 2>&1
  echo "=== $tag rc=$?"
  grep -hE 'TIMING|rel-L2|overflow=|FAILED|Error' "$O/$tag.log" | tail -4
}
# ---- validation: stop before the sweep if any step fails ---------------------------
python -c "import jax; print(jax.__version__, jax.devices())" || exit 2
probe val_1card_p6 1 200000 6 20 local          # vs this box: 11.9 ms at error 6.8e-4
for p in 4 5 6; do probe val_2card_p$p 2 200000 $p 1 cross; done
for p in 4 5 6; do probe val_4card_p$p 4 200000 $p 1 cross; done
python "$BENCH_DIR/slurm/check_validation.py" "$O" || { echo "VALIDATION FAILED"; exit 3; }
# ---- the sweep ---------------------------------------------------------------------
for ndev in 1 2 4; do
  for per in 200000 1000000; do                  # weak scaling: N per device fixed
    arms=$([ $ndev = 1 ] && echo local || echo local,cross)
    probe weak_ndev${ndev}_per${per}_p6 $ndev $((ndev * per)) 6 20 $arms
  done
  for n in 1000000 4000000; do                   # strong scaling: N fixed
    arms=$([ $ndev = 1 ] && echo local || echo local,cross)
    probe strong_ndev${ndev}_n${n}_p6 $ndev $n 6 10 $arms
  done
done
echo "SWEEP DONE $(date +%H:%M:%S)"
