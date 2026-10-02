#!/bin/bash
# HoreKa, ONE node, ONE process driving its 4 GPUs: validation, then the fused-lane sweep.
# Everything marked CONFIRM is a guess made off-cluster; check it against the HoreKa docs
# (or `sinfo -o '%P %G %c %m'`) before the first submission.
#SBATCH --job-name=fused-sweep
#SBATCH --partition=accelerated          # CONFIRM: 4x A100 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64               # CONFIRM: cores per accelerated node
#SBATCH --exclusive
#SBATCH --time=04:00:00
##SBATCH --account=<project>             # CONFIRM
#SBATCH --output=%x-%j.out

set -euo pipefail
# ---- environment (SETUP_HOREKA.md section 1) ---------------------------------------
: "${VENV:=$HOME/venvs/jaccpot-fused}"            # standalone venv, jax[cuda12]==0.10.2
: "${JACCPOT_DIR:=$HOME/src/jaccpot}"             # at the PR head (perf/multigpu-fused-lane)
: "${YGGDRAX_DIR:=$HOME/src/yggdrax}"             # at the PR head
: "${BENCH_DIR:=$HOME/src/Odisseo/benchmark_multigpu}"
: "${OUT_DIR:=${WORK:-$HOME}/fused_sweep/$SLURM_JOB_ID}"
# module load devel/python/3.12                   # CONFIRM, only if the venv needs it
source "$VENV/bin/activate"
export JAX_COMPILATION_CACHE_DIR=${WORK:-$HOME}/jax_cache_fused
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# WHILE/CONDITIONAL command-buffer capture crashes the fused lane (bisected 2026-09-11)
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"
export VENV JACCPOT_DIR YGGDRAX_DIR BENCH_DIR OUT_DIR   # fused_sweep.sh and the probe read them
mkdir -p "$OUT_DIR"
nvidia-smi topo -m > "$OUT_DIR/topology.txt"
python -c "import jax_plugins.xla_cuda12 as p; print(p.__file__)" > "$OUT_DIR/cuda_plugin.txt"

srun --ntasks=1 bash "$BENCH_DIR/slurm/fused_sweep.sh" "$OUT_DIR"
