#!/usr/bin/env bash
# Orchestrate all multi-GPU FMM + tree benchmark phases, picking free GPUs via autocvd.
# Run each phase, then render figures. Safe to run phases individually instead.
#
#   bash benchmark_multigpu/run_all.sh            # full suite (3 GPUs)
#   NDEV=2 N_ACC=20000 bash benchmark_multigpu/run_all.sh
#
# Prereq: the GPU node must be healthy (nvidia-smi responds). If it hangs, the
# driver is wedged and needs an admin reset -- do NOT proceed.
set -uo pipefail

ROOT=/export/home/tbuck/Odisseo-bench-multigpu
PY=/export/home/tbuck/micromamba/envs/odisseo/bin/python
AUTOCVD=/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd
BM=$ROOT/benchmark_multigpu
NDEV=${NDEV:-3}
N_ACC=${N_ACC:-20000}
N_PERF=${N_PERF:-200000}
export JAX_ENABLE_X64=1
cd "$ROOT"

log(){ echo "[bench $(date +%F_%H:%M:%S)] $*"; }

# preflight: refuse to run if nvidia-smi is wedged
if ! timeout 20 nvidia-smi >/dev/null 2>&1; then
  log "FATAL: nvidia-smi does not respond within 20s -- GPU node likely wedged. Aborting."
  exit 1
fi

run() { # run <ngpu> <logtag> <cmd...>
  local ng="$1" tag="$2"; shift 2
  local cvd; cvd=$($AUTOCVD -n "$ng" -l -o -q)
  log "$tag on GPUs [$cvd]: $*"
  CUDA_VISIBLE_DEVICES="$cvd" "$@" 2>&1 | tee "$BM/artifacts/${tag}.log"
  log "$tag rc=${PIPESTATUS[0]}"
}

mkdir -p "$BM/artifacts"
run "$NDEV" accuracy    "$PY" "$BM/fmm/accuracy.py"    --n "$N_ACC" --ndevs "$NDEV"
run "$NDEV" performance "$PY" "$BM/fmm/performance.py" --n "$N_PERF" --ic disk
run "$NDEV" scaling     "$PY" "$BM/fmm/scaling.py"     --mode both
run 1       tree        "$PY" "$BM/tree/tree_bench.py" --ns 10000 100000 1000000

log "rendering figures (no GPU)"
"$PY" "$BM/notebooks/make_figures.py" --which all
log "DONE -> $BM/artifacts"
