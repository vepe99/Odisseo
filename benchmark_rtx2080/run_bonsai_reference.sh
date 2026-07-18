#!/usr/bin/env bash
# =============================================================================
# Bonsai N-body reference on the RTX 2080 Ti, for the SAME 200k disk IC ODISSEO
# integrates (scm8_exp). Live self-gravity (Bonsai BH-tree) + static external
# analytic NFW halo, same G=1 code units, same 4000 steps / t_end=2.0.
# Faithful copy of benchmark_a100/bonsai_reference/run_bonsai_reference.sh, with
# the free-GPU pick routed through pick_gpu.sh so the faulted GPU2 is skipped.
# =============================================================================
set -uo pipefail
REPO=/export/home/tbuck/Odisseo
HERE="$REPO/benchmark_rtx2080"
ENV="${ODISSEO_ENV:-odisseo}"
BON=/export/home/tbuck/Bonsai/runtime/build/bonsai2_slowdust
OUT="$HERE/bonsai_reference"
IC="$OUT/disk_ic.tipsy"
ICNPZ="$REPO/notebooks/scalability/ic_cache/odisseo_agama_ic_200k_scm8_exp.npz"
N2T="$REPO/benchmark_a100/bonsai_reference/npz_to_tipsy.py"
REN="$REPO/benchmark_a100/bonsai_reference/render_bonsai.py"
mkdir -p "$OUT/snapshots"
cd "$OUT"
log(){ echo "[bonsai-2080 $(date +%F_%H:%M:%S)] $*"; }

# tipsy IC from the same npz (matches the ODISSEO run particle-for-particle)
if [ ! -s "$IC" ]; then
  log "generating tipsy IC from $ICNPZ"
  micromamba run -n "$ENV" python "$N2T" --ic-input-path "$ICNPZ" --output "$IC" --n-particles 200000
fi

# NFW params (G=1 code units) from the SAME IC
read NFW_M NFW_RS < <(micromamba run -n "$ENV" python -c "
import numpy as np; z=np.load('$ICNPZ')
print(float(z['halo_mass_code']), float(z['halo_rs_code']))")
log "external NFW: M=$NFW_M r_s=$NFW_RS (G=1 code units)"

GPU=$(bash "$HERE/pick_gpu.sh") || { log "no free GPU"; exit 1; }
log "using GPU $GPU"

log "starting 2 Gyr / 4000-step Bonsai run (dt=5e-4, eps=0.002, theta=0.5, snap 0.005)..."
rm -f snapshots/bonsai_*
CUDA_VISIBLE_DEVICES=$GPU ODISSEO_NFW_G=1.0 ODISSEO_NFW_M=$NFW_M ODISSEO_NFW_RS=$NFW_RS \
  "$BON" -i "$IC" --dev 0 \
  -t 0.0005 -T 2.0 -e 0.002 -o 0.5 -r 1 \
  --snapname snapshots/bonsai --snapiter 0.005 --log 2>&1 | tee "$OUT/bonsai_reference.log"
rc=${PIPESTATUS[0]}
log "Bonsai integration finished rc=$rc; snapshots: $(ls snapshots/bonsai_*-0 2>/dev/null | wc -l)"

# optional movies (comment out to skip)
if [ "${BONSAI_RENDER:-0}" = "1" ]; then
  log "rendering movies (JAX on CPU)..."
  JAX_PLATFORMS=cpu micromamba run -n "$ENV" python "$REN" \
    --snap-glob "snapshots/bonsai_*-0" \
    --movie-path "$OUT/bonsai_ref.gif" \
    --render-projections xy,xz --render-resolution 800 --movie-fps 25 --render-cmap magma \
    --diag-output-prefix "$OUT/bonsai_ref_diagnostics"
fi
log "DONE (grep 'Loop alone took' in bonsai_reference.log for the timing)"
