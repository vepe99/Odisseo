#!/usr/bin/env bash
# Driver: ODISSEO FMM perf + Bonsai reference on the RTX 2080 Ti, then compare.
#   bash benchmark_rtx2080/run_all.sh            # both + compare
#   bash benchmark_rtx2080/run_all.sh odisseo    # ODISSEO perf only
#   bash benchmark_rtx2080/run_all.sh bonsai     # Bonsai only
#   bash benchmark_rtx2080/run_all.sh compare    # just re-print the summary
# Prereq: a HEALTHY CUDA driver (cuInit must succeed). If GPU2 is still faulted,
# every CUDA process fails with CUDA_ERROR_UNKNOWN -- get GPU2 reset first.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
what="${1:-all}"
preflight() {
  echo "[preflight] cuInit check..."
  GPU=$(bash "$HERE/pick_gpu.sh") || { echo "[preflight] no free GPU"; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU" micromamba run -n odisseo python -c \
    "import jax,sys; b=jax.default_backend(); print('backend',b); sys.exit(0 if b=='gpu' else 3)" \
    2>/dev/null || { echo "[preflight] FAIL: JAX cannot init CUDA on this node (GPU2 fault?). Aborting."; exit 3; }
}
case "$what" in
  odisseo) preflight; bash "$HERE/run_odisseo_perf.sh" ;;
  bonsai)  bash "$HERE/run_bonsai_reference.sh" ;;
  compare) micromamba run -n odisseo python "$HERE/compare.py"; exit 0 ;;
  all)     preflight
           bash "$HERE/run_odisseo_perf.sh"
           bash "$HERE/run_bonsai_reference.sh" ;;
  *) echo "usage: $0 [all|odisseo|bonsai|compare]"; exit 2 ;;
esac
micromamba run -n odisseo python "$HERE/compare.py"
