#!/usr/bin/env bash
set -euo pipefail

# Run pytest in the odisseo env, preferring autocvd from that env.
# Usage:
#   tools/run_pytest_gpu.sh [pytest args...]
# Optional env:
#   AUTO_CVD_NUM_GPUS (default: 1)
#   AUTO_CVD_TIMEOUT  (default: 20 seconds)

NUM_GPUS="${AUTO_CVD_NUM_GPUS:-1}"
TIMEOUT_S="${AUTO_CVD_TIMEOUT:-20}"

if micromamba run -n odisseo which autocvd >/dev/null 2>&1; then
  if CUDA_EXPORTS="$(micromamba run -n odisseo autocvd -q -e -n "${NUM_GPUS}" -t "${TIMEOUT_S}" 2>/dev/null)" && [ -n "${CUDA_EXPORTS}" ]; then
    eval "${CUDA_EXPORTS}"
  else
    # Fallback: immediately choose least-used GPU(s) instead of hanging.
    eval "$(micromamba run -n odisseo autocvd -q -e -n "${NUM_GPUS}" -l)"
  fi
fi

exec micromamba run -n odisseo python -m pytest "$@"
