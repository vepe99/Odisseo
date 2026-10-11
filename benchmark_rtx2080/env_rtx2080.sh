# Shared environment for the RTX 2080 Ti (sm_75) benchmark.
# Source this before the GPU runs:  source benchmark_rtx2080/env_rtx2080.sh
#
# Same device-resident fused static-radix fast-lane as the A100 benchmark
# (env_fused.sh) -- this is the built-in default since commit 70811e6 and is
# pure-JAX, so it runs on sm_75. The ONE difference vs the A100 env: Pallas is
# OFF. The Pallas near-field / M2L kernels are gated to compute capability >= 8.0
# (Ampere); the RTX 2080 Ti is sm_75 (Turing), so jaccpot would fall back to the
# pure-JAX path anyway -- we set it explicitly to 0 so the timing is unambiguous.

export JAX_ENABLE_X64=1
export JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=200000
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=64
# neighbor-edge cap sized for 200k on an 11 GB card -- exactly the 2080 Ti case.
export JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=2097152
export JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=131072
# The device-only fused fast lane needs no switches: since jaccpot's 2026-10
# cleanup it is the default, and GPU_MODE, FUSED_MODE, DEVICE_ONLY,
# DISALLOW_HOST_SEGMENT_FALLBACK, REQUIRE_EXACT_CAP_PROFILE_MATCH and
# LARGE_N_COMPILED_STATE_MODE are read nowhere (X4/X6); FLAT_COMPACT_FAR_PAIRS=1
# and RADIX_FAST_PAYLOAD_IN_FUSED=1 are jaccpot's defaults.

# sm_75: no Pallas kernels.
export ODISSEO_FMM_USE_PALLAS=0
