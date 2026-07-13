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
export JACCPOT_STATIC_STRICT_GPU_MODE=on
export JACCPOT_STATIC_STRICT_FUSED_MODE=on
export JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=200000
export JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=64
# neighbor-edge cap sized for 200k on an 11 GB card -- exactly the 2080 Ti case.
export JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP=2097152
export JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH=0

# device-only fused fast-lane (the ~10x win; now the built-in default)
export JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY=1
export JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK=1
export JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS=1
export JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP=131072
export JACCPOT_LARGE_N_COMPILED_STATE_MODE=on
export JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED=1

# sm_75: no Pallas kernels.
export ODISSEO_FMM_USE_PALLAS=0
