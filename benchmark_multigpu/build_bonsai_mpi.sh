#!/usr/bin/env bash
# =============================================================================
# !!! DISABLED / DO NOT RUN (decision 2026-07-11) !!!
# The MPI multi-GPU Bonsai path is NOT used in this benchmark. During its 2-rank
# smoke test it hit CUDA error 700 (illegal address) and WEDGED THE ENTIRE GPU
# NODE (nvidia-smi hung; D-state processes across multiple users; needed an admin
# reset). Bonsai is therefore a SINGLE-GPU baseline only (see common/bonsai.py,
# BONSAI_BIN = runtime/build/bonsai2_slowdust). This script + its findings are
# kept for reference ONLY. If ever revisited: keep ENABLE_SM90=0 (sm_80+PTX;
# native sm_90 is the crash trigger per the NOTE below), start with -I to cap
# iterations, and validate on a NON-shared node -- a repeat wedge affects others.
# =============================================================================
# build_bonsai_mpi.sh
#
# Reproducible build of a MULTI-GPU (MPI-enabled) Bonsai N-body binary, for use
# as a multi-GPU competitor in the ODISSEO benchmark. It builds into a SEPARATE
# directory (runtime/build_mpi) and does NOT touch the existing single-GPU build
# (runtime/build, USE_MPI=OFF), so both binaries coexist.
#
# Host this was verified on: compgpu11, 4x NVIDIA H100 (sm_90), Ubuntu 24.04,
#   CUDA 11.8, OpenMPI 4.1.6 (mpicxx wraps g++-13), nvcc host compiler g++-11.
#
# What it does:
#   1. Applies two tiny, MPI-only source fixes that GCC 13 requires (the MPI code
#      path had bit-rotted: it relied on <cstdint>/<array> being included
#      transitively, which newer libstdc++ no longer does). Idempotent.
#   2. Configures runtime/build_mpi with -DUSE_MPI=ON -DUSE_MPIMT=ON using the
#      MPI compiler wrappers (Bonsai has NO find_package(MPI); it locates MPI
#      purely through CMAKE_CXX_COMPILER=mpicxx / CMAKE_C_COMPILER=mpicc).
#   3. Builds ONLY the `bonsai2_slowdust` target with `make -j`. Building that
#      one target (rather than a bare `make`) avoids a FindCUDA parallel-build
#      race on the shared per-.cu ".depend" temp files between the
#      bonsai2_slowdust and bonsai_amuse targets (harmless "Error copying file
#      ... .depend.tmp to .depend" failures under -j).
#
# GPU architecture:
#   Default GENCODE = sm_80 SASS + compute_80 PTX (this is exactly what the
#   working single-GPU reference binary uses). On the H100 the compute_80 PTX is
#   JIT-compiled to sm_90 at load, so it runs at native speed.
#   NOTE: building EXTRA native sm_90 SASS (ENABLE_SM90=1 below) compiles fine,
#   but at runtime the 2-rank job crashed with "CUDA Runtime API error 700:
#   illegal memory access" in the first (LET) gravity kernel. sm_80+PTX does NOT
#   crash. Leave ENABLE_SM90=0 unless you re-validate native sm_90.
#
# GPU org policy: any GPU execution MUST pick free GPUs via `autocvd`
#   (env `odisseo`), never hard-code device 0. See the run examples at the end.
#
# Usage:
#   bash build_bonsai_mpi.sh            # configure + build (default)
#   ENABLE_SM90=1 bash build_bonsai_mpi.sh   # ALSO emit native sm_90 (see NOTE)
#   CLEAN=1 bash build_bonsai_mpi.sh    # wipe build_mpi first (full reconfigure)
# =============================================================================
set -euo pipefail

# ---- paths / toolchain (edit here if your layout differs) -------------------
BONSAI_SRC=/export/home/tbuck/Bonsai/runtime          # CMake project root
BUILD_DIR="${BONSAI_SRC}/build_mpi"                   # separate MPI build dir
CUDA_HOME=/usr/local/cuda-11.8
CUDA_HOST_COMPILER=/usr/bin/g++-11                    # CUDA 11.8 supports <= gcc 11
MPICXX="$(command -v mpicxx)"
MPICC="$(command -v mpicc)"
JOBS="${JOBS:-8}"
ENABLE_SM90="${ENABLE_SM90:-0}"
CLEAN="${CLEAN:-0}"

export PATH="${CUDA_HOME}/bin:${PATH}"

echo "[build] Bonsai src      : ${BONSAI_SRC}"
echo "[build] build dir       : ${BUILD_DIR}"
echo "[build] mpicxx          : ${MPICXX}  ($(mpicxx --showme:version 2>/dev/null | head -1))"
echo "[build] nvcc            : $(nvcc --version | grep release)"
echo "[build] CUDA host cc     : ${CUDA_HOST_COMPILER}"
echo "[build] ENABLE_SM90     : ${ENABLE_SM90}"

# ---- 1. idempotent MPI-only source fixes (required under GCC 13/libstdc++) ---
IDTYPE_H="${BONSAI_SRC}/include/IDType.h"
OCTREE_CPP="${BONSAI_SRC}/src/octree.cpp"

if ! grep -q '#include <cstdint>' "${IDTYPE_H}"; then
  echo "[patch] adding <cstdint> to IDType.h"
  # insert right after the '#pragma once' line
  sed -i '/#pragma once/a #include <cstdint>   // uint32_t\/uint64_t\/int64_t (GCC 13+ no longer transitively includes this)' "${IDTYPE_H}"
else
  echo "[patch] IDType.h already has <cstdint> (ok)"
fi

if ! grep -q '#include <array>' "${OCTREE_CPP}"; then
  echo "[patch] adding <array>/<algorithm> to octree.cpp"
  # insert after the first include (#include "octree.h")
  sed -i '0,/#include "octree.h"/s//#include "octree.h"\n\n#include <array>       \/\/ std::array (MPI tipsy read path; GCC 13+ needs explicit include)\n#include <algorithm>   \/\/ std::fill/' "${OCTREE_CPP}"
else
  echo "[patch] octree.cpp already has <array> (ok)"
fi

# ---- 2. configure -----------------------------------------------------------
if [[ "${CLEAN}" == "1" ]]; then
  echo "[build] CLEAN=1 -> removing ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

# Extra native gencode (optional; default none). sm_80 + compute_80 PTX come
# from runtime/CMakeLists.txt (GENCODE var) and are ALWAYS built.
NVCC_EXTRA=""
if [[ "${ENABLE_SM90}" == "1" ]]; then
  NVCC_EXTRA="-gencode;arch=compute_90,code=sm_90"
  echo "[build] WARNING: adding native sm_90 -- see NOTE at top (error 700 risk)"
fi

echo "[build] configuring..."
cmake -S "${BONSAI_SRC}" -B "${BUILD_DIR}" \
      -DUSE_MPI=ON -DUSE_MPIMT=ON \
      -DCMAKE_C_COMPILER="${MPICC}" -DCMAKE_CXX_COMPILER="${MPICXX}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
      -DCUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
      -DCUDA_NVCC_FLAGS:STRING="${NVCC_EXTRA}"

# ---- 3. build (single target to dodge the FindCUDA .depend -j race) ---------
echo "[build] compiling bonsai2_slowdust (make -j${JOBS})..."
make -C "${BUILD_DIR}" -j"${JOBS}" bonsai2_slowdust

BIN="${BUILD_DIR}/bonsai2_slowdust"
echo "[build] DONE -> ${BIN}"
echo "[build] MPI linkage:"; ldd "${BIN}" | grep -i mpi || echo "  (no libmpi in ldd -- NOT an MPI build!)"
echo "[build] CUDA archs embedded:"; cuobjdump "${BIN}" 2>/dev/null | grep -E "arch = sm_" | sort | uniq -c

cat <<'EOF'

=============================================================================
 HOW TO RUN THE MULTI-GPU BINARY
=============================================================================
Bonsai assigns GPUs by MPI rank:  devID = rank % (#visible CUDA devices)
(see runtime/src/main.cpp). So every rank must see the SAME set of N GPUs via
CUDA_VISIBLE_DEVICES, and -np N must equal that GPU count -> each rank lands on
a distinct GPU. Do NOT pass --dev in multi-rank mode (it is overridden).

Pick free GPUs with autocvd (org policy) -- lives in the `odisseo` env:

  BIN=/export/home/tbuck/Bonsai/runtime/build_mpi/bonsai2_slowdust
  IC=/export/home/tbuck/Odisseo/benchmark_a100/bonsai_reference/disk_ic.tipsy

  # ---- 2 GPUs / 2 ranks ----
  CVD=$(micromamba run -n odisseo autocvd -n 2 -l -o -q | tail -1)   # e.g. "2,3"
  CUDA_VISIBLE_DEVICES=$CVD mpirun -np 2 -x CUDA_VISIBLE_DEVICES \
    "$BIN" -i "$IC" -t 0.0005 -T 2.0 -e 0.002 -o 0.5 -r 1 --log --prepend-rank

  # ---- 3 GPUs / 3 ranks ----
  CVD=$(micromamba run -n odisseo autocvd -n 3 -l -o -q | tail -1)   # e.g. "1,2,3"
  CUDA_VISIBLE_DEVICES=$CVD mpirun -np 3 -x CUDA_VISIBLE_DEVICES \
    "$BIN" -i "$IC" -t 0.0005 -T 2.0 -e 0.002 -o 0.5 -r 1 --log --prepend-rank

Flags: -t dt, -T t_end, -I max_iterations (cap steps for a smoke test),
       -e softening, -o opening-angle(theta), -r tree-rebuild cadence.
Optional STATIC external NFW halo (matches ODISSEO IC), set on every rank:
   -x ODISSEO_NFW_G=1.0 -x ODISSEO_NFW_M=<Msys> -x ODISSEO_NFW_RS=<rs>
(read via getenv in runtime/src/gpu_iterate.cpp; no-op if ODISSEO_NFW_M unset).

Caveats:
  * Single node only (verified on compgpu11). For multi-node add a hostfile.
  * NOT CUDA-aware MPI: Bonsai stages MPI buffers through host memory, so a
    plain (non-UCX/GDR) OpenMPI is fine.
  * OpenMPI may print a benign "Falling to basic forking method after MPI_Init"
    line -- harmless (Bonsai's per-rank RNG seed helper), the run proceeds.
  * Arch: sm_80 SASS + compute_80 PTX (PTX JITs to sm_90 on H100). Native sm_90
    currently triggers CUDA error 700 in the gravity kernel; keep ENABLE_SM90=0.
=============================================================================
EOF
