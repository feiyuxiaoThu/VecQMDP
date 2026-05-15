#!/usr/bin/env bash
# =============================================================================
# Build script for Rock Sample Vec-QMDP solver only.
# 
# This script builds only the Vec-QMDP_rock_sample_vecqmdp target, which uses
# either the static VecQMDP or dynamic DynVecQMDP solver implementation.
#
# Usage:
#   ./build_vec_qmdp.sh [--debug] [--jobs N] [--build-dir PATH] [--clean] [--static-solver]
#
#   --debug          Build in Debug mode (default: Release)
#   --jobs N         Use N parallel jobs (default: auto-detect)
#   --build-dir PATH Custom build directory (default: <repo>/build)
#   --clean          Remove the build directory before starting
#   --static-solver  Use static pre-allocated VecQMDP solver (default: dynamic)
#   --no-simd        Disable SIMD optimizations
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Script location
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
BUILD_TYPE="Release"
OPT_LEVEL="O3"
ENABLE_SIMD=true
N_JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
BUILD_DIR="${REPO_ROOT}/build"
DO_CLEAN=false
USE_STATIC_SOLVER=false

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-simd)
            ENABLE_SIMD=false
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            OPT_LEVEL="O0"
            shift
            ;;
        --jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --static-solver)
            USE_STATIC_SOLVER=true
            shift
            ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "============================================="
echo " Rock Sample Vec-QMDP Build Script"
echo "============================================="
echo "  Repo root    : ${REPO_ROOT}"
echo "  Build dir    : ${BUILD_DIR}"
echo "  Build type   : ${BUILD_TYPE}"
echo "  Clean build  : ${DO_CLEAN}"
echo "  Parallel jobs: ${N_JOBS}"
echo "  SIMD enabled : ${ENABLE_SIMD}"
echo "  Static solver: ${USE_STATIC_SOLVER}"
echo "============================================="

# ---------------------------------------------------------------------------
# Handle Clean
# ---------------------------------------------------------------------------
if [ "$DO_CLEAN" = true ]; then
    if [ -d "${BUILD_DIR}" ]; then
        echo "Cleaning build directory: ${BUILD_DIR}..."
        rm -rf "${BUILD_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# Handle SIMD compiler flags
# ---------------------------------------------------------------------------
EXTRA_CXX_FLAGS=""
if [ "$ENABLE_SIMD" = true ] && [ "$BUILD_TYPE" == "Release" ]; then
    EXTRA_CXX_FLAGS="-march=native"
    echo "  SIMD optimization: Enabled (-march=native)"
else
    echo "  SIMD optimization: Disabled"
fi

mkdir -p "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# CMake configure
# ---------------------------------------------------------------------------
echo "[1/2] Configuring CMake..."
STATIC_SOLVER_FLAG="OFF"
if [ "$USE_STATIC_SOLVER" = true ]; then
    STATIC_SOLVER_FLAG="ON"
    echo "  Static solver  : Enabled (USE_STATIC_SOLVER=ON)"
else
    echo "  Static solver  : Disabled (default DynVecQMDP)"
fi

cmake \
    -S "${REPO_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -DOPTIMIZATION_LEVEL="${OPT_LEVEL}" \
    -DBUILD_ROCK_SAMPLE_EXAMPLES=ON \
    -DUSE_STATIC_SOLVER="${STATIC_SOLVER_FLAG}" \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BUILD_DIR}/bin" \
    || { echo "Error: CMake configuration failed."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Build Vec-QMDP target only
# ---------------------------------------------------------------------------
echo "[2/2] Building Vec-QMDP target..."

TARGET="Vec-QMDP_rock_sample_vecqmdp"
echo "  Building: ${TARGET}"
cmake --build "${BUILD_DIR}" \
    --target "${TARGET}" \
    --parallel "${N_JOBS}" \
    || { echo "Error: Build failed for target: ${TARGET}"; exit 1; }

echo "============================================="
echo " Vec-QMDP Build complete!"
echo " Binary: ${BUILD_DIR}/bin/${TARGET}"
echo "============================================="
