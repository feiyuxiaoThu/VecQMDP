#!/usr/bin/env bash
# =============================================================================
# Build script for Rock Sample DESPOT and FVRS solvers only.
# 
# This script builds only the DESPOT-based solvers:
#   - Vec-QMDP_rock_sample (DESPOT solver on rock sample)
#   - Vec-QMDP_fvrs (FVRS solver on rock sample)
#
# Usage:
#   ./build_despot.sh [--debug] [--jobs N] [--build-dir PATH] [--clean]
#
#   --debug          Build in Debug mode (default: Release)
#   --jobs N         Use N parallel jobs (default: auto-detect)
#   --build-dir PATH Custom build directory (default: <repo>/build)
#   --clean          Remove the build directory before starting
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
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \?//'
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
echo " Rock Sample DESPOT/FVRS Build Script"
echo "============================================="
echo "  Repo root    : ${REPO_ROOT}"
echo "  Build dir    : ${BUILD_DIR}"
echo "  Build type   : ${BUILD_TYPE}"
echo "  Clean build  : ${DO_CLEAN}"
echo "  Parallel jobs: ${N_JOBS}"
echo "  SIMD enabled : ${ENABLE_SIMD}"
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

cmake \
    -S "${REPO_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -DOPTIMIZATION_LEVEL="${OPT_LEVEL}" \
    -DBUILD_ROCK_SAMPLE_EXAMPLES=ON \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BUILD_DIR}/bin" \
    || { echo "Error: CMake configuration failed."; exit 1; }

echo ""

# ---------------------------------------------------------------------------
# Build DESPOT and FVRS targets only
# ---------------------------------------------------------------------------
echo "[2/2] Building DESPOT and FVRS targets..."

TARGETS=(
    "Vec-QMDP_rock_sample"
    "Vec-QMDP_fvrs"
)

for TARGET in "${TARGETS[@]}"; do
    echo "  Building: ${TARGET}"
    cmake --build "${BUILD_DIR}" \
        --target "${TARGET}" \
        --parallel "${N_JOBS}" \
        || { echo "Error: Build failed for target: ${TARGET}"; exit 1; }
done

echo "============================================="
echo " DESPOT/FVRS Build complete!"
echo " Binaries:"
for TARGET in "${TARGETS[@]}"; do
    echo "   - ${BUILD_DIR}/bin/${TARGET}"
done
echo "============================================="
