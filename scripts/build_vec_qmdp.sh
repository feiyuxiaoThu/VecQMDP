#!/usr/bin/env bash

# Script: build_vec_qmdp.sh
# Description: Unified build script for Vec-QMDP, optionally rebuilding and installing
#              the static GEOS library and then building the vec_qmdp_closed_planner
#              shared library for use from Python.
# Shell: bash

set -euo pipefail

# Colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Move to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo -e "${YELLOW}Starting Vec-QMDP build process...${NC}"

# Determine CPU core count for parallel build
if [[ "${OSTYPE:-}" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=$(nproc)
fi
echo -e "Detected ${GREEN}${NUM_CORES}${NC} CPU cores, using parallel build."

# Detect host architecture and report; CMake will also perform its own detection.
HOST_ARCH=$(uname -m)
case "${HOST_ARCH}" in
    x86_64)
        echo -e "Host architecture: ${GREEN}x86_64${NC} — AVX/AVX2/FMA SIMD path will be used."
        ;;
    aarch64|arm64)
        echo -e "Host architecture: ${GREEN}aarch64${NC} — NEON SIMD path will be used."
        ;;
    *)
        echo -e "${YELLOW}Warning: Unknown host architecture '${HOST_ARCH}'. No architecture-specific SIMD flags will be applied.${NC}"
        ;;
esac


# Check for cmake
if ! command -v cmake &>/dev/null; then
    echo -e "${RED}Error: cmake not found in PATH.${NC}"
    echo -e "${YELLOW}Install cmake via pip (recommended for conda environments):${NC}"
    echo -e "  pip install cmake"
    echo -e "${YELLOW}Or via conda:${NC}"
    echo -e "  conda install cmake"
    exit 1
fi

# Default options
CLEAN_BUILD=0
DEBUG_MODE=0
OPT_LEVEL="O3"
USE_ASAN=0
PERF_PROFILE=0
REBUILD_GEOS=0

show_usage() {
    echo "Usage: $(basename "$0") [options]"
    echo "Options:"
    echo "  --clean, -c          Perform a clean build (remove existing build directory)."
    echo "  --debug, -d          Build in Debug mode."
    echo "  --opt=O1|O2|O3       Set optimization level (default: O3)."
    echo "  --asan, -a           Enable AddressSanitizer for memory error detection."
    echo "  --perf, -p           Enable performance profiling flags (-g -fno-omit-frame-pointer)."
    echo "  --rebuild-geos, -g   Rebuild and install static GEOS library with AVX, then rebuild Vec-QMDP."
    echo "  --help, -h           Show this help message."
    exit 0
}

for arg in "$@"; do
    case "$arg" in
        --clean|-c)
            CLEAN_BUILD=1
            echo -e "${BLUE}Requested clean build...${NC}"
            ;;
        --debug|-d)
            DEBUG_MODE=1
            echo -e "${BLUE}Debug build mode enabled...${NC}"
            ;;
        --opt=*)
            OPT_LEVEL="${arg#*=}"
            if [[ ! "$OPT_LEVEL" =~ ^O[1-3]$ ]]; then
                echo -e "${RED}Error: Invalid optimization level '${OPT_LEVEL}'.${NC}"
                echo -e "${YELLOW}Valid values are: O1, O2, O3.${NC}"
                exit 1
            fi
            echo -e "${BLUE}Using optimization level: ${OPT_LEVEL}${NC}"
            ;;
        --asan|-a)
            USE_ASAN=1
            echo -e "${BLUE}AddressSanitizer enabled...${NC}"
            ;;
        --perf|-p)
            PERF_PROFILE=1
            echo -e "${BLUE}Performance profiling support enabled...${NC}"
            ;;
        --rebuild-geos|-g)
            REBUILD_GEOS=1
            echo -e "${BLUE}Requested static GEOS rebuild and reinstall...${NC}"
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo -e "${RED}Unknown option: ${arg}${NC}"
            show_usage
            ;;
    esac
done

# Optionally rebuild and install static GEOS library with AVX
if [[ "${REBUILD_GEOS}" -eq 1 ]]; then
    echo -e "${YELLOW}=== Rebuilding static GEOS library and Vec-QMDP ===${NC}"

    # Allow overriding the project parent path via environment variable, but default to project root
    VECQMDP_ROOT="$HOME/VecQMDP"

    cd "${VECQMDP_ROOT}/external/geos" || {
            echo -e "${RED}Error: Cannot find ${VECQMDP_ROOT}/external/geos${NC}"
            exit 1
        }
        
    echo -e "${YELLOW}Building GEOS static library with AVX...${NC}"
    bash geos_with_simd.sh || {
        echo -e "${RED}Error: Failed to build GEOS static library with AVX.${NC}"
        exit 1
    }

    echo -e "${YELLOW}Installing GEOS static library...${NC}"
    cd build-avx
    make install || {
        echo -e "${RED}Error: Failed to install GEOS static library.${NC}"
        exit 1
    }

    cd "${PROJECT_ROOT}"
    echo -e "${BLUE}Static GEOS library rebuild and installation completed.${NC}"
fi

# Remove existing shared library if present
EXISTING_LIB="${PROJECT_ROOT}/build/src/vec_qmdp_closed_planner.so"
if [ -f "${EXISTING_LIB}" ]; then
    rm "${EXISTING_LIB}" || {
        echo -e "${RED}Error: Failed to remove existing library ${EXISTING_LIB}.${NC}"
        exit 1
    }
fi

# Handle clean build
if [[ "${CLEAN_BUILD}" -eq 1 && -d "${PROJECT_ROOT}/build" ]]; then
    echo -e "${BLUE}Removing existing build directory...${NC}"
    rm -rf "${PROJECT_ROOT}/build" || {
        echo -e "${RED}Error: Failed to remove build directory.${NC}"
        exit 1
    }
fi

# Create build directory if needed
if [ ! -d "${PROJECT_ROOT}/build" ]; then
    mkdir "${PROJECT_ROOT}/build" || {
        echo -e "${RED}Error: Failed to create build directory.${NC}"
        exit 1
    }
    echo "Created build directory."
fi

# Check VAMP directory presence
if [ ! -d "${PROJECT_ROOT}/external/vamp/src/impl" ]; then
    echo -e "${RED}Warning: VAMP library directory does not exist: ${PROJECT_ROOT}/external/vamp/src/impl${NC}"
    echo -e "${YELLOW}Please ensure the VAMP library is correctly installed or initialized if required.${NC}"
fi

echo -e "${YELLOW}Configuring CMake...${NC}"
BUILD_TYPE="Release"
if [ "${DEBUG_MODE}" -eq 1 ]; then
    BUILD_TYPE="Debug"
fi

# Append conda env prefix to CMAKE_PREFIX_PATH so CMake can find packages installed
# via conda (e.g. Eigen3, GEOS, Boost) without requiring system-wide installation.
# Also explicitly pass PYTHON_EXECUTABLE from CONDA_PREFIX to ensure the Python version
# used for building matches the Boost.Python version installed in the conda environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_PYTHON="${CONDA_PREFIX}/bin/python"
    if [[ ! -x "${CONDA_PYTHON}" ]]; then
        CONDA_PYTHON=$(command -v python3 || command -v python)
    fi
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DOPTIMIZATION_LEVEL=${OPT_LEVEL} -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DPYTHON_EXECUTABLE=${CONDA_PYTHON}"
    echo -e "${BLUE}Using CONDA_PREFIX for CMAKE_PREFIX_PATH: ${CONDA_PREFIX}${NC}"
    echo -e "${BLUE}Using Python executable: ${CONDA_PYTHON}${NC}"
else
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DOPTIMIZATION_LEVEL=${OPT_LEVEL} -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

if [ "${USE_ASAN}" -eq 1 ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DUSE_ASAN=ON"
    echo -e "${YELLOW}Enabling AddressSanitizer in CMake configuration...${NC}"
fi

if [ "${PERF_PROFILE}" -eq 1 ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DUSE_PERF_PROFILE=ON"
    echo -e "${YELLOW}Enabling performance profiling support (-g -fno-omit-frame-pointer)...${NC}"
fi

cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" ${CMAKE_ARGS} || {
    echo -e "${RED}CMake configuration failed.${NC}"
    echo -e "${YELLOW}Please check dependencies and path settings.${NC}"
    exit 1
}

echo -e "${YELLOW}Building project with ${NUM_CORES} parallel jobs...${NC}"
echo -e "${BLUE}Building main target 'vec_qmdp_closed_planner' only.${NC}"

if [ "${DEBUG_MODE}" -eq 1 ]; then
    cmake --build "${PROJECT_ROOT}/build" \
        --target vec_qmdp_closed_planner \
        --parallel "${NUM_CORES}" \
        --config "${BUILD_TYPE}" \
        --verbose 2>&1 | tee "${PROJECT_ROOT}/build/compile_log.txt" || {
        echo -e "${RED}Build failed.${NC}"
        echo -e "${YELLOW}Build log has been saved to build/compile_log.txt.${NC}"
        echo -e "${YELLOW}Please inspect the error messages above.${NC}"
        exit 1
    }
else
    cmake --build "${PROJECT_ROOT}/build" \
        --target vec_qmdp_closed_planner \
        --parallel "${NUM_CORES}" \
        --config "${BUILD_TYPE}" || {
        echo -e "${RED}Build failed.${NC}"
        echo -e "${YELLOW}Re-run with --debug for detailed logs:${NC}"
        echo -e "${YELLOW}  ./scripts/build_vec_qmdp.sh --debug${NC}"
        exit 1
    }
fi

echo -e "${YELLOW}Locating shared library...${NC}"
LIB_EXTENSION="so"
BUILD_DIR="${PROJECT_ROOT}/build"
SRC_LIB_PATH="${BUILD_DIR}/src/vec_qmdp_closed_planner.${LIB_EXTENSION}"
LIB_PATH="${BUILD_DIR}/lib/vec_qmdp_closed_planner.${LIB_EXTENSION}"

if [ -f "${SRC_LIB_PATH}" ]; then
    FOUND_LIB_PATH="${SRC_LIB_PATH}"
    echo -e "${GREEN}Found library in src directory.${NC}"
elif [ -f "${LIB_PATH}" ]; then
    FOUND_LIB_PATH="${LIB_PATH}"
    echo -e "${GREEN}Found library in lib directory.${NC}"
else
    echo -e "${RED}Error: Could not find shared library ${LIB_PATH} or ${SRC_LIB_PATH}.${NC}"
    echo -e "${YELLOW}Please verify the build process and library naming.${NC}"
    exit 1
fi

PYTHON_DIR="${PROJECT_ROOT}/python_planner"
if [ ! -d "${PYTHON_DIR}" ]; then
    echo -e "${RED}Error: Python directory does not exist: ${PYTHON_DIR}.${NC}"
    echo -e "${YELLOW}Please ensure the project structure is complete.${NC}"
    exit 1
fi

echo -e "${YELLOW}Copying shared library to Python directory...${NC}"
cp "${FOUND_LIB_PATH}" "${PYTHON_DIR}/" || {
    echo -e "${RED}Error: Failed to copy shared library to Python directory.${NC}"
    echo -e "${YELLOW}Please check file permissions.${NC}"
    exit 1
}

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "Build details:"
echo -e "  - Build type: ${GREEN}${BUILD_TYPE}${NC}"
echo -e "  - Optimization level: ${GREEN}${OPT_LEVEL}${NC}"

if [ "${USE_ASAN}" -eq 1 ]; then
    echo -e "  - AddressSanitizer: ${GREEN}enabled${NC}"
    echo -e "${YELLOW}Note: AddressSanitizer will slow down execution and is intended for debugging only.${NC}"
    echo -e "You may need to set environment variables when running:"
    echo -e "${YELLOW}export ASAN_OPTIONS=detect_leaks=1:symbolize=1${NC}"
    if [[ "${OSTYPE:-}" != "darwin"* ]]; then
        echo -e "If you see runtime loader order issues with ASan, try:"
        echo -e "${YELLOW}export LD_PRELOAD=\$(gcc -print-file-name=libasan.so)${NC}"
    fi
else
    echo -e "  - AddressSanitizer: ${RED}disabled${NC}"
fi

if [ "${PERF_PROFILE}" -eq 1 ]; then
    echo -e "  - Performance profiling support: ${GREEN}enabled${NC}"
    echo -e "${YELLOW}Debug information is included; you can use perf for profiling.${NC}"
    echo -e "Example perf usage:"
    echo -e "${YELLOW}perf record -g <your_program_command>${NC}"
    echo -e "${YELLOW}perf report${NC}"
else
    echo -e "  - Performance profiling support: ${RED}disabled${NC}"
    echo -e "${YELLOW}To enable profiling, rebuild with the --perf option:${NC}"
    echo -e "${YELLOW}  ./scripts/build_vec_qmdp.sh --perf${NC}"
fi

echo -e "Shared library has been copied to: ${GREEN}${PYTHON_DIR}/$(basename "${FOUND_LIB_PATH}")${NC}"
echo ""
echo -e "You can test importing the library from Python with:"
echo -e "${YELLOW}cd ${PYTHON_DIR}${NC}"
echo -e "${YELLOW}python -c \"import vec_qmdp_closed_planner\"${NC}"
echo ""
echo -e "If you encounter import errors, you may need to extend PYTHONPATH:"
echo -e "${YELLOW}export PYTHONPATH=\$PYTHONPATH:${PYTHON_DIR}${NC}"
echo ""
echo -e "To debug shared library dependencies, you can run:"
echo -e "${YELLOW}ldd ${PYTHON_DIR}/$(basename "${FOUND_LIB_PATH}")${NC}"

