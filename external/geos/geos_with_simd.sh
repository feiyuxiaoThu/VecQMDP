#!/bin/bash

# GEOS with AVX SIMD support build script

set -e  # Exit on any error

echo "=== GEOS AVX SIMD Build Script ==="
echo "Building GEOS with AVX/AVX2 SIMD support..."

# Set build directory
BUILD_DIR="build-avx"
INSTALL_PREFIX="$HOME/.local"

# Clean previous build if exists
# if the argument include -c, clean the build directory
# if [ -d "$BUILD_DIR" ]; then
#     echo "Cleaning previous build directory..."
#     rm -rf "$BUILD_DIR"
# fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring build with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-O3 -mavx -mavx2 -mfma -msse4.1 -msse4.2 -march=native -fPIC" \
    -DCMAKE_C_FLAGS="-O3 -mavx -mavx2 -mfma -msse4.1 -msse4.2 -march=native -fPIC" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=OFF \
    -DGEOS_BUILD_DEVELOPER=OFF

# Build
echo "Building GEOS..."
make -j$(nproc)

# Install
echo "Installing GEOS..."
make install

# # Run tests (optional)
# echo "Running tests..."
# make test

echo "=== Build completed successfully! ==="
echo "To install, run: cd $BUILD_DIR && sudo make install"
echo "Static library files are in: $BUILD_DIR/lib/"
echo "Executable files are in: $BUILD_DIR/bin/"
echo ""
echo "=== Static Library Information ==="
echo "GEOS static libraries will be installed to: $INSTALL_PREFIX/lib/"
echo "- libgeos.a (C++ API)"
echo "- libgeos_c.a (C API)"
echo ""
echo "This will provide better performance and eliminate runtime library dependencies." 