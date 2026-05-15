################################################################################
# GEOS CMakeLists.txt --- Reconstructed for GEOS 3.14.0dev
#
# Simplified flat build: collects all .cpp sources via GLOB_RECURSE,
# generates version headers, builds libgeos.a and libgeos_c.a, and installs.
################################################################################
cmake_minimum_required(VERSION 3.15)

# --- Version constants -------------------------------------------------------
set(VERSION_MAJOR   3)
set(VERSION_MINOR   14)
set(VERSION_PATCH   "0dev")
set(VERSION         "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
set(JTS_PORT        "1.18.0")
set(CAPI_VERSION_MAJOR 1)
set(CAPI_VERSION_MINOR 20)
set(CAPI_VERSION_PATCH 0)
set(CAPI_VERSION    "${CAPI_VERSION_MAJOR}.${CAPI_VERSION_MINOR}.${CAPI_VERSION_PATCH}")

project(GEOS
    VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.0"
    LANGUAGES C CXX)

# --- Build options -----------------------------------------------------------
option(BUILD_SHARED_LIBS    "Build shared libraries"       OFF)
option(BUILD_TESTING        "Build test suite"             OFF)
option(GEOS_BUILD_DEVELOPER "Enable developer build flags" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# --- Generate version headers from .in templates ----------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/geos/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/geos/version.h"
    @ONLY)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/capi/geos_c.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/geos_c.h"
    @ONLY)

# --- Collect C++ sources (all .cpp under src/) --------------------------------
file(GLOB_RECURSE GEOS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Ryu float-to-string helper (pure C, must be compiled as C)
set(GEOS_RYU_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/deps/ryu/d2s.c")

# --- libgeos (C++ core) -------------------------------------------------------
add_library(geos STATIC
    ${GEOS_SOURCES}
    ${GEOS_RYU_SOURCE})

target_include_directories(geos
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/deps/ryu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/deps"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../external/vamp/src/impl")

target_compile_definitions(geos
    PUBLIC
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
        USE_UNSTABLE_GEOS_CPP_API)

target_compile_options(geos PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -ffp-contract=off>)

# Link Eigen3 if available (used by GEOS SIMD geometry operations)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(geos PUBLIC Eigen3::Eigen)
    message(STATUS "GEOS: Linking against Eigen3")
else()
    message(STATUS "GEOS: Eigen3 not found, building without it")
endif()

# --- libgeos_c (C API wrapper) ------------------------------------------------
add_library(geos_c STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/capi/geos_c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/capi/geos_ts_c.cpp")

target_include_directories(geos_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(geos_c PUBLIC geos)

# --- Installation ------------------------------------------------------------
install(TARGETS geos geos_c
    EXPORT geos-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/geos"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/geos/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/geos")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/geos_c.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# --- CMake export targets ----------------------------------------------------
install(EXPORT geos-targets
    FILE        geos-targets.cmake
    NAMESPACE   GEOS::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GEOS")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/geos-config-version.cmake"
"set(PACKAGE_VERSION \"${VERSION_MAJOR}.${VERSION_MINOR}.0\")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
")

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/geos-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/geos-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GEOS")

# --- pkg-config --------------------------------------------------------------
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/geos.pc"
"prefix=${CMAKE_INSTALL_PREFIX}
exec_prefix=\${prefix}
includedir=\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}
libdir=\${prefix}/${CMAKE_INSTALL_LIBDIR}

Name: GEOS
Description: Geometry Engine, Open Source - C API
Requires:
Version: ${VERSION}
Cflags: -I\${includedir}
Libs: -L\${libdir} -lgeos_c
Libs.private: -lgeos -lstdc++ -lm
")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/geos.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
