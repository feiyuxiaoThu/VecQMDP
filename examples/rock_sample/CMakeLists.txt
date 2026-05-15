cmake_minimum_required(VERSION 3.15)
# project(RockSampleVec)

# --- SIMD optimization toggle ---
option(ENABLE_SIMD "Enable SIMD optimizations" ON)
if(ENABLE_SIMD)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ftree-vectorize")
        message(STATUS "SIMD optimizations enabled: -march=native")
    endif()
endif()

# ---------------------------------------------------------------------------
# DESPOT static library (bundled in external/)
# ---------------------------------------------------------------------------
# DESPOT sources use #include <despot/util/...> but the actual include tree
# is external/despot/include/{util,core,...}.  We create a symlink
#   <build>/despot_include/despot  ->  <source>/external/despot/include
# so that the <despot/...> prefix resolves correctly.
set(_DESPOT_INCLUDE_ALIAS "${CMAKE_BINARY_DIR}/despot_include")
file(MAKE_DIRECTORY "${_DESPOT_INCLUDE_ALIAS}")
if(NOT EXISTS "${_DESPOT_INCLUDE_ALIAS}/despot")
    file(CREATE_LINK
        "${PROJECT_SOURCE_DIR}/external/despot/include"
        "${_DESPOT_INCLUDE_ALIAS}/despot"
        SYMBOLIC
    )
endif()

file(GLOB_RECURSE DESPOT_SOURCES
    "${PROJECT_SOURCE_DIR}/external/despot/src/*.cpp"
)

add_library(despot STATIC ${DESPOT_SOURCES})

target_include_directories(despot PUBLIC "${_DESPOT_INCLUDE_ALIAS}")

target_compile_options(despot PRIVATE -w)   # suppress all DESPOT warnings

# ---------------------------------------------------------------------------
# Shared base rock-sample model (used by both despot and fvrs targets)
# ---------------------------------------------------------------------------
set(BASE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/base/base_rock_sample.cpp"
)

# ---------------------------------------------------------------------------
# Target 1: Vec-QMDP_rock_sample  (DESPOT solver on rock sample)
# ---------------------------------------------------------------------------
add_executable(Vec-QMDP_rock_sample
    ${BASE_SOURCES}
    src/rock_sample_despot/rock_sample_despot.cpp
    src/rock_sample_despot/main.cpp
)

target_include_directories(Vec-QMDP_rock_sample PRIVATE
    "${_DESPOT_INCLUDE_ALIAS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(Vec-QMDP_rock_sample PRIVATE despot)

# ---------------------------------------------------------------------------
# Target 2: Vec-QMDP_fvrs  (FVRS solver on rock sample)
# ---------------------------------------------------------------------------
add_executable(Vec-QMDP_fvrs
    ${BASE_SOURCES}
    src/fvrs/fvrs.cpp
    src/fvrs/main.cpp
)

target_include_directories(Vec-QMDP_fvrs PRIVATE
    "${_DESPOT_INCLUDE_ALIAS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(Vec-QMDP_fvrs PRIVATE despot)

# ---------------------------------------------------------------------------
# Target 3: Vec-QMDP_rock_sample_vecqmdp  (Vec-QMDP solver on rock sample)
#
# Solver selection (choose one at cmake configure time):
#   cmake -DUSE_STATIC_SOLVER=ON  ...   →  static pre-allocated VecQMDP
#   cmake -DUSE_STATIC_SOLVER=OFF ...   →  dynamic pool DynVecQMDP (default)
#
# Both solver source files are always compiled so the binary contains both
# implementations.  The active solver is selected at compile time via the
# USE_STATIC_SOLVER macro (defined either here or inside main.cpp).
# ---------------------------------------------------------------------------

option(USE_STATIC_SOLVER
    "Use static (pre-allocated tree) VecQMDP solver instead of DynVecQMDP" OFF)

add_executable(Vec-QMDP_rock_sample_vecqmdp
    ${BASE_SOURCES}
    src/rock_sample_despot/rock_sample_despot.cpp
    src/rock_sample_vecqmdp/rock_sample_dynamic_vecqmdp.cpp
    src/rock_sample_vecqmdp/rock_sample_static_vecqmdp.cpp
    src/rock_sample_vecqmdp/main.cpp
)

if(USE_STATIC_SOLVER)
    target_compile_definitions(Vec-QMDP_rock_sample_vecqmdp PRIVATE USE_STATIC_SOLVER)
    message(STATUS "Vec-QMDP solver: Static VecQMDP (pre-allocated tree)")
else()
    message(STATUS "Vec-QMDP solver: Dynamic VecQMDP (on-demand pool) [default]")
endif()

target_include_directories(Vec-QMDP_rock_sample_vecqmdp PRIVATE
    "${_DESPOT_INCLUDE_ALIAS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(Vec-QMDP_rock_sample_vecqmdp PRIVATE
    vec_qmdp
    despot
)
