# Check if LTO is already set in parent CMakeLists
if(NOT DEFINED VEC_QMDP_LTO)
  option(VEC_QMDP_LTO "Use Link Time Optimization" ON)
endif()

add_library(vec_qmdp_collision OBJECT
    STRtree.cpp
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/contrib/sort/vqsort.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/contrib/sort/vqsort_kv64a.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/contrib/sort/vqsort_kv64d.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/targets.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/per_target.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/abort.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/aligned_allocator.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/print.cc
    ${PROJECT_SOURCE_DIR}/external/highway/hwy/timer.cc
)

# Use globally defined VAMP library path
if(NOT DEFINED VAMP_INCLUDE_DIR)
    set(VAMP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/vamp/src/impl)
endif()

target_include_directories(vec_qmdp_collision
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${VAMP_INCLUDE_DIR}  # Add vamp library include path
    ${PROJECT_SOURCE_DIR}/external/highway  # highway VQSort
)

# Architecture-specific SIMD optimization compile options
if(ARCH_X86_64)
    target_compile_options(vec_qmdp_collision PRIVATE
        -O3
        -mavx -mavx2 -mfma
        -ftree-vectorize
        -Wno-ignored-attributes
    )
elseif(ARCH_AARCH64)
    target_compile_options(vec_qmdp_collision PRIVATE
        -O3
        -march=armv8-a+simd
        -ftree-vectorize
        -Wno-ignored-attributes
    )
else()
    target_compile_options(vec_qmdp_collision PRIVATE
        -O3
        -ftree-vectorize
        -Wno-ignored-attributes
    )
endif()

# Set position-independent code for dynamic library building
set_target_properties(vec_qmdp_collision PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
)

# Enable LTO
if(VEC_QMDP_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT error)
  if(ipo_supported)
    set_property(TARGET vec_qmdp_collision PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "LTO is not supported: ${error}")
  endif()
endif()