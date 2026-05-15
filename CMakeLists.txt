cmake_minimum_required(VERSION 3.15)
project(HiP-Drive VERSION 1.0.0 LANGUAGES CXX)

# Suppress warning for imported targets that may lack IMPORTED_LOCATION
# (commonly triggered by Boost::python / Boost::numpy on some platforms)
if(POLICY CMP0111)
    cmake_policy(SET CMP0111 OLD)
endif()

# ---------------------------------------------------------------------------
# Architecture detection — sets ARCH_X86_64 or ARCH_AARCH64 cache variables
# and propagates the corresponding preprocessor macro to all targets.
# CMAKE_SYSTEM_PROCESSOR reflects the *target* machine, which is correct for
# both native and cross-compilation scenarios.
# ---------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch_lower)
if(_arch_lower MATCHES "x86_64|amd64")
    set(ARCH_X86_64 TRUE CACHE BOOL "Target architecture is x86_64" FORCE)
    set(ARCH_AARCH64 FALSE CACHE BOOL "Target architecture is aarch64" FORCE)
    add_definitions(-DARCH_X86_64=1)
    message(STATUS "Target architecture: x86_64 — macro ARCH_X86_64 defined")
elseif(_arch_lower MATCHES "aarch64|arm64")
    set(ARCH_X86_64 FALSE CACHE BOOL "Target architecture is x86_64" FORCE)
    set(ARCH_AARCH64 TRUE CACHE BOOL "Target architecture is aarch64" FORCE)
    add_definitions(-DARCH_AARCH64=1)
    message(STATUS "Target architecture: aarch64 — macro ARCH_AARCH64 defined")
else()
    set(ARCH_X86_64 FALSE CACHE BOOL "Target architecture is x86_64" FORCE)
    set(ARCH_AARCH64 FALSE CACHE BOOL "Target architecture is aarch64" FORCE)
    message(WARNING "Unknown target architecture '${CMAKE_SYSTEM_PROCESSOR}'. "
                    "Neither ARCH_X86_64 nor ARCH_AARCH64 will be defined.")
endif()
unset(_arch_lower)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(USE_QIZHI_SETTING TRUE)  # Set to FALSE to disable Qizhi-specific settings

# Build type and optimization options
# (can be overridden from command line)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# AddressSanitizer option
option(USE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
if(USE_ASAN)
  message(STATUS "AddressSanitizer enabled")
  message(STATUS "Note: When using ASan with Python extensions:")
  message(STATUS "  1. Use scripts/run_with_asan.sh to run Python scripts")
  message(STATUS "  2. Or manually set: export LD_PRELOAD=\$(gcc -print-file-name=libasan.so)")
  message(STATUS "  3. And set: export ASAN_OPTIONS='detect_leaks=0:verify_asan_link_order=0'")
  
  # Set ASan flags (applicable to both GCC and Clang)
  set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  
  # Special configuration for Python extension modules
  # Do not use -static-libasan; Python modules require dynamic linking
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(ASAN_LINK_FLAGS "-fsanitize=address")
    
    # GCC-specific: use dynamic ASan library (required for Python extension modules)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      # Do not add -static-libasan; keep dynamic linking
      message(STATUS "Using dynamic ASan library for Python compatibility")
    endif()
    
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      # Clang-specific: use shared ASan library
      set(ASAN_LINK_FLAGS "${ASAN_LINK_FLAGS} -shared-libasan")
      message(STATUS "Using shared ASan library for Python compatibility (Clang)")
    endif()
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
    # Modify linker flags so ASan runtime is linked first
    set(CMAKE_EXE_LINKER_FLAGS "${ASAN_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${ASAN_LINK_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
    
    # Add compile definition so code can detect whether ASan is enabled
    add_definitions(-DUSING_ASAN=1)
  endif()
  
  # Print ASan library path for user reference
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
    OUTPUT_VARIABLE ASAN_LIBRARY_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXISTS "${ASAN_LIBRARY_PATH}")
    message(STATUS "ASan library path: ${ASAN_LIBRARY_PATH}")
    message(STATUS "You can use: export LD_PRELOAD=${ASAN_LIBRARY_PATH}")
  endif()
endif()

# Performance profiling option
option(USE_PERF_PROFILE "Enable compiler flags for better performance profiling" OFF)
if(USE_PERF_PROFILE)
  message(STATUS "Performance profiling support enabled")
  # Add debug information and preserve frame pointers
  # set(PERF_FLAGS "-g -fno-omit-frame-pointer")
  set(PERF_FLAGS "-g -fno-omit-frame-pointer -fno-inline-small-functions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PERF_FLAGS}")
  # Preserve these flags even in Release mode
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fno-omit-frame-pointer")
  
  # Ensure debug symbols are not stripped
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
  
  # Reduce optimizations that interfere with performance profiling
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-optimize-sibling-calls")
  endif()
  
  message(STATUS "Performance profiling flags: ${PERF_FLAGS}")
endif()

# Optimization level option (O1, O2, O3)
set(OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level (O1, O2, O3)")
set_property(CACHE OPTIMIZATION_LEVEL PROPERTY STRINGS O1 O2 O3)
message(STATUS "Using optimization level: ${OPTIMIZATION_LEVEL}")

# Link Time Optimization setting (will be used by subdirectories)
option(VEC_QMDP_LTO "Use Link Time Optimization" ON)

# GEOS custom path
set(GEOS_ROOT "$ENV{HOME}/.local")  # Install to ~/.local
set(GEOS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/geos/include)
set(GEOS_LIBRARY_DIR ${GEOS_ROOT}/lib)

# Find required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

if (Eigen3_FOUND)
    message(STATUS "Eigen3 found!")
    message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
    message(STATUS "Eigen3 version: ${EIGEN3_VERSION_STRING}")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# Ensure Eigen include directory is available when using the imported target
# Some Eigen3 Config packages only provide the target and not EIGEN3_INCLUDE_DIR(S)
if(TARGET Eigen3::Eigen)
  get_target_property(_EIGEN3_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  if(_EIGEN3_INCLUDES)
    set(EIGEN3_INCLUDE_DIR ${_EIGEN3_INCLUDES})
  endif()
endif()

# Find Python libraries
if (USE_QIZHI_SETTING)
  find_package(PythonInterp REQUIRED) # QIZHI REQUIRED
endif()
find_package(PythonLibs REQUIRED)
set(PYTHON_VERSION "${PYTHONLIBS_VERSION_STRING}")
# Extract Python major and minor version numbers
string(REGEX REPLACE "^([0-9]+)\\.[0-9]+.*$" "\\1" PYTHON_VERSION_MAJOR "${PYTHON_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" PYTHON_VERSION_MINOR "${PYTHON_VERSION}")
message(STATUS "Found Python: ${PYTHON_VERSION} (Major version: ${PYTHON_VERSION_MAJOR}, Minor version: ${PYTHON_VERSION_MINOR})")

# Find Boost libraries, including Python components
# Select the correct Boost.Python component based on Python version
if(${PYTHON_VERSION_MAJOR} EQUAL 3)
    # Try different Boost Python naming conventions
    find_package(Boost COMPONENTS system filesystem thread python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} QUIET)
    if(NOT Boost_FOUND)
        find_package(Boost COMPONENTS system filesystem thread python${PYTHON_VERSION_MAJOR} numpy${PYTHON_VERSION_MAJOR} QUIET)
    endif()
    if(NOT Boost_FOUND)
        find_package(Boost COMPONENTS system filesystem thread python numpy QUIET)
    endif()
    if(NOT Boost_FOUND)
        find_package(Boost REQUIRED COMPONENTS system filesystem thread python)
    endif()
    
    # Set Boost Python library link target
    if(TARGET Boost::python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        set(BOOST_PYTHON_TARGET Boost::python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    elseif(TARGET Boost::python${PYTHON_VERSION_MAJOR})
        set(BOOST_PYTHON_TARGET Boost::python${PYTHON_VERSION_MAJOR})
    else()
        set(BOOST_PYTHON_TARGET Boost::python)
    endif()
    
    # Set Boost NumPy library link target
    if(TARGET Boost::numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        set(BOOST_NUMPY_TARGET Boost::numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    elseif(TARGET Boost::numpy${PYTHON_VERSION_MAJOR})
        set(BOOST_NUMPY_TARGET Boost::numpy${PYTHON_VERSION_MAJOR})
    else()
        set(BOOST_NUMPY_TARGET Boost::numpy)
    endif()
    
    message(STATUS "Using Boost.Python for Python 3: ${BOOST_PYTHON_TARGET} ${BOOST_NUMPY_TARGET}")
else()
    find_package(Boost REQUIRED COMPONENTS system filesystem thread python numpy)
    set(BOOST_PYTHON_TARGET Boost::python)
    set(BOOST_NUMPY_TARGET Boost::numpy)
    message(STATUS "Using Boost.Python: ${BOOST_PYTHON_TARGET} ${BOOST_NUMPY_TARGET}")
endif()

if (USE_QIZHI_SETTING)
	# Locate NumPy include directory
	# Use a Python script to obtain the NumPy include path
	execute_process(
		COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
		OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE NUMPY_FIND_RESULT
	)

	if(NUMPY_FIND_RESULT EQUAL 0)
		message(STATUS "Found NumPy include dir: ${NUMPY_INCLUDE_DIR}")
	else()
		message(FATAL_ERROR "Could not find NumPy include directory. Please check your Python environment.")
	endif()
endif()

# Find Boost serialization libraries
find_package(Boost REQUIRED COMPONENTS serialization)

# Set compiler flags for optimization and SIMD instructions
set(CMAKE_CXX_FLAGS_RELEASE "-${OPTIMIZATION_LEVEL} -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include(CheckCXXCompilerFlag)

if(ARCH_X86_64)
    # x86_64: probe and enable AVX / AVX2 / FMA extensions
    check_cxx_compiler_flag("-mavx"  COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma"  COMPILER_SUPPORTS_FMA)

    if(COMPILER_SUPPORTS_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    endif()
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    endif()
    if(COMPILER_SUPPORTS_FMA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()
    message(STATUS "SIMD: x86_64 AVX/AVX2/FMA flags enabled")
elseif(ARCH_AARCH64)
    # aarch64: enable ARMv8 NEON/SIMD via -march; NEON is mandatory in ARMv8
    check_cxx_compiler_flag("-march=armv8-a+simd" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    endif()
    message(STATUS "SIMD: aarch64 NEON (-march=armv8-a+simd) flags enabled")
else()
    message(STATUS "SIMD: no architecture-specific SIMD flags applied")
endif()

# Auto-vectorisation — beneficial on both x86_64 and aarch64
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")

# Set VAMP library path
set(VAMP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/vamp/src/impl)

# Set x86-simd-sort (highway) library path
set(HIGH_WAY_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/highway)

# Include directories - ensure all required directories are included
if (USE_QIZHI_SETTING)
	include_directories(
		${PROJECT_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/src
		${VAMP_INCLUDE_DIR}
		${HIGH_WAY_INCLUDE_DIR}
		${EIGEN3_INCLUDE_DIR}
		${Boost_INCLUDE_DIRS}
		${PYTHON_INCLUDE_DIRS}
		${Boost_SERIALIZATION_INCLUDE_DIRS}
		${GEOS_INCLUDE_DIR}
		${NUMPY_INCLUDE_DIR}
	)
else()
	include_directories(
		${PROJECT_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/include
		${PROJECT_SOURCE_DIR}/src
		${VAMP_INCLUDE_DIR}
		${HIGH_WAY_INCLUDE_DIR}
		${EIGEN3_INCLUDE_DIR}
		${Boost_INCLUDE_DIRS}
		${PYTHON_INCLUDE_DIRS}
		${Boost_SERIALIZATION_INCLUDE_DIRS}
		${GEOS_INCLUDE_DIR}
	)
endif()

# Suppress ignored-attributes warning caused by SIMD headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")

# Add subdirectories
add_subdirectory(src)

# Rock Sample POMDP benchmark examples
option(BUILD_ROCK_SAMPLE_EXAMPLES "Build Rock Sample POMDP benchmark examples" OFF)
if(BUILD_ROCK_SAMPLE_EXAMPLES)
    add_subdirectory(examples/rock_sample)
endif()

# # Enable testing
# enable_testing()
# add_subdirectory(tests)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add GEOS library directory
link_directories(${GEOS_LIBRARY_DIR})

# Set GEOS static library names (link order matters)
set(GEOS_LIBRARIES ${GEOS_LIBRARY_DIR}/libgeos.a ${GEOS_LIBRARY_DIR}/libgeos_c.a)

# Static linking does not require RPATH settings
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${GEOS_LIBRARY_DIR}")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${GEOS_LIBRARY_DIR}") 