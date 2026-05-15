add_subdirectory(core)
add_subdirectory(planning)
add_subdirectory(utils)
add_subdirectory(collision)

# Create the main static library
add_library(vec_qmdp STATIC
    $<TARGET_OBJECTS:vec_qmdp_core>
    $<TARGET_OBJECTS:vec_qmdp_planning>
    $<TARGET_OBJECTS:vec_qmdp_utils>
    $<TARGET_OBJECTS:vec_qmdp_collision>
)

target_link_libraries(vec_qmdp
    PUBLIC
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    ${GEOS_LIBRARIES}
)

# Create the shared library for Python bindings
add_library(vec_qmdp_closed_planner SHARED
    $<TARGET_OBJECTS:vec_qmdp_core>
    $<TARGET_OBJECTS:vec_qmdp_planning>
    $<TARGET_OBJECTS:vec_qmdp_utils>
    $<TARGET_OBJECTS:vec_qmdp_collision>
    planning/vec_qmdp_closed_planner.cpp
)

# Link with required libraries including Boost.Python and Python libraries
target_link_libraries(vec_qmdp_closed_planner
    PUBLIC
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
    ${BOOST_PYTHON_TARGET}
    ${BOOST_NUMPY_TARGET}
    Boost::serialization
    ${PYTHON_LIBRARIES}
    # Link GEOS static libraries (use --whole-archive to ensure all symbols are included)
    -Wl,--whole-archive ${GEOS_LIBRARY_DIR}/libgeos_c.a -Wl,--no-whole-archive
    -Wl,--whole-archive ${GEOS_LIBRARY_DIR}/libgeos.a -Wl,--no-whole-archive
    # Add required system libraries
    m  # math library
    pthread  # threading library
)

# Set properties for the shared library
set_target_properties(vec_qmdp_closed_planner PROPERTIES
    PREFIX ""  # Remove "lib" prefix for Python importing
    POSITION_INDEPENDENT_CODE ON  # Required for shared libraries
)

# Statically link GEOS libraries; no RPATH settings required
# set_target_properties(vec_qmdp_closed_planner PROPERTIES
#     INSTALL_RPATH "${GEOS_LIBRARY_DIR}"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )

# If specific Python module name is needed
if(APPLE)
    set_target_properties(vec_qmdp_closed_planner PROPERTIES SUFFIX ".so")  # Use .so extension on macOS
endif() 