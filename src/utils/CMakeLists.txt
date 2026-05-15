add_library(vec_qmdp_utils OBJECT
    geometry_utils.cpp
    global_utils.cpp
    map_utils.cpp
    math_utils.cpp
    occupancy_map.cpp
    params.cpp
    path_utils.cpp
)

target_include_directories(vec_qmdp_utils
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${GEOS_INCLUDE_DIR}
)

# Set position-independent code for dynamic library building
set_target_properties(vec_qmdp_utils PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
)