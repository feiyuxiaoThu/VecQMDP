add_library(vec_qmdp_core OBJECT
    net_belief.cpp
    state.cpp
)

target_include_directories(vec_qmdp_core
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Set position independent code for shared library builds
set_target_properties(vec_qmdp_core PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
)