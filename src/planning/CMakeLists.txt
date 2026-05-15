add_library(vec_qmdp_planning OBJECT
    context_qmdp.cpp
    qmdp_trajectory_planner.cpp
    vec_qmdp_static.cpp
    vec_qmdp_ad.cpp
    vec_qmdp_dynamic.cpp
    reward_functions.cpp
    tracker.cpp
    trajectory_optimization.cpp
)

target_include_directories(vec_qmdp_planning
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Set position independent code for shared library builds
set_target_properties(vec_qmdp_planning PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)