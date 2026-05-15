/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file params.hpp
 * @brief Compile-time constants and tuning parameters for the VecQMDP planner.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace vec_qmdp
{
    namespace utils
    {
        // =========================================================================
        // Debug
        // =========================================================================
        extern int         iteration;
        extern std::string scenario_token;

        extern float offset_x; // ego-centric offset x
        extern float offset_y; // ego-centric offset y
        extern bool  approaching_terminal_point;

        // =========================================================================
        // General Numerical Constants
        // =========================================================================
        constexpr int   RANDOM_SEED = 42;
        constexpr float MAX_VALUE = 10000.0f;
        constexpr float TIME_DECAY_FACTOR = 0.85f;

        // =========================================================================
        // PI Constants
        // =========================================================================
        constexpr float MIN_PI = -3.14159265358979323846f;               // -pi
        constexpr float PI_1_10 = 3.14159265358979323846 * 1.0f / 10.0f; // 18 deg
        constexpr float PI_1_6 = 3.14159265358979323846 * 1.0f / 6.0f;   // 30 deg
        constexpr float PI_1_4 = 3.14159265358979323846 * 1.0f / 4.0f;   // 45 deg
        constexpr float PI_1_3 = 3.14159265358979323846 * 1.0f / 3.0f;   // 60 deg
        constexpr float PI_1_2 = 3.14159265358979323846 * 1.0f / 2.0f;   // 90 deg
        constexpr float PI_2_3 = 3.14159265358979323846 * 2.0f / 3.0f;   // 120 deg
        constexpr float PI_3_4 = 3.14159265358979323846 * 3.0f / 4.0f;   // 135 deg
        constexpr float PI_5_6 = 3.14159265358979323846 * 5.0f / 6.0f;   // 150 deg

        // =========================================================================
        // Time & Discretization
        // =========================================================================
        constexpr float    TIME_STEP = 0.2f;
        constexpr float    PUBLISHED_PROPOSAL_TIME_STEP = 0.1f;
        constexpr float    DISCRETIZATION_TIME = 0.2f;         // discretization time interval for LQR
        constexpr float    DISCRETIZATION_TIME_SQUARE = 0.04f; // square of the discretization time interval
        constexpr float    DISCRETIZATION_TIME_CUBE = 0.008f;
        constexpr float    DISCRETIZATION_TIME_QUAD = 0.0016f;
        constexpr float    STEP_TIME = 2.0;
        constexpr float    ROLLOUT_TIME = 2.0; // rollout len 2s
        constexpr uint32_t STEP_TIME_LEN = STEP_TIME / TIME_STEP;
        constexpr uint32_t ROLLOUT_TIME_LEN = ROLLOUT_TIME / TIME_STEP;

        constexpr float    LOOKAHEAD_TIME = 4.8f;
        constexpr float    LOOKBACK_TIME = 3.0f;
        constexpr float    LOOK_TIME_INTERVAL = 0.6f;
        constexpr uint32_t LOOKAHEAD_STEPS = LOOKAHEAD_TIME / LOOK_TIME_INTERVAL;
        constexpr uint32_t LOOKBACK_STEPS = LOOKBACK_TIME / LOOK_TIME_INTERVAL;
        constexpr uint32_t LOOK_TIME_STEPS = LOOK_TIME_INTERVAL / TIME_STEP;

        // =========================================================================
        // Prediction / Belief Parameters
        // =========================================================================
        constexpr uint32_t DUMMY_TIME_STEPS = 48; // dummy time steps padded to multiples of 8 for memory alignment
        constexpr uint32_t PYTHON_PRED_TIME_STEPS = 80;  // time steps provided by Python predictions
        constexpr uint32_t MAX_PRED_MODES = 6;           // maximum number of prediction modes
        constexpr uint32_t MAX_PRED_TIME_STEPS = 41;     // From 0 to 8 seconds with 0.2s step (41 total time points)
        constexpr uint32_t MAX_EXPECTED_TIME_STEPS = 21; // From 0 to 4 seconds with 0.2s step (21 total time points)

        // =========================================================================
        // Planner Core Parameters
        // =========================================================================
        constexpr float MAX_PLANNING_TIME = 8.0f; // maximum tree search planning time (seconds)
        constexpr int   MAX_ITERATIONS = 100000;

        constexpr uint32_t TREE_HEIGHT = 4;       // planning tree height (affects horizon)
        constexpr uint32_t TOTAL_TREE_HEIGHT = 4; // planning horizon + rollout length

        constexpr uint32_t NUM_REF_PATH = 3;        // number of reference paths used for search
        constexpr uint32_t NUM_LATERAL_OFFSETS = 3; // number of lateral offsets per path (-1.0f, 0.0f, 1.0f)
        constexpr uint32_t NUM_ACTIONS = NUM_REF_PATH * NUM_LATERAL_OFFSETS;  // max branching factor per node
        constexpr float    PATH_OFFSETS_FLOAT[3] = {-1.0f, 0.0f, 1.0f};       // offset values for indices 0, 1, 2
        constexpr uint32_t ACTION_TO_PATH[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};   // action -> path index
        constexpr uint32_t ACTION_TO_OFFSET[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2}; // action -> offset index

        // compute total nodes per tree depth: 1 + 9 + 9^2 + 9^3 + 9^4
        const std::vector<uint32_t> TREE_NODE_SIZES_PER_DEPTH{1, NUM_ACTIONS, NUM_ACTIONS *NUM_ACTIONS,
                                                              NUM_ACTIONS * NUM_ACTIONS * NUM_ACTIONS,
                                                              NUM_ACTIONS * NUM_ACTIONS * NUM_ACTIONS *NUM_ACTIONS};
        constexpr uint32_t          TREE_NODE_SIZE = 1 + NUM_ACTIONS + NUM_ACTIONS * NUM_ACTIONS +
                                                     NUM_ACTIONS * NUM_ACTIONS * NUM_ACTIONS +
                                                     NUM_ACTIONS * NUM_ACTIONS * NUM_ACTIONS * NUM_ACTIONS;

        constexpr uint32_t MAX_OBS_VEHICLES = 192; // max number of observed other vehicles
        constexpr uint32_t MAX_SIM_VEHICLES =
            96; // max number of vehicles used in simulation (multiple of 8 for alignment)
        constexpr uint32_t MAX_NUM_REFLINES =
            5; // max number of reference lines considered in a scene (includes extra ref paths used for filtering)

        constexpr float LEAST_DIST2JUNCTION = 7.0f;              // minimum distance to junction (affects action space)
        constexpr float AGENT_DETECTION_MARGIN = -0.3f;          // offset for minimum distance to other agents
        constexpr float ACTION_CHANGE_DISTANCE_THRESHOLD = 0.7f; // distance threshold to allow action changes

        // =========================================================================
        // Early Termination Parameters
        // =========================================================================
        constexpr int   EARLY_TERM_CHECK_INTERVAL = 5;          // check convergence every N iterations
        constexpr int   EARLY_TERM_MIN_EXPAND_CALLS = 30;       // minimum expand calls before allowing early stop
        constexpr int   EARLY_TERM_STABLE_ITERATIONS = 5;       // number of stable iterations required for best action
        constexpr float EARLY_TERM_Q_CHANGE_THRESHOLD = 0.05f;  // relative Q-value change threshold
        constexpr int   EARLY_TERM_MIN_BEST_ACTION_VISITS = 20; // minimum visits for best action

        // =========================================================================
        // Parallelization Parameters
        // =========================================================================
        constexpr uint32_t NUM_SCENARIOS_PER_THREAD = 8;
        constexpr uint32_t NUM_THREADS = 8; // number of threads
        constexpr uint32_t NUM_SCENARIOS =
            NUM_SCENARIOS_PER_THREAD * NUM_THREADS; // total number of scenarios to consider
        constexpr uint32_t TOTAL_SCNEARIO_ELEMENTS_PARALLEL =
            DUMMY_TIME_STEPS * MAX_SIM_VEHICLES * NUM_SCENARIOS_PER_THREAD;
        constexpr uint32_t TOTAL_SCNEARIO_ELEMENTS_SERIAL = DUMMY_TIME_STEPS * MAX_SIM_VEHICLES * NUM_SCENARIOS;

        // =========================================================================
        // Trajectory Optimization Parameters
        // =========================================================================
        constexpr uint32_t PROPOSAL_DUMMY_SIZE = 48;
        constexpr uint32_t PROPOSAL_TRAJECTORY_SIZE = 41;           // 8.0s
        constexpr uint32_t PROPOSAL_PUBLISHED_TRAJECTORY_SIZE = 81; // 4.0s
        constexpr uint32_t PROPOSAL_EXPECTED_TIME_TRAJECTORY_SIZE = 21;
        constexpr uint32_t CROSS_EVALUATION_LEN = 21;

        constexpr uint32_t NUM_SCENARIOS_TRAJ_OPT_PER_THREAD =
            8; // should be a multiple of 8 for memory alignment and efficient parallelization (max 8 with current
               // NUM_SCENARIOS=64 and NUM_THREADS=8)
        constexpr uint32_t       LATERAL_OFFSETS_NUM = 9;
        const std::vector<float> LATERAL_OFFSETS = {-3.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 3.0f};
        const std::vector<float> ACTUAL_LATERAL_OFFSETS = {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
        constexpr float          LEFT_LATERAL_OFFSET = 0.4f;
        constexpr float          RIGHT_LATERAL_OFFSET = -0.4f;

        constexpr uint32_t NUM_DISPLACEMENTS = PROPOSAL_TRAJECTORY_SIZE - 1;                        // 40 dx points
        constexpr uint32_t NUM_DISPLACEMENTS_EXPECTED = PROPOSAL_EXPECTED_TIME_TRAJECTORY_SIZE - 1; // 20 dx points
        constexpr uint32_t PROPOSAL_BATCH_SIZE =
            NUM_SCENARIOS_TRAJ_OPT_PER_THREAD * LATERAL_OFFSETS_NUM; // 72 scenarios

        // =========================================================================
        // Trajectory Optimization Tuning Parameters
        // =========================================================================

        // Cross-scenario evaluation thresholds
        constexpr float CROSS_EVAL_COLLISION_GAP_THRESHOLD =
            2000.0f; // gap between global and constrained max to prefer global
        constexpr float CROSS_EVAL_SPEED_GAP_THRESHOLD =
            5.0f; // gap between second-edge and constrained max to prefer edge

        // Emergency brake sentinel and TTC thresholds
        constexpr float EMERGENCY_BRAKE_INACTIVE_SENTINEL =
            10.0f;                                            // sentinel value indicating no emergency brake needed
        constexpr int   EMERGENCY_TTC_CRITICAL = 15;          // critical time-to-collision timestep threshold
        constexpr int   EMERGENCY_TTC_EXTENDED = 20;          // extended time-to-collision timestep threshold
        constexpr float EMERGENCY_BRAKE_MAX_EGO_SPEED = 5.0f; // max ego speed for extended emergency brake check

        // Hard brake detection thresholds
        constexpr float HARD_BRAKE_DEC_TOLERANCE = 0.1f;            // tolerance added to HARD_BRAKE_DEC for detection
        constexpr float HARD_BRAKE_LATERAL_OFFSET_THRESHOLD = 1.0f; // max lateral offset for hard brake detection
        constexpr float HARD_BRAKE_VELOCITY_THRESHOLD = 1.5f;       // velocity threshold for hard brake trajectory
        constexpr float HARD_BRAKE_ACC_THRESHOLD = -3.5f;           // acceleration threshold for hard brake trajectory

        // PD controller parameters for emergency braking
        constexpr float PD_VELOCITY_SWITCH_THRESHOLD = 0.2f;  // velocity threshold to switch PD gains
        constexpr float PD_KP_HIGH_SPEED = 10.0f;             // proportional gain at higher speed
        constexpr float PD_KP_LOW_SPEED = 4.0f;               // proportional gain at lower speed
        constexpr float PD_KD_LOW_SPEED = 1.0f;               // derivative gain at lower speed
        constexpr float PD_MAX_ACC_CLAMP = 2.40f;             // upper clamp for PD correcting acceleration
        constexpr float PD_MAX_DEC_CLAMP = -4.05f;            // lower clamp for PD correcting deceleration
        constexpr float PD_VELOCITY_CORRECTION_FACTOR = 1.1f; // scaling factor for correcting velocity

        // Velocity thresholds for emergency brake / static trajectory
        constexpr float LOW_SPEED_THRESHOLD = 3.0f;             // general low-speed threshold for mode/brake decisions
        constexpr float NEAR_STOP_VELOCITY_THRESHOLD = 0.5f;    // near-stop speed threshold
        constexpr float SLOW_VELOCITY_THRESHOLD = 1.0f;         // slow-moving velocity threshold
        constexpr float CREEP_VELOCITY_THRESHOLD = 0.3f;        // creep-speed velocity threshold
        constexpr float VELOCITY_NEAR_ZERO_THRESHOLD = 0.1f;    // near-zero velocity threshold
        constexpr float VELOCITY_ALMOST_ZERO_THRESHOLD = 0.05f; // almost-zero velocity threshold
        constexpr float VELOCITY_SLIGHTLY_NEGATIVE_THRESHOLD = -0.05f; // slightly-negative velocity threshold

        // Step move length values for emergency braking / static trajectory
        constexpr float STEP_MOVE_HARD_BRAKE = -0.3f;     // hard brake step move length
        constexpr float STEP_MOVE_MODERATE_BRAKE = -0.2f; // moderate brake step move length
        constexpr float STEP_MOVE_LIGHT_BRAKE = -0.1f;    // light brake step move length
        constexpr float STEP_MOVE_SLIGHT_BRAKE = -0.05f;  // slight brake step move length
        constexpr float STEP_MOVE_SLIGHT_FORWARD = 0.05f; // slight forward step move length
        constexpr float STEP_MOVE_LIGHT_FORWARD = 0.1f;   // light forward step move length

        // Trajectory length and mode determination thresholds
        constexpr float MIN_TRAJECTORY_LENGTH = 0.1f;          // minimum trajectory length to be considered valid
        constexpr float MODE_VALUE_DIFF_THRESHOLD = -1000.0f;  // value difference threshold for LF/LC mode decision
        constexpr float ON_TARGET_PATH_DISTANCE_MARGIN = 0.3f; // distance margin for on-target-path detection

        // Importance sampling threshold
        constexpr float IMPORTANCE_SAMPLE_PROB_THRESHOLD = 0.05f; // minimum mode probability for importance sampling

        // Best-action selection thresholds
        constexpr float ACTION_VALUE_ZERO_THRESHOLD = 1e-6f;     // threshold for considering action values as zero
        constexpr float ACTION_VALUE_INITIAL_MIN = -1e7f;        // initial worst-case value for best action search
        constexpr float CURRENT_PATH_PREFERENCE_BONUS = 0.0f;    // small bonus added to current path action value
        constexpr float ALL_COLLIDED_VALUE_THRESHOLD = -2000.0f; // value threshold indicating all actions collided
        constexpr float LAST_ACTION_PREFERENCE_MARGIN = 5000.0f; // margin for preferring last action over default

        // AD planner initialization defaults
        constexpr float AD_DEFAULT_DISCOUNT_FACTOR = 1.0f;        // BeliefTreeSearch discount factor (no discounting)
        constexpr float AD_DEFAULT_EXPLORATION_CONSTANT = 100.0f; // UCB exploration constant
        constexpr float AD_DEFAULT_MAX_BACKUP_LAMBDA = 0.001f;    // max-backup softmax temperature
        constexpr float AD_DEFAULT_DEPTH_SYNC_LAMBDA = 10000.0f;  // depth-sync softmax temperature
        constexpr float EGO_LOW_SPEED_THRESHOLD = 3.0f;           // ego speed below which low-speed mode activates

        // =========================================================================
        // STRtree Spatial Index Parameters
        // =========================================================================
        constexpr int    STRTREE_NODE_CAPACITY = 8;      // fan-out per internal node (children per parent)
        constexpr size_t STRTREE_MAX_NODE_NUM = 96;      // max temporary node buffer size
        constexpr size_t STRTREE_SUBTREE_LEAF_SIZE = 64; // max leaf nodes per subtree
        constexpr size_t STRTREE_SUBTREE_INTERNAL_SIZE =
            STRTREE_NODE_CAPACITY; // internal nodes per subtree (equals node capacity)
        constexpr size_t STRTREE_SUBTREE_TOTAL_SIZE = STRTREE_SUBTREE_INTERNAL_SIZE + STRTREE_SUBTREE_LEAF_SIZE; // 72
        constexpr size_t STRTREE_TOTAL_LEAF_SIZE = STRTREE_SUBTREE_LEAF_SIZE * 2;                                // 128
        constexpr size_t STRTREE_TOTAL_TREE_SIZE = STRTREE_SUBTREE_TOTAL_SIZE * 2;                               // 144
        constexpr int STRTREE_UNROLLED_SORT_THRESHOLD = 3; // max candidate count for unrolled insertion sort in query

        // =========================================================================
        // Vehicle Geometry
        // =========================================================================
        constexpr float EGO_BB_EXTENT_X = 1.1485f; // half of ego vehicle width (2.297 m)
        constexpr float EGO_BB_EXTENT_Y = 2.588f;  // half of ego vehicle length (5.176 m)
        constexpr float FRONT_TO_CENTER = 1.628f;
        constexpr float REAR_TO_CENTER = 1.461f;
        constexpr float FRONT_TO_COG = 1.419f;
        constexpr float REAR_TO_COG = 1.67f;
        constexpr float WHEEL_BASE = 3.089f;
        constexpr float REAR_LENGTH = 1.127f;
        constexpr float VEHICLE_REAR_AXLE_TO_CENTER =
            1.461f;                           // distance from vehicle rear axle center to vehicle center
        constexpr float SAFETY_MARGIN = 0.1f; // safety margin for exo agents

        // =========================================================================
        // Vehicle Kinematics
        // =========================================================================
        constexpr float MAX_STEERING = 0.785f;
        constexpr float MAX_STEERING_RATE = 8.0f;
        constexpr float MAX_STEERING_ANGLE = 1.05f; // [rad] maximum steering angle
        constexpr float MAX_YAW_RATE = 0.6f;
        constexpr float MAX_VEL = 13.0f;
        constexpr float MAX_JERK = 1.0f;
        constexpr float MAX_VEL_LATERAL_HIGH_SPEED = 2.0f;
        constexpr float MAX_VEL_LATERAL_LOW_SPEED = 0.5f;
        constexpr float MAX_ACC_LATERAL_HIGH_SPEED = 1.0f;
        constexpr float MAX_ACC_LATERAL_LOW_SPEED = 0.8f;

        // =========================================================================
        // IDM / Longitudinal Dynamics Parameters
        // =========================================================================
        constexpr float MAX_ACC = 1.5f;
        constexpr float MAX_DEC = -2.5f;
        constexpr float HARD_BRAKE_DEC = -3.0f;
        constexpr float COMFORT_DEC = -1.33f;
        constexpr float ACC_EXPONENT = 4.0f;
        constexpr float DESIRED_TIME_HEADWAY = 1.5f;
        constexpr float SAFE_DIST_DIFF = 0.5f;
        constexpr float LEAST_SAFE_DIST = 0.3f;
        const float     SQRT_AB = std::sqrt(MAX_ACC * (-COMFORT_DEC));
        constexpr float LATERAL_COMPENSATION_THRESHOLD = 0.3f;
        constexpr float NOMINAL_ACC_LONG = 0.5f;
        constexpr float MIN_DISTANCE_COMPENSATION = 3.0f;
        constexpr float MAX_DISTANCE_COMPENSATION = 30.0f;
        constexpr float MAX_CONERVATIVE_TRANSITION_TIME = 6.0f;
        constexpr float MAX_NORMINAL_TRANSITION_TIME = 2.5f;
        constexpr float ACCEL_TIME_CONSTANT =
            0.2f; // first-order low-pass filter time constant for acceleration (seconds)
        constexpr float STEERING_ANGLE_TIME_CONSTANT =
            0.05f; // first-order low-pass filter time constant for steering angle (seconds)

        // =================================================================
        // Lead Vehicle Search & Collision Parameters
        // =================================================================
        constexpr float LEAD_VEHICLE_DEFAULT_SPEED = 20.0f;      // default speed when no lead vehicle found
        constexpr float LEAD_VEHICLE_SEARCH_REAR_MARGIN = 2.0f;  // backward offset from ego for lead vehicle search
        constexpr float LEAD_VEHICLE_MIN_SEARCH_VELOCITY = 4.0f; // minimum velocity for search segment length
        constexpr float COLLISION_BRAKING_LOW_SPEED_THRESHOLD = 2.0f; // low ego speed threshold for collision braking

        // =========================================================================
        // Lane-Changing Parameters
        // =========================================================================
        constexpr float LC_ING_ALLOWANCE_LATERAL_OFFSET_THRESHOLD = 0.4f;
        constexpr float LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD = 0.6f;
        constexpr float LC_ING_ALLOWANCE_HEADING_DIFF_THRESHOLD = 0.0872f;       // 5 deg
        constexpr float LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD = 0.1744f; // 10 deg
        constexpr float LC_AHEAD_DISTANCE_THRESHOULD_HIGH = 2.0f;
        constexpr float LC_AHEAD_DISTANCE_THRESHOULD_LOW = 0.5f;
        constexpr float LC_FOLLOW_DISTANCE_COMPENSATION = 1.0f;  // distance added to LC following vehicle gap
        constexpr float LC_RELAXED_TIME_HEADWAY = 0.8f;          // relaxed time headway during LC or slow lead
        constexpr float LC_NORMAL_TIME_HEADWAY = 1.5f;           // normal time headway threshold
        constexpr float LC_SPEED_FOR_DISTANCE_THRESHOLD = 3.0f;  // ego speed threshold for high/low distance selection
        constexpr float LC_RELAXED_ACC_THRESHOLD_MARGIN = 0.1f;  // margin added to MAX_DEC for relaxed acc condition
        constexpr float LC_NORMAL_ACC_THRESHOLD = -2.0f;         // normal (strict) acceleration threshold for LC
        constexpr float LC_UNCONDITIONAL_ALLOW_DISTANCE = 15.0f; // distance beyond which time headway check is skipped
        constexpr float LC_LEAD_LATERAL_COMPENSATION_DISTANCE =
            2.0f; // lateral distance for LC lead vehicle compensation

        // =========================================================================
        // Stanley Controller Parameters
        // =========================================================================
        constexpr float AGGRESIVE_STANLEY_K = 1.0f;
        constexpr float CONSERVATIVE_STANLEY_K = 0.7f;
        constexpr float STANLEY_CURVATURE_THRESHOLD =
            0.05f;                                     // curvature threshold for selecting aggressive/conservative K
        constexpr float STANLEY_MIN_SPEED_QMDP = 0.1f; // minimum speed denominator for Stanley (QMDP batch)
        constexpr float STANLEY_MIN_SPEED_TRAJ = 1.0f; // minimum speed denominator for Stanley (trajectory batch)

        // =========================================================================
        // LQR / Tracking Controller Parameters
        // =========================================================================

        // State vector indices for lateral dynamics
        constexpr uint32_t LATERAL_ERROR =
            0; // [m] lateral error of the vehicle rear axle center relative to the reference path
        constexpr uint32_t HEADING_ERROR = 1;  // [rad] heading error
        constexpr uint32_t STEERING_ANGLE = 2; // [rad] steering angle

        constexpr uint32_t LATERAL_ERROR_IDX =
            0; // [m] lateral error of the vehicle rear axle center relative to the planned centerline
        constexpr uint32_t HEADING_ERROR_IDX = 1;  // [rad] heading error
        constexpr uint32_t STEERING_ANGLE_IDX = 2; // [rad] steering wheel angle relative to vehicle longitudinal axis

        // Motion model state indices
        constexpr uint32_t ACCELERATION_X = 0; // [m/s^2] longitudinal acceleration
        constexpr uint32_t STEERING_RATE = 1;  // [rad/s] steering rate

        // LQR weights
        constexpr float Q_LONGITUDINAL = 10.0f;           // Q weight for longitudinal subsystem
        constexpr float R_LONGITUDINAL = 1.0f;            // R weight for longitudinal subsystem
        constexpr float QRQ_LONGITUDINAL = 20.0f / 41.0f; // combined longitudinal weight
        constexpr float Q_LATERAL_1 = 1.0f;               // Q weight for lateral subsystem component 1
        constexpr float Q_LATERAL_2 = 10.0f;              // Q weight for lateral subsystem component 2
        constexpr float Q_LATERAL_3 = 0.0f;               // Q weight for lateral subsystem component 3
        constexpr float R_LATERAL = 1.0f;                 // R weight for lateral subsystem

        // Tracking controller parameters
        constexpr uint32_t TRACKING_HORIZON = 10;          // tracking horizon (time steps)
        constexpr uint32_t NUM_LATERAL_STATES = 3;         // number of lateral states
        constexpr float    JERK_PENALTY = 0.0001f;         // penalty for jerk (rate of change of acceleration)
        constexpr float    CURVATURE_RATE_PENALTY = 0.01f; // penalty for curvature rate
        constexpr float    INITIAL_CURVATURE_PENALTY =
            1e-6f; // small positive regularization to improve conditioning for least-squares (1e-10 1e-7 1e-5 1e-6)
        constexpr float STOPPING_PROPORTIONAL_GAIN = 0.5f; // proportional gain for stopping controller
        constexpr float STOPPING_VELOCITY =
            0.2f; // speed threshold below which the vehicle is considered stopped and LQR is not used
        constexpr bool EIGEN_FLAG = true; // whether to use Eigen library

        // =========================================================================
        // Reference Line Parameters
        // =========================================================================
        extern float    PATH_SIZE;
        constexpr float PATH_POINT_INTERVAL = 0.2f;
        constexpr int   TRACEBACK_STEPS = 5;

        constexpr float ANCHOR_SEGMENT_LENGTH = 10.0f; // segment length used to query nearest points
        constexpr int   ANCHOR_SEGMENT_LENGTH_SIZE = static_cast<int>(ANCHOR_SEGMENT_LENGTH / PATH_POINT_INTERVAL);

        constexpr float CURVATURE_SEGMENT_LENGTH = 8.0f; // segment length for storing max curvature
        constexpr int CURVATURE_SEGMENT_LENGTH_SIZE = static_cast<int>(CURVATURE_SEGMENT_LENGTH / PATH_POINT_INTERVAL);

        constexpr int LOOKAHEAD_TIME_LENGTH_SIZE_OFFSET =
            static_cast<int>(5.0f / PATH_POINT_INTERVAL); // segment length that stores max desired speed (offset)
        constexpr int LOOKAHEAD_TIME_LENGTH_SIZE =
            static_cast<int>(6.0f / PATH_POINT_INTERVAL); // segment length that stores max desired speed

        const std::vector<float> VEL_TABLE_CURVATURE = {13.00, 10.61, 8.30, 6.50, 6.20, 6.00,
                                                        5.70,  5.50,  5.50, 5.30, 4.00, 3.50};
        const std::vector<float> CURVATURE_TABLE = {0.008, 0.010, 0.02, 0.030, 0.04, 0.05,
                                                    0.06,  0.07,  0.08, 0.10,  0.15, 0.20};

        constexpr float REF_LIGHT_DISTANCE_THRESHOLD = -10.0f; // reference line traffic-light point distance threshold

        // =========================================================================
        // Occupancy Map Parameters
        // =========================================================================
        constexpr float OCCUPANCY_MAP_RADIUS_OFFSET = 30.0f;
        constexpr float DIFFERENCE_YAXIS_OFFSET_THRESHOLD = 5.0f;
        constexpr float EGO_STRTREE_SAFETY_MARGIN_X = 1.0f;
        constexpr float EGO_STRTREE_SAFETY_MARGIN_Y = 1.0f;
        constexpr float EXO_STRTREE_SAFETY_MARGIN = 0.15f;
        constexpr float STRTREE_OFFSET = 0.15f;

        // Map Utilities Parameters
        constexpr float  DRIVABLE_MAP_UPDATE_DISTANCE = 30.0f; // distance threshold triggering drivable area re-update
        constexpr float  VELOCITY_TO_DISTANCE_TIME_HORIZON = 10.0f; // time horizon (s) for velocity-to-distance scaling
        constexpr int    STATIC_MOTION_MAX_ITERATION = 110; // max iteration count for static motion near junction
        constexpr double NEAREST_LANE_QUERY_RADIUS = 3.0;   // radius for nearest lane ID query
        constexpr double DEAD_END_MIN_PATH_LENGTH = 20.0;   // min path length to keep dead-end successor
        constexpr int    MISS_GOAL_TERMINATED_PENALTY = 10; // penalty value indicating fully terminated lane
        constexpr int    MISS_GOAL_HIGH_PENALTY_OFFSET = 5; // threshold/offset for high miss-goal penalty adjustment
        constexpr float  NEIGHBOR_RSQUARED_MIN_THRESHOLD = 0.97f;  // min R-squared for neighbor successor filtering
        constexpr float  FALLBACK_MIN_DESIRED_SPEED = 20.0f;       // default min desired speed when no curvature data
        constexpr float  PATH_LOOKBACK_DISTANCE = 40.0f;           // backward extension distance for path building
        constexpr float  LOOKAHEAD_MIN_DISTANCE = 60.0f;           // minimum extra lookahead distance
        constexpr float  TERMINAL_POINT_LARGE_FRENET_S = 10000.0f; // large frenet_s for terminal path fallback

        // Nearest Point Search Parameters
        constexpr int NEAREST_SEARCH_MAX_STEPS = 25;     // safety upper bound for nearest point search
        constexpr int NEAREST_SEARCH_PATIENCE_LIMIT = 3; // patience limit for nearest point search

        // =========================================================================
        // Reward & Penalty Parameters
        // =========================================================================
        constexpr float LAST_ACTION_PREFERENCE_REWARD_LOW = 1.0f;
        constexpr float LAST_ACTION_PREFERENCE_REWARD_HIGH = 10.0f;
        constexpr float PRUNED_THRESHOLD = -1000.0f;
        constexpr float CRASH_PENALTY = -1000.0f;
        constexpr float CROSS_EVALUATION_PENALTY = -1000.0f;
        constexpr float MOVEMENT_PENALTY = 5.0f;
        constexpr float CROSS_EVALUATION_MOVEMENT_PENALTY_HIGH = 4.0f;
        constexpr float CROSS_EVALUATION_MOVEMENT_PENALTY_MID = 3.5f;
        constexpr float CROSS_EVALUATION_MOVEMENT_PENALTY_LOW = 3.0f;
        constexpr float GOAL_PENALTY_FACTOR_HIGH = -3.0f;
        constexpr float GOAL_PENALTY_FACTOR_LOW = -1.0f;
        constexpr float ACTION_PENALTY = -2.0f;
        constexpr float LATERAL_OFFSET_PENALTY_HIGH = -0.5f;
        constexpr float LATERAL_OFFSET_PENALTY_LOW = -0.2f;
        constexpr float NON_ON_DRIVABLE_AREA_PENALTY_STAGE_1 = -4.0f;
        constexpr float NON_ON_DRIVABLE_AREA_PENALTY_STAGE_2 = -2.0f;
        constexpr float ON_NON_ROUTE_PENALTY_HIGH = -1.5f;
        constexpr float ON_NON_ROUTE_PENALTY_MID = -0.75f;
        constexpr float ON_NON_ROUTE_PENALTY_LOW = -0.5f;
        constexpr float ON_DIFFERENT_PATH_LANES_PENALTY_HIGH = -0.5f;
        constexpr float ON_DIFFERENT_PATH_LANES_PENALTY_LOW = -0.05f;
        constexpr float LATERAL_OFFSET_GUIDANCE = -100.0f;
        constexpr float LATERAL_OFFSET_GUIDANCE_OFFSET = 3.0f * LATERAL_OFFSET_GUIDANCE;
        constexpr float TTC_PENALTY = -3.0f;
        constexpr float DIR_VIOLATION_PENALTY = -1000.0f;
        constexpr float DRIVING_DIRECTION_VIOLATION_SPEED_THRESHOLD = 6.0f / TIME_STEP;
        constexpr float STEERING_PENALTY_WEIGHT_HIGH = -6.0f;
        constexpr float STEERING_PENALTY_WEIGHT_LOW = -3.0f;
        constexpr float STEERING_THRESHOLD = 0.087f; // radians (5 deg)

        // Reward function speed and distance thresholds
        constexpr float REWARD_SPEED_VERY_LOW = 1.0f; // speed below which non-route/different-path penalty is zero
        constexpr float REWARD_SPEED_LOW = 1.5f;      // speed threshold for cross-eval high penalty / steering penalty
        constexpr float REWARD_SPEED_MID = 3.0f;      // moderate speed threshold for penalty selection
        constexpr float REWARD_SPEED_HIGH = 4.0f;     // higher speed threshold for penalty selection
        constexpr float STEERING_PENALTY_SPEED_HIGH = 6.0f;       // speed above which steering penalty weight is high
        constexpr float GOAL_DISTANCE_THRESHOLD = 30.0f;          // distance threshold for goal penalty scaling
        constexpr float GOAL_DISTANCE_MIN_DENOMINATOR = 0.1f;     // minimum denominator for goal distance ratio
        constexpr float LATERAL_OFFSET_HIGH_THRESHOLD = 1.5f;     // lateral offset threshold for high penalty level
        constexpr float CURVATURE_PENALTY_HIGH_THRESHOLD = 0.08f; // high curvature threshold for penalty multiplier
        constexpr float CURVATURE_PENALTY_MED_THRESHOLD = 0.03f;  // medium curvature threshold for penalty multiplier
        constexpr float CURVATURE_PENALTY_HIGH_MULTIPLIER = 4.0f; // multiplier for high curvature regions
        constexpr float CURVATURE_PENALTY_MED_MULTIPLIER = 2.0f;  // multiplier for medium curvature regions

        // =========================================================================
        // Notice Vehicle Parameters
        // Note: vehicles may appear far in Frenet coordinates due to looped reference
        //       lines but are actually nearby in Cartesian space.
        // =========================================================================
        constexpr float         NOTICE_VEHICLE_DISTANCE = 20.0f;
        constexpr float         NOTICE_VEHICLE_DISTANCE_SQUARE = NOTICE_VEHICLE_DISTANCE * NOTICE_VEHICLE_DISTANCE;
        constexpr float         NOTICE_VEHICLE_DISTANCE_FRENET_S = 50.0f;
        constexpr float         NOTICE_VEHICLE_HEADING_DIFF = PI_3_4; // 135 deg
        extern std::vector<int> NOTICE_VEHICLE_IDXS;
        extern float            max_q_value;
        extern float            weighted_q_value;

        // =====================================================================
        // Agent Filtering Parameters (NetBelief)
        // =====================================================================
        constexpr float RED_LIGHT_PROXIMITY_DISTANCE =
            10.0f; // distance beyond red light point to trigger static motion
        constexpr float FILTER_ON_LANE_LATERAL_THRESHOLD = 1.0f; // lateral distance to consider agent "on lane"
        constexpr float FILTER_STATIC_MOTION_LONGITUDINAL_AHEAD =
            15.0f; // max ahead distance for static motion detection
        constexpr float FILTER_STATIC_MOTION_LONGITUDINAL_BEHIND =
            -5.0f; // max behind distance for static motion on other lanes
        constexpr float FILTER_NON_VEHICLE_LATERAL_THRESHOLD =
            3.5f; // lateral distance for keeping non-vehicle obstacles
        constexpr float FILTER_STATIONARY_VEHICLE_LATERAL_MAX =
            5.0f; // max lateral distance for stationary/follow vehicles
        constexpr float FILTER_VEHICLE_LATERAL_THRESHOLD =
            2.5f; // lateral distance for lead/follow vehicle on current path
        constexpr float FILTER_WIDE_LATERAL_KEEP_THRESHOLD =
            4.0f; // wide lateral distance for directly keeping ahead vehicles
        constexpr float FILTER_FOLLOW_LONGITUDINAL_OFFSET =
            3.0f; // longitudinal offset behind ego for follow vehicle detection
        constexpr float FILTER_OTHER_LANE_LATERAL_CUTOFF = 2.8f;    // lateral cutoff for other lane reference lines
        constexpr float FILTER_ADJACENT_LANE_LATERAL_OFFSET = 0.5f; // lateral offset for adjacent lane validation
        constexpr float FILTER_INTERSECTION_LONGITUDINAL_THRESHOLD =
            50.0f; // longitudinal safety distance for intersection check
        constexpr float FILTER_INTERSECTION_LATERAL_THRESHOLD = 3.0f; // lateral safety distance for intersection check
        constexpr float COLLISION_EGO_FRONT_OFFSET = 2.0f;            // offset behind ego for collision detection

        // =====================================================================
        // Junction Proximity & Exo State Parameters
        // =====================================================================
        constexpr float JUNCTION_PROXIMITY_MAX_SPEED = 1.0f; // max ego speed to be considered close to a junction
        constexpr float JUNCTION_EDGE_END_DISTANCE_THRESHOLD =
            20.0f;                                             // max distance to end of edge for junction proximity
        constexpr float JUNCTION_MIN_CONNECTOR_LENGTH = 35.0f; // min lane connector length to qualify as long junction
        constexpr float PEDESTRIAN_EXPANDED_BB_SPEED_THRESHOLD =
            1.0f; // pedestrian speed above which expanded bounding box is used
        constexpr float EXO_FRENET_STOPPED_VELOCITY =
            0.051f; // replacement velocity for near-stopped vehicles in Frenet

        // =================================================================
        // Collision Classification Parameters
        // =================================================================
        constexpr float COLLISION_REAR_HEADING_THRESHOLD =
            0.1744f; // heading angle threshold for rear collision (~10 deg)
        constexpr float COLLISION_REAR_POSITION_THRESHOLD = 0.1760f; // position angle threshold for rear collision
        constexpr int   NO_AT_FAULT_SHORT_HORIZON_ITERATIONS =
            10; // max iteration for short-horizon expanded collision penalty
        constexpr float NO_AT_FAULT_DISTANCE_THRESHOLD = 0.5f; // distance threshold for expanded collision penalty
        constexpr float NO_AT_FAULT_PENALTY_FACTOR = 0.4f;     // factor for expanded collision penalty

        // =============================================================
        // Lateral Motion Compensation Parameters (calculateLateralMotionParams)
        // =============================================================
        constexpr float LATERAL_PARAM_SPEED_THRESHOLD =
            10.0f;                                            // speed threshold separating high/low-speed lateral modes
        constexpr float LATERAL_VEL_RATIO_HIGH_SPEED = 0.15f; // v_lateral = ego_v * ratio at high speed
        constexpr float LATERAL_VEL_RATIO_LOW_SPEED = 0.3f;   // v_lateral = ego_v * ratio at low speed
        constexpr float LATERAL_VEL_MIN_EPSILON = 0.0001f;    // minimum lateral velocity to prevent division by zero
        constexpr float LATERAL_ACC_RATIO_HIGH_SPEED = 0.1f;  // a_lateral_high_speed = ego_v * ratio at high speed
        constexpr float FORWARD_DISP_LOW_SPEED_THRESHOLD =
            1.0f; // speed below which nominal-acc mode is used for forward displacement
        // =========================================================================
        // Utility Functions
        // =========================================================================
        inline float CalInterpolatedMaxSpeed(float curvature)
        {
            if (curvature <= CURVATURE_TABLE.front())
                return MAX_VEL;

            if (curvature >= CURVATURE_TABLE.back())
                return 2.5;

            int pre_index = 0;
            for (; pre_index < CURVATURE_TABLE.size() - 1; pre_index++)
            {
                if (curvature < CURVATURE_TABLE[pre_index + 1])
                    break;
            }
            int   next_index = pre_index + 1;
            float coeff =
                (curvature - CURVATURE_TABLE[pre_index]) / (CURVATURE_TABLE[next_index] - CURVATURE_TABLE[pre_index]);
            return coeff * VEL_TABLE_CURVATURE[next_index] + (1 - coeff) * VEL_TABLE_CURVATURE[pre_index];
        }
    } // namespace utils
} // namespace vec_qmdp
