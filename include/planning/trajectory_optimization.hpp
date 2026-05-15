/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file trajectory_optimization.hpp
 * @brief Vectorized trajectory generation and refinement pipeline.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <core/net_belief.hpp>
#include <core/state.hpp>
#include <fstream>
#include <memory>
#include <planning/context_qmdp.hpp>
#include <planning/tracker.hpp>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <utils/aligned_allocator.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <vamp/vector.hh>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        namespace py = boost::python;
        namespace np = boost::python::numpy;

        class TrajectoryOptimization
        {
          public:
            using Path = utils::Path;
            using FVectorT_traj = utils::FVectorT_traj;
            using IVectorT_traj = utils::IVectorT_traj;
            using FVectorT_traj_1 = utils::FVectorT_traj_1;
            using IVectorT_traj_1 = utils::IVectorT_traj_1;

            TrajectoryOptimization(int num_scenarios);

            TrajectoryOptimization(int num_scenarios, const std::vector<std::shared_ptr<Path>> &ref_paths,
                                   const std::vector<std::shared_ptr<Path>> &extra_ref_paths, size_t curr_ref_path_idx,
                                   size_t target_ref_path_idx);

            ~TrajectoryOptimization();

            void UpdatePathIndex(const std::vector<std::shared_ptr<Path>> &ref_paths,
                                 const std::vector<std::shared_ptr<Path>> &extra_ref_paths, size_t curr_ref_path_idx,
                                 int best_action_idx, int best_target_path_idx, float best_target_offset,
                                 float best_value, int second_best_action_idx, int second_best_target_path_idx,
                                 float second_best_target_offset, float second_best_value, float curr_path_value);

            // Trajectory mode info structure
            struct TrajectoryModeInfo
            {
                bool use_lf_mode;            // true: lane-following mode, false:
                                             // lane-change mode
                bool need_target_path;       // whether target path needs to be
                                             // processed
                size_t ref_path_idx;         // reference path index in use
                int    ref_path_nearest_idx; // nearest point index on reference
                                             // path
                int ref_path_idx_for_frenet; // index for Frenet coords:
                                             // 0=current path, 1=target path
                float init_path_min_dist;    // min distance to current path
                                             // (lane-change mode)
                float target_path_min_dist;  // min distance to target path
                                             // (lane-change mode)
                int init_path_nearest_idx;   // nearest point index on current
                                             // path (lane-change mode)
                int target_path_nearest_idx; // nearest point index on target
                                             // path (lane-change mode)
            };

            // Determine whether reference path is curr path or target path.
            // Returns true for target path, false for curr path.
            bool determineReferencePath(const std::shared_ptr<core::EgoState>  &ego_state,
                                        const std::shared_ptr<core::NetBelief> &belief);

            bool determineReferencePath(const std::shared_ptr<core::EgoState>  &ego_state,
                                        const std::shared_ptr<core::NetBelief> &belief,
                                        const std::shared_ptr<Path> &target_path_ptr, int target_path_idx);

            // Determine feasibility of left/right max lateral offset.
            // Feasible: true, infeasible: false; default: true.
            void determinLateralOffsetSpace(const std::shared_ptr<core::EgoState>  &ego_state,
                                            const std::shared_ptr<core::NetBelief> &belief,
                                            const std::shared_ptr<utils::MapUtils> &map_utils_ptr,
                                            bool &larget_left_offset_cared, bool &larget_right_offset_cared);

            // Determine trajectory generation mode (LF vs LC) and which paths
            // to process.
            TrajectoryModeInfo determineTrajectoryMode(const std::shared_ptr<core::EgoState> &ego_state);

            // Importance-sample scenarios from NetBelief and populate state.
            void importanceSampleScenarios(int thread_id, const std::shared_ptr<core::NetBelief> &belief,
                                           bool need_target_path);

            // Check and generate emergency_brake trajectory.
            float checkAndGenerateEmergencyBrake(const std::shared_ptr<core::EgoState> &ego_state, float lateral_offset,
                                                 float first_acc, float second_acc) const;

            // Generate candidate proposal trajectories.
            void generateProposalTrajectory(const TrajectoryModeInfo &mode_info);

            // Use LQR tracker to generate actual trajectories.
            void trackProposalTrajectoryCppBatch();

            // Perform cross-scenario evaluation.
            std::pair<size_t, float> crossScenarioEvaluation(int                                         thread_id,
                                                             const std::shared_ptr<core::EgoState>      &ego_state,
                                                             const std::shared_ptr<core::NetBelief>     &belief,
                                                             const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
                                                             const std::shared_ptr<utils::MapUtils>     &map_utils_ptr);

            // Get optimized trajectory.
            std::vector<std::vector<float>> getOptimizedTrajectory(const std::shared_ptr<core::EgoState> &ego_state,
                                                                   const size_t &best_scenario) const;

            std::vector<std::vector<float>>
            getOptimizedTrackedTrajectory(const std::shared_ptr<core::EgoState> &ego_state,
                                          const size_t                          &best_scenario) const;

            std::vector<std::vector<float>> getStaticTrajectory(const std::shared_ptr<core::EgoState> &ego_state) const;

            void optimizeTrajectoryStep1_Generate(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                                  const std::shared_ptr<core::NetBelief> &belief,
                                                  const std::shared_ptr<utils::MapUtils> &map_utils_ptr);

            std::pair<size_t, float>
            optimizeTrajectoryStep2_Evaluate(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                             const std::shared_ptr<core::NetBelief>     &belief,
                                             const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
                                             const std::shared_ptr<utils::MapUtils>     &map_utils_ptr);

          private:
            std::shared_ptr<ContextQMDP>       context_qmdp_;
            std::shared_ptr<Tracker>           tracker_;
            size_t                             path_idx_;
            std::vector<std::shared_ptr<Path>> ref_paths_;
            std::vector<std::shared_ptr<Path>> extra_ref_paths_;
            size_t                             ego_curr_path_idx_;
            size_t                             ego_target_path_idx_;
            size_t                             ego_second_target_path_idx_;
            float                              best_target_offset_;
            float                              second_best_target_offset_;
            float                              best_action_value_;
            float                              second_best_action_value_;
            float                              curr_path_value_;
            int                                exo_valid_num_;

            // Ego vehicle state (stored as proposal trajectories for tracker to
            // generate actual trajectories); per scenario.
            float                      ego_initial_x_;
            float                      ego_initial_y_;
            float                      ego_initial_v_;
            float                      ego_initial_a_;
            float                      ego_initial_theta_;
            float                      ego_current_a_;
            float                      ego_initial_steering_angle_;
            float                      ego_initial_steering_rate_;
            std::vector<FVectorT_traj> ego_xs_proposal_;              // ego vehicle x-coordinate
            std::vector<FVectorT_traj> ego_ys_proposal_;              // ego vehicle y-coordinate
            std::vector<FVectorT_traj> ego_vs_proposal_;              // ego vehicle velocity
            std::vector<FVectorT_traj> ego_as_proposal_;              // ego vehicle acceleration
            std::vector<FVectorT_traj> ego_thetas_proposal_;          // ego vehicle heading angle
            std::vector<FVectorT_traj> ego_idm_as_proposal_;          // ego vehicle IDM acceleration
            std::vector<FVectorT_traj> ego_path_curvatures_proposal_; // ego vehicle path curvature

            std::vector<FVectorT_traj> ego_xs_published_proposal_;     // publishable ego x (0.1s interval)
            std::vector<FVectorT_traj> ego_ys_published_proposal_;     // publishable ego y (0.1s interval)
            std::vector<FVectorT_traj> ego_vs_published_proposal_;     // publishable ego velocity (0.1s
                                                                       // interval)
            std::vector<FVectorT_traj> ego_as_published_proposal_;     // publishable ego acceleration
                                                                       // (0.1s interval)
            std::vector<FVectorT_traj> ego_thetas_published_proposal_; // publishable ego heading angle
                                                                       // (0.1s interval)

            std::vector<FVectorT_traj> ego_xs_traj_cpp_;
            std::vector<FVectorT_traj> ego_ys_traj_cpp_;
            std::vector<FVectorT_traj> ego_vs_traj_cpp_;
            std::vector<FVectorT_traj> ego_as_traj_cpp_;
            std::vector<FVectorT_traj> ego_thetas_traj_cpp_;

            FVectorT_traj ego_lateral_offsets_;
            FVectorT_traj ego_find_exo_lateral_offsets_;

            // Collision time (used for emergency_brake).
            int time_to_infraction_;

            // Exo vehicle state data
            AlignedVectorFloat exo_xs_flat_;               // exo vehicle x [s][v][t] -> flat array
            AlignedVectorFloat exo_ys_flat_;               // exo vehicle y [s][v][t] -> flat array
            AlignedVectorFloat exo_vs_flat_;               // exo vehicle velocity [s][v][t] -> flat array
            AlignedVectorFloat exo_thetas_flat_;           // exo vehicle heading angle
                                                           // [s][v][t] -> flat array
            AlignedVectorFloat exo_cos_thetas_flat_;       // exo vehicle cos(heading)
                                                           // [s][v][t] -> flat array
            AlignedVectorFloat exo_sin_thetas_flat_;       // exo vehicle sin(heading)
                                                           // [s][v][t] -> flat array
            AlignedVectorFloat exo_original_bb_extent_xs_; // exo vehicle width (bb_extent_x)
                                                           // [v] -> flat array
            AlignedVectorFloat exo_original_bb_extent_ys_; // exo vehicle length (bb_extent_y)
                                                           // [v] -> flat array
            AlignedVectorFloat exo_expanded_bb_extent_xs_; // exo expanded width
                                                           // [v] -> flat array
            AlignedVectorFloat exo_expanded_bb_extent_ys_; // exo expanded length [v] -> flat
                                                           // array
            std::vector<AlignedVectorFloat> exo_ss_flat_;  // exo s-coordinate [ref path idx][s][t][v] ->
                                                           // flat
            std::vector<AlignedVectorFloat> exo_ls_flat_;  // exo l-coordinate [ref path idx][s][t][v] ->
                                                           // flat
            std::vector<AlignedVectorFloat>
                exo_ls_projected_radius_flat_; // exo l projected radius [ref
                                               // path idx][s][t][v] -> flat
                                               // std::vector<std::vector<std::shared_ptr<STRtree>>>
            std::vector<std::vector<std::shared_ptr<STRtree>>> exo_expanded_strtrees_; // exo expanded STRtree [ref path
                                                                                       // idx][s][t]

            // Importance sampling weights
            std::vector<double> importance_weights_;
            AlignedVectorInt    init_nearest_idxs_;
            AlignedVectorInt    target_nearest_idxs_;

            // Generate proposal
            std::array<int, utils::MAX_SIM_VEHICLES> exo_active_flags_arr_;

            // evaluation
            std::array<float, utils::PROPOSAL_BATCH_SIZE> values_array;

            // Dimension info
            uint32_t scenario_size_;
            uint32_t global_time_size_; // number of time steps (DUMMY_TIME_STEPS 24)
            uint32_t step_time_size_;   // step duration
            uint32_t vehicle_max_size_; // vehicle count
            uint32_t exo_total_size_;   // total exo vehicles (time_size *
                                        // vehicle_size)

            // Get flattened index: scenario s, time t, vehicle v
            inline uint32_t getExoIdxTimeVehicle(uint32_t s = 0, uint32_t t = 0, uint32_t v = 0) const
            {
                return s * exo_total_size_ + t * vehicle_max_size_ + v;
            }

            // Get flattened index (vehicle-major layout): scenario s, vehicle
            // v, time t
            inline uint32_t getExoIdxVehicleTime(uint32_t s = 0, uint32_t v = 0, uint32_t t = 0) const
            {
                return s * exo_total_size_ + v * global_time_size_ + t;
            }

            // Initialize data structures
            void initializeDataStructures();
        };
    } // namespace planning
} // namespace vec_qmdp