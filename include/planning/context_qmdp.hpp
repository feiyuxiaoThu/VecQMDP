/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file context_qmdp.hpp
 * @brief Ego-environment interaction: control commands, state transitions, and trajectory evaluation.
 */

#pragma once

#include <Eigen/Dense>
#include <bitset>
#include <collision/STRtree.hpp>
#include <memory>
#include <planning/reward_functions.hpp>
#include <utils/aligned_allocator.hpp>
#include <utils/math_utils.hpp>
#include <utils/occupancy_map.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>
#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

namespace vec_qmdp
{
    namespace planning
    {
        using FVectorT_qmdp = utils::FVectorT_qmdp;
        using IVectorT_qmdp = utils::IVectorT_qmdp;
        using FVectorT_traj = utils::FVectorT_traj;
        using IVectorT_traj = utils::IVectorT_traj;
        using FVectorT_traj_1 = utils::FVectorT_traj_1;
        using IVectorT_traj_1 = utils::IVectorT_traj_1;
        using FVectorT_1 = utils::FVectorT_1;
        using IVectorT_1 = utils::IVectorT_1;
        using IVectorT_12 = utils::IVectorT_12;
        using FVectorT_12 = utils::FVectorT_12;
        using AlignedVectorFloat = utils::AlignedVectorFloat;
        using AlignedVectorInt = utils::AlignedVectorInt;
        using AlignedVectorBool = utils::AlignedVectorBool;
        using STRtree = collision::STRtree;
        using Path = utils::Path;

        class ContextQMDP
        {
          public:
            ContextQMDP();
            ~ContextQMDP();

            /**
             * @brief Single-step planning
             * @param scenario_num number of scenarios
             * @param path_size number of paths
             * @param curr_stepped_time_idx current stepped time index
             * @param ego_xs ego vehicle x positions
             * @param ego_ys ego vehicle y positions
             * @param ego_thetas ego vehicle headings
             * @param ego_vs ego vehicle velocities
             * @param ego_as ego vehicle accelerations
             * @param ego_init_path_idxs ego initial path indices
             * @param ego_paths ego reference paths
             * @param exo_xs exo vehicle x positions
             * @param exo_ys exo vehicle y positions
             * @param exo_ss exo vehicle Frenet s coordinates
             * @param exo_ls exo vehicle Frenet l coordinates
             * @param exo_vs exo vehicle velocities
             * @param exo_cos_thetas exo vehicle heading cosines
             * @param exo_sin_thetas exo vehicle heading sines
             * @param exo_bb_extent_xs exo vehicle bounding box half-widths
             * @param exo_bb_extent_ys exo vehicle bounding box half-lengths
             * @param exo_strtrees exo vehicle STRtrees
             * @param ego_target_path_idxs ego target path indices
             * @param rewards accumulated rewards
             * @param exo_size number of exo vehicles
             * @param time_size number of time steps
             */
            void StepBatch(
                const int &scenario_num, const size_t &path_size, bool low_speed_mode,
                IVectorT_qmdp &curr_stepped_time_idx, FVectorT_qmdp &ego_xs, FVectorT_qmdp &ego_ys,
                FVectorT_qmdp &ego_thetas, FVectorT_qmdp &ego_vs, FVectorT_qmdp &ego_as,
                IVectorT_qmdp &ego_init_path_idxs, FVectorT_qmdp &ego_init_lateral_offsets,
                const std::vector<std::shared_ptr<Path>> &ego_paths, // 0: left path, 1: middle path, 2: right path
                const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys,
                const std::vector<AlignedVectorFloat> &exo_ss, const std::vector<AlignedVectorFloat> &exo_ls,
                const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
                const AlignedVectorFloat &exo_bb_extent_xs, const AlignedVectorFloat &exo_bb_extent_ys,
                const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
                const IVectorT_qmdp &ego_target_path_idxs, // action, 0: left, 1: middle, 2: right
                const FVectorT_qmdp &ego_target_offsets,   // lateral offset, -1.0f, 0.0f, 1.0f
                FVectorT_qmdp &rewards, FVectorT_qmdp &rollout_rewards, uint32_t time_size, IVectorT_qmdp &active_flags,
                std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags);

            void generateProposalTrajectoryLFBatch(
                const int &scenario_num, const size_t &path_size, float ego_x, float ego_y, float ego_theta,
                float ego_v, float ego_a, const FVectorT_traj &lateral_offsets,
                const FVectorT_traj &find_exo_lateral_offsets, std::shared_ptr<Path> ego_path, int path_nearest_idx,
                const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_ss,
                const AlignedVectorFloat &exo_ls, const AlignedVectorFloat &exo_ls_projected_radius,
                const AlignedVectorFloat &exo_vs, const AlignedVectorFloat &exo_cos_thetas,
                const AlignedVectorFloat &exo_sin_thetas, const AlignedVectorFloat &exo_bb_extent_xs,
                const AlignedVectorFloat &exo_bb_extent_ys, const std::vector<std::shared_ptr<STRtree>> &exo_strtrees,
                const FVectorT_12 &exo_active_flags, std::vector<FVectorT_traj> &ego_xs_traj,
                std::vector<FVectorT_traj> &ego_ys_traj, std::vector<FVectorT_traj> &ego_thetas_traj,
                std::vector<FVectorT_traj> &ego_vs_traj, std::vector<FVectorT_traj> &ego_as_traj,
                std::vector<FVectorT_traj> &ego_idm_a_traj, std::vector<FVectorT_traj> &ego_path_curvatures_proposal,
                std::vector<FVectorT_traj> &ego_xs_published_proposal,
                std::vector<FVectorT_traj> &ego_ys_published_proposal,
                std::vector<FVectorT_traj> &ego_thetas_published_proposal,
                std::vector<FVectorT_traj> &ego_vs_published_proposal,
                std::vector<FVectorT_traj> &ego_as_published_proposal);

            void generateProposalTrajectoryLCBatch(
                int scenario_num, size_t path_size, FVectorT_traj ego_xs, FVectorT_traj ego_ys,
                FVectorT_traj ego_thetas, FVectorT_traj ego_vs, FVectorT_traj ego_as, bool turn_left,
                const FVectorT_traj &lateral_offsets, const FVectorT_traj &find_exo_lateral_offsets,
                bool                  immediate_assumed_at_target_path,
                std::shared_ptr<Path> ego_init_path,      // init path
                std::shared_ptr<Path> ego_target_path,    // target path
                IVectorT_traj         ego_curr_path_idxs, // 0: init path, 1: target path
                IVectorT_traj init_path_nearest_idxs, IVectorT_traj target_path_nearest_idxs,
                const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys,
                const std::vector<AlignedVectorFloat> &exo_ss, const std::vector<AlignedVectorFloat> &exo_ls,
                const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
                const AlignedVectorFloat &exo_bb_extent_xs, const AlignedVectorFloat &exo_bb_extent_ys,
                const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
                const FVectorT_12 &exo_active_flags, std::vector<FVectorT_traj> &ego_xs_traj,
                std::vector<FVectorT_traj> &ego_ys_traj, std::vector<FVectorT_traj> &ego_thetas_traj,
                std::vector<FVectorT_traj> &ego_vs_traj, std::vector<FVectorT_traj> &ego_as_traj,
                std::vector<FVectorT_traj> &ego_idm_a_traj, std::vector<FVectorT_traj> &ego_xs_published_proposal,
                std::vector<FVectorT_traj> &ego_ys_published_proposal,
                std::vector<FVectorT_traj> &ego_thetas_published_proposal,
                std::vector<FVectorT_traj> &ego_vs_published_proposal,
                std::vector<FVectorT_traj> &ego_as_published_proposal);

            FVectorT_traj crossScenarioEvaluationBatch(
                int thread_id, const std::vector<double> &importance_sampling_weights,
                const FVectorT_traj &lateral_offsets_proposal, float ego_initial_v,
                const std::vector<FVectorT_traj> &ego_xs_proposal, const std::vector<FVectorT_traj> &ego_ys_proposal,
                const std::vector<FVectorT_traj> &ego_as_proposal, const std::vector<FVectorT_traj> &ego_vs_proposal,
                const std::vector<FVectorT_traj> &ego_xs_traj, const std::vector<FVectorT_traj> &ego_ys_traj,
                const std::vector<FVectorT_traj> &ego_vs_traj, const std::vector<FVectorT_traj> &ego_as_traj,
                const std::vector<FVectorT_traj> &ego_thetas_traj,
                const std::vector<FVectorT_traj> &ego_path_curvatures_proposal,
                std::shared_ptr<Path>             ego_path, // reference path
                const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_ss, const AlignedVectorFloat &exo_ls,
                const AlignedVectorFloat &exo_thetas, const AlignedVectorFloat &exo_cos_thetas,
                const AlignedVectorFloat &exo_sin_thetas, const AlignedVectorFloat &exo_original_bb_extent_xs,
                const AlignedVectorFloat                    &exo_original_bb_extent_ys,
                const AlignedVectorFloat                    &exo_expanded_bb_extent_xs,
                const AlignedVectorFloat                    &exo_expanded_bb_extent_ys,
                const std::vector<std::shared_ptr<STRtree>> &exo_strtrees,
                const std::shared_ptr<utils::OccupancyMap> &occupancy_map, IVectorT_traj &collided_timesteps,
                std::vector<int> &ttc_timesteps);

            /**
             * @brief SIMD-parallelized batch version of FindLeadOrFollowVehicleCloseToEgoReferencePath.
             * Determines whether exo vehicles are ahead of / behind the ego and close to its reference path (Frenet
             * frame).
             * @param find_lead_vehicle whether to search for a lead vehicle
             * @param start_time_index starting time index
             * @param curr_nearest_idxs current nearest-point indices
             * @param ego_bb_extent_x ego bounding box half-width
             * @param exo_ss exo vehicle Frenet s coordinates
             * @param exo_ls exo vehicle Frenet l coordinates
             * @param exo_vs exo vehicle velocities
             */
            template <bool FIND_LEAD_VEHICLE>
            void FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
                std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const IVectorT_qmdp &start_time_indice,
                const IVectorT_qmdp &curr_nearest_idxs, const IVectorT_qmdp &ego_ref_path_idxs,
                const FVectorT_qmdp &ego_bb_extent_x, const FVectorT_qmdp &ego_bb_extent_y,
                const std::vector<AlignedVectorFloat> &exo_ss, const std::vector<AlignedVectorFloat> &exo_ls,
                const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
                const FVectorT_qmdp &lateral_offsets, int scenario_num, const IVectorT_qmdp &active_flags,
                const std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags);

            template <bool FIND_LEAD_VEHICLE>
            void FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
                std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, size_t t,
                const IVectorT_traj &path_nearest_idxs, float ego_bb_extent_x, float ego_bb_extent_y,
                const AlignedVectorFloat &exo_ss, const AlignedVectorFloat &exo_ls,
                const AlignedVectorFloat &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
                const FVectorT_traj &lateral_offsets, int scenario_num, const FVectorT_12 &exo_active_flags);

            template <bool FIND_LEAD_VEHICLE, bool CHECK_CURRENT_PATH>
            void FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
                std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, size_t t,
                const IVectorT_traj &curr_nearest_idxs, const IVectorT_traj &ego_ref_path_idxs, float ego_bb_extent_x,
                float ego_bb_extent_y, const std::vector<AlignedVectorFloat> &exo_ss,
                const std::vector<AlignedVectorFloat> &exo_ls,
                const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
                const FVectorT_traj &lateral_offsets, int scenario_num, const IVectorT_traj &active_flags,
                const FVectorT_12 &exo_active_flags);

            /**
             * @brief SIMD-parallelized batch version of FindLeadVehicleIntersectedBatch.
             * Checks whether the ego vehicle collides with exo vehicles over a time window.
             * @param t starting time index (already-stepped time index within the current Step call)
             * @param path reference path
             * @param curr_nearest_idxs current nearest-point indices
             * @param ego_vs ego vehicle velocities
             * @param ego_bb_extent_x ego bounding box half-width
             * @param ego_bb_extent_y ego bounding box half-length
             * @param exo_vs exo vehicle velocities
             * @param exo_strtrees exo vehicle STRtrees
             * @param valid_count number of valid scenarios
             * @param curr_stepped_time_idx global stepped time index (from root node)
             */
            void FindLeadVehicleIntersectedBatch(
                std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const size_t &path_size,
                const IVectorT_qmdp &curr_stepped_time_idxs, const IVectorT_qmdp &curr_nearest_idxs,
                const IVectorT_qmdp &ego_ref_path_idxs, const FVectorT_qmdp &ego_vs, float ego_bb_extent_x,
                float ego_bb_extent_y, const AlignedVectorFloat &exo_vs, const AlignedVectorFloat &exo_bb_extent_y,
                const std::vector<AlignedVectorFloat>                    &exo_ss,
                const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
                const FVectorT_qmdp &lateral_offsets, int scenario_num, const IVectorT_qmdp &active_flags,
                const std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags);

            void FindLeadVehicleIntersectedBatch(std::pair<AlignedVectorFloat, AlignedVectorFloat> &results,
                                                 const size_t &path_size, const size_t &curr_stepped_time_idx,
                                                 const IVectorT_traj &path_nearest_idxs, const FVectorT_traj &ego_vs,
                                                 float ego_bb_extent_x, float ego_bb_extent_y,
                                                 const AlignedVectorFloat                    &exo_vs,
                                                 const AlignedVectorFloat                    &exo_bb_extent_y,
                                                 const AlignedVectorFloat                    &exo_ss,
                                                 const std::vector<std::shared_ptr<STRtree>> &exo_strtrees,
                                                 const FVectorT_traj &lateral_offsets, int scenario_num);

            void FindLeadVehicleIntersectedBatch(
                std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const size_t &path_size,
                const size_t &curr_stepped_time_idx, const IVectorT_traj &curr_nearest_idxs,
                const IVectorT_traj &ego_ref_path_idxs, // 0: init path; 1: target path;
                const FVectorT_traj &ego_vs, float ego_bb_extent_x, float ego_bb_extent_y,
                const AlignedVectorFloat &exo_vs, const AlignedVectorFloat &exo_bb_extent_y,
                const std::vector<AlignedVectorFloat>                    &exo_ss,
                const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
                const FVectorT_traj &lateral_offsets, int scenario_num, const IVectorT_traj &active_flags);

            bool
            LCAllowanceCheckSerial(const std::pair<float, float> &LC_lead_vehicles_close_to_ego_reference_path,
                                   const std::pair<float, float> &LC_following_vehicles_close_to_ego_reference_path,
                                   float ego_v, float ego_desired_v, bool active_flag);

            /**
             * @brief Lane change allowance check
             * @param LF_lead_vehicles_close_to_ego_reference_path lead vehicle info on the current lane
             * @param LC_lead_vehicles_close_to_ego_reference_path lead vehicle info on the target lane
             * @param LC_following_vehicles_close_to_ego_reference_path following vehicle info on the target lane
             * @param ego_vs ego vehicle velocities
             * @param ego_desired_vs ego desired velocities
             * @param active_flags active scenario flags
             */
            IVectorT_qmdp LCAllowanceCheckBatch(
                const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path,
                const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path,
                const std::pair<AlignedVectorFloat, AlignedVectorFloat>
                                    &LC_following_vehicles_close_to_ego_reference_path,
                const FVectorT_qmdp &ego_vs, const FVectorT_qmdp &ego_thetas, const FVectorT_qmdp &ego_desired_vs,
                const FVectorT_qmdp &ego_vs_lateral, const IVectorT_qmdp &active_flags,
                const IVectorT_qmdp &consider_left_LC_mask, const IVectorT_qmdp &consider_right_LC_mask,
                const FVectorT_qmdp &ego_curr_path_ls, const FVectorT_qmdp &ego_curr_path_thetas,
                IVectorT_qmdp &LF_lead_allowance_flags);

            IVectorT_traj LCAllowanceCheckBatch(
                const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path,
                const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path,
                const std::pair<AlignedVectorFloat, AlignedVectorFloat>
                                    &LC_following_vehicles_close_to_ego_reference_path,
                const FVectorT_traj &ego_vs, const FVectorT_traj &ego_thetas, const FVectorT_traj &ego_desired_vs,
                const FVectorT_traj &ego_vs_lateral, const IVectorT_traj &active_flags, bool turn_left,
                const FVectorT_traj &ego_curr_path_ls, const FVectorT_traj &ego_curr_path_thetas,
                IVectorT_traj &LF_lead_allowance_flags);

            float calculateLateralMotionParams(float ego_vs, float actual_lateral_distances_abs,
                                               bool simplified = false);

            float calculateLateralMotionParams(float ego_vs, float actual_lateral_distances_abs, float v_lateral,
                                               bool simplified = false, bool for_LF = false);

            FVectorT_qmdp calculateLateralMotionParams(const FVectorT_qmdp &ego_vs,
                                                       const FVectorT_qmdp &actual_lateral_distances_abs,
                                                       bool                 simplified = false);

            FVectorT_qmdp calculateLateralMotionParams(const FVectorT_qmdp &ego_vs,
                                                       const FVectorT_qmdp &actual_lateral_distances_abs,
                                                       const FVectorT_qmdp &v_lateral, bool simplified = false,
                                                       bool for_LF = false);

            FVectorT_traj calculateLateralMotionParams(const FVectorT_traj &ego_vs,
                                                       const FVectorT_traj &actual_lateral_distances_abs,
                                                       bool                 simplified = false);

            FVectorT_traj calculateLateralMotionParams(const FVectorT_traj &ego_vs,
                                                       const FVectorT_traj &actual_lateral_distances_abs,
                                                       const FVectorT_traj &v_lateral, bool simplified = false,
                                                       bool for_LF = false);

            float IDMSerial(float leading_vehicle_v, float following_vehicle_v, float relative_distance,
                            float desired_speed);

            /**
             * @brief Compute acceleration using the IDM (Intelligent Driver Model)
             * @param leading_vehicle_vs lead vehicle velocities
             * @param following_vehicle_vs following vehicle velocities
             * @param relative_distance relative distance to lead vehicle
             * @param desired_speed desired cruising speed
             */
            FVectorT_qmdp IDMBatch(const FVectorT_qmdp &leading_vehicle_vs, const FVectorT_qmdp &following_vehicle_vs,
                                   const FVectorT_qmdp &relative_distance, const FVectorT_qmdp &desired_speed);

            FVectorT_traj IDMBatch(const FVectorT_traj &leading_vehicle_vs, const FVectorT_traj &following_vehicle_vs,
                                   const FVectorT_traj &relative_distance, const FVectorT_traj &desired_speed);

            FVectorT_traj CurveAndLaneChangeAwareIDMBatch(const FVectorT_traj &leading_vehicle_vs,
                                                          const FVectorT_traj &following_vehicle_vs,
                                                          const FVectorT_traj &relative_distance,
                                                          const FVectorT_traj &desired_speed,
                                                          const IVectorT_traj &lane_change_flag);

            /**
             * @brief Compute steering angle using the Stanley controller
             * @param ref_point_xs reference point x positions
             * @param ref_point_ys reference point y positions
             * @param ref_point_headings reference point headings
             * @param future_curvatures path curvatures ahead
             * @param lateral_offsets lateral offsets from reference path
             * @param ego_xs ego vehicle x positions
             * @param ego_ys ego vehicle y positions
             * @param ego_vs ego vehicle velocities
             * @param ego_thetas ego vehicle headings
             * @param ego_thetas_cos cosines of ego headings
             * @param ego_thetas_sin sines of ego headings
             */
            FVectorT_qmdp
            GetSteeringByStanleyBatch(const FVectorT_qmdp &ref_point_xs, const FVectorT_qmdp &ref_point_ys,
                                      const FVectorT_qmdp &ref_point_headings, const FVectorT_qmdp &future_curvatures,
                                      const FVectorT_qmdp &lateral_offsets, const FVectorT_qmdp &ego_xs,
                                      const FVectorT_qmdp &ego_ys, const FVectorT_qmdp &ego_vs,
                                      const FVectorT_qmdp &ego_thetas, const FVectorT_qmdp &ego_thetas_cos,
                                      const FVectorT_qmdp &ego_thetas_sin, bool print_debug = false);

            FVectorT_traj
            GetSteeringByStanleyBatch(const FVectorT_traj &ref_point_xs, const FVectorT_traj &ref_point_ys,
                                      const FVectorT_traj &ref_point_thetas, const FVectorT_traj &future_curvatures,
                                      const FVectorT_traj &lateral_offsets, const FVectorT_traj &ego_xs,
                                      const FVectorT_traj &ego_ys, const FVectorT_traj &ego_vs,
                                      const FVectorT_traj &ego_thetas, const FVectorT_traj &ego_thetas_cos,
                                      const FVectorT_traj &ego_thetas_sin);

            /**
             * @brief Update ego vehicle state using bicycle model
             * @param ego_xs ego x coordinates
             * @param ego_ys ego y coordinates
             * @param ego_vs ego velocities
             * @param ego_accs ego accelerations
             * @param ego_thetas ego heading angles
             * @param ego_thetas_cos cosines of ego headings
             * @param ego_thetas_sin sines of ego headings
             * @param accs acceleration commands
             * @param steerings steering angle commands
             */
            void UpdateStateBatch(FVectorT_qmdp &ego_xs, FVectorT_qmdp &ego_ys, FVectorT_qmdp &ego_vs,
                                  FVectorT_qmdp &ego_accs, FVectorT_qmdp &ego_thetas, FVectorT_qmdp &ego_thetas_cos,
                                  FVectorT_qmdp &ego_thetas_sin, const FVectorT_qmdp &cal_accs,
                                  const FVectorT_qmdp &cal_steerings);

            void UpdateStateBatch(FVectorT_traj &ego_xs, FVectorT_traj &ego_ys, FVectorT_traj &ego_vs,
                                  FVectorT_traj &ego_as, FVectorT_traj &ego_thetas, FVectorT_traj &ego_thetas_cos,
                                  FVectorT_traj &ego_thetas_sin, const FVectorT_traj &cal_accs,
                                  const FVectorT_traj &cal_steerings, float time_step = utils::TIME_STEP);

            void UpdateSpeedBatch(FVectorT_traj &ego_vs, FVectorT_traj &ego_avg_vs, FVectorT_traj &ego_as,
                                  const FVectorT_traj &cal_accs, float time_step = utils::TIME_STEP);

            void UpdateHeadingBatch(FVectorT_traj &ego_xs, FVectorT_traj &ego_ys, FVectorT_traj &ego_avg_speeds,
                                    FVectorT_traj &ego_thetas, FVectorT_traj &ego_thetas_cos,
                                    FVectorT_traj &ego_thetas_sin, const FVectorT_traj &cal_steerings,
                                    float time_step = utils::TIME_STEP);

            // Reward / penalty functions and collision classification have been moved to
            // include/planning/reward_functions.hpp and are available as free functions
            // in the vec_qmdp::planning namespace.

            inline void UpdateExoSize(const int &exo_size) { exo_size_ = exo_size; }

            inline int GetExoSize() const { return exo_size_; }

          public:
            // Compute ego vehicle corners
            void computeEgoCornersBatch(const FVectorT_qmdp &ego_xs, const FVectorT_qmdp &ego_ys,
                                        const FVectorT_qmdp &ego_cos_thetas, const FVectorT_qmdp &ego_sin_thetas,
                                        float ego_bb_extent_x, float ego_bb_extent_y, FVectorT_qmdp *corners_xs,
                                        FVectorT_qmdp *corners_ys);

            void computeEgoCornersBatch(const FVectorT_traj &ego_xs, const FVectorT_traj &ego_ys,
                                        const FVectorT_traj &ego_cos_thetas, const FVectorT_traj &ego_sin_thetas,
                                        float ego_bb_extent_x, float ego_bb_extent_y, FVectorT_traj *corners_xs,
                                        FVectorT_traj *corners_ys);

            // Check ego-obstacle collision using Separating Axis Theorem (SAT)
            // True: no collision, False: collision
            FVectorT_1 checkSATCollisionBatch(const FVectorT_1 &ego_xs, const FVectorT_1 &ego_ys,
                                              const FVectorT_1 &ego_cos_thetas, const FVectorT_1 &ego_sin_thetas,
                                              const FVectorT_1 &exo_xs, const FVectorT_1 &exo_ys,
                                              const FVectorT_1 &exo_cos_thetas, const FVectorT_1 &exo_sin_thetas,
                                              float ego_bb_extent_x, float ego_bb_extent_y,
                                              const FVectorT_1 &exo_bb_extent_xs, const FVectorT_1 &exo_bb_extent_ys);

            /**
             * @brief Check ego-obstacle collision using Separating Axis Theorem (SAT)
             * @param ego_x ego vehicle x coordinate
             * @param ego_y ego vehicle y coordinate
             * @param ego_cos_theta cosine of ego heading
             * @param ego_sin_theta sine of ego heading
             * @param ego_bb_extent_x ego bounding box half-width
             * @param ego_bb_extent_y ego bounding box half-length
             * @return collided vehicle index, -1 if no collision
             */
            std::vector<int> checkSATCollisionBatch(float ego_x, float ego_y, float ego_cos_theta, float ego_sin_theta,
                                                    const FVectorT_qmdp &exo_xs, const FVectorT_qmdp &exo_ys,
                                                    const FVectorT_qmdp &exo_cos_thetas,
                                                    const FVectorT_qmdp &exo_sin_thetas, float ego_bb_extent_x,
                                                    float ego_bb_extent_y, const FVectorT_qmdp &exo_bb_extent_xs,
                                                    const FVectorT_qmdp &exo_bb_extent_ys);

            FVectorT_qmdp checkSATCollisionBatch(float ego_x, float ego_y, float ego_cos_theta, float ego_sin_theta,
                                                 const FVectorT_qmdp &exo_xs, const FVectorT_qmdp &exo_ys,
                                                 const FVectorT_qmdp &exo_cos_thetas,
                                                 const FVectorT_qmdp &exo_sin_thetas, float ego_bb_extent_x,
                                                 float ego_bb_extent_y, const FVectorT_qmdp &exo_bb_extent_xs,
                                                 const FVectorT_qmdp &exo_bb_extent_ys, FVectorT_qmdp &min_gaps);

            FVectorT_1 checkSATCollisionBatch1(float ego_x, float ego_y, float ego_cos_theta, float ego_sin_theta,
                                               const FVectorT_1 &exo_xs, const FVectorT_1 &exo_ys,
                                               const FVectorT_1 &exo_cos_thetas, const FVectorT_1 &exo_sin_thetas,
                                               float ego_bb_extent_x, float ego_bb_extent_y,
                                               const FVectorT_1 &exo_bb_extent_xs, const FVectorT_1 &exo_bb_extent_ys);

            // 0: not collided; 0xFFFFFFFF: collided
            FVectorT_traj_1 checkSATCollisionBatchTraj(const FVectorT_traj_1 &ego_xs, const FVectorT_traj_1 &ego_ys,
                                                       const FVectorT_traj_1 &ego_cos_thetas,
                                                       const FVectorT_traj_1 &ego_sin_thetas, float exo_x, float exo_y,
                                                       float exo_cos_theta, float exo_sin_theta, float ego_bb_extent_x,
                                                       float ego_bb_extent_y, float exo_bb_extent_x,
                                                       float exo_bb_extent_y, FVectorT_traj_1 &min_gaps);

            int HasCollisionBatch(float ego_rear_x, float ego_rear_y, float ego_x, float ego_y, float ego_theta,
                                  float ego_cos_theta, float ego_sin_theta, const AlignedVectorFloat &exo_xs,
                                  const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_cos_thetas,
                                  const AlignedVectorFloat &exo_sin_thetas, const AlignedVectorFloat &exo_bb_extent_xs,
                                  const AlignedVectorFloat       &exo_bb_extent_ys,
                                  const std::shared_ptr<STRtree> &exo_strtree, const size_t &scenario_idx,
                                  float ego_bb_extent_x, float ego_bb_extent_y, float query_aabb_min_x,
                                  float query_aabb_min_y, float query_aabb_max_x, float query_aabb_max_y,
                                  const size_t                             &curr_stepped_time_idx,
                                  std::array<int, utils::MAX_SIM_VEHICLES> &exo_inactive_flag);

            void HasCollisionBatch(
                int thread_id, size_t scenario_offset, const FVectorT_traj &ego_xs, const FVectorT_traj &ego_ys,
                const FVectorT_traj &ego_cos_thetas, const FVectorT_traj &ego_sin_thetas,
                const FVectorT_traj &query_aabb_min_xs, const FVectorT_traj &query_aabb_min_ys,
                const FVectorT_traj &query_aabb_max_xs, const FVectorT_traj &query_aabb_max_ys,
                const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_vs,
                const AlignedVectorFloat &exo_thetas, const AlignedVectorFloat &exo_cos_thetas,
                const AlignedVectorFloat &exo_sin_thetas, const AlignedVectorFloat &exo_original_bb_extent_xs,
                const AlignedVectorFloat                    &exo_original_bb_extent_ys,
                const AlignedVectorFloat                    &exo_expanded_bb_extent_xs,
                const AlignedVectorFloat                    &exo_expanded_bb_extent_ys,
                const std::vector<std::shared_ptr<STRtree>> &strtrees, float ego_bb_extent_x, float ego_bb_extent_y,
                AlignedVectorInt &has_original_collision, AlignedVectorInt &has_expanded_collision,
                AlignedVectorFloat &collided_min_distances, AlignedVectorInt &collided_exo_idxs,
                AlignedVectorFloat &collided_exo_xs, AlignedVectorFloat &collided_exo_ys,
                AlignedVectorFloat &collided_exo_vs, AlignedVectorFloat &collided_exo_thetas,
                AlignedVectorFloat &collided_exo_cos_thetas, AlignedVectorFloat &collided_exo_sin_thetas,
                AlignedVectorFloat &collided_exo_expanded_bb_extent_xs,
                AlignedVectorFloat &collided_exo_expanded_bb_extent_ys, int curr_stepped_time_idx,
                const IVectorT_traj                                   &active_flags,
                std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags, bool print);

          private:
            // Compute exo vehicle index (for Frenet s, l, nearest idx)
            inline uint32_t getExoIdxTimeVehicle(uint32_t s, uint32_t t, uint32_t v) const
            {
                return s * exo_total_size_ + t * vehicle_size_ + v;
            }

            // Compute exo vehicle index (for other attributes)
            inline uint32_t getExoIdxVehicleTime(uint32_t s, uint32_t v, uint32_t t) const
            {
                return s * exo_total_size_ + v * total_time_size_ + t;
            }

            // Compute STRtree index
            inline uint32_t getStrtreeIdx(uint32_t s, uint32_t t) const { return s * total_time_size_ + t; }

            // Compute exo vehicle index for proposals (for Frenet s, l, nearest idx)
            inline uint32_t getExoIdxTimeVehicleProposal(uint32_t s, uint32_t t, uint32_t v) const
            {
                return s * exo_total_size_proposal_ + t * vehicle_size_ + v;
            }

            // Compute exo vehicle index for proposals (for other attributes)
            inline uint32_t getExoIdxVehicleTimeProposal(uint32_t s, uint32_t v, uint32_t t) const
            {
                return s * exo_total_size_proposal_ + v * total_time_size_proposal_ + t;
            }

            // Compute STRtree index for proposals
            inline uint32_t getStrtreeIdxProposal(uint32_t s, uint32_t t) const
            {
                return s * total_time_size_proposal_ + t;
            }

          private:
            int vector_qmdp_width = FVectorT_qmdp::num_scalars_per_row;

            int vehicle_size_;             // Maximum number of exo vehicles
            int total_time_size_;          // Number of time steps (dummy)
            int exo_size_;                 // Number of valid exo vehicles
            int exo_total_size_;           // Total exo vehicles across all time steps
            int total_time_size_proposal_; // Number of time steps for trajectory optimization (dummy)
            int exo_total_size_proposal_;  // Total exo vehicles for trajectory optimization
        };

    } // namespace planning
} // namespace vec_qmdp