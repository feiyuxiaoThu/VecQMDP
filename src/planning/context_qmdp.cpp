/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <planning/context_qmdp.hpp>
#include <utils/params.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        ContextQMDP::ContextQMDP()
        {
            vehicle_size_ = utils::MAX_SIM_VEHICLES;
            total_time_size_ = utils::DUMMY_TIME_STEPS;
            exo_total_size_ = utils::DUMMY_TIME_STEPS * utils::MAX_SIM_VEHICLES;
            total_time_size_proposal_ = utils::PROPOSAL_DUMMY_SIZE;
            exo_total_size_proposal_ = utils::PROPOSAL_DUMMY_SIZE * utils::MAX_SIM_VEHICLES;
        }

        ContextQMDP::~ContextQMDP() {}

        void ContextQMDP::StepBatch(
            const int &scenario_num, const size_t &path_size, bool low_speed_mode, IVectorT_qmdp &curr_stepped_time_idx,
            FVectorT_qmdp &ego_xs, FVectorT_qmdp &ego_ys, FVectorT_qmdp &ego_thetas, FVectorT_qmdp &ego_vs,
            FVectorT_qmdp &ego_as, IVectorT_qmdp &ego_init_path_idxs, FVectorT_qmdp &ego_init_lateral_offsets,
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
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags)
        {
            // if there is no active flag, return
            if (active_flags.none())
            {
                return;
            }

            constexpr float ego_bb_extent_x = utils::EGO_BB_EXTENT_X;
            constexpr float ego_bb_extent_y = utils::EGO_BB_EXTENT_Y;

            static thread_local utils::StepBatchWorkspace ws(scenario_num);
            ws.reset(scenario_num);

            // Compute the time index at the end of the step
            IVectorT_qmdp &step_end_time_idxs = ws.step_end_time_idxs;

            {
                step_end_time_idxs = curr_stepped_time_idx + static_cast<int32_t>(time_size);

                if ((step_end_time_idxs >= utils::MAX_PRED_TIME_STEPS).all())
                    return;
            }

            IVectorT_qmdp &curr_path_is_left = ws.curr_path_is_left;
            IVectorT_qmdp &curr_path_is_middle = ws.curr_path_is_middle;
            IVectorT_qmdp &curr_path_is_right = ws.curr_path_is_right;
            IVectorT_qmdp &target_path_is_left = ws.target_path_is_left;
            IVectorT_qmdp &target_path_is_middle = ws.target_path_is_middle;
            IVectorT_qmdp &target_path_is_right = ws.target_path_is_right;
            IVectorT_qmdp &left_path_mask = ws.left_path_mask;
            IVectorT_qmdp &middle_path_mask = ws.middle_path_mask;
            IVectorT_qmdp &right_path_mask = ws.right_path_mask;
            IVectorT_qmdp &left_path_nearest_idxs = ws.left_path_nearest_idxs;
            IVectorT_qmdp &middle_path_nearest_idxs = ws.middle_path_nearest_idxs;
            IVectorT_qmdp &right_path_nearest_idxs = ws.right_path_nearest_idxs;
            IVectorT_qmdp &ego_curr_path_idxs = ws.ego_curr_path_idxs;
            FVectorT_qmdp &ego_curr_lateral_offsets =
                ws.ego_curr_lateral_offsets; // Lateral offset to apply on current path (if already on target path, use
                                             // target offset; otherwise, use offset needed for lane change)
            IVectorT_qmdp &should_switch_path = ws.should_switch_path;
            IVectorT_qmdp &ego_curr_path_nearest_idxs = ws.ego_curr_path_nearest_idxs;
            IVectorT_qmdp &ego_target_path_nearest_idxs = ws.ego_target_path_nearest_idxs;

            FVectorT_qmdp &ego_cos_thetas = ws.ego_cos_thetas;
            FVectorT_qmdp &ego_sin_thetas = ws.ego_sin_thetas;
            FVectorT_qmdp &left_path_min_distance = ws.left_path_min_distance;
            FVectorT_qmdp &middle_path_min_distance = ws.middle_path_min_distance;
            FVectorT_qmdp &right_path_min_distance = ws.right_path_min_distance;
            FVectorT_qmdp &target_path_min_distance = ws.target_path_min_distance;
            FVectorT_qmdp &ego_curr_path_ls = ws.ego_curr_path_ls;
            FVectorT_qmdp &future_curvatures = ws.future_curvatures;
            FVectorT_qmdp &ego_desired_vs = ws.ego_desired_vs;
            FVectorT_qmdp &ego_curr_path_heading = ws.ego_curr_path_heading;

            FVectorT_qmdp &curr_path_thetas_sin = ws.curr_path_thetas_sin;
            FVectorT_qmdp &curr_path_thetas_cos = ws.curr_path_thetas_cos;
            FVectorT_qmdp &ego_proj_radius_ls = ws.ego_proj_radius_ls;
            FVectorT_qmdp &v_lateral = ws.v_lateral;
            FVectorT_qmdp &relative_distance = ws.relative_distance;
            FVectorT_qmdp &leading_exo_vs = ws.leading_exo_vs;
            FVectorT_qmdp &ref_point_xs = ws.ref_point_xs;
            FVectorT_qmdp &ref_point_ys = ws.ref_point_ys;
            FVectorT_qmdp &ref_point_thetas = ws.ref_point_thetas;
            FVectorT_qmdp &cal_accs = ws.cal_accs;
            FVectorT_qmdp &cal_steerings = ws.cal_steerings;
            FVectorT_qmdp &curr_path_min_distance = ws.curr_path_min_distance;
            FVectorT_qmdp &points_on_path_left = ws.points_on_path_left;
            FVectorT_qmdp &ego_curr_path_ss = ws.ego_curr_path_ss;
            FVectorT_qmdp &min_s_values = ws.min_s_values;
            FVectorT_qmdp &min_l_values = ws.min_l_values;
            FVectorT_qmdp &max_s_values = ws.max_s_values;
            FVectorT_qmdp &max_l_values = ws.max_l_values;
            FVectorT_qmdp &ego_rear_xs = ws.ego_rear_xs;
            FVectorT_qmdp &ego_rear_ys = ws.ego_rear_ys;
            FVectorT_qmdp &active_flags_float = ws.active_flags_float;
            FVectorT_qmdp &collision_flags_float = ws.collision_flags_float;
            FVectorT_qmdp &collision_penalty = ws.collision_penalty;
            FVectorT_qmdp &movement_penalty = ws.movement_penalty;
            FVectorT_qmdp &miss_goal_rewards = ws.miss_goal_rewards;
            FVectorT_qmdp &action_penalties = ws.action_penalties;
            FVectorT_qmdp &rollout_step_reward = ws.rollout_step_reward;
            FVectorT_qmdp &miss_goal_pen = ws.miss_goal_pen;
            FVectorT_qmdp &goal_diff = ws.goal_diff;

            IVectorT_qmdp &consider_LC_mask = ws.consider_LC_mask;
            IVectorT_qmdp &consider_left_LC_mask = ws.consider_left_LC_mask;
            IVectorT_qmdp &consider_right_LC_mask = ws.consider_right_LC_mask;
            IVectorT_qmdp &LC_allowance_flags = ws.LC_allowance_flags;
            IVectorT_qmdp &LF_lead_allowance_flags = ws.LF_lead_allowance_flags;
            IVectorT_qmdp &collision_flags = ws.collision_flags;

            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path =
                ws.LF_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path =
                ws.LC_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_following_vehicles_close_to_ego_reference_path =
                ws.LC_following_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_future_path_intersected =
                ws.LF_lead_vehicles_future_path_intersected;
            AlignedVectorFloat &final_relative_distance_vec = ws.final_relative_distance_vec;

            AlignedVectorFloat &final_exo_vs_vec = ws.final_exo_vs_vec;
            AlignedVectorFloat &ref_point_xs_vec = ws.ref_point_xs_vec;
            AlignedVectorFloat &ref_point_ys_vec = ws.ref_point_ys_vec;
            AlignedVectorFloat &ref_point_thetas_vec = ws.ref_point_thetas_vec;
            AlignedVectorFloat &points_on_path_left_vec = ws.points_on_path_left_vec;

            {
                // 1. Basic data preparation
                ego_cos_thetas = ego_thetas.cos();
                ego_sin_thetas = ego_thetas.sin();

                // 2. Path Mask preparation
                curr_path_is_left = ego_init_path_idxs == 0;
                curr_path_is_middle = ego_init_path_idxs == 1;
                curr_path_is_right = ego_init_path_idxs == 2;

                target_path_is_left = ego_target_path_idxs == 0;
                target_path_is_middle = ego_target_path_idxs == 1;
                target_path_is_right = ego_target_path_idxs == 2;

                // Combined Mask: compute left lane data if current is left OR target is left
                left_path_mask = (target_path_is_left | curr_path_is_left) & active_flags;
                middle_path_mask = (target_path_is_middle | curr_path_is_middle) & active_flags;
                right_path_mask = (target_path_is_right | curr_path_is_right) & active_flags;

                // 3. Batch compute Nearest Index (NearestBatch internally applies Mask pruning)
                left_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);
                middle_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);
                right_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);

                // Define temporary variables to store results for three paths
                FVectorT_qmdp curv_0(0.0f), curv_1(0.0f), curv_2(0.0f);
                FVectorT_qmdp speed_0(utils::MAX_VEL), speed_1(utils::MAX_VEL), speed_2(utils::MAX_VEL);
                FVectorT_qmdp curr_k_0(0.0f), curr_k_1(0.0f), curr_k_2(0.0f);
                FVectorT_qmdp theta_0(0.0f), theta_1(0.0f), theta_2(0.0f);
                FVectorT_qmdp side_0(0.0f), side_1(0.0f), side_2(0.0f);

                // Note: if ego_paths[x] is null, initialize to 0 to prevent Gather out-of-bounds
                if (ego_paths[0] && left_path_mask.any())
                {
                    left_path_nearest_idxs = ego_paths[0]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        ego_xs, ego_ys, left_path_mask, left_path_min_distance);

                    // Batch Gather Theta
                    theta_0 = ego_paths[0]->GetThetaBatch(left_path_nearest_idxs);
                    // Batch compute PointOnLeft (returns 1.0 or -1.0)
                    side_0 = ego_paths[0]->PointOnLeft(ego_xs, ego_ys, left_path_nearest_idxs);
                }
                else
                {
                    left_path_nearest_idxs = IVectorT_qmdp::fill(0);
                }

                if (ego_paths[1] && middle_path_mask.any())
                {
                    middle_path_nearest_idxs = ego_paths[1]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        ego_xs, ego_ys, middle_path_mask, middle_path_min_distance);

                    // Batch Gather Theta
                    theta_1 = ego_paths[1]->GetThetaBatch(middle_path_nearest_idxs);
                    // Batch compute PointOnLeft (returns 1.0 or -1.0)
                    side_1 = ego_paths[1]->PointOnLeft(ego_xs, ego_ys, middle_path_nearest_idxs);
                }
                else
                {
                    middle_path_nearest_idxs = IVectorT_qmdp::fill(0);
                }

                if (ego_paths[2] && right_path_mask.any())
                {
                    right_path_nearest_idxs = ego_paths[2]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        ego_xs, ego_ys, right_path_mask, right_path_min_distance);

                    // Batch Gather Theta
                    theta_2 = ego_paths[2]->GetThetaBatch(right_path_nearest_idxs);
                    // Batch compute PointOnLeft (returns 1.0 or -1.0)
                    side_2 = ego_paths[2]->PointOnLeft(ego_xs, ego_ys, right_path_nearest_idxs);
                }
                else
                {
                    right_path_nearest_idxs = IVectorT_qmdp::fill(0);
                }

                // 4. Path switching logic (pure SIMD)
                target_path_min_distance =
                    FVectorT_qmdp::select(target_path_is_left.template as<FVectorT_qmdp>(), left_path_min_distance,
                                          FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(),
                                                                middle_path_min_distance, right_path_min_distance));

                FVectorT_qmdp target_path_chosen_side = FVectorT_qmdp::select(
                    target_path_is_left.template as<FVectorT_qmdp>(), side_0,
                    FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(), side_1, side_2));

                auto target_path_min_distance_offset =
                    (target_path_min_distance * target_path_chosen_side - ego_target_offsets).abs();

                should_switch_path = (target_path_min_distance < utils::ACTION_CHANGE_DISTANCE_THRESHOLD |
                                      target_path_min_distance_offset < utils::ACTION_CHANGE_DISTANCE_THRESHOLD)
                                         .template as<IVectorT_qmdp>() &
                                     (ego_init_path_idxs != ego_target_path_idxs);

                // Update current path index
                ego_curr_path_idxs =
                    IVectorT_qmdp::select(should_switch_path, ego_target_path_idxs, ego_init_path_idxs);
                ego_curr_lateral_offsets = FVectorT_qmdp::select(
                    (ego_curr_path_idxs == ego_target_path_idxs).template as<FVectorT_qmdp>(), ego_target_offsets,
                    ego_init_lateral_offsets); // For vehicles on the same path, directly use target offset
                                               // (ego_target_offsets)

                // Recompute Mask (since ego_curr_path_idxs may have changed)
                // Note: the above Masks were for NearestBatch (Current | Target),
                // here we need the exact "Current Path Mask" to determine which path's data to use
                curr_path_is_left = (ego_curr_path_idxs == 0);
                curr_path_is_middle = (ego_curr_path_idxs == 1);
                curr_path_is_right = (ego_curr_path_idxs == 2);
                left_path_mask = (target_path_is_left | curr_path_is_left) & active_flags;
                middle_path_mask = (target_path_is_middle | curr_path_is_middle) & active_flags;
                right_path_mask = (target_path_is_right | curr_path_is_right) & active_flags;

                // --- Path 0 (Left) ---
                // Only compute when Path 0 exists and vehicles are actually using Path 0
                if (ego_paths[0] && left_path_mask.any())
                {
                    // 1. Batch compute Curvature & Speed (using optimized Batch function)
                    // Note: pass left_path_mask for possible internal pruning (though the Batch function may not use
                    // the mask)
                    auto res = ego_paths[0]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        left_path_nearest_idxs, ego_vs, curr_k_0, curr_path_is_left & active_flags);
                    curv_0 = res.first;
                    speed_0 = res.second;
                }

                // --- Path 1 (Middle) ---
                if (ego_paths[1] && middle_path_mask.any())
                {
                    auto res = ego_paths[1]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        middle_path_nearest_idxs, ego_vs, curr_k_1, curr_path_is_middle & active_flags);
                    curv_1 = res.first;
                    speed_1 = res.second;
                }

                // --- Path 2 (Right) ---
                if (ego_paths[2] && right_path_mask.any())
                {
                    auto res = ego_paths[2]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                        right_path_nearest_idxs, ego_vs, curr_k_2, curr_path_is_right & active_flags);
                    curv_2 = res.first;
                    speed_2 = res.second;
                }

                // ========================================================
                // Data multiplexing (Select) - replaces switch-case
                // ========================================================

                // 1. Future Curvatures (Max Curvature)
                // Logic: if Left select 0, else (if Middle select 1, else select 2)
                future_curvatures = FVectorT_qmdp::select(
                    curr_path_is_left.template as<FVectorT_qmdp>(), curv_0,
                    FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), curv_1, curv_2));

                // 2. Desired Speed
                ego_desired_vs = FVectorT_qmdp::select(
                    curr_path_is_left.template as<FVectorT_qmdp>(), speed_0,
                    FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), speed_1, speed_2));

                // 3. Ego Current Path Nearest Idxs
                ego_curr_path_nearest_idxs = IVectorT_qmdp::select(
                    curr_path_is_left, left_path_nearest_idxs,
                    IVectorT_qmdp::select(curr_path_is_middle, middle_path_nearest_idxs, right_path_nearest_idxs));

                // 4. Ego Target Path Nearest Idxs
                // Note: target_path_is_... masks are used here
                ego_target_path_nearest_idxs = IVectorT_qmdp::select(
                    target_path_is_left, left_path_nearest_idxs,
                    IVectorT_qmdp::select(target_path_is_middle, middle_path_nearest_idxs, right_path_nearest_idxs));

                // 5. Ego Current Path Heading
                ego_curr_path_heading = FVectorT_qmdp::select(
                    curr_path_is_left.template as<FVectorT_qmdp>(), theta_0,
                    FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), theta_1, theta_2));

                // 6. Ego Current Path L (Frenet L)
                // L = min_distance * PointOnLeft_Sign
                FVectorT_qmdp chosen_min_dist =
                    FVectorT_qmdp::select(curr_path_is_left.template as<FVectorT_qmdp>(), left_path_min_distance,
                                          FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(),
                                                                middle_path_min_distance, right_path_min_distance));

                FVectorT_qmdp chosen_side = FVectorT_qmdp::select(
                    curr_path_is_left.template as<FVectorT_qmdp>(), side_0,
                    FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), side_1, side_2));

                ego_curr_path_ls = chosen_min_dist * chosen_side;
            }

            for (uint32_t t = 0; t < time_size; ++t)
            {
                AlignedVectorInt &has_collision = ws.has_collision;

                {
                    consider_LC_mask = active_flags & (ego_curr_path_idxs != ego_target_path_idxs);

                    // Left lane change: Current > Target (e.g. 1 -> 0)
                    consider_left_LC_mask = active_flags & (ego_curr_path_idxs > ego_target_path_idxs);
                    // Right lane change: Current < Target (e.g. 1 -> 2)
                    consider_right_LC_mask = active_flags & (ego_curr_path_idxs < ego_target_path_idxs);

                    // Logic: take max(Action_Offset, 0.4).
                    // If Action is 1.0, keep 1.0 (further left, facilitating lane change).
                    // If Action is 0.0 or -1.0, force to 0.4 (pull to lane edge in preparation).
                    if (consider_left_LC_mask.any())
                    {
                        ego_curr_lateral_offsets = FVectorT_qmdp::select(
                            consider_left_LC_mask.template as<FVectorT_qmdp>(),
                            ego_curr_lateral_offsets.max(utils::LEFT_LATERAL_OFFSET), ego_curr_lateral_offsets);
                    }

                    if (consider_right_LC_mask.any())
                    {
                        ego_curr_lateral_offsets = FVectorT_qmdp::select(
                            consider_right_LC_mask.template as<FVectorT_qmdp>(),
                            ego_curr_lateral_offsets.min(utils::RIGHT_LATERAL_OFFSET), ego_curr_lateral_offsets);
                    }
                }

                {
                    /* =========================
                     * Find leading vehicles (relative distance, exo speed; {max, max} if not found) on the current path
                     * =========================*/
                    curr_path_thetas_sin = ego_curr_path_heading.sin();
                    curr_path_thetas_cos = ego_curr_path_heading.cos();
                    ego_proj_radius_ls =
                        (-ego_cos_thetas * curr_path_thetas_sin + ego_sin_thetas * curr_path_thetas_cos).abs() *
                            ego_bb_extent_y +
                        (ego_sin_thetas * curr_path_thetas_sin + ego_cos_thetas * curr_path_thetas_cos).abs() *
                            ego_bb_extent_x;
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<true>(
                        LF_lead_vehicles_close_to_ego_reference_path, curr_stepped_time_idx, ego_curr_path_nearest_idxs,
                        ego_curr_path_idxs, ego_proj_radius_ls, ego_bb_extent_y, exo_ss, exo_ls,
                        exo_ls_projected_radius, exo_vs, exo_bb_extent_xs, exo_bb_extent_ys, ego_curr_lateral_offsets,
                        scenario_num, active_flags, exo_inactive_flags);
                }

                {
                    /* =========================
                     * Check LC allowance (only considers vehicles with incomplete LC)
                     * =========================*/
                    // LC mask: only consider active scenarios that have not yet reached the target path (i.e., LC
                    // incomplete)
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<true>(
                        LC_lead_vehicles_close_to_ego_reference_path, curr_stepped_time_idx,
                        ego_target_path_nearest_idxs, ego_target_path_idxs, ego_bb_extent_x, ego_bb_extent_y, exo_ss,
                        exo_ls, exo_ls_projected_radius, exo_vs, exo_bb_extent_xs, exo_bb_extent_ys, 0.0f, scenario_num,
                        consider_LC_mask, exo_inactive_flags);
                }

                {
                    // find following vehicle close to ego reference path
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<false>(
                        LC_following_vehicles_close_to_ego_reference_path, curr_stepped_time_idx,
                        ego_target_path_nearest_idxs, ego_target_path_idxs, ego_bb_extent_x, ego_bb_extent_y, exo_ss,
                        exo_ls, exo_ls_projected_radius, exo_vs, exo_bb_extent_xs, exo_bb_extent_ys, 0.0f, scenario_num,
                        consider_LC_mask, exo_inactive_flags);
                }

                {
                    // Perform MOBIL check and verify whether the lead vehicle on the current path is close enough
                    // Maximum lateral velocity
                    v_lateral =
                        FVectorT_qmdp::select(
                            ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                            (ego_vs * utils::LATERAL_VEL_RATIO_HIGH_SPEED).min(utils::MAX_VEL_LATERAL_HIGH_SPEED),
                            (ego_vs * utils::LATERAL_VEL_RATIO_LOW_SPEED).max(utils::MAX_VEL_LATERAL_LOW_SPEED))
                            .max(utils::LATERAL_VEL_MIN_EPSILON);

                    LC_allowance_flags = LCAllowanceCheckBatch(
                        LF_lead_vehicles_close_to_ego_reference_path, LC_lead_vehicles_close_to_ego_reference_path,
                        LC_following_vehicles_close_to_ego_reference_path, ego_vs, ego_thetas, ego_desired_vs,
                        v_lateral, consider_LC_mask, consider_left_LC_mask, consider_right_LC_mask, ego_curr_path_ls,
                        ego_curr_path_heading, LF_lead_allowance_flags);
                }

                {
                    /* =========================
                     * Find leading vehicles (relative distance, exo speed; {max, max} if not found) of the future path
                     * For vehicles that need to perform LF and vehicles that cannot execute LC
                     * =========================*/
                    // *Note: when finding lead vehicles for LC, future path intersection may also need to be considered
                    // LF mask: only consider active scenarios where neither LK nor LC can proceed
                    FindLeadVehicleIntersectedBatch(
                        LF_lead_vehicles_future_path_intersected, path_size, curr_stepped_time_idx,
                        ego_curr_path_nearest_idxs, ego_curr_path_idxs, ego_vs, ego_bb_extent_x, ego_bb_extent_y,
                        exo_vs, exo_bb_extent_ys, exo_ss, exo_strtrees, ego_curr_lateral_offsets, scenario_num,
                        active_flags & ~LC_allowance_flags, exo_inactive_flags);
                }

                {
                    // 1. Prepare base mask
                    // Logic: if LC is allowed, or (LC is being considered and lead vehicle allows lane change), use LC
                    // lead vehicle data
                    auto use_LC_data_mask = LC_allowance_flags | (consider_LC_mask & LF_lead_allowance_flags);

                    // Logic: decide whether to use the reference point from the target path or from the current path
                    // Original logic: if LC_allowance_flags is true, use the target path; otherwise use the current
                    // path
                    const auto &use_target_path_ref_mask = LC_allowance_flags;

                    // 2. Compute data for the LK (Lane Keep / Follow) scenario
                    // Compare LF_lead and LF_intersect and take the minimum distance (Min)
                    // Corresponds to the original logic: if (LF_lead.first < LF_intersect.first) ...
                    auto distance_LF_closed = FVectorT_qmdp(LF_lead_vehicles_close_to_ego_reference_path.first.data());
                    auto v_LF_closed = FVectorT_qmdp(LF_lead_vehicles_close_to_ego_reference_path.second.data());
                    auto distance_LF_intersected = FVectorT_qmdp(LF_lead_vehicles_future_path_intersected.first.data());
                    auto v_LF_intersected = FVectorT_qmdp(LF_lead_vehicles_future_path_intersected.second.data());

                    FVectorT_qmdp dist_LK = distance_LF_closed.min(distance_LF_intersected);
                    FVectorT_qmdp v_LK = FVectorT_qmdp::select(distance_LF_closed < distance_LF_intersected,
                                                               v_LF_closed, v_LF_intersected);

                    // 3. Fuse LC and LK data to obtain the final lead-vehicle information
                    auto distance_LC = FVectorT_qmdp(LC_lead_vehicles_close_to_ego_reference_path.first.data());
                    auto v_LC = FVectorT_qmdp(LC_lead_vehicles_close_to_ego_reference_path.second.data());
                    FVectorT_qmdp raw_relative_distance =
                        FVectorT_qmdp::select(use_LC_data_mask.template as<FVectorT_qmdp>(), distance_LC, dist_LK);
                    FVectorT_qmdp raw_exo_vs =
                        FVectorT_qmdp::select(use_LC_data_mask.template as<FVectorT_qmdp>(), v_LC, v_LK);

                    // 4. Obtain the reference point (X, Y, Theta)
                    // We need to decide whether to use the target index or the current index based on
                    // use_target_path_ref_mask For this kind of "fetch data from a pointer array using a variable
                    // index", we must use a calculate-all-and-select strategy

                    // Define temporary variables
                    FVectorT_qmdp x_target(0.f), y_target(0.f), theta_target(0.f);
                    FVectorT_qmdp x_curr(0.f), y_curr(0.f), theta_curr(0.f);

                    // 4.1 Get the reference point on the target path (compute only when needed)
                    // Use active_flags for a bit of pruning, but rely mainly on select
                    if (use_target_path_ref_mask.any())
                    {
                        FVectorT_qmdp x0, x1, x2, y0, y1, y2, th0, th1, th2;

                        if (ego_paths[0] && target_path_is_left.any())
                        {
                            x0 = ego_paths[0]->GetXBatch(ego_target_path_nearest_idxs);
                            y0 = ego_paths[0]->GetYBatch(ego_target_path_nearest_idxs);
                            th0 = ego_paths[0]->GetThetaBatch(ego_target_path_nearest_idxs);
                        }

                        if (ego_paths[1] && target_path_is_middle.any())
                        {
                            x1 = ego_paths[1]->GetXBatch(ego_target_path_nearest_idxs);
                            y1 = ego_paths[1]->GetYBatch(ego_target_path_nearest_idxs);
                            th1 = ego_paths[1]->GetThetaBatch(ego_target_path_nearest_idxs);
                        }

                        if (ego_paths[2] && target_path_is_right.any())
                        {
                            x2 = ego_paths[2]->GetXBatch(ego_target_path_nearest_idxs);
                            y2 = ego_paths[2]->GetYBatch(ego_target_path_nearest_idxs);
                            th2 = ego_paths[2]->GetThetaBatch(ego_target_path_nearest_idxs);
                        }

                        x_target = FVectorT_qmdp::select(
                            target_path_is_left.template as<FVectorT_qmdp>(), x0,
                            FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(), x1, x2));

                        y_target = FVectorT_qmdp::select(
                            target_path_is_left.template as<FVectorT_qmdp>(), y0,
                            FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(), y1, y2));

                        theta_target = FVectorT_qmdp::select(
                            target_path_is_left.template as<FVectorT_qmdp>(), th0,
                            FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(), th1, th2));
                    }

                    // 4.2 Get the reference point on the current path (same logic as above, but indices are for current
                    // path) In most cases the current path is needed, so the mask can be active_flags
                    if (!use_target_path_ref_mask.all())
                    {
                        FVectorT_qmdp x0, x1, x2, y0, y1, y2, th0, th1, th2;
                        // Note: here we use the current nearest indices computed by NearestBatch
                        if (ego_paths[0] && curr_path_is_left.any())
                        {
                            x0 = ego_paths[0]->GetXBatch(ego_curr_path_nearest_idxs);
                            y0 = ego_paths[0]->GetYBatch(ego_curr_path_nearest_idxs);
                            th0 = ego_paths[0]->GetThetaBatch(ego_curr_path_nearest_idxs);
                        }

                        if (ego_paths[1] && curr_path_is_middle.any())
                        {
                            x1 = ego_paths[1]->GetXBatch(ego_curr_path_nearest_idxs);
                            y1 = ego_paths[1]->GetYBatch(ego_curr_path_nearest_idxs);
                            th1 = ego_paths[1]->GetThetaBatch(ego_curr_path_nearest_idxs);
                        }

                        if (ego_paths[2] && curr_path_is_right.any())
                        {
                            x2 = ego_paths[2]->GetXBatch(ego_curr_path_nearest_idxs);
                            y2 = ego_paths[2]->GetYBatch(ego_curr_path_nearest_idxs);
                            th2 = ego_paths[2]->GetThetaBatch(ego_curr_path_nearest_idxs);
                        }

                        x_curr = FVectorT_qmdp::select(
                            curr_path_is_left.template as<FVectorT_qmdp>(), x0,
                            FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), x1, x2));

                        y_curr = FVectorT_qmdp::select(
                            curr_path_is_left.template as<FVectorT_qmdp>(), y0,
                            FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), y1, y2));

                        theta_curr = FVectorT_qmdp::select(
                            curr_path_is_left.template as<FVectorT_qmdp>(), th0,
                            FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), th1, th2));
                    }

                    // 4.3 Final selection of reference point
                    ref_point_xs =
                        FVectorT_qmdp::select(use_target_path_ref_mask.template as<FVectorT_qmdp>(), x_target, x_curr);
                    ref_point_ys =
                        FVectorT_qmdp::select(use_target_path_ref_mask.template as<FVectorT_qmdp>(), y_target, y_curr);
                    ref_point_thetas = FVectorT_qmdp::select(use_target_path_ref_mask.template as<FVectorT_qmdp>(),
                                                             theta_target, theta_curr);

                    // 5. Handle traffic lights
                    // Logic: if (dist >= traffic_dist && traffic_dist >= threshold), then set dist = traffic_dist and v
                    // = 0 Get the red-light position on the current path
                    float red_light_s_0 = ego_paths[0] ? ego_paths[0]->red_light_point_s_ : 9999.f;
                    float red_light_s_1 = ego_paths[1] ? ego_paths[1]->red_light_point_s_ : 9999.f;
                    float red_light_s_2 = ego_paths[2] ? ego_paths[2]->red_light_point_s_ : 9999.f;

                    FVectorT_qmdp current_red_light_s =
                        FVectorT_qmdp::select((curr_path_is_left.template as<FVectorT_qmdp>()), red_light_s_0,
                                              FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(),
                                                                    red_light_s_1, red_light_s_2));

                    // Compute distance to the traffic light
                    FVectorT_qmdp traffic_light_dist =
                        current_red_light_s -
                        ego_curr_path_nearest_idxs.template convert<FVectorT_qmdp>() * utils::PATH_POINT_INTERVAL -
                        utils::EGO_BB_EXTENT_Y;

                    // Build mask for the traffic-light condition
                    auto tl_cond = (raw_relative_distance >= traffic_light_dist) &
                                   (traffic_light_dist >= utils::REF_LIGHT_DISTANCE_THRESHOLD) &
                                   (traffic_light_dist < 1000.0f);

                    // Apply traffic-light mask to distance and speed
                    relative_distance = FVectorT_qmdp::select(tl_cond, traffic_light_dist, raw_relative_distance);
                    leading_exo_vs = FVectorT_qmdp::select(tl_cond, 0.0f, raw_exo_vs);
                }

                {
                    /* =========================
                     * Calculate and update commands (acc, steering)
                     * =========================*/
                    // Compute acceleration
                    cal_accs =
                        IDMBatch(leading_exo_vs, ego_vs, relative_distance - utils::SAFE_DIST_DIFF, ego_desired_vs);

                    // Compute steering
                    FVectorT_qmdp refline_lateral_offsets = FVectorT_qmdp::select(
                        LC_allowance_flags.template as<FVectorT_qmdp>(), ego_target_offsets, ego_curr_lateral_offsets);
                    cal_steerings = GetSteeringByStanleyBatch(
                        ref_point_xs, ref_point_ys, ref_point_thetas, future_curvatures, refline_lateral_offsets,
                        ego_xs, ego_ys, ego_vs, ego_thetas, ego_cos_thetas, ego_sin_thetas);

                    // Update ego states
                    UpdateStateBatch(ego_xs, ego_ys, ego_vs, ego_as, ego_thetas, ego_cos_thetas, ego_sin_thetas,
                                     cal_accs, cal_steerings);
                }

                // Update curr_stepped_time_idx
                curr_stepped_time_idx = curr_stepped_time_idx + 1;

                // Define accumulators
                FVectorT_qmdp f_curv_0(0.f), f_curv_1(0.f), f_curv_2(0.f);
                FVectorT_qmdp des_v_0(utils::MAX_VEL), des_v_1(utils::MAX_VEL), des_v_2(utils::MAX_VEL);
                FVectorT_qmdp theta_0(0.f), theta_1(0.f), theta_2(0.f);
                FVectorT_qmdp side_0(0.f), side_1(0.f), side_2(0.f);
                FVectorT_qmdp curr_k_0(0.f), curr_k_1(0.f),
                    curr_k_2(0.f); // Used to store current curvature (if the batch interface returns it)

                {
                    /* =========================
                     * Update Nearest Path Index and ego_curr_path_idx
                     * (whether the agent is on the target path or on the current path)
                     * =========================*/

                    // 1. Remove static and define locally; for a constant like StepDirection we can use const
                    const IVectorT_qmdp step_directions = IVectorT_qmdp::fill(1);

                    // 2. Reset distances (important: avoid using stale data from the previous frame)
                    left_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);
                    middle_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);
                    right_path_min_distance = FVectorT_qmdp::fill(utils::MAX_VALUE);

                    // 3. Compute nearest points (with mask-based pruning)
                    if (ego_paths[0] && left_path_mask.any())
                    {
                        ego_paths[0]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(ego_xs, ego_ys, left_path_nearest_idxs,
                                                                                 left_path_min_distance,
                                                                                 step_directions, left_path_mask);
                        theta_0 = ego_paths[0]->GetThetaBatch(left_path_nearest_idxs);
                        side_0 = ego_paths[0]->PointOnLeft(ego_xs, ego_ys, left_path_nearest_idxs);
                    }
                    if (ego_paths[1] && middle_path_mask.any())
                    {
                        ego_paths[1]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(
                            ego_xs, ego_ys, middle_path_nearest_idxs, middle_path_min_distance, step_directions,
                            middle_path_mask);
                        theta_1 = ego_paths[1]->GetThetaBatch(middle_path_nearest_idxs);
                        side_1 = ego_paths[1]->PointOnLeft(ego_xs, ego_ys, middle_path_nearest_idxs);
                    }
                    if (ego_paths[2] && right_path_mask.any())
                    {
                        ego_paths[2]->NearestBatch<FVectorT_qmdp, IVectorT_qmdp>(
                            ego_xs, ego_ys, right_path_nearest_idxs, right_path_min_distance, step_directions,
                            right_path_mask);
                        theta_2 = ego_paths[2]->GetThetaBatch(right_path_nearest_idxs);
                        side_2 = ego_paths[2]->PointOnLeft(ego_xs, ego_ys, right_path_nearest_idxs);
                    }

                    // 4. Extract minimum distances for current path and target path
                    curr_path_min_distance =
                        FVectorT_qmdp::select(curr_path_is_left.template as<FVectorT_qmdp>(), left_path_min_distance,
                                              FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(),
                                                                    middle_path_min_distance, right_path_min_distance));

                    target_path_min_distance =
                        FVectorT_qmdp::select(target_path_is_left.template as<FVectorT_qmdp>(), left_path_min_distance,
                                              FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(),
                                                                    middle_path_min_distance, right_path_min_distance));

                    auto target_path_chosen_side = FVectorT_qmdp::select(
                        target_path_is_left.template as<FVectorT_qmdp>(), side_0,
                        FVectorT_qmdp::select(target_path_is_middle.template as<FVectorT_qmdp>(), side_1, side_2));

                    auto target_path_min_distance_offset =
                        (target_path_min_distance * target_path_chosen_side - ego_target_offsets).abs();

                    // 5. Decide whether to switch path
                    constexpr float distance_threshold = 0.3f;

                    should_switch_path = ((curr_path_min_distance - target_path_min_distance) > distance_threshold |
                                          target_path_min_distance < utils::ACTION_CHANGE_DISTANCE_THRESHOLD |
                                          target_path_min_distance_offset < utils::ACTION_CHANGE_DISTANCE_THRESHOLD)
                                             .template as<IVectorT_qmdp>() &
                                         (ego_curr_path_idxs != ego_target_path_idxs);

                    // 7. Update indices
                    ego_curr_path_idxs =
                        IVectorT_qmdp::select(should_switch_path, ego_target_path_idxs, ego_curr_path_idxs);
                    ego_curr_lateral_offsets = FVectorT_qmdp::select(
                        (ego_curr_path_idxs == ego_target_path_idxs).template as<FVectorT_qmdp>(), ego_target_offsets,
                        ego_curr_lateral_offsets); // For paths that are the same, directly use the target offset
                                                   // (ego_target_offsets)

                    curr_path_is_left = ego_curr_path_idxs == 0;
                    curr_path_is_middle = ego_curr_path_idxs == 1;
                    curr_path_is_right = ego_curr_path_idxs == 2;
                    left_path_mask = (target_path_is_left | curr_path_is_left) & active_flags;
                    middle_path_mask = (target_path_is_middle | curr_path_is_middle) & active_flags;
                    right_path_mask = (target_path_is_right | curr_path_is_right) & active_flags;
                }

                {
                    // 1. Prepare target nearest indices
                    // Directly use select without a switch
                    ego_target_path_nearest_idxs =
                        IVectorT_qmdp::select(target_path_is_left, left_path_nearest_idxs,
                                              IVectorT_qmdp::select(target_path_is_middle, middle_path_nearest_idxs,
                                                                    right_path_nearest_idxs));

                    // 2. Compute path properties (curvature, desired speed, heading, L, goal)
                    // --- Path 0 ---
                    if (ego_paths[0] && curr_path_is_left.any())
                    {
                        // Call the optimized batch interface
                        auto res = ego_paths[0]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                            left_path_nearest_idxs, ego_vs, curr_k_0, curr_path_is_left & active_flags);
                        f_curv_0 = res.first;
                        des_v_0 = res.second;
                    }
                    // --- Path 1 ---
                    if (ego_paths[1] && curr_path_is_middle.any())
                    {
                        auto res = ego_paths[1]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                            middle_path_nearest_idxs, ego_vs, curr_k_1, curr_path_is_middle & active_flags);
                        f_curv_1 = res.first;
                        des_v_1 = res.second;
                    }
                    // --- Path 2 ---
                    if (ego_paths[2] && (ego_curr_path_idxs == 2).any())
                    {
                        auto res = ego_paths[2]->GetMaxCurvatureAndMinDesiredSpeedBatch<FVectorT_qmdp, IVectorT_qmdp>(
                            right_path_nearest_idxs, ego_vs, curr_k_2, curr_path_is_right & active_flags);
                        f_curv_2 = res.first;
                        des_v_2 = res.second;
                    }

                    // 4. Fuse results (multiplexing)
                    future_curvatures = FVectorT_qmdp::select(
                        curr_path_is_left.template as<FVectorT_qmdp>(), f_curv_0,
                        FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), f_curv_1, f_curv_2));

                    ego_desired_vs = FVectorT_qmdp::select(
                        curr_path_is_left.template as<FVectorT_qmdp>(), des_v_0,
                        FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), des_v_1, des_v_2));

                    ego_curr_path_heading = FVectorT_qmdp::select(
                        curr_path_is_left.template as<FVectorT_qmdp>(), theta_0,
                        FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), theta_1, theta_2));

                    ego_curr_path_nearest_idxs = IVectorT_qmdp::select(
                        curr_path_is_left, left_path_nearest_idxs,
                        IVectorT_qmdp::select(curr_path_is_middle, middle_path_nearest_idxs, right_path_nearest_idxs));

                    // 5. Compute L (Frenet lateral coordinate)
                    // L = MinDist * Sign(PointOnLeft)
                    FVectorT_qmdp chosen_min_dist =
                        FVectorT_qmdp::select(curr_path_is_left.template as<FVectorT_qmdp>(), left_path_min_distance,
                                              FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(),
                                                                    middle_path_min_distance, right_path_min_distance));

                    points_on_path_left = FVectorT_qmdp::select(
                        curr_path_is_left.template as<FVectorT_qmdp>(), side_0,
                        FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), side_1, side_2));

                    ego_curr_path_ls = chosen_min_dist * points_on_path_left;

                    // 6. Compute penalty related data (goal difference and missing goal)
                    // Since the penalty is a member variable of Path rather than a vector, we can load it directly
                    // But because we do not use switch, we need to retrieve penalties for all three paths and then
                    // select
                    float pen_0, pen_1, pen_2;
                    float goal_s_0, goal_s_1, goal_s_2;
                    if (ego_paths[0])
                    {
                        pen_0 = ego_paths[0]->miss_goal_penalty_;
                        goal_s_0 = ego_paths[0]->goal_frenet_s_;
                    }
                    if (ego_paths[1])
                    {
                        pen_1 = ego_paths[1]->miss_goal_penalty_;
                        goal_s_1 = ego_paths[1]->goal_frenet_s_;
                    }
                    if (ego_paths[2])
                    {
                        pen_2 = ego_paths[2]->miss_goal_penalty_;
                        goal_s_2 = ego_paths[2]->goal_frenet_s_;
                    }

                    miss_goal_pen = FVectorT_qmdp::select(
                        curr_path_is_left.template as<FVectorT_qmdp>(), pen_0,
                        FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(), pen_1, pen_2));

                    goal_diff =
                        FVectorT_qmdp::select(curr_path_is_left.template as<FVectorT_qmdp>(), goal_s_0,
                                              FVectorT_qmdp::select(curr_path_is_middle.template as<FVectorT_qmdp>(),
                                                                    goal_s_1, goal_s_2)) -
                        ego_curr_path_nearest_idxs.template convert<FVectorT_qmdp>() * utils::PATH_POINT_INTERVAL;
                }

                /* =========================
                 * Collision check
                 * =========================*/
                std::fill(has_collision.begin(), has_collision.end(), 0);
                if (exo_size_ > 0)
                {
                    {
                        ego_curr_path_ss =
                            ego_curr_path_nearest_idxs.template convert<FVectorT_qmdp>() * utils::PATH_POINT_INTERVAL;
                        ego_curr_path_ls = curr_path_min_distance * points_on_path_left;

                        min_s_values = ego_curr_path_ss - ego_bb_extent_y - utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                        min_l_values = ego_curr_path_ls - ego_bb_extent_x - utils::EGO_STRTREE_SAFETY_MARGIN_X;
                        max_s_values = ego_curr_path_ss + ego_bb_extent_y + utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                        max_l_values = ego_curr_path_ls + ego_bb_extent_x + utils::EGO_STRTREE_SAFETY_MARGIN_X;

                        ego_rear_xs = ego_xs - utils::REAR_TO_CENTER * ego_cos_thetas;
                        ego_rear_ys = ego_ys - utils::REAR_TO_CENTER * ego_sin_thetas;
                    }

                    {
                        for (size_t i = 0; i < scenario_num; ++i)
                        {
                            const auto idx = utils::div_mod<size_t>(i, vector_qmdp_width);
                            if (active_flags[idx])
                            {
                                const std::shared_ptr<STRtree> &exo_strtree =
                                    exo_strtrees[ego_curr_path_idxs[idx]]
                                                [i * total_time_size_ + curr_stepped_time_idx[idx]];

                                // Compute collision result
                                has_collision[i] = HasCollisionBatch(
                                    ego_rear_xs[idx], ego_rear_ys[idx], ego_xs[idx], ego_ys[idx], ego_thetas[idx],
                                    ego_cos_thetas[idx], ego_sin_thetas[idx], exo_xs, exo_ys, exo_cos_thetas,
                                    exo_sin_thetas, exo_bb_extent_xs, exo_bb_extent_ys, exo_strtree, i, ego_bb_extent_x,
                                    ego_bb_extent_y, min_s_values[idx], min_l_values[idx], max_s_values[idx],
                                    max_l_values[idx], curr_stepped_time_idx[idx], exo_inactive_flags[i]);
                            }
                        }
                    }
                }

                /* =========================
                 * Update penalties
                 * =========================*/
                {
                    active_flags_float = (active_flags.template convert<FVectorT_qmdp>()).abs();

                    // Collision penalty
                    collision_flags = IVectorT_qmdp(has_collision.data());
                    collision_flags_float = collision_flags.template convert<FVectorT_qmdp>().abs();
                    collision_penalty = GetCollisionPenalty<FVectorT_qmdp>(ego_vs, collision_flags_float);
                    rewards = rewards + collision_penalty;

                    // Speed penalty
                    movement_penalty = GetMovementPenalty<FVectorT_qmdp>(ego_vs, active_flags_float);
                    rewards = rewards + movement_penalty;

                    // Goal penalty
                    miss_goal_rewards =
                        GetGoalPenalty<FVectorT_qmdp>(low_speed_mode, miss_goal_pen, goal_diff, active_flags_float);
                    rewards = rewards + miss_goal_rewards;

                    // Action penalty
                    action_penalties = GetSteeringPenalty<FVectorT_qmdp>(ego_vs, cal_steerings, active_flags_float);
                    rewards = rewards + action_penalties;

                    // Rollout penalty after collision
                    if (collision_flags.any())
                    {
                        rollout_step_reward = collision_flags_float * (collision_penalty + action_penalties +
                                                                       miss_goal_rewards + GetMovementPenalty(0.0f));
                        rewards = rewards +
                                  rollout_step_reward *
                                      (step_end_time_idxs - curr_stepped_time_idx).template convert<FVectorT_qmdp>();
                        rollout_rewards =
                            rollout_rewards +
                            rollout_step_reward *
                                (utils::MAX_PRED_TIME_STEPS - 1 - step_end_time_idxs).template convert<FVectorT_qmdp>();
                    }
                    /* =========================
                     * Update active flags and scene indices
                     * If a collision occurs, set active_flags to 0 for that scenario
                     * We do not need to consider whether step length exceeds rollout length here,
                     * because active_flags has already been initialized according to rollout length when passed in
                     * =========================*/
                    active_flags = active_flags & (~collision_flags);
                }

                if (active_flags.none())
                {
                    break;
                }
            }

            ego_init_path_idxs = ego_curr_path_idxs;
            ego_init_lateral_offsets = ego_curr_lateral_offsets;
        }

        void ContextQMDP::generateProposalTrajectoryLFBatch(
            const int &scenario_num, const size_t &path_size, float ego_x, float ego_y, float ego_theta, float ego_v,
            float ego_a, const FVectorT_traj &lateral_offsets, const FVectorT_traj &find_exo_lateral_offsets,
            std::shared_ptr<Path> ego_path, int path_nearest_idx, const AlignedVectorFloat &exo_xs,
            const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_ss, const AlignedVectorFloat &exo_ls,
            const AlignedVectorFloat &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
            const AlignedVectorFloat &exo_bb_extent_xs, const AlignedVectorFloat &exo_bb_extent_ys,
            const std::vector<std::shared_ptr<STRtree>> &exo_strtrees, const FVectorT_12 &exo_active_flags,
            std::vector<FVectorT_traj> &ego_xs_traj, std::vector<FVectorT_traj> &ego_ys_traj,
            std::vector<FVectorT_traj> &ego_thetas_traj, std::vector<FVectorT_traj> &ego_vs_traj,
            std::vector<FVectorT_traj> &ego_as_traj, std::vector<FVectorT_traj> &ego_idm_a_traj,
            std::vector<FVectorT_traj> &ego_path_curvatures_proposal,
            std::vector<FVectorT_traj> &ego_xs_published_proposal,
            std::vector<FVectorT_traj> &ego_ys_published_proposal,
            std::vector<FVectorT_traj> &ego_thetas_published_proposal,
            std::vector<FVectorT_traj> &ego_vs_published_proposal,
            std::vector<FVectorT_traj> &ego_as_published_proposal)
        {
            constexpr float ego_bb_extent_x = utils::EGO_BB_EXTENT_X;
            constexpr float ego_bb_extent_y = utils::EGO_BB_EXTENT_Y;

            int proposal_idx = 0;
            int published_proposal_idx = 0;
            int min_path_curvature_idx = 0;

            float interpolated_x, interpolated_y, interpolated_theta, interpolated_path_idx, initial_lateral_distance,
                interpolated_theta_sin, interpolated_theta_cos;

            static thread_local utils::GenerateProposalLFWorkspace ws(scenario_num);
            ws.reset(scenario_num);

            IVectorT_traj &path_nearest_idxs = ws.path_nearest_idxs;
            FVectorT_traj &actual_lateral_distances_abs = ws.actual_lateral_distances_abs;
            FVectorT_traj &ego_vs = ws.ego_vs;
            FVectorT_traj &ego_as = ws.ego_as;
            FVectorT_traj &interpolated_path_idxs = ws.interpolated_path_idxs;

            {
                // Compute the nearest point of the ego vehicle on the current path
                ego_path->InterpolatedNearest(ego_x, ego_y, path_nearest_idx, interpolated_x, interpolated_y,
                                              interpolated_theta, interpolated_path_idx);

                // Compute lateral distance in the Frenet coordinate frame
                initial_lateral_distance = ego_path->PointOnLeft(ego_x, ego_y, path_nearest_idx) *
                                           utils::Distance(ego_x, ego_y, interpolated_x, interpolated_y);

                // Compute initial state
                interpolated_theta_sin = std::sin(interpolated_theta);
                interpolated_theta_cos = std::cos(interpolated_theta);
                ego_xs_traj[proposal_idx] = -lateral_offsets * interpolated_theta_sin + interpolated_x;
                ego_ys_traj[proposal_idx] = lateral_offsets * interpolated_theta_cos + interpolated_y;
                ego_xs_published_proposal[published_proposal_idx] = ego_xs_traj[proposal_idx];
                ego_ys_published_proposal[published_proposal_idx] = ego_ys_traj[proposal_idx];

                // Compute actual lateral distances for each proposal
                actual_lateral_distances_abs = (-lateral_offsets + initial_lateral_distance).abs();

                ego_thetas_traj[proposal_idx] = interpolated_theta;
                ego_vs_traj[proposal_idx] = ego_v;
                ego_as_traj[proposal_idx] = ego_a;
                ego_idm_a_traj[proposal_idx] = ego_a;

                ego_thetas_published_proposal[published_proposal_idx] = interpolated_theta;
                ego_vs_published_proposal[published_proposal_idx] = ego_v;
                ego_as_published_proposal[published_proposal_idx] = ego_a;

                ego_vs = FVectorT_traj::fill(ego_v);
                ego_as = FVectorT_traj::fill(ego_a);
                interpolated_path_idxs = FVectorT_traj::fill(interpolated_path_idx);
                path_nearest_idxs = IVectorT_traj::fill(path_nearest_idx);

                ++proposal_idx;
                ++published_proposal_idx;
            }

            FVectorT_traj &path_curr_curvatures = ws.path_curr_curvatures;
            FVectorT_traj &ego_desired_vs = ws.ego_desired_vs;
            FVectorT_traj &relative_distance = ws.relative_distance;
            FVectorT_traj &leading_exo_vs = ws.leading_exo_vs;
            FVectorT_traj &v_lateral = ws.v_lateral;
            FVectorT_traj &compensated_distance = ws.compensated_distance;
            FVectorT_traj &cal_accs = ws.cal_accs;
            FVectorT_traj &curvature_scale_factor = ws.curvature_scale_factor;
            FVectorT_traj &ego_avg_vs_ref = ws.ego_avg_vs_ref;
            FVectorT_traj &interpolated_xs = ws.interpolated_xs;
            FVectorT_traj &interpolated_ys = ws.interpolated_ys;
            FVectorT_traj &interpolated_thetas = ws.interpolated_thetas;

            std::pair<AlignedVectorFloat, AlignedVectorFloat> &lead_vehicles_close_to_ego_reference_path =
                ws.lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &lead_vehicles_future_path_intersected =
                ws.lead_vehicles_future_path_intersected;

            // For 20 time steps, the last time step does not require computation
            size_t t = 0;
            for (; t < utils::PROPOSAL_TRAJECTORY_SIZE - 1; ++t, ++proposal_idx, ++published_proposal_idx)
            {
                {
                    /* =========================
                     * Find desired speed and max curvature
                     * =========================*/
                    auto res = ego_path->GetMaxCurvatureAndMinDesiredSpeedBatch(
                        path_nearest_idxs, ego_vs, path_curr_curvatures, IVectorT_traj::fill(0xFFFFFFFF));

                    ego_path_curvatures_proposal[t] = std::move(res.first);
                    ego_desired_vs = std::move(res.second);
                }

                {
                    /* =========================
                     * Find leading vehicles close to ref_path (relative distance, exo speed; {max, max} if not found)
                     * on the current path
                     * =========================*/
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<true>(
                        lead_vehicles_close_to_ego_reference_path, t, path_nearest_idxs, ego_bb_extent_x,
                        ego_bb_extent_y, exo_ss, exo_ls, exo_ls_projected_radius, exo_vs, exo_bb_extent_xs,
                        exo_bb_extent_ys, find_exo_lateral_offsets, scenario_num, exo_active_flags);
                }

                {
                    /* =========================
                     * Find leading vehicles, intersected (relative distance, exo speed; {max, max} if not found) of the
                     * future path
                     * =========================*/
                    // Consider the LF mask so that only active scenarios are used, along with scenarios where LK is
                    // used and LC is not feasible
                    FindLeadVehicleIntersectedBatch(lead_vehicles_future_path_intersected, path_size, t,
                                                    path_nearest_idxs, ego_vs, ego_bb_extent_x, ego_bb_extent_y, exo_vs,
                                                    exo_bb_extent_ys, exo_ss, exo_strtrees, find_exo_lateral_offsets,
                                                    scenario_num);
                }
                {
                    // [Optimization] Fully vectorized logic for merging results and traffic light handling
                    // 1. Load scalar results into SIMD vectors
                    FVectorT_traj dist_close(lead_vehicles_close_to_ego_reference_path.first.data());
                    FVectorT_traj v_close(lead_vehicles_close_to_ego_reference_path.second.data());

                    FVectorT_traj dist_intersect(lead_vehicles_future_path_intersected.first.data());
                    FVectorT_traj v_intersect(lead_vehicles_future_path_intersected.second.data());

                    // 2. Merge Logic: Take the smaller distance
                    // if (dist_intersect < dist_close) use intersect
                    auto use_intersect_mask = dist_intersect < dist_close;
                    relative_distance = FVectorT_traj::select(use_intersect_mask, dist_intersect, dist_close);
                    leading_exo_vs = FVectorT_traj::select(use_intersect_mask, v_intersect, v_close);

                    // 3. Traffic Light Logic (Vectorized)
                    FVectorT_traj traffic_light_distance = ego_path->red_light_point_s_ -
                                                           interpolated_path_idxs * utils::PATH_POINT_INTERVAL -
                                                           utils::EGO_BB_EXTENT_Y;

                    FVectorT_traj rel_v = (ego_vs - leading_exo_vs).max(0.0001f);
                    FVectorT_traj ego_v_safe = ego_vs + 0.0001f;

                    // Condition 1: distance check
                    auto cond_dist = relative_distance >= traffic_light_distance;
                    // Condition 2: time-to-collision/arrival check
                    auto cond_ttc = (relative_distance / rel_v) > (traffic_light_distance / ego_v_safe);
                    // Condition 3: validity check
                    auto cond_valid = (traffic_light_distance >= utils::REF_LIGHT_DISTANCE_THRESHOLD) &
                                      (traffic_light_distance < 1000.0f);

                    // Combined mask: (cond1 || cond2) && cond3
                    auto traffic_light_mask = (cond_dist | cond_ttc) & cond_valid;

                    // Apply traffic light override
                    relative_distance =
                        FVectorT_traj::select(traffic_light_mask, traffic_light_distance, relative_distance);
                    leading_exo_vs = FVectorT_traj::select(traffic_light_mask, 0.0f, leading_exo_vs);
                }

                {
                    // Maximum lateral speed
                    v_lateral =
                        FVectorT_traj::select(
                            ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                            (ego_vs * utils::LATERAL_VEL_RATIO_HIGH_SPEED).min(utils::MAX_VEL_LATERAL_HIGH_SPEED),
                            (ego_vs * utils::LATERAL_VEL_RATIO_LOW_SPEED).max(utils::MAX_VEL_LATERAL_LOW_SPEED))
                            .max(utils::LATERAL_VEL_MIN_EPSILON);
                    compensated_distance =
                        calculateLateralMotionParams(ego_vs, actual_lateral_distances_abs, v_lateral, false, true);
                    relative_distance = relative_distance + compensated_distance;
                }

                {
                    /* =========================
                     * Calculate and update commands (acc, steering)
                     * =========================*/
                    // Compute acceleration
                    cal_accs = CurveAndLaneChangeAwareIDMBatch(leading_exo_vs, ego_vs,
                                                               relative_distance - utils::SAFE_DIST_DIFF,
                                                               ego_desired_vs, IVectorT_traj::fill(0));

                    // Compute curvature scaling factor: v_actual = v_ref * (1 + κ * l)
                    // Consider the influence of lateral offset on arc length so that proposals with different offsets
                    // move forward at correct speeds
                    curvature_scale_factor = (1.0f + path_curr_curvatures * lateral_offsets).max(0.1f);

                    // *at the first time interval
                    // Update ego speed (based on reference-line speed)
                    UpdateSpeedBatch(ego_vs, ego_avg_vs_ref, ego_as, cal_accs, utils::PUBLISHED_PROPOSAL_TIME_STEP);

                    // Update progress along the path (using reference-line speed)
                    interpolated_path_idxs = (ego_avg_vs_ref * curvature_scale_factor *
                                                  utils::PUBLISHED_PROPOSAL_TIME_STEP / utils::PATH_POINT_INTERVAL +
                                              interpolated_path_idxs)
                                                 .min(path_size - 1.0f);
                    path_nearest_idxs = interpolated_path_idxs.template convert<IVectorT_traj>();

                    // update ego states
                    ego_path->InterpolatedNearestBatch(interpolated_path_idxs, path_nearest_idxs, interpolated_xs,
                                                       interpolated_ys, interpolated_thetas);

                    // Store the published proposal (store the actual speed)
                    ego_xs_published_proposal[published_proposal_idx] =
                        -lateral_offsets * interpolated_thetas.sin() + interpolated_xs;
                    ego_ys_published_proposal[published_proposal_idx] =
                        lateral_offsets * interpolated_thetas.cos() + interpolated_ys;
                    ego_thetas_published_proposal[published_proposal_idx] = interpolated_thetas;
                    ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                    ego_as_published_proposal[published_proposal_idx] = ego_as;
                    ++published_proposal_idx;

                    // *at the second time interval
                    UpdateSpeedBatch(ego_vs, ego_avg_vs_ref, ego_as, cal_accs, utils::PUBLISHED_PROPOSAL_TIME_STEP);

                    // Update progress along the path (using reference-line speed)
                    interpolated_path_idxs = (ego_avg_vs_ref * curvature_scale_factor *
                                                  utils::PUBLISHED_PROPOSAL_TIME_STEP / utils::PATH_POINT_INTERVAL +
                                              interpolated_path_idxs)
                                                 .min(path_size - 1.0f);
                    path_nearest_idxs = interpolated_path_idxs.template convert<IVectorT_traj>();

                    // update ego states
                    ego_path->InterpolatedNearestBatch(interpolated_path_idxs, path_nearest_idxs, interpolated_xs,
                                                       interpolated_ys, interpolated_thetas);

                    // Update ego actual lateral offset (assuming uniform linear lateral motion)
                    actual_lateral_distances_abs =
                        (actual_lateral_distances_abs - v_lateral * utils::TIME_STEP).max(0.0f);
                }

                {
                    /* =========================
                     * Store proposal
                     * =========================*/
                    ego_xs_traj[proposal_idx] = -lateral_offsets * interpolated_thetas.sin() + interpolated_xs;
                    ego_ys_traj[proposal_idx] = lateral_offsets * interpolated_thetas.cos() + interpolated_ys;
                    ego_thetas_traj[proposal_idx] = interpolated_thetas;
                    ego_vs_traj[proposal_idx] = ego_vs;
                    ego_as_traj[proposal_idx] = ego_as;
                    ego_idm_a_traj[proposal_idx] = cal_accs;

                    ego_xs_published_proposal[published_proposal_idx] = ego_xs_traj[proposal_idx];
                    ego_ys_published_proposal[published_proposal_idx] = ego_ys_traj[proposal_idx];
                    ego_thetas_published_proposal[published_proposal_idx] = ego_thetas_traj[proposal_idx];
                    ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                    ego_as_published_proposal[published_proposal_idx] = ego_as;
                }
            }

            ego_path_curvatures_proposal[t] = ego_path_curvatures_proposal[t - 1];
        }

        void ContextQMDP::generateProposalTrajectoryLCBatch(
            int scenario_num, size_t path_size, FVectorT_traj ego_xs, FVectorT_traj ego_ys, FVectorT_traj ego_thetas,
            FVectorT_traj ego_vs, FVectorT_traj ego_as, bool turn_left, const FVectorT_traj &lateral_offsets,
            const FVectorT_traj &find_exo_lateral_offsets, bool immediate_assumed_at_target_path,
            std::shared_ptr<Path> ego_init_path,      // init path
            std::shared_ptr<Path> ego_target_path,    // target path
            IVectorT_traj         ego_curr_path_idxs, // 0: init path, 1: target path
            IVectorT_traj init_path_nearest_idxs, IVectorT_traj target_path_nearest_idxs,
            const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys,
            const std::vector<AlignedVectorFloat> &exo_ss, const std::vector<AlignedVectorFloat> &exo_ls,
            const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
            const AlignedVectorFloat &exo_bb_extent_xs, const AlignedVectorFloat &exo_bb_extent_ys,
            const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees, const FVectorT_12 &exo_active_flags,
            std::vector<FVectorT_traj> &ego_xs_traj, std::vector<FVectorT_traj> &ego_ys_traj,
            std::vector<FVectorT_traj> &ego_thetas_traj, std::vector<FVectorT_traj> &ego_vs_traj,
            std::vector<FVectorT_traj> &ego_as_traj, std::vector<FVectorT_traj> &ego_idm_a_traj,
            std::vector<FVectorT_traj> &ego_xs_published_proposal,
            std::vector<FVectorT_traj> &ego_ys_published_proposal,
            std::vector<FVectorT_traj> &ego_thetas_published_proposal,
            std::vector<FVectorT_traj> &ego_vs_published_proposal,
            std::vector<FVectorT_traj> &ego_as_published_proposal)
        {
            constexpr float ego_bb_extent_x = utils::EGO_BB_EXTENT_X;
            constexpr float ego_bb_extent_y = utils::EGO_BB_EXTENT_Y;

            float LC_lateral_shift;

            int proposal_idx = 0, published_proposal_idx = 0;

            static thread_local utils::GenerateProposalLCWorkspace ws(scenario_num);
            ws.reset(scenario_num);

            IVectorT_traj &curr_path_nearest_idxs = ws.curr_path_nearest_idxs;
            IVectorT_traj &on_target_path_mask = ws.on_target_path_mask;
            IVectorT_traj &assumed_exactly_on_target_path_mask = ws.assumed_exactly_on_target_path_mask;

            FVectorT_traj &ego_cos_thetas = ws.ego_cos_thetas;
            FVectorT_traj &ego_sin_thetas = ws.ego_sin_thetas;
            FVectorT_traj &actual_lateral_distances_abs = ws.actual_lateral_distances_abs;
            FVectorT_traj &interpolated_target_path_idxs = ws.interpolated_target_path_idxs;

            {
                ego_cos_thetas = ego_thetas.cos();
                ego_sin_thetas = ego_thetas.sin();

                curr_path_nearest_idxs = init_path_nearest_idxs;
                on_target_path_mask = (ego_curr_path_idxs == 1);           // the ego agent is on the target path
                assumed_exactly_on_target_path_mask = on_target_path_mask; // the ego agent is exactly on the target
                                                                           // path (to avoid sudden jumps in proposals)

                LC_lateral_shift = turn_left ? utils::LEFT_LATERAL_OFFSET : utils::RIGHT_LATERAL_OFFSET;

                // Initialize actual lateral distances (for compensating lateral motion)
                actual_lateral_distances_abs = FVectorT_traj::fill(0.0f);

                interpolated_target_path_idxs = target_path_nearest_idxs.template convert<FVectorT_traj>();

                // Store initial state
                ego_xs_traj[proposal_idx] = ego_xs;
                ego_ys_traj[proposal_idx] = ego_ys;
                ego_thetas_traj[proposal_idx] = ego_thetas;
                ego_vs_traj[proposal_idx] = ego_vs;
                ego_as_traj[proposal_idx] = ego_as;
                ego_idm_a_traj[proposal_idx] = ego_as;

                ego_xs_published_proposal[published_proposal_idx] = ego_xs;
                ego_ys_published_proposal[published_proposal_idx] = ego_ys;
                ego_thetas_published_proposal[published_proposal_idx] = ego_thetas;
                ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                ego_as_published_proposal[published_proposal_idx] = ego_as;

                ++proposal_idx;
                ++published_proposal_idx;
            }

            IVectorT_traj &step_directions = ws.step_directions;
            IVectorT_traj &tmp_target_path_nearest_idxs = ws.tmp_target_path_nearest_idxs;
            IVectorT_traj &should_switch_path = ws.should_switch_path;
            IVectorT_traj &newly_on_target = ws.newly_on_target;
            IVectorT_traj &consider_LC_mask = ws.consider_LC_mask;
            IVectorT_traj &LC_allowance_flags = ws.LC_allowance_flags;
            IVectorT_traj &LF_lead_allowance_flags = ws.LF_lead_allowance_flags;
            IVectorT_traj &in_lane_change_flag_simd = ws.in_lane_change_flag_simd;
            IVectorT_traj &tmp_nearest_path_idxs = ws.tmp_nearest_path_idxs;

            FVectorT_traj &init_path_min_distance = ws.init_path_min_distance;
            FVectorT_traj &target_path_min_distance = ws.target_path_min_distance;
            FVectorT_traj &curr_path_thetas = ws.curr_path_thetas;
            FVectorT_traj &init_path_ls = ws.init_path_ls;
            FVectorT_traj &target_path_ls = ws.target_path_ls;
            FVectorT_traj &real_target_path_distance = ws.real_target_path_distance;
            FVectorT_traj &distance_diff = ws.distance_diff;
            FVectorT_traj &path_curr_curvatures = ws.path_curr_curvatures;
            FVectorT_traj &path_curvatures = ws.path_curvatures;
            FVectorT_traj &ego_desired_vs = ws.ego_desired_vs;
            FVectorT_traj &lateral_offsets_for_LF = ws.lateral_offsets_for_LF;
            FVectorT_traj &v_lateral = ws.v_lateral;
            FVectorT_traj &distance_compensation = ws.distance_compensation;
            FVectorT_traj &relative_distance = ws.relative_distance;
            FVectorT_traj &leading_exo_vs = ws.leading_exo_vs;
            FVectorT_traj &ref_point_xs = ws.ref_point_xs;
            FVectorT_traj &ref_point_ys = ws.ref_point_ys;
            FVectorT_traj &ref_point_thetas = ws.ref_point_thetas;
            FVectorT_traj &cal_accs = ws.cal_accs;
            FVectorT_traj &target_path_xs = ws.target_path_xs;
            FVectorT_traj &target_path_ys = ws.target_path_ys;
            FVectorT_traj &target_path_thetas = ws.target_path_thetas;
            FVectorT_traj &curr_lateral_offsets = ws.curr_lateral_offsets;
            FVectorT_traj &curvature_scale_factor = ws.curvature_scale_factor;
            FVectorT_traj &ego_avg_vs_ref = ws.ego_avg_vs_ref;
            FVectorT_traj &interpolated_xs = ws.interpolated_xs;
            FVectorT_traj &interpolated_ys = ws.interpolated_ys;
            FVectorT_traj &interpolated_thetas = ws.interpolated_thetas;
            FVectorT_traj &ego_offset_xs = ws.ego_offset_xs;
            FVectorT_traj &ego_offset_ys = ws.ego_offset_ys;
            FVectorT_traj &cal_steerings = ws.cal_steerings;

            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path =
                ws.LF_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path =
                ws.LC_lead_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_following_vehicles_close_to_ego_reference_path =
                ws.LC_following_vehicles_close_to_ego_reference_path;
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_future_path_intersected =
                ws.LF_lead_vehicles_future_path_intersected;

            // For 20 steps, the last time step does not require computation
            for (size_t t = 0; t < utils::PROPOSAL_TRAJECTORY_SIZE - 1; ++t, ++proposal_idx, ++published_proposal_idx)
            {
                // update ego states using InterpolatedNearestBatc
                {
                    /* =========================
                     * Update path_nearest_idxs
                     * =========================*/
                    init_path_min_distance = FVectorT_traj::fill(utils::MAX_VALUE);
                    target_path_min_distance = FVectorT_traj::fill(utils::MAX_VALUE);
                    ego_init_path->NearestBatch(ego_xs, ego_ys, init_path_nearest_idxs, init_path_min_distance,
                                                step_directions, ~IVectorT_traj::fill(0));
                    tmp_target_path_nearest_idxs = target_path_nearest_idxs;
                    ego_target_path->NearestBatch(ego_xs, ego_ys, tmp_target_path_nearest_idxs,
                                                  target_path_min_distance, step_directions,
                                                  ~assumed_exactly_on_target_path_mask);

                    target_path_nearest_idxs = IVectorT_traj::select(
                        ~assumed_exactly_on_target_path_mask, tmp_target_path_nearest_idxs, target_path_nearest_idxs);
                }

                {
                    // Compute signed lateral distance (left is positive, right is negative)
                    curr_path_thetas = FVectorT_traj::gather(ego_init_path->thetas_.data(), init_path_nearest_idxs);
                    init_path_ls = init_path_min_distance *
                                   ego_init_path->PointOnLeft(ego_xs, ego_ys, curr_path_thetas, init_path_nearest_idxs);
                    target_path_ls = target_path_min_distance *
                                     ego_target_path->PointOnLeft(ego_xs, ego_ys, target_path_nearest_idxs);

                    // Compute actual distance to the target "offset lane"
                    real_target_path_distance = (target_path_ls - lateral_offsets).abs();

                    // ==========================================
                    // 2. Path switching logic
                    // ==========================================

                    // Optimization: directly compute "distance gain"
                    // Logic: init_dist - min(dist_to_lane, dist_to_center)
                    // Physical meaning: how much closer we get if we switch to the target lane (or its center) compared
                    // with the current one If the actual distance to the target path (considering both offset and
                    // nearest distance) is smaller than that to the current path, switch the current path to the target
                    // path
                    auto min_target_dist = real_target_path_distance.min(target_path_ls.abs());
                    auto distance_diff = init_path_min_distance - min_target_dist;

                    // Set a distance threshold to avoid frequent switching due to tiny differences
                    // Hysteresis threshold: distance gain > 0.3 m or already very close to the target (<
                    // ChangeThreshold)
                    constexpr float distance_threshold = 0.3f; // 0.3 m threshold that can be tuned
                    should_switch_path = (distance_diff > distance_threshold |
                                          real_target_path_distance < utils::ACTION_CHANGE_DISTANCE_THRESHOLD)
                                             .template as<IVectorT_traj>() |
                                         on_target_path_mask;

                    // Update indices
                    // Note: here we assume indices only take values 0 or 1. If there are more states, we must preserve
                    // the else branch of ego_curr_path_idxs
                    ego_curr_path_idxs = IVectorT_traj::select(should_switch_path, 1, ego_curr_path_idxs);
                    curr_path_nearest_idxs =
                        IVectorT_traj::select(should_switch_path, target_path_nearest_idxs, init_path_nearest_idxs);

                    // Update mask for the next frame
                    on_target_path_mask = (ego_curr_path_idxs == 1);

                    // ==========================================
                    // 3. "Exactly on target" snapshot logic
                    // ==========================================

                    // Check whether we enter the fine range (< 0.2 m)
                    // Check whether the ego is exactly on the target path. The threshold is small because once assumed
                    // exactly, the ego is treated as fully on the target lane; too large a threshold would cause large
                    // jumps in proposals
                    IVectorT_traj is_close_enough = (real_target_path_distance < 0.2f).template as<IVectorT_traj>();
                    IVectorT_traj final_exact_mask = is_close_enough | assumed_exactly_on_target_path_mask;

                    // Detect rising edge: the state just becomes "close" in this frame while it was not in the previous
                    // one
                    newly_on_target = final_exact_mask & ~assumed_exactly_on_target_path_mask;

                    // Optimization: fully branchless update
                    // Even if newly_on_target is entirely false, the cost of select instructions is still much lower
                    // than pipeline stalls from if(any) branches
                    actual_lateral_distances_abs =
                        FVectorT_traj::select(newly_on_target.template as<FVectorT_traj>(), real_target_path_distance,
                                              actual_lateral_distances_abs);
                    interpolated_target_path_idxs = FVectorT_traj::select(
                        newly_on_target.template as<FVectorT_traj>(),
                        target_path_nearest_idxs.template convert<FVectorT_traj>(), interpolated_target_path_idxs);

                    // Latch state for the next iteration
                    assumed_exactly_on_target_path_mask = final_exact_mask;

                    // Merge masks
                    on_target_path_mask = on_target_path_mask | assumed_exactly_on_target_path_mask;
                }

                {
                    /* =========================
                     * Find desired speed
                     * =========================*/
                    FVectorT_traj curv_init(0.0f), curv_target(0.0f);
                    auto [max_curv_init, min_speed_init] = ego_init_path->GetMaxCurvatureAndMinDesiredSpeedBatch(
                        init_path_nearest_idxs, ego_vs, curv_init, IVectorT_traj::fill(0xFFFFFFFF));
                    auto [max_curv_target, min_speed_target] = ego_target_path->GetMaxCurvatureAndMinDesiredSpeedBatch(
                        target_path_nearest_idxs, ego_vs, curv_target, IVectorT_traj::fill(0xFFFFFFFF));

                    auto use_init_mask = (ego_curr_path_idxs == 0).template as<FVectorT_traj>();
                    path_curr_curvatures = FVectorT_traj::select(use_init_mask, curv_init, curv_target);
                    path_curvatures = FVectorT_traj::select(use_init_mask, max_curv_init, max_curv_target);
                    ego_desired_vs = FVectorT_traj::select(use_init_mask, min_speed_init, min_speed_target);
                }

                {
                    /* =========================
                     * Find leading vehicles close to ref_path (relative distance, exo speed; {max, max} if not found)
                     * on the current path
                     * =========================*/
                    // If the current path is the target path, use lateral_offsets; otherwise use LC_lateral_shift (the
                    // pre-lane-change lateral shift)
                    lateral_offsets_for_LF = FVectorT_traj::select(on_target_path_mask.template as<FVectorT_traj>(),
                                                                   find_exo_lateral_offsets, LC_lateral_shift);

                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<true, true>(
                        LF_lead_vehicles_close_to_ego_reference_path, t, curr_path_nearest_idxs, ego_curr_path_idxs,
                        ego_bb_extent_x, ego_bb_extent_y, exo_ss, exo_ls, exo_ls_projected_radius, exo_vs,
                        exo_bb_extent_xs, exo_bb_extent_ys, lateral_offsets_for_LF, scenario_num, 0xFFFFFFFF,
                        exo_active_flags);
                }

                {
                    /* =========================
                     * Check LC allowance (only consider vehicles for which LC has not yet finished)
                     * =========================*/
                    consider_LC_mask = ~on_target_path_mask;

                    // find leading vehicle close to ego reference path
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<true, false>(
                        LC_lead_vehicles_close_to_ego_reference_path, t, target_path_nearest_idxs, 1, ego_bb_extent_x,
                        ego_bb_extent_y, exo_ss, exo_ls, exo_ls_projected_radius, exo_vs, exo_bb_extent_xs,
                        exo_bb_extent_ys, find_exo_lateral_offsets, scenario_num, consider_LC_mask, exo_active_flags);
                }

                {
                    // find following vehicle close to ego reference path
                    FindLeadOrFollowVehicleCloseToEgoReferencePathBatch<false, false>(
                        LC_following_vehicles_close_to_ego_reference_path, t, target_path_nearest_idxs, 1,
                        ego_bb_extent_x, ego_bb_extent_y, exo_ss, exo_ls, exo_ls_projected_radius, exo_vs,
                        exo_bb_extent_xs, exo_bb_extent_ys, find_exo_lateral_offsets, scenario_num, consider_LC_mask,
                        exo_active_flags);
                }

                {
                    // Perform MOBIL check and also verify whether the lead vehicle on the current path is sufficiently
                    // close - optimized version Maximum lateral speed
                    v_lateral =
                        FVectorT_traj::select(
                            ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                            (ego_vs * utils::LATERAL_VEL_RATIO_HIGH_SPEED).min(utils::MAX_VEL_LATERAL_HIGH_SPEED),
                            (ego_vs * utils::LATERAL_VEL_RATIO_LOW_SPEED).max(utils::MAX_VEL_LATERAL_LOW_SPEED))
                            .max(utils::LATERAL_VEL_MIN_EPSILON);

                    LC_allowance_flags = LCAllowanceCheckBatch(
                        LF_lead_vehicles_close_to_ego_reference_path, LC_lead_vehicles_close_to_ego_reference_path,
                        LC_following_vehicles_close_to_ego_reference_path, ego_vs, ego_thetas, ego_desired_vs,
                        v_lateral, consider_LC_mask, turn_left, init_path_ls, curr_path_thetas,
                        LF_lead_allowance_flags);
                }

                {
                    /* =========================
                     * Find intersected leading vehicles (relative distance, exo speed; {max, max} if not found) on the
                     * future path
                     * =========================*/
                    // Consider the LF mask to only use active scenarios, and combine LK with scenarios where LC is not
                    // feasible
                    FindLeadVehicleIntersectedBatch(LF_lead_vehicles_future_path_intersected, path_size, t,
                                                    curr_path_nearest_idxs, ego_curr_path_idxs, ego_vs, ego_bb_extent_x,
                                                    ego_bb_extent_y, exo_vs, exo_bb_extent_ys, exo_ss, exo_strtrees,
                                                    lateral_offsets_for_LF, scenario_num, ~LC_allowance_flags);
                }

                {
                    if (immediate_assumed_at_target_path)
                    {
                        // If LC is allowed, treat the vehicle as already being on the target lane
                        IVectorT_traj new_on_target_path_mask =
                            assumed_exactly_on_target_path_mask | LC_allowance_flags;

                        // Detect vehicles that enter the target path for the first time and compute actual lateral
                        // distance (rising edge: newly changed to true)
                        newly_on_target = new_on_target_path_mask & ~assumed_exactly_on_target_path_mask;

                        actual_lateral_distances_abs =
                            FVectorT_traj::select(newly_on_target.template as<FVectorT_traj>(),
                                                  real_target_path_distance, actual_lateral_distances_abs);

                        interpolated_target_path_idxs = FVectorT_traj::select(
                            newly_on_target.template as<FVectorT_traj>(),
                            target_path_nearest_idxs.template convert<FVectorT_traj>(), interpolated_target_path_idxs);

                        assumed_exactly_on_target_path_mask = new_on_target_path_mask;
                    }

                    // Compute lateral velocity and compensation (only for vehicles on the target path)
                    if (assumed_exactly_on_target_path_mask.any())
                    {
                        distance_compensation =
                            calculateLateralMotionParams(ego_vs, actual_lateral_distances_abs, v_lateral);

                        // Update lateral distances assuming uniform linear lateral motion
                        actual_lateral_distances_abs = FVectorT_traj::select(
                            assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                            (actual_lateral_distances_abs - v_lateral * utils::TIME_STEP).max(0.0f),
                            actual_lateral_distances_abs);
                    }
                }

                {
                    // [Optimization] Fully vectorized info collection

                    // 1. Calculate masks for different cases
                    IVectorT_traj mask_LC_success = LC_allowance_flags;
                    IVectorT_traj mask_trying_LC = consider_LC_mask & ~mask_LC_success;
                    IVectorT_traj mask_LF_lead_allowed = LF_lead_allowance_flags;

                    // 2. Load Data
                    FVectorT_traj lf_close_d(LF_lead_vehicles_close_to_ego_reference_path.first.data());
                    FVectorT_traj lf_close_v(LF_lead_vehicles_close_to_ego_reference_path.second.data());
                    FVectorT_traj lc_lead_d(LC_lead_vehicles_close_to_ego_reference_path.first.data());
                    FVectorT_traj lc_lead_v(LC_lead_vehicles_close_to_ego_reference_path.second.data());
                    FVectorT_traj lf_inter_d(LF_lead_vehicles_future_path_intersected.first.data());
                    FVectorT_traj lf_inter_v(LF_lead_vehicles_future_path_intersected.second.data());

                    // 3. Determine LF (LF-only case logic: min of close and intersect + comp)
                    auto          mask_lf_inter_closer = lf_inter_d < lf_close_d;
                    FVectorT_traj lf_min_d = FVectorT_traj::select(mask_lf_inter_closer, lf_inter_d, lf_close_d);
                    FVectorT_traj lf_min_v = FVectorT_traj::select(mask_lf_inter_closer, lf_inter_v, lf_close_v);
                    FVectorT_traj d_case_lf = lf_min_d + distance_compensation;
                    FVectorT_traj v_case_lf = lf_min_v;

                    // 4. Determine Logic for LC Success
                    FVectorT_traj d_case_lc_success = FVectorT_traj::select(
                        mask_LF_lead_allowed.template as<FVectorT_traj>(), lc_lead_d + distance_compensation,
                        lf_close_d); // If !allowed, fallback to LF close? (Code says so)
                    FVectorT_traj v_case_lc_success =
                        FVectorT_traj::select(mask_LF_lead_allowed.template as<FVectorT_traj>(), lc_lead_v, lf_close_v);
                    // Note: Original code: if (LF_lead_allowance_flags) LC... else LF... (Wait, if LC allowed, why
                    // check LF allowance? Ah, MOBIL check passed but maybe physical space on current lane blocked?
                    // Original code logic at 3390)

                    // 5. Determine Logic for Trying LC
                    // If LF allowed -> LC lead; else -> min(LF close, LF inter)
                    FVectorT_traj d_case_trying =
                        FVectorT_traj::select(mask_LF_lead_allowed.template as<FVectorT_traj>(), lc_lead_d,
                                              lf_min_d); // Original code 3412 vs 3421/3426
                    // Note: Original code at 3412 says: if (LF_lead_allowance_flags) -> LC_lead...
                    // But at 3419: compare LF_close vs LF_inter
                    FVectorT_traj v_case_trying =
                        FVectorT_traj::select(mask_LF_lead_allowed.template as<FVectorT_traj>(), lc_lead_v, lf_min_v);

                    // 6. Combine Results
                    // If LC Success -> Case 1
                    // If Trying LC -> Case 2
                    // Else (LF Only) -> Case 3
                    FVectorT_traj final_d = FVectorT_traj::select(
                        mask_LC_success.template as<FVectorT_traj>(), d_case_lc_success,
                        FVectorT_traj::select(mask_trying_LC.template as<FVectorT_traj>(), d_case_trying, d_case_lf));
                    FVectorT_traj final_v = FVectorT_traj::select(
                        mask_LC_success.template as<FVectorT_traj>(), v_case_lc_success,
                        FVectorT_traj::select(mask_trying_LC.template as<FVectorT_traj>(), v_case_trying, v_case_lf));

                    // 7. Reference Points & Offsets
                    // Case 1 (LC Success) -> Target Path
                    // Case 2 (Trying LC) -> Init Path
                    // Case 3 (LF Only) -> Target Path
                    // So use Init Path ONLY if (Trying LC)
                    IVectorT_traj mask_use_init = mask_trying_LC;

                    FVectorT_traj init_xs = FVectorT_traj::gather(ego_init_path->xs_.data(), init_path_nearest_idxs);
                    FVectorT_traj init_ys = FVectorT_traj::gather(ego_init_path->ys_.data(), init_path_nearest_idxs);
                    FVectorT_traj init_th =
                        FVectorT_traj::gather(ego_init_path->thetas_.data(), init_path_nearest_idxs);
                    FVectorT_traj target_xs =
                        FVectorT_traj::gather(ego_target_path->xs_.data(), target_path_nearest_idxs);
                    FVectorT_traj target_ys =
                        FVectorT_traj::gather(ego_target_path->ys_.data(), target_path_nearest_idxs);
                    FVectorT_traj target_th =
                        FVectorT_traj::gather(ego_target_path->thetas_.data(), target_path_nearest_idxs);

                    ref_point_xs =
                        FVectorT_traj::select(mask_use_init.template as<FVectorT_traj>(), init_xs, target_xs);
                    ref_point_ys =
                        FVectorT_traj::select(mask_use_init.template as<FVectorT_traj>(), init_ys, target_ys);
                    ref_point_thetas =
                        FVectorT_traj::select(mask_use_init.template as<FVectorT_traj>(), init_th, target_th);

                    curr_lateral_offsets = FVectorT_traj::select(mask_use_init.template as<FVectorT_traj>(),
                                                                 LC_lateral_shift, lateral_offsets);

                    in_lane_change_flag_simd = mask_LC_success | (mask_trying_LC & mask_LF_lead_allowed);

                    // 8. Traffic Light (Vectorized)
                    FVectorT_traj tl_s =
                        FVectorT_traj::select((ego_curr_path_idxs == 0).template as<FVectorT_traj>(),
                                              ego_init_path->red_light_point_s_, ego_target_path->red_light_point_s_);
                    FVectorT_traj current_s =
                        FVectorT_traj::select((ego_curr_path_idxs == 0).template as<FVectorT_traj>(),
                                              init_path_nearest_idxs.template convert<FVectorT_traj>(),
                                              target_path_nearest_idxs.template convert<FVectorT_traj>()) *
                        utils::PATH_POINT_INTERVAL;
                    FVectorT_traj tl_dist = tl_s - current_s - FVectorT_traj(utils::EGO_BB_EXTENT_Y);

                    auto mask_tl =
                        (final_d >= tl_dist) & (tl_dist >= utils::REF_LIGHT_DISTANCE_THRESHOLD) & (tl_dist < 1000.0f);
                    final_d = FVectorT_traj::select(mask_tl, tl_dist, final_d);
                    final_v = FVectorT_traj::select(mask_tl, FVectorT_traj(0.0f), final_v);

                    relative_distance = final_d;
                    leading_exo_vs = final_v;
                }

                {

                    /* =========================
                     * Calculate and update commands (acc, steering)
                     * =========================*/
                    // Compute acceleration
                    cal_accs = CurveAndLaneChangeAwareIDMBatch(leading_exo_vs, ego_vs,
                                                               relative_distance - utils::SAFE_DIST_DIFF,
                                                               ego_desired_vs, in_lane_change_flag_simd);

                    // Update ego states - for vehicles already on the target path or with LC allowed, assume the
                    // vehicle is on the target lane
                    if (assumed_exactly_on_target_path_mask.any())
                    {
                        if (t == 0)
                        {
                            target_path_xs =
                                FVectorT_traj::gather(ego_target_path->xs_.data(), target_path_nearest_idxs);
                            target_path_ys =
                                FVectorT_traj::gather(ego_target_path->ys_.data(), target_path_nearest_idxs);
                            target_path_thetas =
                                FVectorT_traj::gather(ego_target_path->thetas_.data(), target_path_nearest_idxs);
                            ego_xs_traj[0] = FVectorT_traj::select(
                                assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                -curr_lateral_offsets * target_path_thetas.sin() + target_path_xs, ego_xs);
                            ego_ys_traj[0] = FVectorT_traj::select(
                                assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                curr_lateral_offsets * target_path_thetas.cos() + target_path_ys, ego_ys);
                            ego_thetas_traj[0] =
                                FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                      target_path_thetas, ego_thetas);

                            ego_xs_published_proposal[0] = ego_xs_traj[0];
                            ego_ys_published_proposal[0] = ego_ys_traj[0];
                            ego_thetas_published_proposal[0] = ego_thetas_traj[0];
                        }

                        // Separate vehicles already on the target path from those not yet on it
                        bool any_not_on_target_path_mask = (~assumed_exactly_on_target_path_mask).any();

                        // Compute curvature scaling factor (only effective for vehicles on the target path)
                        // For vehicles on the target path, use their corresponding curvature and lateral offset
                        curvature_scale_factor =
                            (FVectorT_traj::fill(1.0f) + path_curr_curvatures * curr_lateral_offsets).max(0.1f);

                        // *At the first time interval*
                        // Update ego speed (based on reference-line speed)
                        UpdateSpeedBatch(ego_vs, ego_avg_vs_ref, ego_as, cal_accs, utils::PUBLISHED_PROPOSAL_TIME_STEP);

                        // First handle vehicles on the target path (since they are already on the target path, update
                        // states with LF logic) Update progress along the path (using reference-line speed)
                        interpolated_target_path_idxs =
                            (ego_avg_vs_ref * curvature_scale_factor * utils::PUBLISHED_PROPOSAL_TIME_STEP /
                                 utils::PATH_POINT_INTERVAL +
                             interpolated_target_path_idxs)
                                .min(path_size - 1.0f);
                        tmp_nearest_path_idxs = interpolated_target_path_idxs.template convert<IVectorT_traj>();

                        // update ego states using InterpolatedNearestBatch
                        ego_target_path->InterpolatedNearestBatch(interpolated_target_path_idxs, tmp_nearest_path_idxs,
                                                                  interpolated_xs, interpolated_ys,
                                                                  interpolated_thetas);

                        // Adjust position according to lateral offsets
                        ego_offset_xs = -curr_lateral_offsets * interpolated_thetas.sin() + interpolated_xs;
                        ego_offset_ys = curr_lateral_offsets * interpolated_thetas.cos() + interpolated_ys;

                        // Update states of vehicles that are not yet on the target path
                        if (any_not_on_target_path_mask)
                        {
                            // Compute steering
                            cal_steerings = GetSteeringByStanleyBatch(
                                ref_point_xs, ref_point_ys, ref_point_thetas, path_curvatures, curr_lateral_offsets,
                                ego_xs, ego_ys, ego_vs, ego_thetas, ego_cos_thetas, ego_sin_thetas);

                            // Only update vehicles that are not on the target path
                            UpdateHeadingBatch(ego_xs, ego_ys, ego_avg_vs_ref, ego_thetas, ego_cos_thetas,
                                               ego_sin_thetas, cal_steerings, utils::PUBLISHED_PROPOSAL_TIME_STEP);
                        }

                        ego_xs = FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                       ego_offset_xs, ego_xs);
                        ego_ys = FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                       ego_offset_ys, ego_ys);
                        ego_thetas =
                            FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                  interpolated_thetas, ego_thetas);

                        // Store published proposal (store actual speed and apply curvature correction for vehicles on
                        // the target path)
                        ego_xs_published_proposal[published_proposal_idx] = ego_xs;
                        ego_ys_published_proposal[published_proposal_idx] = ego_ys;
                        ego_thetas_published_proposal[published_proposal_idx] = ego_thetas;
                        ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                        ego_as_published_proposal[published_proposal_idx] = ego_as;
                        ++published_proposal_idx;

                        // *at the second time interval
                        UpdateSpeedBatch(ego_vs, ego_avg_vs_ref, ego_as, cal_accs, utils::PUBLISHED_PROPOSAL_TIME_STEP);

                        // First handle vehicles on the target path (update using LF logic)
                        // Update progress along the path (using reference-line speed)
                        interpolated_target_path_idxs =
                            (ego_avg_vs_ref * curvature_scale_factor * utils::PUBLISHED_PROPOSAL_TIME_STEP /
                                 utils::PATH_POINT_INTERVAL +
                             interpolated_target_path_idxs)
                                .min(path_size - 1.0f);
                        tmp_nearest_path_idxs = interpolated_target_path_idxs.template convert<IVectorT_traj>();
                        target_path_nearest_idxs = IVectorT_traj::select(
                            assumed_exactly_on_target_path_mask, tmp_nearest_path_idxs, target_path_nearest_idxs);

                        // update ego states using InterpolatedNearestBatch
                        ego_target_path->InterpolatedNearestBatch(interpolated_target_path_idxs, tmp_nearest_path_idxs,
                                                                  interpolated_xs, interpolated_ys,
                                                                  interpolated_thetas);

                        // Adjust position according to lateral offsets
                        ego_offset_xs = -curr_lateral_offsets * interpolated_thetas.sin() + interpolated_xs;
                        ego_offset_ys = curr_lateral_offsets * interpolated_thetas.cos() + interpolated_ys;

                        // Update states of vehicles that are not yet on the target path
                        if (any_not_on_target_path_mask)
                        {
                            // Only update vehicles that are not on the target path
                            UpdateHeadingBatch(ego_xs, ego_ys, ego_avg_vs_ref, ego_thetas, ego_cos_thetas,
                                               ego_sin_thetas, cal_steerings, utils::PUBLISHED_PROPOSAL_TIME_STEP);
                        }

                        // [Optimization] Vectorized state update for target path vehicles (second step)
                        ego_xs = FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                       ego_offset_xs, ego_xs);
                        ego_ys = FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                       ego_offset_ys, ego_ys);
                        ego_thetas =
                            FVectorT_traj::select(assumed_exactly_on_target_path_mask.template as<FVectorT_traj>(),
                                                  interpolated_thetas, ego_thetas);
                    }
                    else
                    {
                        // Compute steering
                        cal_steerings = GetSteeringByStanleyBatch(ref_point_xs, ref_point_ys, ref_point_thetas,
                                                                  path_curvatures, curr_lateral_offsets, ego_xs, ego_ys,
                                                                  ego_vs, ego_thetas, ego_cos_thetas, ego_sin_thetas);

                        // *at the first time interval
                        // When no vehicle is on the target path, update states using the conventional method
                        UpdateStateBatch(ego_xs, ego_ys, ego_vs, ego_as, ego_thetas, ego_cos_thetas, ego_sin_thetas,
                                         cal_accs, cal_steerings, utils::PUBLISHED_PROPOSAL_TIME_STEP);

                        ego_xs_published_proposal[published_proposal_idx] = ego_xs;
                        ego_ys_published_proposal[published_proposal_idx] = ego_ys;
                        ego_thetas_published_proposal[published_proposal_idx] = ego_thetas;
                        ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                        ego_as_published_proposal[published_proposal_idx] = ego_as;
                        ++published_proposal_idx;

                        // *at the second time interval
                        UpdateStateBatch(ego_xs, ego_ys, ego_vs, ego_as, ego_thetas, ego_cos_thetas, ego_sin_thetas,
                                         cal_accs, cal_steerings, utils::PUBLISHED_PROPOSAL_TIME_STEP);
                    }
                }

                {
                    /* =========================
                     * Store proposal
                     * =========================*/
                    ego_xs_traj[proposal_idx] = ego_xs;
                    ego_ys_traj[proposal_idx] = ego_ys;
                    ego_thetas_traj[proposal_idx] = ego_thetas;
                    ego_vs_traj[proposal_idx] = ego_vs;
                    ego_as_traj[proposal_idx] = ego_as;
                    ego_idm_a_traj[proposal_idx] = cal_accs;

                    ego_xs_published_proposal[published_proposal_idx] = ego_xs;
                    ego_ys_published_proposal[published_proposal_idx] = ego_ys;
                    ego_thetas_published_proposal[published_proposal_idx] = ego_thetas;
                    ego_vs_published_proposal[published_proposal_idx] = ego_vs;
                    ego_as_published_proposal[published_proposal_idx] = ego_as;
                }
            }
        }

        FVectorT_traj ContextQMDP::crossScenarioEvaluationBatch(
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
            const AlignedVectorFloat &exo_ss, const AlignedVectorFloat &exo_ls, const AlignedVectorFloat &exo_thetas,
            const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
            const AlignedVectorFloat &exo_original_bb_extent_xs, const AlignedVectorFloat &exo_original_bb_extent_ys,
            const AlignedVectorFloat &exo_expanded_bb_extent_xs, const AlignedVectorFloat &exo_expanded_bb_extent_ys,
            const std::vector<std::shared_ptr<STRtree>> &exo_strtrees,
            const std::shared_ptr<utils::OccupancyMap> &occupancy_map, IVectorT_traj &collided_timesteps,
            std::vector<int> &ttc_timesteps)
        {
            constexpr float ego_bb_extent_x = utils::EGO_BB_EXTENT_X;
            constexpr float ego_bb_extent_y = utils::EGO_BB_EXTENT_Y;

            int                                                         evaluation_len, drivable_area_evaluation_len;
            static thread_local utils::CrossScenarioEvaluationWorkspace ws(utils::PROPOSAL_BATCH_SIZE);
            ws.reset(utils::PROPOSAL_BATCH_SIZE);

            FVectorT_traj &path_min_dists = ws.path_min_dists;
            FVectorT_traj &path_left_flags = ws.path_left_flags;
            FVectorT_traj &ego_path_ss = ws.ego_path_ss;
            FVectorT_traj &ego_path_ls = ws.ego_path_ls;

            FVectorT_traj &precomputed_non_on_drivable_area_penalties = ws.precomputed_non_on_drivable_area_penalties;
            FVectorT_traj &ego_oncoming_culmulative_vs = ws.ego_oncoming_culmulative_vs;
            FVectorT_traj &value_array_sum = ws.value_array_sum;

            IVectorT_traj &path_nearest_idxs = ws.path_nearest_idxs;

            auto &corners_xs = ws.corners_xs;
            auto &corners_ys = ws.corners_ys;

            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_non_drivable_area_masks =
                ws.on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &left_corners_on_non_drivable_area_masks =
                ws.left_corners_on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &right_corners_on_non_drivable_area_masks =
                ws.right_corners_on_non_drivable_area_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_coming_traffic_masks =
                ws.on_coming_traffic_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_intersection_masks = ws.on_intersection_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_multiple_lane_masks = ws.on_multiple_lane_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_different_path_lanes_masks =
                ws.on_different_path_lanes_masks;
            std::array<IVectorT_traj, utils::CROSS_EVALUATION_LEN> &on_non_route_masks = ws.on_non_route_masks;

            std::array<FVectorT_traj, utils::PROPOSAL_TRAJECTORY_SIZE> &ego_cos_thetas_traj = ws.ego_cos_thetas_traj;
            std::array<FVectorT_traj, utils::PROPOSAL_TRAJECTORY_SIZE> &ego_sin_thetas_traj = ws.ego_sin_thetas_traj;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &precomputed_penalties = ws.precomputed_penalties;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_rear_axle_xs = ws.ego_rear_axle_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_rear_axle_ys = ws.ego_rear_axle_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_front_left_corner_xs =
                ws.ego_front_left_corner_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_front_left_corner_ys =
                ws.ego_front_left_corner_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_front_right_corner_xs =
                ws.ego_front_right_corner_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &ego_front_right_corner_ys =
                ws.ego_front_right_corner_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &nuplan_metric_penalties =
                ws.nuplan_metric_penalties;

            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &query_aabb_min_xs = ws.query_aabb_min_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &query_aabb_min_ys = ws.query_aabb_min_ys;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &query_aabb_max_xs = ws.query_aabb_max_xs;
            std::array<FVectorT_traj, utils::CROSS_EVALUATION_LEN> &query_aabb_max_ys = ws.query_aabb_max_ys;

            {
                if (ego_initial_v < 3.0f)
                {
                    evaluation_len = utils::CROSS_EVALUATION_LEN;
                    drivable_area_evaluation_len = utils::CROSS_EVALUATION_LEN;
                }
                else if (ego_initial_v < 5.0f)
                {
                    evaluation_len = utils::CROSS_EVALUATION_LEN;
                    drivable_area_evaluation_len = 10;
                }
                else
                {
                    evaluation_len = 16;
                    drivable_area_evaluation_len = 10;
                }

                // Precompute cosine and sine of the ego heading
                for (size_t i = 0; i < ego_thetas_traj.size(); ++i)
                {
                    ego_cos_thetas_traj[i] = ego_thetas_traj[i].cos();
                    ego_sin_thetas_traj[i] = ego_thetas_traj[i].sin();
                }

                path_min_dists = FVectorT_traj::fill(utils::MAX_VALUE);
                path_nearest_idxs = ego_path->NearestBatch(ego_xs_traj[0], ego_ys_traj[0],
                                                           IVectorT_traj::fill(0xFFFFFFFF), path_min_dists);
                path_left_flags = ego_path->PointOnLeft(ego_xs_traj[0], ego_ys_traj[0], path_nearest_idxs);
                ego_path_ss = path_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
                ego_path_ls = path_min_dists * path_left_flags;

                query_aabb_min_xs[0] = ego_path_ss - ego_bb_extent_y - utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                query_aabb_min_ys[0] = ego_path_ls - ego_bb_extent_x - utils::EGO_STRTREE_SAFETY_MARGIN_X;
                query_aabb_max_xs[0] = ego_path_ss + ego_bb_extent_y + utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                query_aabb_max_ys[0] = ego_path_ls + ego_bb_extent_x + utils::EGO_STRTREE_SAFETY_MARGIN_X;

                for (size_t t = 1; t < evaluation_len; ++t)
                {
                    path_min_dists = FVectorT_traj::fill(utils::MAX_VALUE);
                    ego_path->NearestBatch(ego_xs_traj[t], ego_ys_traj[t], path_nearest_idxs, path_min_dists,
                                           IVectorT_traj::fill(1), IVectorT_traj::fill(0xFFFFFFFF));
                    path_left_flags = ego_path->PointOnLeft(ego_xs_traj[t], ego_ys_traj[t], path_nearest_idxs);
                    ego_path_ss = path_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
                    ego_path_ls = path_min_dists * path_left_flags;

                    query_aabb_min_xs[t] = ego_path_ss - ego_bb_extent_y - utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                    query_aabb_min_ys[t] = ego_path_ls - ego_bb_extent_x - utils::EGO_STRTREE_SAFETY_MARGIN_X;
                    query_aabb_max_xs[t] = ego_path_ss + ego_bb_extent_y + utils::EGO_STRTREE_SAFETY_MARGIN_Y;
                    query_aabb_max_ys[t] = ego_path_ls + ego_bb_extent_x + utils::EGO_STRTREE_SAFETY_MARGIN_X;
                }
            }

            // Precompute lateral offset penalty
            FVectorT_traj lateral_offset_penalty =
                GetLateralOffsetPenaltyBatch(ego_vs_traj[0], lateral_offsets_proposal);

            for (size_t t = 0; t < evaluation_len; ++t)
            {
                precomputed_penalties[t] = 0.0f;
                nuplan_metric_penalties[t] = 0.0f;

                // *** Compute ego position masks ***
                // Compute ego corners
                {
                    computeEgoCornersBatch(ego_xs_traj[t], ego_ys_traj[t], ego_cos_thetas_traj[t],
                                           ego_sin_thetas_traj[t], ego_bb_extent_x, ego_bb_extent_y, corners_xs,
                                           corners_ys);

                    // Compute ego rear axle
                    ego_rear_axle_xs[t] = ego_xs_traj[t] - ego_cos_thetas_traj[t] * utils::REAR_TO_CENTER;
                    ego_rear_axle_ys[t] = ego_ys_traj[t] - ego_sin_thetas_traj[t] * utils::REAR_TO_CENTER;
                }

                {
                    // Compute ego position masks
                    occupancy_map->ContainsPointsInDrivableAreaBatch(
                        ego_xs_traj[t], ego_ys_traj[t], ego_rear_axle_xs[t], ego_rear_axle_ys[t], corners_xs,
                        corners_ys, on_non_drivable_area_masks[t], left_corners_on_non_drivable_area_masks[t],
                        right_corners_on_non_drivable_area_masks[t], on_coming_traffic_masks[t],
                        on_intersection_masks[t], on_multiple_lane_masks[t], on_different_path_lanes_masks[t],
                        on_non_route_masks[t]);
                }

                {
                    const FVectorT_traj &ego_path_curvature_proposal_t = (ego_path_curvatures_proposal.empty())
                                                                             ? FVectorT_traj::fill(0.0f)
                                                                             : ego_path_curvatures_proposal[t];

                    precomputed_non_on_drivable_area_penalties =
                        precomputed_non_on_drivable_area_penalties +
                        GetOnNonRoutePenalty(ego_initial_v, on_non_route_masks[t], ego_path_curvature_proposal_t) +
                        GetOnDifferentPathLanePenaltyBatch(ego_initial_v, on_different_path_lanes_masks[t],
                                                           on_non_route_masks[t], ego_path_curvature_proposal_t);

                    // Precompute speed penalty and lateral offset penalty
                    precomputed_penalties[t] = precomputed_penalties[t] +
                                               GetCrossEvaluationMovementPenaltyBatch(ego_vs_traj[t]) +
                                               lateral_offset_penalty;

                    // Precompute non-drivable area penalty
                    if (t <= drivable_area_evaluation_len)
                    {
                        nuplan_metric_penalties[t] =
                            nuplan_metric_penalties[t] +
                            GetNonDrivableAreaPenaltyBatch<FVectorT_traj, IVectorT_traj>(on_non_drivable_area_masks[t]);
                    }

                    // Precompute direction-compliance penalty
                    if (t <= 10)
                    {
                        auto direction_compliance_penality = GetDirectionCompliancePenaltyBatch(
                            ego_vs_traj[t], on_coming_traffic_masks[t], on_non_drivable_area_masks[t],
                            ego_oncoming_culmulative_vs);
                        nuplan_metric_penalties[t] = nuplan_metric_penalties[t] + direction_compliance_penality;
                    }

                    precomputed_penalties[t] = precomputed_penalties[t] + nuplan_metric_penalties[t];

                    ego_front_left_corner_xs[t] = corners_xs[0];
                    ego_front_left_corner_ys[t] = corners_ys[0];
                    ego_front_right_corner_xs[t] = corners_xs[1];
                    ego_front_right_corner_ys[t] = corners_ys[1];
                }
            }

            {
                // Precompute penalty for being on non-drivable areas (considering future steps, penalty decreases over
                // time)
                float time_decay_weight = 1.0f; // penalty decay factor per time step

                for (size_t t = 0; t < 10; ++t)
                {
                    time_decay_weight *= utils::TIME_DECAY_FACTOR;

                    if (on_non_drivable_area_masks[t].any())
                    {
                        // For any contact with non-drivable areas, apply an offset-based penalty function
                        precomputed_non_on_drivable_area_penalties =
                            precomputed_non_on_drivable_area_penalties +
                            (GetLateralOffsetOnNonDrivableAreaPenaltyBatch(
                                 lateral_offsets_proposal, left_corners_on_non_drivable_area_masks[t],
                                 right_corners_on_non_drivable_area_masks[t]) +
                             on_non_drivable_area_masks[t].template convert<FVectorT_traj>().abs() *
                                 utils::NON_ON_DRIVABLE_AREA_PENALTY_STAGE_1) *
                                time_decay_weight;
                    }
                }

                for (size_t t = 10; t < drivable_area_evaluation_len; ++t)
                {
                    // Compute time decay weight; the later the time step, the smaller the weight
                    time_decay_weight *= utils::TIME_DECAY_FACTOR;

                    // Compute ego position masks
                    IVectorT_traj on_non_drivable_area_mask, left_corners_on_non_drivable_area_mask,
                        right_corners_on_non_drivable_area_mask;

                    if (t < evaluation_len)
                    {
                        on_non_drivable_area_mask = on_non_drivable_area_masks[t];
                        left_corners_on_non_drivable_area_mask = left_corners_on_non_drivable_area_masks[t];
                        right_corners_on_non_drivable_area_mask = right_corners_on_non_drivable_area_masks[t];
                    }
                    else
                    {
                        computeEgoCornersBatch(ego_xs_traj[t], ego_ys_traj[t], ego_cos_thetas_traj[t],
                                               ego_sin_thetas_traj[t], ego_bb_extent_x, ego_bb_extent_y, corners_xs,
                                               corners_ys);

                        // Compute ego rear axle
                        FVectorT_traj ego_rear_axle_x = ego_xs_traj[t] - ego_cos_thetas_traj[t] * ego_bb_extent_x;
                        FVectorT_traj ego_rear_axle_y = ego_ys_traj[t] - ego_sin_thetas_traj[t] * ego_bb_extent_x;

                        IVectorT_traj dummy_mask, on_different_path_lanes_mask, on_non_route_mask;

                        occupancy_map->ContainsPointsInDrivableAreaBatch(
                            ego_xs_traj[t], ego_ys_traj[t], ego_rear_axle_x, ego_rear_axle_y, corners_xs, corners_ys,
                            on_non_drivable_area_mask, left_corners_on_non_drivable_area_mask,
                            right_corners_on_non_drivable_area_mask, dummy_mask, dummy_mask, dummy_mask,
                            on_different_path_lanes_mask, on_non_route_mask);

                        const FVectorT_traj &ego_path_curvature_proposal_t = (ego_path_curvatures_proposal.empty())
                                                                                 ? FVectorT_traj::fill(0.0f)
                                                                                 : ego_path_curvatures_proposal[t];

                        precomputed_non_on_drivable_area_penalties =
                            precomputed_non_on_drivable_area_penalties +
                            GetOnNonRoutePenalty(ego_initial_v, on_non_route_mask, ego_path_curvature_proposal_t) +
                            GetOnDifferentPathLanePenaltyBatch(ego_initial_v, on_different_path_lanes_mask,
                                                               on_non_route_mask, ego_path_curvature_proposal_t);
                    }

                    if (on_non_drivable_area_mask.any())
                    {
                        // For any contact with non-drivable areas, use an offset-based penalty function
                        precomputed_non_on_drivable_area_penalties =
                            precomputed_non_on_drivable_area_penalties +
                            (GetLateralOffsetOnNonDrivableAreaPenaltyBatch(lateral_offsets_proposal,
                                                                           left_corners_on_non_drivable_area_mask,
                                                                           right_corners_on_non_drivable_area_mask) +
                             on_non_drivable_area_mask.template convert<FVectorT_traj>().abs() *
                                 utils::NON_ON_DRIVABLE_AREA_PENALTY_STAGE_2) *
                                time_decay_weight;
                    }
                }
            }

            // Record inactive colliding vehicles for each trajectory and scenario during evaluation
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags = ws.exo_inactive_flags;

            // Pre-allocate buffers for collision detection
            AlignedVectorInt &has_original_collision = ws.has_original_collision;
            AlignedVectorInt &has_expanded_collision = ws.has_expanded_collision;
            AlignedVectorInt &collided_agent_idxs = ws.collided_agent_idxs;

            AlignedVectorFloat &collided_exo_xs_v = ws.collided_exo_xs_v;
            AlignedVectorFloat &collided_exo_ys_v = ws.collided_exo_ys_v;
            AlignedVectorFloat &collided_exo_vs_v = ws.collided_exo_vs_v;
            AlignedVectorFloat &collided_exo_thetas_v = ws.collided_exo_thetas_v;
            AlignedVectorFloat &collided_exo_cos_thetas_v = ws.collided_exo_cos_thetas_v;
            AlignedVectorFloat &collided_exo_sin_thetas_v = ws.collided_exo_sin_thetas_v;
            AlignedVectorFloat &collided_exo_expanded_bb_extent_xs_v = ws.collided_exo_expanded_bb_extent_xs_v;
            AlignedVectorFloat &collided_exo_expanded_bb_extent_ys_v = ws.collided_exo_expanded_bb_extent_ys_v;
            AlignedVectorFloat &collided_min_distances_v = ws.collided_min_distances_v;

            double total_weight_sum = 0.0f;
            for (size_t scenario_offset = 0; scenario_offset < utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD;
                 ++scenario_offset)
            {
                std::fill(exo_inactive_flags.begin(), exo_inactive_flags.end(),
                          std::array<int, utils::MAX_SIM_VEHICLES>{});

                IVectorT_traj active_flags = IVectorT_traj::fill(0xFFFFFFFF); // Active scenarios in the current scene
                FVectorT_traj rewards = precomputed_non_on_drivable_area_penalties;

                for (int t = 0; t < evaluation_len; ++t)
                {
                    // Speed penalty
                    rewards = rewards + precomputed_penalties[t] * active_flags.template convert<FVectorT_traj>().abs();

                    if (exo_size_ > 0)
                    {
                        std::fill(has_original_collision.begin(), has_original_collision.end(), 0);
                        std::fill(has_expanded_collision.begin(), has_expanded_collision.end(), 0);
                        std::fill(collided_agent_idxs.begin(), collided_agent_idxs.end(), -1);

                        // Collision penalty
                        HasCollisionBatch(
                            thread_id, scenario_offset, ego_xs_traj[t], ego_ys_traj[t], ego_cos_thetas_traj[t],
                            ego_sin_thetas_traj[t], query_aabb_min_xs[t], query_aabb_min_ys[t], query_aabb_max_xs[t],
                            query_aabb_max_ys[t], exo_xs, exo_ys, exo_vs, exo_thetas, exo_cos_thetas, exo_sin_thetas,
                            exo_original_bb_extent_xs, exo_original_bb_extent_ys, exo_expanded_bb_extent_xs,
                            exo_expanded_bb_extent_ys, exo_strtrees, ego_bb_extent_x, ego_bb_extent_y,
                            has_original_collision, has_expanded_collision, collided_min_distances_v,
                            collided_agent_idxs, collided_exo_xs_v, collided_exo_ys_v, collided_exo_vs_v,
                            collided_exo_thetas_v, collided_exo_cos_thetas_v, collided_exo_sin_thetas_v,
                            collided_exo_expanded_bb_extent_xs_v, collided_exo_expanded_bb_extent_ys_v, t, active_flags,
                            exo_inactive_flags, true);

                        IVectorT_traj has_original_collision_flag(has_original_collision.data());
                        IVectorT_traj has_expanded_collision_flag(has_expanded_collision.data());

                        if (has_expanded_collision_flag.any())
                        {
                            FVectorT_traj collided_exo_xs(collided_exo_xs_v.data());
                            FVectorT_traj collided_exo_ys(collided_exo_ys_v.data());
                            FVectorT_traj collided_exo_vs(collided_exo_vs_v.data());
                            FVectorT_traj collided_exo_thetas(collided_exo_thetas_v.data());
                            FVectorT_traj collided_exo_cos_thetas(collided_exo_cos_thetas_v.data());
                            FVectorT_traj collided_exo_sin_thetas(collided_exo_sin_thetas_v.data());
                            FVectorT_traj collided_exo_bb_extent_xs(collided_exo_expanded_bb_extent_xs_v.data());
                            FVectorT_traj collided_exo_bb_extent_ys(collided_exo_expanded_bb_extent_ys_v.data());
                            FVectorT_traj collided_min_distances(collided_min_distances_v.data());

                            IVectorT_traj in_front_mask =
                                ~utils::IsAngleWithinThreshold<FVectorT_traj>(
                                     collided_exo_xs - ego_rear_axle_xs[t], collided_exo_ys - ego_rear_axle_ys[t],
                                     -ego_cos_thetas_traj[t], -ego_sin_thetas_traj[t],
                                     utils::PI_1_3)
                                     .template as<IVectorT_traj>(); // Within a 240-degree frontal cone

                            IVectorT_traj collision_flags;
                            FVectorT_traj collision_penalty = GetNoAtFaultCollisionPenaltyBatch(
                                scenario_offset, has_original_collision_flag, has_expanded_collision_flag,
                                collision_flags, on_different_path_lanes_masks[t], on_non_drivable_area_masks[t],
                                ego_rear_axle_xs[t], ego_rear_axle_ys[t], ego_front_left_corner_xs[t],
                                ego_front_left_corner_ys[t], ego_front_right_corner_xs[t], ego_front_right_corner_ys[t],
                                ego_vs_traj[t], ego_thetas_traj[t], ego_cos_thetas_traj[t], ego_sin_thetas_traj[t],
                                collided_exo_xs, collided_exo_ys, collided_exo_vs, collided_exo_thetas,
                                collided_exo_cos_thetas, collided_exo_sin_thetas, collided_exo_bb_extent_xs,
                                collided_exo_bb_extent_ys, collided_min_distances, collided_agent_idxs,
                                exo_inactive_flags, t);
                            rewards = rewards + collision_penalty;

                            // Update collision timesteps (apply strong braking only if the colliding vehicle is in
                            // front, or if ego speed is low)
                            collided_timesteps = IVectorT_traj::select(
                                collision_flags &
                                    (in_front_mask | (ego_vs_traj[0] <= utils::COLLISION_BRAKING_LOW_SPEED_THRESHOLD)
                                                         .template as<IVectorT_traj>()),
                                collided_timesteps.min(t), collided_timesteps);

                            // Collision rollout penalty
                            FVectorT_traj collision_flags_float =
                                collision_flags.template convert<FVectorT_traj>().abs();
                            rewards = rewards +
                                      collision_flags_float *
                                          (collision_penalty + nuplan_metric_penalties[t] + GetMovementPenalty(0.0f)) *
                                          (float)(evaluation_len - t - 1);

                            // Update active flags
                            active_flags = active_flags & (~collision_flags);
                        }
                    }

                    if (active_flags.none())
                    {
                        break;
                    }
                }

                total_weight_sum += importance_sampling_weights[scenario_offset];
                value_array_sum = value_array_sum + importance_sampling_weights[scenario_offset] * rewards;
            }

            value_array_sum = value_array_sum / total_weight_sum;

            return value_array_sum;
        }

        // Compute ego corners
        void ContextQMDP::computeEgoCornersBatch(const FVectorT_qmdp &ego_xs, const FVectorT_qmdp &ego_ys,
                                                 const FVectorT_qmdp &ego_cos_thetas,
                                                 const FVectorT_qmdp &ego_sin_thetas, float ego_bb_extent_x,
                                                 float ego_bb_extent_y, FVectorT_qmdp *corners_xs,
                                                 FVectorT_qmdp *corners_ys)
        {
            FVectorT_qmdp ego_relative_cos_dist_x = ego_cos_thetas * ego_bb_extent_x;
            FVectorT_qmdp ego_relative_sin_dist_x = ego_sin_thetas * ego_bb_extent_x;
            FVectorT_qmdp ego_relative_cos_dist_y = ego_cos_thetas * ego_bb_extent_y;
            FVectorT_qmdp ego_relative_sin_dist_y = ego_sin_thetas * ego_bb_extent_y;

            // Front-left corner
            corners_xs[0] = ego_xs + ego_relative_cos_dist_y - ego_relative_sin_dist_x;
            corners_ys[0] = ego_ys + ego_relative_sin_dist_y + ego_relative_cos_dist_x;

            // Front-right corner
            corners_xs[1] = ego_xs + ego_relative_cos_dist_y + ego_relative_sin_dist_x,
            corners_ys[1] = ego_ys + ego_relative_sin_dist_y - ego_relative_cos_dist_x;

            // Rear-left corner
            corners_xs[2] = ego_xs - ego_relative_cos_dist_y - ego_relative_sin_dist_x,
            corners_ys[2] = ego_ys - ego_relative_sin_dist_y + ego_relative_cos_dist_x;

            // Rear-right corner
            corners_xs[3] = ego_xs - ego_relative_cos_dist_y + ego_relative_sin_dist_x,
            corners_ys[3] = ego_ys - ego_relative_sin_dist_y - ego_relative_cos_dist_x;

            // Middle of the left side
            corners_xs[4] = ego_xs - ego_relative_sin_dist_x;
            corners_ys[4] = ego_ys + ego_relative_cos_dist_x;

            // Middle of the right side
            corners_xs[5] = ego_xs + ego_relative_sin_dist_x;
            corners_ys[5] = ego_ys - ego_relative_cos_dist_x;
        }

        // Compute ego corners
        void ContextQMDP::computeEgoCornersBatch(const FVectorT_traj &ego_xs, const FVectorT_traj &ego_ys,
                                                 const FVectorT_traj &ego_cos_thetas,
                                                 const FVectorT_traj &ego_sin_thetas, float ego_bb_extent_x,
                                                 float ego_bb_extent_y, FVectorT_traj *corners_xs,
                                                 FVectorT_traj *corners_ys)
        {
            FVectorT_traj ego_relative_cos_dist_x = ego_cos_thetas * ego_bb_extent_x;
            FVectorT_traj ego_relative_sin_dist_x = ego_sin_thetas * ego_bb_extent_x;
            FVectorT_traj ego_relative_cos_dist_y = ego_cos_thetas * ego_bb_extent_y;
            FVectorT_traj ego_relative_sin_dist_y = ego_sin_thetas * ego_bb_extent_y;

            // Front-left corner
            corners_xs[0] = ego_xs + ego_relative_cos_dist_y - ego_relative_sin_dist_x;
            corners_ys[0] = ego_ys + ego_relative_sin_dist_y + ego_relative_cos_dist_x;

            // Front-right corner
            corners_xs[1] = ego_xs + ego_relative_cos_dist_y + ego_relative_sin_dist_x,
            corners_ys[1] = ego_ys + ego_relative_sin_dist_y - ego_relative_cos_dist_x;

            // Rear-left corner
            corners_xs[2] = ego_xs - ego_relative_cos_dist_y - ego_relative_sin_dist_x,
            corners_ys[2] = ego_ys - ego_relative_sin_dist_y + ego_relative_cos_dist_x;

            // Rear-right corner
            corners_xs[3] = ego_xs - ego_relative_cos_dist_y + ego_relative_sin_dist_x,
            corners_ys[3] = ego_ys - ego_relative_sin_dist_y - ego_relative_cos_dist_x;

            // Middle of the left side
            corners_xs[4] = ego_xs - ego_relative_sin_dist_x;
            corners_ys[4] = ego_ys + ego_relative_cos_dist_x;

            // Middle of the right side
            corners_xs[5] = ego_xs + ego_relative_sin_dist_x;
            corners_ys[5] = ego_ys - ego_relative_cos_dist_x;
        }

        FVectorT_1 ContextQMDP::checkSATCollisionBatch(const FVectorT_1 &ego_xs, const FVectorT_1 &ego_ys,
                                                       const FVectorT_1 &ego_cos_thetas,
                                                       const FVectorT_1 &ego_sin_thetas, const FVectorT_1 &exo_xs,
                                                       const FVectorT_1 &exo_ys, const FVectorT_1 &exo_cos_thetas,
                                                       const FVectorT_1 &exo_sin_thetas, float ego_bb_extent_x,
                                                       float ego_bb_extent_y, const FVectorT_1 &exo_bb_extent_xs,
                                                       const FVectorT_1 &exo_bb_extent_ys)
        {
            // Initialize collision flags, assuming collision by default
            FVectorT_1 non_collision_flag = FVectorT_1::fill(0x00000000);

            // Compute vector from ego center to exo center
            FVectorT_1 ego_to_exo_x = exo_xs - ego_xs;
            FVectorT_1 ego_to_exo_y = exo_ys - ego_ys;

            // Check along exo heading axis (exo_cos_theta, exo_sin_theta)
            // Compute ego's projection radius along this axis
            FVectorT_1 ego_proj_on_exo_heading =
                (ego_cos_thetas * exo_cos_thetas + ego_sin_thetas * exo_sin_thetas).abs() * ego_bb_extent_y +
                (-ego_sin_thetas * exo_cos_thetas + ego_cos_thetas * exo_sin_thetas).abs() * ego_bb_extent_x;

            // Compute distance between centers projected on this axis
            FVectorT_1 center_proj_on_exo_heading =
                (ego_to_exo_x * exo_cos_thetas + ego_to_exo_y * exo_sin_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_heading + exo_bb_extent_ys < center_proj_on_exo_heading);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to exo heading (-exo_sin_theta, exo_cos_theta)
            FVectorT_1 ego_proj_on_exo_normal =
                ego_bb_extent_y * (-ego_cos_thetas * exo_sin_thetas + ego_sin_thetas * exo_cos_thetas).abs() +
                ego_bb_extent_x * (ego_sin_thetas * exo_sin_thetas + ego_cos_thetas * exo_cos_thetas).abs();

            // Compute distance between centers projected on this axis
            FVectorT_1 center_proj_on_exo_normal =
                (-ego_to_exo_x * exo_sin_thetas + ego_to_exo_y * exo_cos_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_normal + exo_bb_extent_xs < center_proj_on_exo_normal);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check along ego heading axis (ego_cos_theta, ego_sin_theta)
            FVectorT_1 exo_proj_on_ego_heading =
                exo_bb_extent_ys * (exo_cos_thetas * ego_cos_thetas + exo_sin_thetas * ego_sin_thetas).abs() +
                exo_bb_extent_xs * (-exo_sin_thetas * ego_cos_thetas + exo_cos_thetas * ego_sin_thetas).abs();

            FVectorT_1 center_proj_on_ego_heading =
                (ego_to_exo_x * ego_cos_thetas + ego_to_exo_y * ego_sin_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_heading + ego_bb_extent_y < center_proj_on_ego_heading);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to ego heading (-ego_sin_theta, ego_cos_theta)
            FVectorT_1 exo_proj_on_ego_normal =
                exo_bb_extent_ys * (-exo_cos_thetas * ego_sin_thetas + exo_sin_thetas * ego_cos_thetas).abs() +
                exo_bb_extent_xs * (exo_sin_thetas * ego_sin_thetas + exo_cos_thetas * ego_cos_thetas).abs();

            FVectorT_1 center_proj_on_ego_normal =
                (-ego_to_exo_x * ego_sin_thetas + ego_to_exo_y * ego_cos_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_normal + ego_bb_extent_x < center_proj_on_ego_normal);

            return non_collision_flag;
        }

        // Perform collision detection for objects
        std::vector<int> ContextQMDP::checkSATCollisionBatch(
            float ego_x, float ego_y, float ego_cos_theta, float ego_sin_theta, const FVectorT_qmdp &exo_xs,
            const FVectorT_qmdp &exo_ys, const FVectorT_qmdp &exo_cos_thetas, const FVectorT_qmdp &exo_sin_thetas,
            float ego_bb_extent_x, float ego_bb_extent_y, const FVectorT_qmdp &exo_bb_extent_xs,
            const FVectorT_qmdp &exo_bb_extent_ys)
        {
            // Initialize collision flags, assuming collision by default
            std::vector<int> collided_agent_idxs;
            FVectorT_qmdp    non_collision_flag = FVectorT_qmdp::fill(0x00000000);

            // Compute vector from ego center to exo center
            FVectorT_qmdp ego_to_exo_x = exo_xs - ego_x;
            FVectorT_qmdp ego_to_exo_y = exo_ys - ego_y;

            // Check along exo heading axis (exo_cos_theta, exo_sin_theta)
            // Compute ego's projection radius along this axis
            FVectorT_qmdp ego_proj_on_exo_heading =
                (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() * ego_bb_extent_y +
                (exo_cos_thetas * -ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() * ego_bb_extent_x;

            // Compute distance between centers projected on this axis
            FVectorT_qmdp center_proj_on_exo_heading =
                (ego_to_exo_x * exo_cos_thetas + ego_to_exo_y * exo_sin_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_heading + exo_bb_extent_ys < center_proj_on_exo_heading);

            // If all are non-colliding, return early (-1: no collision)
            if (non_collision_flag.all())
                return collided_agent_idxs;

            // Check axis perpendicular to exo heading (-exo_sin_theta, exo_cos_theta)
            FVectorT_qmdp ego_proj_on_exo_normal =
                ego_bb_extent_y * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                ego_bb_extent_x * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            // Compute distance between centers projected on this axis
            FVectorT_qmdp center_proj_on_exo_normal =
                (-ego_to_exo_x * exo_sin_thetas + ego_to_exo_y * exo_cos_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_normal + exo_bb_extent_xs < center_proj_on_exo_normal);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return collided_agent_idxs;

            // Check along ego heading axis (ego_cos_theta, ego_sin_theta)
            FVectorT_qmdp exo_proj_on_ego_heading =
                exo_bb_extent_ys * (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() +
                exo_bb_extent_xs * (-exo_sin_thetas * ego_cos_theta + exo_cos_thetas * ego_sin_theta).abs();

            FVectorT_qmdp center_proj_on_ego_heading =
                (ego_to_exo_x * ego_cos_theta + ego_to_exo_y * ego_sin_theta).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_heading + ego_bb_extent_y < center_proj_on_ego_heading);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return collided_agent_idxs;

            // Check axis perpendicular to ego heading (-ego_sin_theta, ego_cos_theta)
            FVectorT_qmdp exo_proj_on_ego_normal =
                exo_bb_extent_ys * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                exo_bb_extent_xs * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            FVectorT_qmdp center_proj_on_ego_normal =
                (-ego_to_exo_x * ego_sin_theta + ego_to_exo_y * ego_cos_theta).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_normal + ego_bb_extent_x < center_proj_on_ego_normal);

            for (int i = 0; i < FVectorT_qmdp::num_scalars; ++i)
            {
                if (non_collision_flag[{0, i}] == 0)
                {
                    collided_agent_idxs.emplace_back(i);
                }
            }

            return collided_agent_idxs;
        }

        // Perform collision detection for objects
        FVectorT_qmdp ContextQMDP::checkSATCollisionBatch(
            float ego_x, float ego_y, float ego_cos_theta, float ego_sin_theta, const FVectorT_qmdp &exo_xs,
            const FVectorT_qmdp &exo_ys, const FVectorT_qmdp &exo_cos_thetas, const FVectorT_qmdp &exo_sin_thetas,
            float ego_bb_extent_x, float ego_bb_extent_y, const FVectorT_qmdp &exo_bb_extent_xs,
            const FVectorT_qmdp &exo_bb_extent_ys, FVectorT_qmdp &min_gaps)
        {
            // Initialize collision flags, assuming collision by default
            FVectorT_qmdp non_collision_flag = FVectorT_qmdp::fill(0x00000000);

            // Compute vector from ego center to exo center
            FVectorT_qmdp ego_to_exo_x = exo_xs - ego_x;
            FVectorT_qmdp ego_to_exo_y = exo_ys - ego_y;

            // Check along exo heading axis (exo_cos_theta, exo_sin_theta)
            // Compute ego's projection radius along this axis
            FVectorT_qmdp ego_proj_on_exo_heading =
                (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() * ego_bb_extent_y +
                (exo_cos_thetas * -ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() * ego_bb_extent_x;

            // Compute distance between centers projected on this axis
            FVectorT_qmdp center_proj_on_exo_heading =
                (ego_to_exo_x * exo_cos_thetas + ego_to_exo_y * exo_sin_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            FVectorT_qmdp gaps_on_exo_heading =
                center_proj_on_exo_heading - (ego_proj_on_exo_heading + exo_bb_extent_ys);
            non_collision_flag = non_collision_flag | (gaps_on_exo_heading > 0.0f);

            min_gaps = FVectorT_qmdp::select(gaps_on_exo_heading > 0.0f, min_gaps.min(gaps_on_exo_heading), min_gaps);

            // If all are non-colliding, return early (-1: no collision)
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to exo heading (-exo_sin_theta, exo_cos_theta)
            FVectorT_qmdp ego_proj_on_exo_normal =
                ego_bb_extent_y * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                ego_bb_extent_x * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            // Compute distance between centers projected on this axis
            FVectorT_qmdp center_proj_on_exo_normal =
                (-ego_to_exo_x * exo_sin_thetas + ego_to_exo_y * exo_cos_thetas).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            FVectorT_qmdp gaps_on_exo_normal = center_proj_on_exo_normal - (ego_proj_on_exo_normal + exo_bb_extent_xs);
            non_collision_flag = non_collision_flag | (gaps_on_exo_normal > 0.0f);

            min_gaps = FVectorT_qmdp::select(gaps_on_exo_normal > 0.0f, min_gaps.min(gaps_on_exo_normal), min_gaps);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check along ego heading axis (ego_cos_theta, ego_sin_theta)
            FVectorT_qmdp exo_proj_on_ego_heading =
                exo_bb_extent_ys * (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() +
                exo_bb_extent_xs * (-exo_sin_thetas * ego_cos_theta + exo_cos_thetas * ego_sin_theta).abs();

            FVectorT_qmdp center_proj_on_ego_heading =
                (ego_to_exo_x * ego_cos_theta + ego_to_exo_y * ego_sin_theta).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            FVectorT_qmdp gaps_on_ego_heading =
                center_proj_on_ego_heading - (exo_proj_on_ego_heading + ego_bb_extent_y);
            non_collision_flag = non_collision_flag | (gaps_on_ego_heading > 0.0f);

            min_gaps = FVectorT_qmdp::select(gaps_on_ego_heading > 0.0f, min_gaps.min(gaps_on_ego_heading), min_gaps);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to ego heading (-ego_sin_theta, ego_cos_theta)
            FVectorT_qmdp exo_proj_on_ego_normal =
                exo_bb_extent_ys * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                exo_bb_extent_xs * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            FVectorT_qmdp center_proj_on_ego_normal =
                (-ego_to_exo_x * ego_sin_theta + ego_to_exo_y * ego_cos_theta).abs();

            // If center distance exceeds the sum of projection radii, they are separated on this axis
            FVectorT_qmdp gaps_on_ego_normal = center_proj_on_ego_normal - (exo_proj_on_ego_normal + ego_bb_extent_x);
            non_collision_flag = non_collision_flag | (gaps_on_ego_normal > 0.0f);

            min_gaps = FVectorT_qmdp::select(gaps_on_ego_normal > 0.0f, min_gaps.min(gaps_on_ego_normal), min_gaps);

            return non_collision_flag;
        }

        FVectorT_1 ContextQMDP::checkSATCollisionBatch1(float ego_x, float ego_y, float ego_cos_theta,
                                                        float ego_sin_theta, const FVectorT_1 &exo_xs,
                                                        const FVectorT_1 &exo_ys, const FVectorT_1 &exo_cos_thetas,
                                                        const FVectorT_1 &exo_sin_thetas, float ego_bb_extent_x,
                                                        float ego_bb_extent_y, const FVectorT_1 &exo_bb_extent_xs,
                                                        const FVectorT_1 &exo_bb_extent_ys)
        {
            // Initialize collision flags, assuming collision by default
            FVectorT_1 non_collision_flag = FVectorT_1::fill(0x00000000);

            // Compute vector from ego center to exo center
            FVectorT_1 ego_to_exo_x = exo_xs - ego_x;
            FVectorT_1 ego_to_exo_y = exo_ys - ego_y;

            // Check along exo heading axis (exo_cos_theta, exo_sin_theta)
            // Compute ego's projection radius along this axis
            FVectorT_1 ego_proj_on_exo_heading =
                (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() * ego_bb_extent_y +
                (exo_cos_thetas * -ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() * ego_bb_extent_x;

            // Compute distance between centers projected on this axis
            FVectorT_1 center_proj_on_exo_heading =
                (ego_to_exo_x * exo_cos_thetas + ego_to_exo_y * exo_sin_thetas).abs();

            // If center distance exceeds the sum of projection half-lengths, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_heading + exo_bb_extent_ys < center_proj_on_exo_heading);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to exo heading (-exo_sin_theta, exo_cos_theta)
            FVectorT_1 ego_proj_on_exo_normal =
                ego_bb_extent_y * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                ego_bb_extent_x * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            // Compute distance between centers projected on this axis
            FVectorT_1 center_proj_on_exo_normal =
                (-ego_to_exo_x * exo_sin_thetas + ego_to_exo_y * exo_cos_thetas).abs();

            // If center distance exceeds the sum of projection half-lengths, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (ego_proj_on_exo_normal + exo_bb_extent_xs < center_proj_on_exo_normal);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check along ego heading axis (ego_cos_theta, ego_sin_theta)
            FVectorT_1 exo_proj_on_ego_heading =
                exo_bb_extent_ys * (exo_cos_thetas * ego_cos_theta + exo_sin_thetas * ego_sin_theta).abs() +
                exo_bb_extent_xs * (-exo_sin_thetas * ego_cos_theta + exo_cos_thetas * ego_sin_theta).abs();

            FVectorT_1 center_proj_on_ego_heading = (ego_to_exo_x * ego_cos_theta + ego_to_exo_y * ego_sin_theta).abs();

            // If center distance exceeds the sum of projection half-lengths, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_heading + ego_bb_extent_y < center_proj_on_ego_heading);

            // If all are non-colliding, return early
            if (non_collision_flag.all())
                return non_collision_flag;

            // Check axis perpendicular to ego heading (-ego_sin_theta, ego_cos_theta)
            FVectorT_1 exo_proj_on_ego_normal =
                exo_bb_extent_ys * (-exo_cos_thetas * ego_sin_theta + exo_sin_thetas * ego_cos_theta).abs() +
                exo_bb_extent_xs * (exo_sin_thetas * ego_sin_theta + exo_cos_thetas * ego_cos_theta).abs();

            FVectorT_1 center_proj_on_ego_normal = (-ego_to_exo_x * ego_sin_theta + ego_to_exo_y * ego_cos_theta).abs();

            // If center distance exceeds the sum of projection half-lengths, they are separated on this axis
            non_collision_flag =
                non_collision_flag | (exo_proj_on_ego_normal + ego_bb_extent_x < center_proj_on_ego_normal);

            return non_collision_flag;
        }

        FVectorT_traj_1 ContextQMDP::checkSATCollisionBatchTraj(
            const FVectorT_traj_1 &ego_xs, const FVectorT_traj_1 &ego_ys, const FVectorT_traj_1 &ego_cos_thetas,
            const FVectorT_traj_1 &ego_sin_thetas, float exo_x, float exo_y, float exo_cos_theta, float exo_sin_theta,
            float ego_bb_extent_x, float ego_bb_extent_y, float exo_bb_extent_x, float exo_bb_extent_y,
            FVectorT_traj_1 &min_gaps)
        {
            // 1. Transform Exo to Ego Local Frame
            FVectorT_traj_1 dx = exo_x - ego_xs;
            FVectorT_traj_1 dy = exo_y - ego_ys;

            // Local Position
            FVectorT_traj_1 local_x = dx * ego_cos_thetas + dy * ego_sin_thetas;
            FVectorT_traj_1 local_y = -dx * ego_sin_thetas + dy * ego_cos_thetas;

            // Relative Orientation
            FVectorT_traj_1 local_exo_cos = exo_cos_theta * ego_cos_thetas + exo_sin_theta * ego_sin_thetas;
            FVectorT_traj_1 local_exo_sin = exo_sin_theta * ego_cos_thetas - exo_cos_theta * ego_sin_thetas;
            FVectorT_traj_1 abs_local_cos = local_exo_cos.abs();
            FVectorT_traj_1 abs_local_sin = local_exo_sin.abs();

            // --- SAT Axis 1: Ego X ---
            FVectorT_traj_1 proj_exo_x = exo_bb_extent_y * abs_local_cos + exo_bb_extent_x * abs_local_sin;
            FVectorT_traj_1 overlap_x = local_x.abs() - ego_bb_extent_y - proj_exo_x;
            FVectorT_traj_1 no_col = (overlap_x > 0.0f);

            min_gaps = FVectorT_traj_1::select(no_col, min_gaps.min(overlap_x), min_gaps);

            // --- SAT Axis 2: Ego Y ---
            FVectorT_traj_1 proj_exo_y = exo_bb_extent_y * abs_local_sin + exo_bb_extent_x * abs_local_cos;
            FVectorT_traj_1 overlap_y = local_y.abs() - ego_bb_extent_x - proj_exo_y;
            no_col = no_col | (overlap_y > 0.0f);
            min_gaps = FVectorT_traj_1::select(overlap_y > 0.0f, min_gaps.min(overlap_y), min_gaps);

            // Optimization: Early exit mask check could be done here, but usually faster to stream through
            if (no_col.all())
                return ~no_col;

            // --- SAT Axis 3: Exo X ---
            FVectorT_traj_1 proj_ego_x = ego_bb_extent_y * abs_local_cos + ego_bb_extent_x * abs_local_sin;
            FVectorT_traj_1 dist_exo_x = (local_x * local_exo_cos + local_y * local_exo_sin).abs();
            FVectorT_traj_1 overlap_exo_x = dist_exo_x - exo_bb_extent_y - proj_ego_x;
            no_col = no_col | (overlap_exo_x > 0.0f);
            min_gaps = FVectorT_traj_1::select(overlap_exo_x > 0.0f, min_gaps.min(overlap_exo_x), min_gaps);

            // --- SAT Axis 4: Exo Y ---
            FVectorT_traj_1 proj_ego_y = ego_bb_extent_y * abs_local_sin + ego_bb_extent_x * abs_local_cos;
            FVectorT_traj_1 dist_exo_y = (-local_x * local_exo_sin + local_y * local_exo_cos).abs();
            FVectorT_traj_1 overlap_exo_y = dist_exo_y - exo_bb_extent_x - proj_ego_y;
            no_col = no_col | (overlap_exo_y > 0.0f);
            min_gaps = FVectorT_traj_1::select(overlap_exo_y > 0.0f, min_gaps.min(overlap_exo_y), min_gaps);

            return ~no_col;
        }

        // 0: not collided; 0xFFFFFFFF: collided
        int ContextQMDP::HasCollisionBatch(
            float ego_rear_x, float ego_rear_y, float ego_x, float ego_y, float ego_theta, float ego_cos_theta,
            float ego_sin_theta, const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys,
            const AlignedVectorFloat &exo_cos_thetas, const AlignedVectorFloat &exo_sin_thetas,
            const AlignedVectorFloat &exo_bb_extent_xs, const AlignedVectorFloat &exo_bb_extent_ys,
            const std::shared_ptr<STRtree> &exo_strtree, const size_t &scenario_idx, float ego_bb_extent_x,
            float ego_bb_extent_y, float query_aabb_min_x, float query_aabb_min_y, float query_aabb_max_x,
            float query_aabb_max_y, const size_t &curr_stepped_time_idx,
            std::array<int, utils::MAX_SIM_VEHICLES> &exo_inactive_flag)
        {
            // 1. Allocate a temporary buffer on the stack
            int candidate_size = 0;

            // 2. STRtree query
            // Note: if queryBatch returns a vector, there will still be heap allocations.
            // It is recommended to optimize queryBatch to accept a reference buffer or to return a SmallVector.
            const std::array<int, 8> &candidate_idxs_buffer = exo_strtree->queryBatch<8>(
                query_aabb_min_x, query_aabb_min_y, query_aabb_max_x, query_aabb_max_y, candidate_size);

            if (candidate_size == 0)
                return 0;

            // 3. Prepare SIMD data
            IVectorT_1 candidate_idxs_v(candidate_idxs_buffer.data());

            // Compute gather indices
            IVectorT_1 data_idxs = IVectorT_1::fill(scenario_idx * exo_total_size_ + curr_stepped_time_idx) +
                                   candidate_idxs_v * total_time_size_;

            // Gather Exo States
            FVectorT_1 exo_xs_v = FVectorT_1::gather(exo_xs.data(), data_idxs);
            FVectorT_1 exo_ys_v = FVectorT_1::gather(exo_ys.data(), data_idxs);
            FVectorT_1 exo_cos_v = FVectorT_1::gather(exo_cos_thetas.data(), data_idxs);
            FVectorT_1 exo_sin_v = FVectorT_1::gather(exo_sin_thetas.data(), data_idxs);
            FVectorT_1 exo_ext_x_v = FVectorT_1::gather(exo_bb_extent_xs.data(), candidate_idxs_v);
            FVectorT_1 exo_ext_y_v = FVectorT_1::gather(exo_bb_extent_ys.data(), candidate_idxs_v);

            // 4. SAT collision detection (ego-local optimized) & InBehind integration
            // 4.1 Coordinate transform: Global -> Ego Local
            FVectorT_1 dx = exo_xs_v - ego_x;
            FVectorT_1 dy = exo_ys_v - ego_y;

            // Rotation matrix R^T * d
            FVectorT_1 local_x = dx * ego_cos_theta + dy * ego_sin_theta;
            FVectorT_1 local_y = -dx * ego_sin_theta + dy * ego_cos_theta;

            // Compute exo heading in the local coordinate frame (cos(a-b) = cos*cos + sin*sin)
            FVectorT_1 local_cos = exo_cos_v * ego_cos_theta + exo_sin_v * ego_sin_theta;
            FVectorT_1 local_sin =
                exo_sin_v * ego_cos_theta - exo_cos_v * ego_sin_theta; // sin(a-b) check sign carefully
            FVectorT_1 abs_local_cos = local_cos.abs();
            FVectorT_1 abs_local_sin = local_sin.abs();

            // --- SAT Check 1: Ego X axis (local X) ---
            // Projection distance: abs(local_x)
            // Sum of radii: ego_ext_x + (exo_ext_x * |cos'| + exo_ext_y * |sin'|)
            FVectorT_1 proj_exo_on_ego_x = exo_ext_y_v * abs_local_cos + exo_ext_x_v * abs_local_sin;
            FVectorT_1 no_col_mask = (local_x.abs() > (ego_bb_extent_y + proj_exo_on_ego_x));

            // --- SAT Check 2: Ego Y axis (local Y) ---
            // Projection distance: abs(local_y)
            FVectorT_1 proj_exo_on_ego_y = exo_ext_y_v * abs_local_sin + exo_ext_x_v * abs_local_cos;
            no_col_mask = no_col_mask | (local_y.abs() > (ego_bb_extent_x + proj_exo_on_ego_y));

            // Optimization: Early exit mask check could be done here, but usually faster to stream through
            if (no_col_mask.all())
            {
                return 0;
            }

            // --- SAT Check 3: Exo X axis ---
            // Axis vector in local coordinates: (local_cos, local_sin)
            // Ego projection: ego_ext_x * |cos'| + ego_ext_y * |sin'|
            // Exo projection: exo_ext_x
            // Distance projection: |local_x * cos' + local_y * sin'|
            FVectorT_1 proj_ego_on_exo_x = ego_bb_extent_y * abs_local_cos + ego_bb_extent_x * abs_local_sin;
            FVectorT_1 dist_on_exo_x = (local_x * local_cos + local_y * local_sin).abs();
            no_col_mask = no_col_mask | (dist_on_exo_x > (exo_ext_y_v + proj_ego_on_exo_x));

            // --- SAT Check 4: Exo Y axis ---
            // Axis vector in local coordinates: (-local_sin, local_cos)
            FVectorT_1 proj_ego_on_exo_y = ego_bb_extent_y * abs_local_sin + ego_bb_extent_x * abs_local_cos;
            FVectorT_1 dist_on_exo_y = (-local_x * local_sin + local_y * local_cos).abs();
            no_col_mask = no_col_mask | (dist_on_exo_y > (exo_ext_x_v + proj_ego_on_exo_y));

            if (no_col_mask.all())
            {
                return 0;
            }

            // Invert mask to obtain the collision mask
            IVectorT_1 is_colliding = (~no_col_mask).template as<IVectorT_1>();

            // 5. InBehind check
            // The key is to determine whether exo is behind ego within a certain angle range
            IVectorT_1 is_behind =
                utils::IsAngleWithinThreshold<FVectorT_1>(exo_xs_v - ego_rear_x, exo_ys_v - ego_rear_y, -ego_cos_theta,
                                                          -ego_sin_theta, utils::PI_1_4)
                    .template as<IVectorT_1>();

            // 6. Process results
            // Truly dangerous collision = collision AND (NOT behind) AND (NOT already inactive)
            // Gather inactive flags
            IVectorT_1 exo_inactive_v =
                IVectorT_1::gather(exo_inactive_flag.data(), candidate_idxs_v); // Gather int array needs helper

            IVectorT_1 real_collision = is_colliding & (~is_behind) & (exo_inactive_v == 0);

            // Need to mark as inactive = collision AND behind AND (NOT already inactive)
            IVectorT_1 need_set_inactive = is_colliding & is_behind & (exo_inactive_v == 0);

            // Mask handling: only consider the first candidate_size results
            // Create a mask where only the first candidate_size bits are 1
            // (A simple approach is to handle this within the loop below)

            // 7. Write back & return (scalar processing because the amount is small and has side effects)
            // Store SIMD results back into stack arrays
            auto col_res = real_collision.to_array();
            auto inactive_res = need_set_inactive.to_array();

            for (int k = 0; k < candidate_size; ++k)
            {
                if (col_res[k])
                    return 0xFFFFFFFF; // Found a real collision, return immediately
                if (inactive_res[k])
                {
                    // Only this part performs memory writes
                    exo_inactive_flag[candidate_idxs_buffer[k]] = 0xFFFFFFFF;
                }
            }

            return 0;
        }

        // for cross-scenario evaluation
        void ContextQMDP::HasCollisionBatch(
            int thread_id, size_t scenario_offset, const FVectorT_traj &ego_xs, const FVectorT_traj &ego_ys,
            const FVectorT_traj &ego_cos_thetas, const FVectorT_traj &ego_sin_thetas,
            const FVectorT_traj &query_aabb_min_xs, const FVectorT_traj &query_aabb_min_ys,
            const FVectorT_traj &query_aabb_max_xs, const FVectorT_traj &query_aabb_max_ys,
            const AlignedVectorFloat &exo_xs, const AlignedVectorFloat &exo_ys, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_thetas, const AlignedVectorFloat &exo_cos_thetas,
            const AlignedVectorFloat &exo_sin_thetas, const AlignedVectorFloat &exo_original_bb_extent_xs,
            const AlignedVectorFloat &exo_original_bb_extent_ys, const AlignedVectorFloat &exo_expanded_bb_extent_xs,
            const AlignedVectorFloat &exo_expanded_bb_extent_ys, const std::vector<std::shared_ptr<STRtree>> &strtrees,
            float ego_bb_extent_x, float ego_bb_extent_y, AlignedVectorInt &has_original_collision,
            AlignedVectorInt &has_expanded_collision, AlignedVectorFloat &collided_min_distances,
            AlignedVectorInt &collided_exo_idxs, AlignedVectorFloat &collided_exo_xs,
            AlignedVectorFloat &collided_exo_ys, AlignedVectorFloat &collided_exo_vs,
            AlignedVectorFloat &collided_exo_thetas, AlignedVectorFloat &collided_exo_cos_thetas,
            AlignedVectorFloat &collided_exo_sin_thetas, AlignedVectorFloat &collided_exo_expanded_bb_extent_xs,
            AlignedVectorFloat &collided_exo_expanded_bb_extent_ys, int curr_stepped_time_idx,
            const IVectorT_traj                                   &active_flags,
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags, bool print)
        {
            // 1. STRtree is fixed
            // All rows (batches) share the same scene structure, so the STRtree pointer is fixed
            const std::shared_ptr<STRtree> &strtree =
                strtrees[scenario_offset * total_time_size_proposal_ + curr_stepped_time_idx];

            // Perform a separate query for each row (each SIMD batch)
            // FVectorT_traj::num_rows = FVectorT_traj::num_scalars / FVectorT_traj::width (for example 96 / 8 = 12)
            for (int row = 0; row < FVectorT_traj::num_rows; ++row)
            {
                // Compute the starting index of the current row (relative to the internal storage of FVectorT_traj)
                int base_idx =
                    row *
                    FVectorT_traj::num_scalars_per_row;  // Offset of the current batch in FVectorT_traj (0, 8, 16, ...)
                int global_scenario_base_idx = base_idx; // Used to access global arrays such as has_collision

                // 2.1 Build active mask
                const FVectorT_traj_1 &row_active_mask = active_flags[row].template as<FVectorT_traj_1>();

                if (row_active_mask.none())
                    continue;

                // 2.2 Load ego data
                const FVectorT_traj_1 &ego_xs_v = ego_xs[row];
                const FVectorT_traj_1 &ego_ys_v = ego_ys[row];
                const FVectorT_traj_1 &ego_cos_thetas_v = ego_cos_thetas[row];
                const FVectorT_traj_1 &ego_sin_thetas_v = ego_sin_thetas[row];

                // 2.3 Union AABB (Point 1)
                // Load query AABB for the current batch
                // Note: query_aabb arrays are laid out in a global flat index
                FVectorT_traj_1 min_x_v = query_aabb_min_xs[row];
                FVectorT_traj_1 min_y_v = query_aabb_min_ys[row];
                FVectorT_traj_1 max_x_v = query_aabb_max_xs[row];
                FVectorT_traj_1 max_y_v = query_aabb_max_ys[row];

                // Mask inactive lanes (fill with extreme values)
                min_x_v = FVectorT_traj_1::select(row_active_mask, min_x_v, utils::MAX_VALUE);
                min_y_v = FVectorT_traj_1::select(row_active_mask, min_y_v, utils::MAX_VALUE);
                max_x_v = FVectorT_traj_1::select(row_active_mask, max_x_v, -utils::MAX_VALUE);
                max_y_v = FVectorT_traj_1::select(row_active_mask, max_y_v, -utils::MAX_VALUE);

                // Horizontal Reduction
                float union_min_x = min_x_v.hmin();
                float union_min_y = min_y_v.hmin();
                float union_max_x = max_x_v.hmax();
                float union_max_y = max_y_v.hmax();

                // 2.4 Query STRtree
                // Use stack buffer (size 16, point 1)
                constexpr int                       QUERY_LIMIT = 16;
                int                                 candidate_size = 0;
                const std::array<int, QUERY_LIMIT> &candidates = strtree->queryBatch<QUERY_LIMIT>(
                    union_min_x, union_min_y, union_max_x, union_max_y, candidate_size);

                if (candidate_size == 0 && utils::NOTICE_VEHICLE_IDXS.empty())
                    continue;

                // 2.5 Initialize SIMD state registers (registers for best collision)
                IVectorT_traj_1 mask_found_original = IVectorT_traj_1::fill(0x00000000); // initialized as false
                IVectorT_traj_1 mask_found_expanded = IVectorT_traj_1::fill(0x00000000); // initialized as false

                IVectorT_traj_1 res_idx = IVectorT_traj_1::fill(0);
                FVectorT_traj_1 res_x = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_y = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_v = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_th = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_c = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_s = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_ex_x = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_ex_y = FVectorT_traj_1::fill(0.0f);
                FVectorT_traj_1 res_dist = FVectorT_traj_1::fill(10.0f);

                // capture all by reference [&] to access context variables like ego_xs_v and mask_found_original
                constexpr float kCoarseCheckDistSq = 50.0f;
                auto            process_agent = [&](int cand_idx, bool enable_coarse_check)
                {
                    // 3.1 Calculate Data Index
                    int data_idx = scenario_offset * exo_total_size_proposal_ + cand_idx * total_time_size_proposal_ +
                                   curr_stepped_time_idx;

                    // 3.2 Splat Exo Data (Scalar -> Vector)
                    float ex_x = exo_xs[data_idx];
                    float ex_y = exo_ys[data_idx];

                    // --- Coarse distance filter for "noticed vehicles" ---
                    // If coarse check is enabled, first compute Euclidean distance. If too far, early-return to avoid
                    // expensive SAT tests. Note: this uses SIMD vectors because ego is vectorized.
                    FVectorT_traj_1 valid_mask = row_active_mask; // initially the row's active mask

                    if (enable_coarse_check)
                    {
                        // Compute distance from ego (vector) to agent (scalar)
                        FVectorT_traj_1 dx = ego_xs_v - ex_x;
                        FVectorT_traj_1 dy = ego_ys_v - ex_y;
                        FVectorT_traj_1 dist_sq = dx * dx + dy * dy;

                        // Generate distance mask: only lanes closer than the threshold continue
                        FVectorT_traj_1 dist_mask = dist_sq < kCoarseCheckDistSq;
                        valid_mask = valid_mask & dist_mask;

                        // If all lanes are filtered out by distance, skip this agent
                        if (valid_mask.none())
                            return;
                    }

                    // --- Continue loading remaining data ---
                    float ex_cos = exo_cos_thetas[data_idx];
                    float ex_sin = exo_sin_thetas[data_idx];
                    float ex_v = exo_vs[data_idx];
                    float ex_th = exo_thetas[data_idx];
                    float ex_orig_ext_x = exo_original_bb_extent_xs[cand_idx];
                    float ex_orig_ext_y = exo_original_bb_extent_ys[cand_idx];
                    float ex_exp_ext_x = exo_expanded_bb_extent_xs[cand_idx];
                    float ex_exp_ext_y = exo_expanded_bb_extent_ys[cand_idx];

                    // 3.3 Build inactive mask
                    int inactive_bits = 0;
                    for (int lane = 0; lane < FVectorT_traj::num_scalars_per_row; ++lane)
                    {
                        if (exo_inactive_flags[global_scenario_base_idx + lane][cand_idx])
                        {
                            inactive_bits |= (1 << lane);
                        }
                    }
                    static const IVectorT_traj_1 powers_of_two = []()
                    {
                        std::array<int, FVectorT_traj::num_scalars_per_row> arr;
                        for (int i = 0; i < FVectorT_traj::num_scalars_per_row; ++i)
                        {
                            arr[i] = (1 << i);
                        }
                        return IVectorT_traj_1(arr);
                    }();
                    FVectorT_traj_1 inactive_mask =
                        ((IVectorT_traj_1(inactive_bits) & powers_of_two) != 0).template as<FVectorT_traj_1>();

                    // Update valid_mask (combine active, coarse-check, and inactive results)
                    valid_mask = valid_mask & (~inactive_mask);
                    if (valid_mask.none())
                        return;

                    // 3.4 Expanded Check (Splat)
                    FVectorT_traj_1 dummy_dist = FVectorT_traj_1::fill(10.0f);
                    FVectorT_traj_1 is_exp_col = checkSATCollisionBatchTraj(
                        ego_xs_v, ego_ys_v, ego_cos_thetas_v, ego_sin_thetas_v, ex_x, ex_y, ex_cos, ex_sin,
                        ego_bb_extent_x, ego_bb_extent_y, ex_exp_ext_x, ex_exp_ext_y, dummy_dist);

                    is_exp_col = is_exp_col & valid_mask;

                    if (is_exp_col.none())
                        return;

                    // 3.5 Original Check (Splat)
                    FVectorT_traj_1 dist = FVectorT_traj_1::fill(10.0f);
                    FVectorT_traj_1 is_orig_col = checkSATCollisionBatchTraj(
                        ego_xs_v, ego_ys_v, ego_cos_thetas_v, ego_sin_thetas_v, ex_x, ex_y, ex_cos, ex_sin,
                        ego_bb_extent_x, ego_bb_extent_y, ex_orig_ext_x, ex_orig_ext_y, dist);

                    is_orig_col = is_orig_col & is_exp_col;

                    // 3.6 State update (select logic)
                    // mask_found_original is captured by reference outside the lambda and ensures deduplication
                    // (if this candidate was already processed and a collision was found, update_orig will be all
                    // zeros)

                    IVectorT_traj_1 update_orig = is_orig_col.template as<IVectorT_traj_1>() & (~mask_found_original);
                    IVectorT_traj_1 update_exp = (is_exp_col & (~is_orig_col)).template as<IVectorT_traj_1>() &
                                                 (~mask_found_expanded) & (~mask_found_original);

                    FVectorT_traj_1 update_mask = (update_orig | update_exp).template as<FVectorT_traj_1>();

                    if (update_mask.any())
                    {
                        res_idx =
                            IVectorT_traj_1::select(update_mask.template as<IVectorT_traj_1>(), cand_idx, res_idx);
                        res_x = FVectorT_traj_1::select(update_mask, ex_x, res_x);
                        res_y = FVectorT_traj_1::select(update_mask, ex_y, res_y);
                        res_v = FVectorT_traj_1::select(update_mask, ex_v, res_v);
                        res_th = FVectorT_traj_1::select(update_mask, ex_th, res_th);
                        res_c = FVectorT_traj_1::select(update_mask, ex_cos, res_c);
                        res_s = FVectorT_traj_1::select(update_mask, ex_sin, res_s);
                        res_ex_x = FVectorT_traj_1::select(update_mask, ex_exp_ext_x, res_ex_x);
                        res_ex_y = FVectorT_traj_1::select(update_mask, ex_exp_ext_y, res_ex_y);
                        res_dist = FVectorT_traj_1::select(update_mask, dist, res_dist);

                        mask_found_original = mask_found_original | is_orig_col.template as<IVectorT_traj_1>();
                        mask_found_expanded = mask_found_expanded | is_exp_col.template as<IVectorT_traj_1>();
                    }
                };

                // 1. Iterate over regular STRtree candidates (no coarse filtering, STRtree already narrowed by s)
                for (int k = 0; k < candidate_size; ++k)
                {
                    process_agent(candidates[k], false); // enable_coarse_check = false

                    // Check early exit (if all active lanes have found original collisions)
                    if ((mask_found_original | (~row_active_mask.template as<IVectorT_traj_1>())).all())
                        goto write_back;
                }

                // 2. Iterate over "noticed vehicles" (requires coarse filtering: physically close but s might be far)
                for (int idx : utils::NOTICE_VEHICLE_IDXS)
                {
                    process_agent(idx, true); // enable_coarse_check = true

                    // Check Early Exit
                    if ((mask_found_original | (~row_active_mask.template as<IVectorT_traj_1>())).all())
                        goto write_back;
                }

            write_back:
                // 4. Write Back (Point 5 - Movemask)
                if (mask_found_expanded.any())
                {
                    // Store registers to temporary stack arrays
                    alignas(32) float buf_x[FVectorT_traj::num_scalars_per_row],
                        buf_y[FVectorT_traj::num_scalars_per_row], buf_v[FVectorT_traj::num_scalars_per_row];
                    alignas(32) float buf_th[FVectorT_traj::num_scalars_per_row],
                        buf_c[FVectorT_traj::num_scalars_per_row], buf_s[FVectorT_traj::num_scalars_per_row];
                    alignas(32) float buf_ex[FVectorT_traj::num_scalars_per_row],
                        buf_ey[FVectorT_traj::num_scalars_per_row], buf_d[FVectorT_traj::num_scalars_per_row];
                    alignas(32) int32_t buf_idx[FVectorT_traj::num_scalars_per_row];

                    // Bulk store from SIMD vectors to temporary buffers
                    res_idx.to_array(buf_idx);
                    res_x.to_array(buf_x);
                    res_y.to_array(buf_y);
                    res_v.to_array(buf_v);
                    res_th.to_array(buf_th);
                    res_c.to_array(buf_c);
                    res_s.to_array(buf_s);
                    res_ex_x.to_array(buf_ex);
                    res_ex_y.to_array(buf_ey);
                    res_dist.to_array(buf_d);

                    // Process in segments of 8 (standard AVX width).
                    int segment_idx = 0;
                    for (int i = 0; i < FVectorT_traj::num_scalars_per_row; i += 8, ++segment_idx)
                    {
                        // Extract 8-bit mask from the current SIMD segment (8 lanes)
                        uint32_t exp_mask =
                            _mm256_movemask_ps(mask_found_expanded.template as<FVectorT_traj_1>().data[segment_idx]);

                        // Performance Optimization: Skip the entire 8-lane block if no collisions are found
                        if (exp_mask == 0)
                            continue;

                        uint32_t orig_mask =
                            _mm256_movemask_ps(mask_found_original.template as<FVectorT_traj_1>().data[segment_idx]);

                        // Iterate only over the set bits (lanes where collision occurred)
                        while (exp_mask)
                        {
                            // Get index of the lowest set bit using Hardware-accelerated Trailing Zero Count
                            int lane_idx = __builtin_ctz(exp_mask);
                            int current_idx = i + lane_idx;
                            int global_idx = base_idx + current_idx;

                            // Write back expanded collision data
                            has_expanded_collision[global_idx] = 0xFFFFFFFF;
                            collided_exo_idxs[global_idx] = buf_idx[current_idx];
                            collided_exo_xs[global_idx] = buf_x[current_idx];
                            collided_exo_ys[global_idx] = buf_y[current_idx];
                            collided_exo_vs[global_idx] = buf_v[current_idx];
                            collided_exo_thetas[global_idx] = buf_th[current_idx];
                            collided_exo_cos_thetas[global_idx] = buf_c[current_idx];
                            collided_exo_sin_thetas[global_idx] = buf_s[current_idx];
                            collided_exo_expanded_bb_extent_xs[global_idx] = buf_ex[current_idx];
                            collided_exo_expanded_bb_extent_ys[global_idx] = buf_ey[current_idx];
                            collided_min_distances[global_idx] = buf_d[current_idx];

                            // Check if the original (non-expanded) collision also triggered for this lane
                            if (orig_mask & (1 << lane_idx))
                            {
                                has_original_collision[global_idx] = 0xFFFFFFFF;
                            }

                            // Clear the lowest set bit to move to the next collision in the mask
                            exp_mask &= (exp_mask - 1);
                        }
                    }
                }
            }
        }

        void ContextQMDP::FindLeadVehicleIntersectedBatch(
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const size_t &path_size,
            const IVectorT_qmdp &curr_stepped_time_idxs, const IVectorT_qmdp &curr_nearest_idxs,
            const IVectorT_qmdp &ego_ref_path_idxs, const FVectorT_qmdp &ego_vs, float ego_bb_extent_x,
            float ego_bb_extent_y, const AlignedVectorFloat &exo_vs, const AlignedVectorFloat &exo_bb_extent_y,
            const std::vector<AlignedVectorFloat>                    &exo_ss,
            const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
            const FVectorT_qmdp &lateral_offsets, int scenario_num, const IVectorT_qmdp &active_flags,
            const std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags)
        {
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), utils::LEAD_VEHICLE_DEFAULT_SPEED);

            IVectorT_qmdp future_stepped_time_idxs = curr_stepped_time_idxs + utils::LOOK_TIME_STEPS;
            IVectorT_qmdp idxs_valid_mask = (future_stepped_time_idxs < utils::MAX_PRED_TIME_STEPS) & active_flags;

            if (idxs_valid_mask.none())
            {
                return;
            }

            auto ego_ref_path_idxs_arr = ego_ref_path_idxs.to_array();
            auto curr_stepped_time_idxs_arr = curr_stepped_time_idxs.to_array();

            // Track whether each scene has already found a lead vehicle (non-active scenes are treated as already
            // having one)
            IVectorT_qmdp found_lead_vehicle = ~active_flags; // false
            auto          found_lead_vehicle_arr = found_lead_vehicle.to_array();

            // Compute initial ego Frenet s
            FVectorT_qmdp ego_curr_ss =
                curr_nearest_idxs.template convert<FVectorT_qmdp>() * utils::PATH_POINT_INTERVAL;
            auto ego_init_ss_arr = (ego_curr_ss - utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN).to_array();

            // Compute Frenet s and l lengths for each segment
            FVectorT_qmdp max_vel_time_interval_len =
                ego_vs.max(utils::LEAD_VEHICLE_MIN_SEARCH_VELOCITY) * utils::LOOK_TIME_INTERVAL;
            FVectorT_qmdp half_segment_lengths_ss = max_vel_time_interval_len / 2.0f + ego_bb_extent_y;
            FVectorT_qmdp max_ls = lateral_offsets + ego_bb_extent_x;
            FVectorT_qmdp min_ls = lateral_offsets - ego_bb_extent_x;
            auto          min_ls_arr = min_ls.to_array();
            auto          max_ls_arr = max_ls.to_array();

            // Iterate over time steps
            for (size_t t = 1; t <= utils::LOOKAHEAD_STEPS; ++t)
            {
                // Get ego Frenet coordinates along the path
                ego_curr_ss = ego_curr_ss + max_vel_time_interval_len;

                // Compute AABB bounds
                auto min_s_arr = (ego_curr_ss - half_segment_lengths_ss).to_array();
                auto max_s_arr = (ego_curr_ss + half_segment_lengths_ss).to_array();

                // Move all data from SIMD registers into L1 cache/stack
                auto future_stepped_time_idxs_arr = future_stepped_time_idxs.to_array();

                // Combine masks from all SIMD vectors to support num_scalars_per_row > FloatVectorWidth
                uint32_t loop_mask = 0;
                for (size_t v = 0; v < IVectorT_qmdp::num_vectors_per_row; ++v)
                    loop_mask |=
                        (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(idxs_valid_mask.data[v])))
                         << (v * utils::FloatVectorWidth));

                bool any_new_found = false;

                // Collect AABBs to query and their corresponding scenario indices
                while (loop_mask)
                {
                    int i = __builtin_ctz(loop_mask); // Get the actual scenario index

                    uint32_t strtree_idx = getStrtreeIdx(i, future_stepped_time_idxs_arr[i]);

                    // Get the STRtree for the corresponding scenario and time
                    int                             ref_path_idx = ego_ref_path_idxs_arr[i];
                    const std::shared_ptr<STRtree> &strtree = exo_strtrees[ref_path_idx][strtree_idx];

                    int                       candidate_size = 0;
                    const std::array<int, 8> &candidates = strtree->queryBatch<8>(
                        min_s_arr[i], min_ls_arr[i], max_s_arr[i], max_ls_arr[i], candidate_size);

                    // If candidate vehicles are found, iterate over them
                    for (int j = 0; j < candidate_size; ++j)
                    {
                        int candidate_idx = candidates[j];

                        if (exo_inactive_flags[i][candidate_idx])
                        {
                            continue;
                        }

                        // Load the candidate vehicle's current state
                        size_t exo_data_idx = getExoIdxVehicleTime(i, candidate_idx, future_stepped_time_idxs_arr[i]);

                        size_t exo_frenet_data_idx =
                            getExoIdxTimeVehicle(i, curr_stepped_time_idxs_arr[i], candidate_idx);

                        // Consider only the slowest vehicle ahead
                        if (exo_vs[exo_data_idx] < results.second[i] &&
                            exo_ss[ref_path_idx][exo_frenet_data_idx] - ego_init_ss_arr[i] >
                                -utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN)
                        {
                            results.first[i] = exo_ss[ref_path_idx][exo_frenet_data_idx] - ego_init_ss_arr[i] -
                                               exo_bb_extent_y[candidate_idx] - ego_bb_extent_y;
                            results.second[i] = exo_vs[exo_data_idx];
                            found_lead_vehicle_arr[i] = 0xFFFFFFFF;
                            any_new_found = true;
                        }
                    }

                    loop_mask &= (loop_mask - 1);
                }

                if (any_new_found)
                {
                    found_lead_vehicle = IVectorT_qmdp(found_lead_vehicle_arr.data());
                }

                // Stop if the time horizon is exceeded, or if all scenes have found a lead vehicle
                future_stepped_time_idxs = future_stepped_time_idxs + utils::LOOK_TIME_STEPS;
                idxs_valid_mask = (future_stepped_time_idxs < utils::MAX_PRED_TIME_STEPS) & ~found_lead_vehicle;

                if (idxs_valid_mask.none())
                    break;
            }
        }

        void ContextQMDP::FindLeadVehicleIntersectedBatch(std::pair<AlignedVectorFloat, AlignedVectorFloat> &results,
                                                          const size_t &path_size, const size_t &curr_stepped_time_idx,
                                                          const IVectorT_traj &path_nearest_idxs,
                                                          const FVectorT_traj &ego_vs, float ego_bb_extent_x,
                                                          float ego_bb_extent_y, const AlignedVectorFloat &exo_vs,
                                                          const AlignedVectorFloat                    &exo_bb_extent_y,
                                                          const AlignedVectorFloat                    &exo_ss,
                                                          const std::vector<std::shared_ptr<STRtree>> &exo_strtrees,
                                                          const FVectorT_traj &lateral_offsets, int scenario_num)
        {
            // Result arrays: store per-scenario lead-vehicle info (relative distance, speed)
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), utils::LEAD_VEHICLE_DEFAULT_SPEED);

            size_t future_stepped_time_idx = curr_stepped_time_idx + utils::LOOK_TIME_STEPS;
            if (future_stepped_time_idx >= utils::PROPOSAL_TRAJECTORY_SIZE)
                return;

            // Track whether each scene has already found a lead vehicle (non-active scenes are treated as already
            // having one)
            IVectorT_traj not_found_lead_vehicle = IVectorT_traj::fill(0xFFFFFFFF); // true
            auto          not_found_lead_vehicle_arr = not_found_lead_vehicle.to_array();

            // Compute initial ego Frenet s
            FVectorT_traj ego_curr_ss =
                path_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
            auto ego_init_ss_arr = (ego_curr_ss - utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN).to_array();

            // Compute Frenet s and l lengths for each segment
            FVectorT_traj max_vel_time_interval_len =
                ego_vs.max(utils::LEAD_VEHICLE_MIN_SEARCH_VELOCITY) * utils::LOOK_TIME_INTERVAL;
            FVectorT_traj half_segment_lengths_ss = max_vel_time_interval_len / 2.0f + ego_bb_extent_y;
            FVectorT_traj max_ls = lateral_offsets + ego_bb_extent_x;
            FVectorT_traj min_ls = lateral_offsets - ego_bb_extent_x;
            auto          min_ls_arr = min_ls.to_array();
            auto          max_ls_arr = max_ls.to_array();

            // Iterate over time steps
            for (size_t t = 1; t <= utils::LOOKAHEAD_STEPS; ++t)
            {
                // Get ego Frenet coordinates along the path
                ego_curr_ss = ego_curr_ss + max_vel_time_interval_len;

                // Compute AABB bounds
                auto min_ss_arr = (ego_curr_ss - half_segment_lengths_ss).to_array();
                auto max_ss_arr = (ego_curr_ss + half_segment_lengths_ss).to_array();

                bool any_new_found = false;

                // Use bit masks to iterate over active scenes
                for (size_t row_idx = 0; row_idx < IVectorT_traj::num_rows; ++row_idx)
                {
                    // Combine masks from all SIMD vectors in this row to support num_scalars_per_row > FloatVectorWidth
                    uint32_t loop_mask = 0;
                    for (size_t v = 0; v < IVectorT_traj::num_vectors_per_row; ++v)
                        loop_mask |= (static_cast<uint32_t>(_mm256_movemask_ps(
                                          _mm256_castsi256_ps(not_found_lead_vehicle[row_idx].data[v])))
                                      << (v * utils::FloatVectorWidth));

                    while (loop_mask)
                    {
                        int lane_idx = __builtin_ctz(loop_mask);
                        int i = row_idx * IVectorT_traj::num_scalars_per_row + lane_idx;

                        uint32_t strtree_idx = getStrtreeIdxProposal(lane_idx, future_stepped_time_idx);

                        // Get STRtree for the corresponding scenario and time
                        const std::shared_ptr<STRtree> &strtree = exo_strtrees[strtree_idx];

                        int                       candidate_size = 0;
                        const std::array<int, 8> &candidates = strtree->queryBatch<8>(
                            min_ss_arr[i], min_ls_arr[i], max_ss_arr[i], max_ls_arr[i], candidate_size);

                        // Iterate over candidate vehicles
                        for (int j = 0; j < candidate_size; ++j)
                        {
                            int candidate_idx = candidates[j];

                            // Load candidate vehicle state
                            size_t exo_data_idx =
                                getExoIdxVehicleTimeProposal(lane_idx, candidate_idx, future_stepped_time_idx);

                            size_t exo_frenet_data_idx =
                                getExoIdxTimeVehicleProposal(lane_idx, curr_stepped_time_idx, candidate_idx);

                            // Consider only the slowest vehicle ahead
                            if (exo_vs[exo_data_idx] < results.second[i] &&
                                exo_ss[exo_frenet_data_idx] - ego_init_ss_arr[i] >
                                    -utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN)
                            {
                                results.first[i] = exo_ss[exo_frenet_data_idx] - ego_init_ss_arr[i] -
                                                   exo_bb_extent_y[candidate_idx] - ego_bb_extent_y;
                                results.second[i] = exo_vs[exo_data_idx];
                                not_found_lead_vehicle_arr[i] = 0;
                                any_new_found = true;
                            }
                        }

                        loop_mask &= (loop_mask - 1);
                    }
                }

                if (any_new_found)
                {
                    not_found_lead_vehicle = IVectorT_traj(not_found_lead_vehicle_arr.data());
                }

                future_stepped_time_idx = future_stepped_time_idx + utils::LOOK_TIME_STEPS;

                if (future_stepped_time_idx >= utils::PROPOSAL_TRAJECTORY_SIZE || not_found_lead_vehicle.none())
                {
                    break;
                }
            }
        }

        // for LC proposal only
        void ContextQMDP::FindLeadVehicleIntersectedBatch(
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const size_t &path_size,
            const size_t &curr_stepped_time_idx, const IVectorT_traj &path_nearest_idxs,
            const IVectorT_traj &ego_ref_path_idxs, // 0: init path; 1: target path;
            const FVectorT_traj &ego_vs, float ego_bb_extent_x, float ego_bb_extent_y, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_bb_extent_y, const std::vector<AlignedVectorFloat> &exo_ss,
            const std::vector<std::vector<std::shared_ptr<STRtree>>> &exo_strtrees,
            const FVectorT_traj &lateral_offsets, int scenario_num, const IVectorT_traj &active_flags)
        {
            // Result arrays: store per-scenario lead-vehicle info (relative distance, speed)
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), utils::LEAD_VEHICLE_DEFAULT_SPEED);

            size_t future_stepped_time_idx = curr_stepped_time_idx + utils::LOOK_TIME_STEPS;
            if (future_stepped_time_idx >= utils::PROPOSAL_TRAJECTORY_SIZE || active_flags.none())
                return;

            // Track whether each scenario has already found a lead vehicle (non-active scenarios are treated as already
            // having a lead vehicle)
            IVectorT_traj not_found_lead_vehicle = active_flags; // true
            auto          not_found_lead_vehicle_arr = not_found_lead_vehicle.to_array();

            auto ego_ref_path_idxs_arr = ego_ref_path_idxs.to_array();

            // Compute the ego vehicle's initial frenet_s
            FVectorT_traj ego_curr_ss =
                path_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
            auto ego_init_ss_arr = (ego_curr_ss - utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN).to_array();

            // Compute the frenet_s and frenet_l length of each segment
            FVectorT_traj max_vel_time_interval_len =
                ego_vs.max(utils::LEAD_VEHICLE_MIN_SEARCH_VELOCITY) * utils::LOOK_TIME_INTERVAL;
            FVectorT_traj half_segment_lengths_ss = max_vel_time_interval_len / 2.0f + ego_bb_extent_y;
            FVectorT_traj max_ls = lateral_offsets + ego_bb_extent_x;
            FVectorT_traj min_ls = lateral_offsets - ego_bb_extent_x;
            auto          min_ls_arr = min_ls.to_array();
            auto          max_ls_arr = max_ls.to_array();

            // Iterate over each time step
            for (size_t t = 1; t <= utils::LOOKAHEAD_STEPS; ++t)
            {
                // Get the ego vehicle's Frenet coordinates on the path
                ego_curr_ss = ego_curr_ss + max_vel_time_interval_len;

                // Compute the AABB boundaries
                auto min_ss_arr = (ego_curr_ss - half_segment_lengths_ss).to_array();
                auto max_ss_arr = (ego_curr_ss + half_segment_lengths_ss).to_array();

                bool any_new_found = false;

                // Batch query the STRtree for each scenario
                for (size_t row_idx = 0; row_idx < IVectorT_traj::num_rows; ++row_idx)
                {
                    // Combine masks from all SIMD vectors in this row to support num_scalars_per_row > FloatVectorWidth
                    uint32_t loop_mask = 0;
                    for (size_t v = 0; v < IVectorT_traj::num_vectors_per_row; ++v)
                        loop_mask |= (static_cast<uint32_t>(_mm256_movemask_ps(
                                          _mm256_castsi256_ps(not_found_lead_vehicle[row_idx].data[v])))
                                      << (v * utils::FloatVectorWidth));

                    while (loop_mask)
                    {
                        int lane_idx = __builtin_ctz(loop_mask);
                        int i = row_idx * IVectorT_traj::num_scalars_per_row + lane_idx;

                        uint32_t strtree_idx = getStrtreeIdxProposal(lane_idx, future_stepped_time_idx);
                        const std::shared_ptr<STRtree> &strtree = exo_strtrees[ego_ref_path_idxs_arr[i]][strtree_idx];

                        // Query potential collision objects
                        int                       candidate_size = 0;
                        const std::array<int, 8> &candidates = strtree->queryBatch<8>(
                            min_ss_arr[i], min_ls_arr[i], max_ss_arr[i], max_ls_arr[i], candidate_size);

                        // Iterate over all candidate vehicles
                        for (int j = 0; j < candidate_size; ++j)
                        {
                            int candidate_idx = candidates[j];

                            // Get the current state of the candidate vehicle
                            size_t exo_data_idx =
                                getExoIdxVehicleTimeProposal(lane_idx, candidate_idx, future_stepped_time_idx);

                            size_t exo_frenet_data_idx =
                                getExoIdxTimeVehicleProposal(lane_idx, curr_stepped_time_idx, candidate_idx);

                            // Only consider the slowest vehicle in front
                            if (exo_vs[exo_data_idx] < results.second[i] &&
                                exo_ss[ego_ref_path_idxs_arr[i]][exo_frenet_data_idx] - ego_init_ss_arr[i] >
                                    -utils::LEAD_VEHICLE_SEARCH_REAR_MARGIN)
                            {
                                results.first[i] = exo_ss[ego_ref_path_idxs_arr[i]][exo_frenet_data_idx] -
                                                   ego_init_ss_arr[i] - exo_bb_extent_y[candidate_idx] -
                                                   ego_bb_extent_y;
                                results.second[i] = exo_vs[exo_data_idx];
                                not_found_lead_vehicle_arr[i] = 0;
                                any_new_found = true;
                            }
                        }

                        loop_mask &= (loop_mask - 1);
                    }
                }

                if (any_new_found)
                {
                    not_found_lead_vehicle = IVectorT_traj(not_found_lead_vehicle_arr.data());
                }

                future_stepped_time_idx = future_stepped_time_idx + utils::LOOK_TIME_STEPS;
                if (future_stepped_time_idx >= utils::PROPOSAL_TRAJECTORY_SIZE || not_found_lead_vehicle.none())
                {
                    break;
                }
            }
        }

        // 12x8 large matrix
        // When each ego agent follows a different path, comparisons should use ego progress on the target path, and exo
        // information should also be collected based on that target path
        template <bool FIND_LEAD_VEHICLE>
        void ContextQMDP::FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, const IVectorT_qmdp &start_time_idxs,
            const IVectorT_qmdp &curr_nearest_idxs, const IVectorT_qmdp &ego_ref_path_idxs,
            const FVectorT_qmdp &ego_bb_extent_xs, const FVectorT_qmdp &ego_bb_extent_ys,
            const std::vector<AlignedVectorFloat> &exo_ss, const std::vector<AlignedVectorFloat> &exo_ls,
            const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
            const FVectorT_qmdp &lateral_offsets, int scenario_num, const IVectorT_qmdp &active_flags,
            const std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags)
        {
            // Result arrays, storing per-scenario lead-vehicle information (relative distance, speed)
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), 0.0f);

            auto exo_bb_extent_y_v = FVectorT_12(exo_bb_extent_y.data());

            // Compute current ego progress
            FVectorT_qmdp ego_progress_vec =
                curr_nearest_idxs.template convert<FVectorT_qmdp>() * utils::PATH_POINT_INTERVAL;

            // Iterate over all scenes
            for (int i = 0; i < scenario_num; ++i)
            {
                // Get current scene index
                const auto pair_idx = utils::div_mod<size_t>(i, vector_qmdp_width);

                // Skip invalid scenes
                if (!active_flags[pair_idx])
                {
                    continue;
                }

                // Get exo information for the current reference path index
                const auto &exo_ss_ref = exo_ss[ego_ref_path_idxs[pair_idx]];
                const auto &exo_ls_ref = exo_ls[ego_ref_path_idxs[pair_idx]];
                const auto &exo_ls_projected_radius_ref = exo_ls_projected_radius[ego_ref_path_idxs[pair_idx]];
                auto        lateral_offset = lateral_offsets[pair_idx];
                auto        ego_bb_extent_x = ego_bb_extent_xs[pair_idx];
                auto        ego_bb_extent_y = ego_bb_extent_ys[pair_idx];

                float ego_progress = ego_progress_vec[pair_idx];

                // Collect exo_ss and exo_vs for the current scene and time step
                size_t start_idx = getExoIdxTimeVehicle(i, start_time_idxs[pair_idx], 0);

                // Use gather to read multiple child-node boundary data in one shot
                auto delta_exo_ss_v = FVectorT_12::load_contiguous(exo_ss_ref.data(), start_idx) - ego_progress;
                auto relative_diff_l_v_abs =
                    (FVectorT_12::load_contiguous(exo_ls_ref.data(), start_idx) - lateral_offset).abs();
                auto exo_ls_projected_radius_v =
                    FVectorT_12::load_contiguous(exo_ls_projected_radius_ref.data(), start_idx);
                auto actual_delta_exo_ss_v = delta_exo_ss_v.abs() - exo_bb_extent_y_v - ego_bb_extent_y;
                auto modified_ego_extent_xs =
                    FVectorT_12::select(actual_delta_exo_ss_v < 0.0f, utils::EGO_BB_EXTENT_X, ego_bb_extent_x);
                auto inactive_mask = IVectorT_12(exo_inactive_flags[i].data()).template as<FVectorT_12>();

                // Use SIMD to check for intersection
                FVectorT_12 intersects_v;
                if constexpr (FIND_LEAD_VEHICLE)
                {
                    intersects_v = (delta_exo_ss_v >= 0.0f) &
                                   (relative_diff_l_v_abs <= exo_ls_projected_radius_v + modified_ego_extent_xs +
                                                                 utils::AGENT_DETECTION_MARGIN) &
                                   ~inactive_mask;
                }
                else
                {
                    intersects_v = (delta_exo_ss_v < 0.0f) &
                                   (relative_diff_l_v_abs <= exo_ls_projected_radius_v + modified_ego_extent_xs +
                                                                 utils::AGENT_DETECTION_MARGIN) &
                                   ~inactive_mask;
                }

                FVectorT_12 actual_delta_exo_ss_v_clamped =
                    FVectorT_12::select(intersects_v, actual_delta_exo_ss_v, utils::MAX_VALUE);

                // 6. Horizontal reduction: get the actual minimum distance for the current scene (single serialized
                // pass) This step compares all distance elements inside the SIMD vector and produces the final scalar
                // minimum value.

                float current_min_dist = actual_delta_exo_ss_v_clamped.hmin();

                // 7. Update the result arrays
                if (current_min_dist < results.first[i])
                {
                    results.first[i] = current_min_dist;

                    // 8. Scalar search: find the speed corresponding to the minimum distance
                    // This step is necessary because SIMD reduction for the minimum value cannot efficiently return the
                    // associated index. We iterate over all vehicles again, only checking those whose distance equals
                    // the minimum distance, and retrieve their speeds.

                    // Broadcast the minimum value for comparison
                    FVectorT_1 target_val_v = FVectorT_1::fill(current_min_dist);

#pragma unroll
                    for (int batch_idx = 0; batch_idx < 12; ++batch_idx)
                    {
                        // Take the batch_idx-th 256-bit vector
                        // Note: directly read the precomputed clamped result here to leverage the L1 cache
                        const auto &dist_vec = actual_delta_exo_ss_v_clamped[batch_idx];

                        // SIMD comparison: check which float equals the minimum value
                        auto match_mask = (dist_vec == target_val_v);

                        if (match_mask.any())
                        {
                            // 1. Extract mask bits (8 bits)
                            uint32_t bitmask = _mm256_movemask_ps(match_mask.data[0]);

                            // 2. Find the offset (0-7) of the first bit that is 1
                            // __builtin_ctz is a single-cycle instruction
                            int lane_idx = __builtin_ctz(bitmask);

                            // 3. Compute the global relative index
                            // batch_idx * 8 is the base, lane_idx is the offset
                            results.second[i] =
                                exo_vs[getExoIdxVehicleTime(i, (batch_idx * 8) + lane_idx, start_time_idxs[pair_idx])];

                            // Exit as soon as we find it
                            break;
                        }
                    }
                }
            }
        }

        // Consider the case where each ego agent follows the same path (LF proposal)
        template <bool FIND_LEAD_VEHICLE>
        void ContextQMDP::FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, size_t start_time_idx,
            const IVectorT_traj &path_nearest_idxs, float ego_bb_extent_x, float ego_bb_extent_y,
            const AlignedVectorFloat &exo_ss, const AlignedVectorFloat &exo_ls,
            const AlignedVectorFloat &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
            const FVectorT_traj &lateral_offsets, int scenario_num, const FVectorT_12 &exo_active_flags)
        {
            // Result arrays: store per-scenario lead-vehicle info (relative distance, speed)
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), 0.0f);

            // Precompute ego progress (SIMD matrix operations remain unchanged)
            // ego_progress_vec is also a 9x8 structure
            FVectorT_traj ego_progress_vec =
                path_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
            auto ego_progress_vec_arr = ego_progress_vec.to_array();
            auto lateral_offsets_arr = lateral_offsets.to_array();

            auto exo_bb_extent_y_v = FVectorT_12(exo_bb_extent_y.data());

            constexpr int NUM_ROWS = FVectorT_traj::num_rows;
            constexpr int NUM_WIDTH = FVectorT_traj::num_scalars_per_row;

            // Iterate over each scene
            for (int s = 0; s < utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD; ++s)
            {
                // ==================================================================
                // Phase A: scene-level data preloading (the heaviest part, now done only 8 times)
                // ==================================================================

                size_t exo_start_idx = getExoIdxTimeVehicleProposal(s, start_time_idx, 0);

                // These variables stay in registers and are reused by the inner ego loop
                auto exo_ss_v = FVectorT_12::load_contiguous(exo_ss.data(), exo_start_idx);
                auto exo_ls_v = FVectorT_12::load_contiguous(exo_ls.data(), exo_start_idx);
                auto exo_ls_radius_v = FVectorT_12::load_contiguous(exo_ls_projected_radius.data(), exo_start_idx);

                // ==================================================================
                // Phase B: iterate over all ego vehicles that belong to this scene (iterate over 9 rows)
                // ==================================================================
                // Matrix structure FVectorT_traj (9 rows x 8 columns), column s corresponds to scene s
                for (int row = 0; row < NUM_ROWS; ++row)
                {
                    // Compute the global index
                    int global_ego_idx = row * NUM_WIDTH + s;

                    // --------------------------------------------------------------
                    // 1. Scalar extraction
                    // Extract the scalar value belonging to the current scene s from the SIMD vector
                    // --------------------------------------------------------------

                    // Note: operator[] returns a SIMD vector; we need the s-th element
                    // Direct memory access is usually faster than _mm256_extract_ps because the data is in the L1 cache
                    float ego_progress = ego_progress_vec_arr[global_ego_idx];
                    float lateral_offset = lateral_offsets_arr[global_ego_idx];

                    // --------------------------------------------------------------
                    // 2. Compute the logic (reuse exo data stored in registers)
                    // --------------------------------------------------------------
                    auto delta_exo_ss_v = exo_ss_v - ego_progress;
                    auto relative_diff_l_v_abs = (exo_ls_v - lateral_offset).abs();

                    auto actual_delta_exo_ss_v = delta_exo_ss_v.abs() - exo_bb_extent_y_v - ego_bb_extent_y;
                    auto modified_ego_extent_xs =
                        FVectorT_12::select(actual_delta_exo_ss_v < 0.0f, utils::EGO_BB_EXTENT_X, ego_bb_extent_x);

                    FVectorT_12 intersects_v;
                    if constexpr (FIND_LEAD_VEHICLE)
                    {
                        intersects_v = (delta_exo_ss_v >= 0.0f) &
                                       (relative_diff_l_v_abs <=
                                        exo_ls_radius_v + modified_ego_extent_xs + utils::AGENT_DETECTION_MARGIN) &
                                       exo_active_flags;
                    }
                    else
                    {
                        intersects_v = (delta_exo_ss_v < 0.0f) &
                                       (relative_diff_l_v_abs <=
                                        exo_ls_radius_v + modified_ego_extent_xs + utils::AGENT_DETECTION_MARGIN) &
                                       exo_active_flags;
                    }

                    auto actual_delta_exo_ss_v_clamped =
                        FVectorT_12::select(intersects_v, actual_delta_exo_ss_v, utils::MAX_VALUE);

                    // --------------------------------------------------------------
                    // 3. Reduction and update (keep the bit-scan optimization)
                    // --------------------------------------------------------------
                    float current_min_dist = actual_delta_exo_ss_v_clamped.hmin();

                    if (current_min_dist < results.first[global_ego_idx])
                    {
                        results.first[global_ego_idx] = current_min_dist;

                        FVectorT_1 target_val_v = FVectorT_1::fill(current_min_dist);

#pragma unroll
                        for (int batch_idx = 0; batch_idx < 12; ++batch_idx)
                        {
                            const auto &dist_vec = actual_delta_exo_ss_v_clamped[batch_idx];
                            auto        match_mask = (dist_vec == target_val_v);

                            if (match_mask.any())
                            {
                                // 1. Extract mask bits (8 bits)
                                uint32_t bitmask = _mm256_movemask_ps(match_mask.data[0]);

                                // 2. Find the offset (0-7) of the first bit that is 1
                                int lane_idx = __builtin_ctz(bitmask);

                                // Get the speed
                                results.second[global_ego_idx] =
                                    exo_vs[getExoIdxVehicleTimeProposal(s, (batch_idx * 8) + lane_idx, start_time_idx)];

                                // Exit as soon as we find it
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Consider the case where each ego agent follows different paths.
        // Comparisons should use ego progress on the target path, and exo information should also be collected based on
        // that target path (LC proposal). ego_ref_path_idxs is either all 1 (CHECK_CURRENT_PATH = false), or mixed
        // (CHECK_CURRENT_PATH = true). When CHECK_CURRENT_PATH is true, active_flags is all 0xFFFFFFFF; otherwise it is
        // mixed.
        template <bool FIND_LEAD_VEHICLE, bool CHECK_CURRENT_PATH>
        void ContextQMDP::FindLeadOrFollowVehicleCloseToEgoReferencePathBatch(
            std::pair<AlignedVectorFloat, AlignedVectorFloat> &results, size_t start_time_idx,
            const IVectorT_traj &curr_nearest_idxs, const IVectorT_traj &ego_ref_path_idxs, float ego_bb_extent_x,
            float ego_bb_extent_y, const std::vector<AlignedVectorFloat> &exo_ss,
            const std::vector<AlignedVectorFloat> &exo_ls,
            const std::vector<AlignedVectorFloat> &exo_ls_projected_radius, const AlignedVectorFloat &exo_vs,
            const AlignedVectorFloat &exo_bb_extent_x, const AlignedVectorFloat &exo_bb_extent_y,
            const FVectorT_traj &lateral_offsets, int scenario_num, const IVectorT_traj &active_flags,
            const FVectorT_12 &exo_active_flags)
        {
            // Result arrays: store lead-vehicle info per scene (relative distance, speed)
            std::fill(results.first.begin(), results.first.end(), utils::MAX_VALUE);
            std::fill(results.second.begin(), results.second.end(), 0.0f);

            // Precompute ego progress (SIMD matrix operations remain unchanged)
            // ego_progress_vec is also a 9x8 structure
            FVectorT_traj ego_progress_vec =
                curr_nearest_idxs.template convert<FVectorT_traj>() * utils::PATH_POINT_INTERVAL;
            auto ego_progress_vec_arr = ego_progress_vec.to_array();
            auto lateral_offsets_arr = lateral_offsets.to_array();

            auto exo_bb_extent_y_v = FVectorT_12(exo_bb_extent_y.data());

            // Assume FVectorT_traj has a num_rows() method or is known to be 9
            constexpr int NUM_ROWS = FVectorT_traj::num_rows;
            constexpr int NUM_WIDTH = FVectorT_traj::num_scalars_per_row;

            // ========================================================================
            // 1. Pre-computation phase
            // Before entering the s loop, convert SIMD data into a scalar-friendly format
            // ========================================================================

            // This array stores row-filtering masks
            // Case A (current path): store the "Is_Path_1" mask (Active is all ones, so it does not need to be stored)
            // Case B (target path): store the "Is_Active" mask (Path is all ones, so it does not need to be stored)
            std::array<uint32_t, NUM_ROWS> row_filtering_masks;

            // Global demand masks: bit s being 1 means scene s needs data for the corresponding path
            uint32_t global_needs_path0 = 0;
            uint32_t global_needs_path1 = 0;

            for (int row = 0; row < NUM_ROWS; ++row)
            {
                if constexpr (CHECK_CURRENT_PATH)
                {
                    // --- Current path mode (mixed paths, all active) ---
                    // We only need to compute the path mask
                    // Assume ego_ref_path_idxs[row] == 1 generates the mask
                    // Here we assume active_flags is all ones and never read it, saving memory bandwidth
                    // Combine masks from all SIMD vectors in this row to support num_scalars_per_row > FloatVectorWidth
                    uint32_t p1_mask = 0;
                    for (size_t v = 0; v < IVectorT_traj::num_vectors_per_row; ++v)
                        p1_mask |= (static_cast<uint32_t>(
                                        _mm256_movemask_ps(_mm256_castsi256_ps((ego_ref_path_idxs[row] == 1).data[v])))
                                    << (v * utils::FloatVectorWidth));

                    row_filtering_masks[row] = p1_mask;

                    // Aggregated demand: path 1 is needed where p1_mask is true; path 0 is needed where p1_mask is
                    // false
                    global_needs_path1 |= p1_mask;
                    global_needs_path0 |= (~p1_mask);
                }
                else
                {
                    // --- Target path mode (path 1 only, mixed active) ---
                    // We only need to compute the active mask
                    // Assume ego_ref_path_idxs is all ones and never read it
                    // Combine masks from all SIMD vectors in this row to support num_scalars_per_row > FloatVectorWidth
                    uint32_t act_mask = 0;
                    for (size_t v = 0; v < IVectorT_traj::num_vectors_per_row; ++v)
                        act_mask |=
                            (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(active_flags[row].data[v])))
                             << (v * utils::FloatVectorWidth));

                    row_filtering_masks[row] = act_mask;

                    // Aggregated demand: only path 1 may be needed
                    global_needs_path1 |= act_mask;
                    // global_needs_path0 stays at its default value 0
                }
            }

            // ========================================================================
            // Core Calculation Lambda
            // ========================================================================
            auto process_ego_logic = [&](int row, int s, const FVectorT_12 &exo_ss_v, const FVectorT_12 &exo_ls_v,
                                         const FVectorT_12 &exo_ls_radius_v) __attribute__((always_inline))
            {
                int global_ego_idx = row * NUM_WIDTH + s;

                float ego_progress = ego_progress_vec_arr[global_ego_idx];
                float lateral_offset = lateral_offsets_arr[global_ego_idx];

                auto delta_exo_ss_v = exo_ss_v - ego_progress;
                auto relative_diff_l_v_abs = (exo_ls_v - lateral_offset).abs();
                auto actual_delta_exo_ss_v = delta_exo_ss_v.abs() - exo_bb_extent_y_v - ego_bb_extent_y;
                auto modified_ego_extent_xs =
                    FVectorT_12::select(actual_delta_exo_ss_v < 0.0f, utils::EGO_BB_EXTENT_X, ego_bb_extent_x);

                // Use SIMD operations to check for intersection
                FVectorT_12 intersects_v;
                if constexpr (FIND_LEAD_VEHICLE)
                {
                    intersects_v = (delta_exo_ss_v >= 0.0f) &
                                   (relative_diff_l_v_abs <=
                                    exo_ls_radius_v + modified_ego_extent_xs + utils::AGENT_DETECTION_MARGIN) &
                                   exo_active_flags;
                }
                else
                {
                    intersects_v = (delta_exo_ss_v < 0.0f) &
                                   (relative_diff_l_v_abs <=
                                    exo_ls_radius_v + modified_ego_extent_xs + utils::AGENT_DETECTION_MARGIN) &
                                   exo_active_flags;
                }

                auto actual_delta_exo_ss_v_clamped =
                    FVectorT_12::select(intersects_v, actual_delta_exo_ss_v, utils::MAX_VALUE);

                // Horizontal reduction: get the actual minimum distance for the current scene
                float current_min_dist = actual_delta_exo_ss_v_clamped.hmin();

                if (current_min_dist < results.first[global_ego_idx])
                {
                    results.first[global_ego_idx] = current_min_dist;

                    FVectorT_1 target_val_v = FVectorT_1::fill(current_min_dist);

#pragma unroll
                    for (int batch_idx = 0; batch_idx < 12; ++batch_idx)
                    {
                        // Take the batch_idx-th 256-bit vector
                        const auto &dist_vec = actual_delta_exo_ss_v_clamped[batch_idx];

                        // SIMD comparison: check which float equals the minimum value
                        auto match_mask = (dist_vec == target_val_v);

                        if (match_mask.any())
                        {
                            uint32_t bitmask = _mm256_movemask_ps(match_mask.data[0]);
                            int      lane_idx = __builtin_ctz(bitmask);
                            results.second[global_ego_idx] =
                                exo_vs[getExoIdxVehicleTimeProposal(s, (batch_idx * 8) + lane_idx, start_time_idx)];
                            break;
                        }
                    }
                }
            };

            // ========================================================================
            // 2. Loop phase
            // ========================================================================
            for (int s = 0; s < utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD; ++s)
            {
                uint32_t s_bit = (1 << s);

                // --------------------------------------------------------------------
                // Block Path 0 (Only runs in LC mode)
                // --------------------------------------------------------------------
                if constexpr (CHECK_CURRENT_PATH)
                {
                    if (global_needs_path0 & s_bit)
                    {
                        size_t start_idx = getExoIdxTimeVehicleProposal(s, start_time_idx, 0);
                        auto   exo_ss_v = FVectorT_12::load_contiguous(exo_ss[0].data(), start_idx);
                        auto   exo_ls_v = FVectorT_12::load_contiguous(exo_ls[0].data(), start_idx);
                        auto   exo_ls_radius_v =
                            FVectorT_12::load_contiguous(exo_ls_projected_radius[0].data(), start_idx);

                        for (int row = 0; row < NUM_ROWS; ++row)
                        {
                            // Logic: in current-path mode, row_filtering_masks stores "Is Path 1"
                            // Therefore we are looking for "Not Path 1" (i.e., path 0)
                            // Since Active is always true, there is no need to check Active
                            if ((~row_filtering_masks[row]) & s_bit)
                            {
                                process_ego_logic(row, s, exo_ss_v, exo_ls_v, exo_ls_radius_v);
                            }
                        }
                    }
                }

                // --------------------------------------------------------------------
                // Block Path 1 (Runs in both modes, but logic differs slightly)
                // --------------------------------------------------------------------
                if (global_needs_path1 & s_bit)
                {
                    size_t start_idx = getExoIdxTimeVehicleProposal(s, start_time_idx, 0);
                    auto   exo_ss_v = FVectorT_12::load_contiguous(exo_ss[1].data(), start_idx);
                    auto   exo_ls_v = FVectorT_12::load_contiguous(exo_ls[1].data(), start_idx);
                    auto   exo_ls_radius_v = FVectorT_12::load_contiguous(exo_ls_projected_radius[1].data(), start_idx);

                    for (int row = 0; row < NUM_ROWS; ++row)
                    {
                        if constexpr (CHECK_CURRENT_PATH)
                        {
                            // Current-path mode: row_filtering_masks is "Is Path 1"
                            // Check whether it is path 1 (Active is implicitly true)
                            if (row_filtering_masks[row] & s_bit)
                            {
                                process_ego_logic(row, s, exo_ss_v, exo_ls_v, exo_ls_radius_v);
                            }
                        }
                        else
                        {
                            // Target-path mode: row_filtering_masks is "Is Active"
                            // Check whether it is active (Path is implicitly 1)
                            if (row_filtering_masks[row] & s_bit)
                            {
                                process_ego_logic(row, s, exo_ss_v, exo_ls_v, exo_ls_radius_v);
                            }
                        }
                    }
                }
            }
        }

        bool ContextQMDP::LCAllowanceCheckSerial(
            const std::pair<float, float> &LC_lead_vehicles_close_to_ego_reference_path,
            const std::pair<float, float> &LC_following_vehicles_close_to_ego_reference_path, float ego_v,
            float ego_desired_v, bool active_flag)
        {
            if (!active_flag)
            {
                return false;
            }

            // 1. Check the lead vehicle on the target lane; if the target-lane acceleration is greater than the comfort
            // deceleration threshold, lane change is not allowed
            float LC_acc = IDMSerial(LC_lead_vehicles_close_to_ego_reference_path.second, ego_v,
                                     LC_lead_vehicles_close_to_ego_reference_path.first, ego_desired_v);

            if (LC_acc <= utils::COMFORT_DEC)
            {
                return false;
            }

            // 2. Check the following vehicle on the target lane; if the target-lane following-vehicle acceleration is
            // greater than the comfort deceleration threshold, lane change is not allowed
            float LC_follow_acc = IDMSerial(ego_v, LC_following_vehicles_close_to_ego_reference_path.second,
                                            LC_following_vehicles_close_to_ego_reference_path.first +
                                                utils::LC_FOLLOW_DISTANCE_COMPENSATION,
                                            ego_desired_v);

            if (LC_follow_acc <= utils::COMFORT_DEC)
            {
                return false;
            }

            return true;
        }

        IVectorT_qmdp ContextQMDP::LCAllowanceCheckBatch(
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path,
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path,
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_following_vehicles_close_to_ego_reference_path,
            const FVectorT_qmdp &ego_vs, const FVectorT_qmdp &ego_thetas, const FVectorT_qmdp &ego_desired_vs,
            const FVectorT_qmdp &ego_vs_lateral, const IVectorT_qmdp &active_flags,
            const IVectorT_qmdp &consider_left_LC_mask, const IVectorT_qmdp &consider_right_LC_mask,
            const FVectorT_qmdp &ego_curr_path_ls, const FVectorT_qmdp &ego_curr_path_thetas,
            IVectorT_qmdp &LF_lead_allowance_flags)
        {
            // If there are no active scenarios, directly return false
            if (active_flags.none())
            {
                return active_flags;
            }

            // Initialize lane-change allowance flags; default is allowed
            IVectorT_qmdp allowance_flags = active_flags;

            FVectorT_qmdp LF_lead_vehicles_relative_distance =
                FVectorT_qmdp(LF_lead_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_qmdp LF_lead_vehicles_vs =
                FVectorT_qmdp(LF_lead_vehicles_close_to_ego_reference_path.second.data());

            // Compute IDM acceleration for the ego vehicle on the current lane (to decide whether a lane change is
            // needed)
            FVectorT_qmdp LF_acc =
                IDMBatch(LF_lead_vehicles_vs, ego_vs, LF_lead_vehicles_relative_distance, ego_desired_vs);

            // Compute the time headway for the ego vehicle on the current lane
            FVectorT_qmdp LF_time_headway = LF_lead_vehicles_relative_distance / (ego_vs - LF_lead_vehicles_vs);

            // Determine whether the vehicle is currently in a lane change (LC)
            FVectorT_qmdp theta_diff = utils::NormalizeAngleSIMD<FVectorT_qmdp>(ego_thetas - ego_curr_path_thetas);

            FVectorT_qmdp left_mask = consider_left_LC_mask.template as<FVectorT_qmdp>();
            FVectorT_qmdp right_mask = consider_right_LC_mask.template as<FVectorT_qmdp>();

            // Whether the vehicle is changing lanes and whether the lane change is completed
            FVectorT_qmdp LCing = (((ego_curr_path_ls > utils::LC_ING_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) |
                                    (theta_diff > utils::LC_ING_ALLOWANCE_HEADING_DIFF_THRESHOLD)) &
                                   left_mask) // left
                                  | (((ego_curr_path_ls < -utils::LC_ING_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) |
                                      (theta_diff < -utils::LC_ING_ALLOWANCE_HEADING_DIFF_THRESHOLD)) &
                                     right_mask); // right

            // ----------------------
            // LC completed condition split
            // ----------------------
            FVectorT_qmdp LC_completed_angle =
                (theta_diff > utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD) & left_mask |
                (theta_diff < -utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD) & right_mask;

            FVectorT_qmdp LC_completed_offset =
                ((ego_curr_path_ls > utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) & left_mask) |
                ((ego_curr_path_ls < -utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) & right_mask);

            FVectorT_qmdp LC_completed = LC_completed_angle | LC_completed_offset;

            // ----------------------
            // Relaxed condition
            // ----------------------
            FVectorT_qmdp relaxed_condition = LCing | (LF_lead_vehicles_vs <= 0.0f);

            // ----------------------
            // Time headway threshold
            // ----------------------
            FVectorT_qmdp LF_time_headway_threshold = FVectorT_qmdp::select(
                LC_completed, // Large heading or lateral offset → time threshold = 0
                0.0f,
                FVectorT_qmdp::select(relaxed_condition, // During lane change or with a slower lead vehicle → relaxed
                                      utils::LC_RELAXED_TIME_HEADWAY,
                                      utils::LC_NORMAL_TIME_HEADWAY // Normal case
                                      ));

            // ----------------------
            // Distance threshold
            // ----------------------
            FVectorT_qmdp LF_distance_threshold = FVectorT_qmdp::select(
                LC_completed_angle, // Large heading difference → distance threshold = 0
                0.0f,
                FVectorT_qmdp::select(
                    ego_vs >
                        utils::LC_SPEED_FOR_DISTANCE_THRESHOLD, // Otherwise choose high/low threshold based on speed
                    utils::LC_AHEAD_DISTANCE_THRESHOULD_HIGH, utils::LC_AHEAD_DISTANCE_THRESHOULD_LOW));

            FVectorT_qmdp LF_acc_threshold =
                FVectorT_qmdp::select(relaxed_condition, utils::MAX_DEC + utils::LC_RELAXED_ACC_THRESHOLD_MARGIN,
                                      utils::LC_NORMAL_ACC_THRESHOLD); // When the lead vehicle moves in the opposite
                                                                       // direction, only consider time headway

            // If the current-lane time headway is smaller than the threshold, lane change is not allowed
            LF_lead_allowance_flags =
                (LF_lead_vehicles_relative_distance > utils::LC_UNCONDITIONAL_ALLOW_DISTANCE | LF_time_headway < 0.0f |
                 (LF_lead_vehicles_relative_distance >= LF_distance_threshold &
                  LF_time_headway >= LF_time_headway_threshold))
                    .template as<IVectorT_qmdp>();
            allowance_flags = allowance_flags & LF_lead_allowance_flags;
            LF_lead_allowance_flags =
                LF_lead_allowance_flags & (LF_acc >= LF_acc_threshold).template as<IVectorT_qmdp>();

            // std::cout << std::endl;
            if (allowance_flags.none())
            {
                return allowance_flags;
            }

            // 2. Check the lead vehicle on the target lane
            FVectorT_qmdp LC_lead_vehicles_relative_distance =
                FVectorT_qmdp(LC_lead_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_qmdp LC_lead_vehicles_vs =
                FVectorT_qmdp(LC_lead_vehicles_close_to_ego_reference_path.second.data());

            // Add lateral compensation distance
            LC_lead_vehicles_relative_distance =
                LC_lead_vehicles_relative_distance +
                calculateLateralMotionParams(ego_vs, FVectorT_qmdp::fill(utils::LC_LEAD_LATERAL_COMPENSATION_DISTANCE),
                                             ego_vs_lateral, true);

            // Compute IDM acceleration for the ego vehicle on the target lane (to judge whether lane change is safe)
            FVectorT_qmdp LC_acc =
                IDMBatch(LC_lead_vehicles_vs, ego_vs, LC_lead_vehicles_relative_distance, ego_desired_vs);

            // If the target-lane acceleration is smaller than the comfort deceleration threshold, lane change is not
            // allowed
            IVectorT_qmdp LC_lead_allowance_flags = (LC_acc > utils::COMFORT_DEC).template as<IVectorT_qmdp>();
            allowance_flags = allowance_flags & LC_lead_allowance_flags;

            if (allowance_flags.none())
            {
                return allowance_flags;
            }

            // 3. Check the following vehicle on the target lane (note: here we treat the ego vehicle as the lead and
            // the other as the following vehicle)
            FVectorT_qmdp LC_follow_vehicles_relative_distance =
                FVectorT_qmdp(LC_following_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_qmdp LC_follow_vehicles_vs =
                FVectorT_qmdp(LC_following_vehicles_close_to_ego_reference_path.second.data());

            // Add lateral compensation distance
            LC_follow_vehicles_relative_distance =
                LC_follow_vehicles_relative_distance + utils::LC_FOLLOW_DISTANCE_COMPENSATION;

            // Important: here we treat the ego vehicle as the lead vehicle when computing the IDM model for the
            // following vehicle The relative distance has already been correctly computed by
            // FindLeadOrFollowVehicleCloseToEgoReferencePathBatch We need to compute the following vehicle's
            // acceleration, so the order is: ego (lead), other (following), assuming the other vehicle's desired speed
            // equals its current speed
            FVectorT_qmdp LC_follow_acc =
                IDMBatch(ego_vs, LC_follow_vehicles_vs, LC_follow_vehicles_relative_distance, LC_follow_vehicles_vs);

            // If the following vehicle needs hard braking, lane change is not allowed
            IVectorT_qmdp LC_follow_allowance_flags =
                (LC_follow_acc > utils::COMFORT_DEC | LC_follow_vehicles_vs <= 0.0f).template as<IVectorT_qmdp>();
            allowance_flags = allowance_flags & LC_follow_allowance_flags;

            return allowance_flags;
        }

        IVectorT_traj ContextQMDP::LCAllowanceCheckBatch(
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LF_lead_vehicles_close_to_ego_reference_path,
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_lead_vehicles_close_to_ego_reference_path,
            const std::pair<AlignedVectorFloat, AlignedVectorFloat> &LC_following_vehicles_close_to_ego_reference_path,
            const FVectorT_traj &ego_vs, const FVectorT_traj &ego_thetas, const FVectorT_traj &ego_desired_vs,
            const FVectorT_traj &ego_vs_lateral, const IVectorT_traj &active_flags, bool turn_left,
            const FVectorT_traj &ego_curr_path_ls, const FVectorT_traj &ego_curr_path_thetas,
            IVectorT_traj &LF_lead_allowance_flags)
        {
            // If there are no active scenarios, directly return false
            if (active_flags.none())
            {
                return active_flags;
            }

            // Initialize lane-change allowance flags; default is allowed
            IVectorT_traj allowance_flags = active_flags;

            // 1. Check the lead vehicle on the current lane
            FVectorT_traj LF_lead_vehicles_relative_distance(LF_lead_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_traj LF_lead_vehicles_vs(LF_lead_vehicles_close_to_ego_reference_path.second.data());

            // Compute IDM acceleration for the ego vehicle on the current lane (to decide whether a lane change is
            // needed)
            FVectorT_traj LF_acc =
                IDMBatch(LF_lead_vehicles_vs, ego_vs, LF_lead_vehicles_relative_distance, ego_desired_vs);
            // Compute the time headway for the ego vehicle on the current lane
            FVectorT_traj LF_time_headway = LF_lead_vehicles_relative_distance / (ego_vs - LF_lead_vehicles_vs);

            FVectorT_traj theta_diff = utils::NormalizeAngleSIMD(ego_thetas - ego_curr_path_thetas);
            FVectorT_traj LCing, LC_completed_angle, LC_completed;
            if (turn_left)
            {
                LCing = (ego_curr_path_ls > utils::LC_ING_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) |
                        (theta_diff > utils::LC_ING_ALLOWANCE_HEADING_DIFF_THRESHOLD);
                LC_completed_angle = (theta_diff > utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD);
                LC_completed =
                    (ego_curr_path_ls > utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) | LC_completed_angle;
            }
            else
            {
                LCing = (ego_curr_path_ls < -utils::LC_ING_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) |
                        (theta_diff < -utils::LC_ING_ALLOWANCE_HEADING_DIFF_THRESHOLD);
                LC_completed_angle = (theta_diff < -utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD);
                LC_completed =
                    (ego_curr_path_ls < -utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD) | LC_completed_angle;
            }

            // If the vehicle is in a lane change, use relaxed conditions
            FVectorT_traj relaxed_condition = LCing | (LF_lead_vehicles_vs <= 0.0f);

            FVectorT_traj LF_time_headway_threshold =
                FVectorT_traj::select(LC_completed, 0.0f,
                                      FVectorT_traj::select(relaxed_condition, utils::LC_RELAXED_TIME_HEADWAY,
                                                            utils::LC_NORMAL_TIME_HEADWAY));

            FVectorT_traj LF_distance_threshold =
                FVectorT_traj::select(LC_completed_angle, 0.0f,
                                      FVectorT_traj::select(ego_vs > utils::LC_SPEED_FOR_DISTANCE_THRESHOLD,
                                                            utils::LC_AHEAD_DISTANCE_THRESHOULD_HIGH,
                                                            utils::LC_AHEAD_DISTANCE_THRESHOULD_LOW));

            FVectorT_traj LF_acc_threshold =
                FVectorT_traj::select(relaxed_condition, utils::MAX_DEC + utils::LC_RELAXED_ACC_THRESHOLD_MARGIN,
                                      utils::LC_NORMAL_ACC_THRESHOLD); // When the lead vehicle moves in the opposite
                                                                       // direction, only consider time headway

            // If the current-lane time headway is smaller than the threshold, lane change is not allowed
            LF_lead_allowance_flags =
                (LF_lead_vehicles_relative_distance > utils::LC_UNCONDITIONAL_ALLOW_DISTANCE | LF_time_headway < 0.0f |
                 (LF_lead_vehicles_relative_distance >= LF_distance_threshold &
                  LF_time_headway >= LF_time_headway_threshold))
                    .template as<IVectorT_traj>();
            allowance_flags = allowance_flags & LF_lead_allowance_flags;

            LF_lead_allowance_flags =
                LF_lead_allowance_flags & (LF_acc >= LF_acc_threshold).template as<IVectorT_traj>();

            if (allowance_flags.none())
            {
                return allowance_flags;
            }

            FVectorT_traj LC_lead_vehicles_relative_distance(LC_lead_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_traj LC_lead_vehicles_vs(LC_lead_vehicles_close_to_ego_reference_path.second.data());

            // Add lateral compensation distance
            LC_lead_vehicles_relative_distance =
                LC_lead_vehicles_relative_distance +
                calculateLateralMotionParams(ego_vs, FVectorT_traj::fill(utils::LC_LEAD_LATERAL_COMPENSATION_DISTANCE),
                                             ego_vs_lateral, true);

            // Compute IDM acceleration for ego on the target lane (to judge post-LC safety)
            FVectorT_traj LC_acc =
                IDMBatch(LC_lead_vehicles_vs, ego_vs, LC_lead_vehicles_relative_distance, ego_desired_vs);

            // If target-lane acceleration is below the comfort deceleration threshold, disallow lane change
            IVectorT_traj LC_lead_allowance_flags = (LC_acc > utils::COMFORT_DEC).template as<IVectorT_traj>();
            allowance_flags = allowance_flags & LC_lead_allowance_flags;

            if (allowance_flags.none())
            {
                return allowance_flags;
            }

            // 3. Check trailing vehicle on the target lane (note: ego is the lead; trailing car is the following
            // vehicle)
            FVectorT_traj LC_follow_vehicles_relative_distance(
                LC_following_vehicles_close_to_ego_reference_path.first.data());
            FVectorT_traj LC_follow_vehicles_vs(LC_following_vehicles_close_to_ego_reference_path.second.data());

            // Add lateral compensation distance
            LC_follow_vehicles_relative_distance =
                LC_follow_vehicles_relative_distance + utils::LC_FOLLOW_DISTANCE_COMPENSATION;

            // Important: treat ego as the lead vehicle and compute the follower's IDM model
            // Relative distance is already correctly computed by FindLeadOrFollowVehicleCloseToEgoReferencePathBatch
            // We need the follower's acceleration; order: ego (lead), follower; assume follower's desired speed equals
            // its current speed
            FVectorT_traj LC_follow_acc =
                IDMBatch(ego_vs, LC_follow_vehicles_vs, LC_follow_vehicles_relative_distance, LC_follow_vehicles_vs);

            // If the follower requires harsh deceleration, disallow lane change
            IVectorT_traj LC_follow_allowance_flags =
                (LC_follow_acc > utils::COMFORT_DEC | LC_follow_vehicles_vs <= 0.0f).template as<IVectorT_traj>();
            allowance_flags = allowance_flags & LC_follow_allowance_flags;

            // *4. Overall decision: margin-of-safety check (additional), may also consider the lane-change gap
            return allowance_flags;
        }

        float ContextQMDP::calculateLateralMotionParams(float ego_v, float actual_lateral_distance_abs, bool simplified)
        {
            // Compute lateral speed parameters
            float v_lateral;
            if (ego_v > utils::LATERAL_PARAM_SPEED_THRESHOLD)
            {
                v_lateral = std::min(ego_v * utils::LATERAL_VEL_RATIO_HIGH_SPEED, utils::MAX_VEL_LATERAL_HIGH_SPEED);
            }
            else
            {
                v_lateral = std::max(ego_v * utils::LATERAL_VEL_RATIO_LOW_SPEED, utils::MAX_VEL_LATERAL_LOW_SPEED);
            }
            v_lateral = std::max(v_lateral, utils::LATERAL_VEL_MIN_EPSILON);

            // ----------------------
            // 1) Compute lateral transition time t_transition
            // ----------------------
            // Estimate total movement time
            float t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distance_abs / v_lateral;
            }
            else
            {
                // Compute lateral acceleration parameters
                float a_lateral_high_speed =
                    std::min(ego_v * utils::LATERAL_ACC_RATIO_HIGH_SPEED, utils::MAX_ACC_LATERAL_HIGH_SPEED);
                float a_lateral = (ego_v > utils::LATERAL_PARAM_SPEED_THRESHOLD) ? a_lateral_high_speed
                                                                                 : utils::MAX_ACC_LATERAL_LOW_SPEED;
                float t_acc =
                    std::min(std::sqrt(2.0f * actual_lateral_distance_abs / a_lateral), v_lateral / a_lateral);
                t_transition =
                    std::min(t_acc + (actual_lateral_distance_abs - 0.5f * a_lateral * t_acc * t_acc) / v_lateral,
                             actual_lateral_distance_abs / v_lateral);
            }

            float conservative_t_transition = std::min(t_transition, utils::MAX_CONERVATIVE_TRANSITION_TIME);
            float normal_t_transition = std::min(t_transition, utils::MAX_NORMINAL_TRANSITION_TIME);

            // ----------------------
            // 2) Compute forward displacement estimate s_forward
            //    - conservative: v * t
            //    - nominal (use accel): v * t + 0.5 * a_nominal * t^2
            //    For very small v (even 0), use the nominal form to avoid v*t being 0
            // ----------------------
            float s_forward;
            if (ego_v < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD)
            {
                s_forward = ego_v * normal_t_transition +
                            0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition;
            }
            else
            {
                s_forward = ego_v * conservative_t_transition;
            }
            s_forward = std::min(s_forward, utils::MAX_DISTANCE_COMPENSATION);

            return s_forward;
        }

        float ContextQMDP::calculateLateralMotionParams(float ego_v, float actual_lateral_distance_abs, float v_lateral,
                                                        bool simplified, bool for_LF)
        {
            // ----------------------
            // 1) Compute lateral transition time t_transition
            // ----------------------
            // Estimate total movement time
            float t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distance_abs / v_lateral;
            }
            else
            {
                // Compute lateral acceleration parameters
                float a_lateral_high_speed =
                    std::min(ego_v * utils::LATERAL_ACC_RATIO_HIGH_SPEED, utils::MAX_ACC_LATERAL_HIGH_SPEED);
                float a_lateral = (ego_v > utils::LATERAL_PARAM_SPEED_THRESHOLD) ? a_lateral_high_speed
                                                                                 : utils::MAX_ACC_LATERAL_LOW_SPEED;
                float t_acc =
                    std::min(std::sqrt(2.0f * actual_lateral_distance_abs / a_lateral), v_lateral / a_lateral);
                t_transition =
                    std::min(t_acc + (actual_lateral_distance_abs - 0.5f * a_lateral * t_acc * t_acc) / v_lateral,
                             actual_lateral_distance_abs / v_lateral);
            }

            if (for_LF)
            {
                return ego_v * t_transition;
            }

            float conservative_t_transition = std::min(t_transition, utils::MAX_CONERVATIVE_TRANSITION_TIME);
            float normal_t_transition = std::min(t_transition, utils::MAX_NORMINAL_TRANSITION_TIME);

            // ----------------------
            // 2) Compute forward displacement estimate s_forward
            //    - conservative: v * t
            //    - nominal (use accel): v * t + 0.5 * a_nominal * t^2
            //    For very small v (even 0), use the nominal form to avoid v*t being 0
            // ----------------------
            float s_forward;
            if (ego_v < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD)
            {
                s_forward = ego_v * normal_t_transition +
                            0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition;
            }
            else
            {
                s_forward = ego_v * conservative_t_transition;
            }
            s_forward = std::min(s_forward, utils::MAX_DISTANCE_COMPENSATION);

            return s_forward;
        }

        FVectorT_traj ContextQMDP::calculateLateralMotionParams(const FVectorT_traj &ego_vs,
                                                                const FVectorT_traj &actual_lateral_distances_abs,
                                                                bool                 simplified)
        {
            // Compute lateral speed parameters
            FVectorT_traj v_lateral =
                FVectorT_traj::select(
                    ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                    (ego_vs * utils::LATERAL_VEL_RATIO_HIGH_SPEED).min(utils::MAX_VEL_LATERAL_HIGH_SPEED),
                    (ego_vs * utils::LATERAL_VEL_RATIO_LOW_SPEED).max(utils::MAX_VEL_LATERAL_LOW_SPEED))
                    .max(utils::LATERAL_VEL_MIN_EPSILON);

            FVectorT_traj inv_v_lateral = 1.0f / v_lateral;

            // ----------------------
            // 1) Compute lateral transition time t_transition
            // ----------------------
            // Estimate total movement time
            FVectorT_traj t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distances_abs * inv_v_lateral;
            }
            else
            {
                // Compute lateral acceleration parameters
                FVectorT_traj a_lateral_high_speed =
                    (ego_vs * utils::LATERAL_ACC_RATIO_HIGH_SPEED).min(utils::MAX_ACC_LATERAL_HIGH_SPEED);
                FVectorT_traj a_lateral = FVectorT_traj::select(ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                                                                a_lateral_high_speed, utils::MAX_ACC_LATERAL_LOW_SPEED);

                FVectorT_traj inv_a_lateral =
                    1.0f / a_lateral; // Assume a_lateral is always > 0 (guaranteed by constants)

                FVectorT_traj t_acc =
                    (2.0f * actual_lateral_distances_abs * inv_a_lateral).sqrt().min(v_lateral * inv_a_lateral);
                t_transition =
                    (t_acc + (actual_lateral_distances_abs - 0.5 * a_lateral * t_acc * t_acc) * inv_v_lateral)
                        .min(actual_lateral_distances_abs * inv_v_lateral);
            }

            FVectorT_traj conservative_t_transition = t_transition.min(utils::MAX_CONERVATIVE_TRANSITION_TIME);
            FVectorT_traj normal_t_transition = t_transition.min(utils::MAX_NORMINAL_TRANSITION_TIME);

            // ----------------------
            // 2)  s_forward
            //    - conservative: v * t
            //    - nominal (use accel): v * t + 0.5 * a_nominal * t^2
            // ----------------------
            FVectorT_traj s_forward =
                FVectorT_traj::select(ego_vs < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD,
                                      ego_vs * normal_t_transition +
                                          0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition,
                                      ego_vs * conservative_t_transition)
                    .min(utils::MAX_DISTANCE_COMPENSATION);

            return s_forward;
        }

        FVectorT_traj ContextQMDP::calculateLateralMotionParams(const FVectorT_traj &ego_vs,
                                                                const FVectorT_traj &actual_lateral_distances_abs,
                                                                const FVectorT_traj &v_lateral, bool simplified,
                                                                bool for_LF)
        {
            FVectorT_traj inv_v_lateral = 1.0f / v_lateral;

            // ----------------------
            // 1) Compute lateral transition time t_transition
            // ----------------------
            // Estimate total motion time
            FVectorT_traj t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distances_abs * inv_v_lateral;
            }
            else
            {
                // Compute lateral acceleration parameters
                FVectorT_traj a_lateral_high_speed =
                    (ego_vs * utils::LATERAL_ACC_RATIO_HIGH_SPEED).min(utils::MAX_ACC_LATERAL_HIGH_SPEED);
                FVectorT_traj a_lateral = FVectorT_traj::select(ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                                                                a_lateral_high_speed, utils::MAX_ACC_LATERAL_LOW_SPEED);
                FVectorT_traj inv_a_lateral =
                    1.0f / a_lateral; // Assume a_lateral is always > 0 (guaranteed by constants)

                FVectorT_traj t_acc =
                    (2.0f * actual_lateral_distances_abs * inv_a_lateral).sqrt().min(v_lateral * inv_a_lateral);
                t_transition =
                    (t_acc + (actual_lateral_distances_abs - 0.5 * a_lateral * t_acc * t_acc) * inv_v_lateral)
                        .min(actual_lateral_distances_abs * inv_v_lateral);
            }

            if (for_LF)
            {
                return ego_vs * t_transition;
            }

            FVectorT_traj conservative_t_transition = t_transition.min(utils::MAX_CONERVATIVE_TRANSITION_TIME);
            FVectorT_traj normal_t_transition = t_transition.min(utils::MAX_NORMINAL_TRANSITION_TIME);

            // ----------------------
            // 2) Compute forward displacement estimate s_forward
            //    - conservative: v * t
            //    - nominal (use accel): v * t + 0.5 * a_nominal * t^2
            //    For very small v (even 0), use the nominal form to avoid v*t being 0
            //    When in LF mode, use it directly
            // ----------------------
            FVectorT_traj s_forward =
                FVectorT_traj::select(ego_vs < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD,
                                      ego_vs * normal_t_transition +
                                          0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition,
                                      ego_vs * conservative_t_transition)
                    .min(utils::MAX_DISTANCE_COMPENSATION);

            return s_forward;
        }

        FVectorT_qmdp ContextQMDP::calculateLateralMotionParams(const FVectorT_qmdp &ego_vs,
                                                                const FVectorT_qmdp &actual_lateral_distances_abs,
                                                                bool                 simplified)
        {
            FVectorT_qmdp v_lateral =
                FVectorT_qmdp::select(
                    ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                    (ego_vs * utils::LATERAL_VEL_RATIO_HIGH_SPEED).min(utils::MAX_VEL_LATERAL_HIGH_SPEED),
                    (ego_vs * utils::LATERAL_VEL_RATIO_LOW_SPEED).max(utils::MAX_VEL_LATERAL_LOW_SPEED))
                    .max(utils::LATERAL_VEL_MIN_EPSILON);
            FVectorT_qmdp inv_v_lateral = 1.0f / v_lateral;
            FVectorT_qmdp t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distances_abs * inv_v_lateral;
            }
            else
            {
                FVectorT_qmdp a_lateral_high_speed =
                    (ego_vs * utils::LATERAL_ACC_RATIO_HIGH_SPEED).min(utils::MAX_ACC_LATERAL_HIGH_SPEED);
                FVectorT_qmdp a_lateral = FVectorT_qmdp::select(ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                                                                a_lateral_high_speed, utils::MAX_ACC_LATERAL_LOW_SPEED);
                FVectorT_qmdp inv_a_lateral = 1.0f / a_lateral;
                FVectorT_qmdp t_acc =
                    (2.0f * actual_lateral_distances_abs * inv_a_lateral).sqrt().min(v_lateral * inv_a_lateral);
                t_transition =
                    (t_acc + (actual_lateral_distances_abs - 0.5f * a_lateral * t_acc * t_acc) * inv_v_lateral)
                        .min(actual_lateral_distances_abs * inv_v_lateral);
            }
            FVectorT_qmdp conservative_t_transition = t_transition.min(utils::MAX_CONERVATIVE_TRANSITION_TIME);
            FVectorT_qmdp normal_t_transition = t_transition.min(utils::MAX_NORMINAL_TRANSITION_TIME);
            FVectorT_qmdp s_forward =
                FVectorT_qmdp::select(ego_vs < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD,
                                      ego_vs * normal_t_transition +
                                          0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition,
                                      ego_vs * conservative_t_transition)
                    .min(utils::MAX_DISTANCE_COMPENSATION);
            return s_forward;
        }

        FVectorT_qmdp ContextQMDP::calculateLateralMotionParams(const FVectorT_qmdp &ego_vs,
                                                                const FVectorT_qmdp &actual_lateral_distances_abs,
                                                                const FVectorT_qmdp &v_lateral, bool simplified,
                                                                bool for_LF)
        {
            FVectorT_qmdp inv_v_lateral = 1.0f / v_lateral;
            FVectorT_qmdp t_transition;
            if (simplified)
            {
                t_transition = actual_lateral_distances_abs * inv_v_lateral;
            }
            else
            {
                FVectorT_qmdp a_lateral_high_speed =
                    (ego_vs * utils::LATERAL_ACC_RATIO_HIGH_SPEED).min(utils::MAX_ACC_LATERAL_HIGH_SPEED);
                FVectorT_qmdp a_lateral = FVectorT_qmdp::select(ego_vs > utils::LATERAL_PARAM_SPEED_THRESHOLD,
                                                                a_lateral_high_speed, utils::MAX_ACC_LATERAL_LOW_SPEED);
                FVectorT_qmdp inv_a_lateral = 1.0f / a_lateral;
                FVectorT_qmdp t_acc =
                    (2.0f * actual_lateral_distances_abs * inv_a_lateral).sqrt().min(v_lateral * inv_a_lateral);
                t_transition =
                    (t_acc + (actual_lateral_distances_abs - 0.5f * a_lateral * t_acc * t_acc) * inv_v_lateral)
                        .min(actual_lateral_distances_abs * inv_v_lateral);
            }
            if (for_LF)
            {
                return ego_vs * t_transition;
            }
            FVectorT_qmdp conservative_t_transition = t_transition.min(utils::MAX_CONERVATIVE_TRANSITION_TIME);
            FVectorT_qmdp normal_t_transition = t_transition.min(utils::MAX_NORMINAL_TRANSITION_TIME);
            FVectorT_qmdp s_forward =
                FVectorT_qmdp::select(ego_vs < utils::FORWARD_DISP_LOW_SPEED_THRESHOLD,
                                      ego_vs * normal_t_transition +
                                          0.5f * utils::NOMINAL_ACC_LONG * normal_t_transition * normal_t_transition,
                                      ego_vs * conservative_t_transition)
                    .min(utils::MAX_DISTANCE_COMPENSATION);
            return s_forward;
        }

        float ContextQMDP::IDMSerial(float leading_vehicle_v, float following_vehicle_v, float relative_distance,
                                     float desired_speed)
        {
            float relative_vs = following_vehicle_v - leading_vehicle_v;
            float desired_gaps = std::max(0.0f, following_vehicle_v * utils::DESIRED_TIME_HEADWAY +
                                                    following_vehicle_v * relative_vs / (2.0f * utils::SQRT_AB)) +
                                 utils::LEAST_SAFE_DIST;

            float acc =
                utils::MAX_ACC * (1 - std::pow(following_vehicle_v / (desired_speed + 0.0001f), utils::ACC_EXPONENT) -
                                  std::pow(desired_gaps / std::max(relative_distance, 0.0001f), 2));
            return std::clamp(acc, utils::MAX_DEC, utils::MAX_ACC);
        }

        FVectorT_qmdp ContextQMDP::IDMBatch(const FVectorT_qmdp &leading_vehicle_vs,
                                            const FVectorT_qmdp &following_vehicle_vs,
                                            const FVectorT_qmdp &relative_distance, const FVectorT_qmdp &desired_speed)
        {
            FVectorT_qmdp relative_vs = following_vehicle_vs - leading_vehicle_vs;
            FVectorT_qmdp desired_gaps = (following_vehicle_vs * utils::DESIRED_TIME_HEADWAY +
                                          following_vehicle_vs * relative_vs / (2.0f * utils::SQRT_AB))
                                             .max(0.0f) +
                                         utils::LEAST_SAFE_DIST;

            FVectorT_qmdp acc =
                utils::MAX_ACC * (1 - (following_vehicle_vs / (desired_speed + 0.0001f)).pow(utils::ACC_EXPONENT) -
                                  (desired_gaps / relative_distance.max(0.0001f)).square());
            return acc.clamp(utils::MAX_DEC, utils::MAX_ACC);
        }

        FVectorT_traj ContextQMDP::IDMBatch(const FVectorT_traj &leading_vehicle_vs,
                                            const FVectorT_traj &following_vehicle_vs,
                                            const FVectorT_traj &relative_distance, const FVectorT_traj &desired_speed)
        {
            FVectorT_traj relative_vs = following_vehicle_vs - leading_vehicle_vs;
            FVectorT_traj desired_gaps = (following_vehicle_vs * utils::DESIRED_TIME_HEADWAY +
                                          following_vehicle_vs * relative_vs / (2.0f * utils::SQRT_AB))
                                             .max(0.0f) +
                                         utils::LEAST_SAFE_DIST;

            FVectorT_traj acc =
                utils::MAX_ACC * (1 - (following_vehicle_vs / (desired_speed + 0.0001f)).pow(utils::ACC_EXPONENT) -
                                  (desired_gaps / relative_distance.max(0.0001f)).square());
            return acc.clamp(utils::MAX_DEC, utils::MAX_ACC);
        }

        FVectorT_traj ContextQMDP::CurveAndLaneChangeAwareIDMBatch(const FVectorT_traj &leading_vehicle_vs,
                                                                   const FVectorT_traj &following_vehicle_vs,
                                                                   const FVectorT_traj &relative_distance,
                                                                   const FVectorT_traj &desired_speed,
                                                                   const IVectorT_traj &lane_change_flag)
        {
            FVectorT_traj relative_vs = following_vehicle_vs - leading_vehicle_vs;
            FVectorT_traj desired_gaps = (following_vehicle_vs * utils::DESIRED_TIME_HEADWAY +
                                          following_vehicle_vs * relative_vs / (2.0f * utils::SQRT_AB))
                                             .max(0.0f) +
                                         utils::LEAST_SAFE_DIST;

            FVectorT_traj scaled_speed = following_vehicle_vs / (desired_speed + 0.0001f);

            FVectorT_traj acc = utils::MAX_ACC * (1 - scaled_speed.pow(utils::ACC_EXPONENT) -
                                                  (desired_gaps / relative_distance.max(0.0001f)).square());

            acc = acc.min(utils::MAX_ACC);
            FVectorT_traj on_sharp_curve_or_lane_change =
                scaled_speed > 1.3f | lane_change_flag.template as<FVectorT_traj>();
            FVectorT_traj min_acc =
                FVectorT_traj::select(on_sharp_curve_or_lane_change, utils::MAX_DEC, utils::HARD_BRAKE_DEC);
            acc = acc.max(min_acc);

            return acc;
        }

        FVectorT_qmdp ContextQMDP::GetSteeringByStanleyBatch(
            const FVectorT_qmdp &ref_point_xs, const FVectorT_qmdp &ref_point_ys, const FVectorT_qmdp &ref_point_thetas,
            const FVectorT_qmdp &future_curvatures, const FVectorT_qmdp &lateral_offsets, const FVectorT_qmdp &ego_xs,
            const FVectorT_qmdp &ego_ys, const FVectorT_qmdp &ego_vs, const FVectorT_qmdp &ego_thetas,
            const FVectorT_qmdp &ego_thetas_cos, const FVectorT_qmdp &ego_thetas_sin, bool print_debug)
        {
            // Compute the front-wheel position
            FVectorT_qmdp ego_front_rear_xs = ego_xs + ego_thetas_cos * utils::FRONT_TO_CENTER;
            FVectorT_qmdp ego_front_rear_ys = ego_ys + ego_thetas_sin * utils::FRONT_TO_CENTER;

            // Compute the vector from the front wheel to the target point
            FVectorT_qmdp target_2_front_xs = ego_front_rear_xs - ref_point_xs;
            FVectorT_qmdp target_2_front_ys = ego_front_rear_ys - ref_point_ys;

            // Compute lateral error
            FVectorT_qmdp lateral_errors =
                target_2_front_xs * ego_thetas_sin - target_2_front_ys * ego_thetas_cos + lateral_offsets;

            // Compute heading error
            FVectorT_qmdp theta_fs = ref_point_thetas - ego_thetas;

            // Use different k values to handle curvature
            FVectorT_qmdp k_values = FVectorT_qmdp::select(future_curvatures > utils::STANLEY_CURVATURE_THRESHOLD,
                                                           FVectorT_qmdp::fill(utils::AGGRESIVE_STANLEY_K),
                                                           FVectorT_qmdp::fill(utils::CONSERVATIVE_STANLEY_K));

            // Use SIMD to compute the Stanley lateral control term in parallel
            FVectorT_qmdp numerator = k_values * lateral_errors;
            FVectorT_qmdp denominator = ego_vs.max(
                utils::STANLEY_MIN_SPEED_QMDP); // Add a lower bound to speed to avoid excessive amplification

            // Use the atan2_approx function
            FVectorT_qmdp lateral_control = numerator.atan2_approx(denominator);

            // Combine control terms
            FVectorT_qmdp steering = lateral_control + theta_fs;

            // Normalize angle - using SIMD parallel version
            steering = utils::NormalizeAngleSIMD<FVectorT_qmdp>(steering);

            // Clamp to maximum steering angle
            steering = steering.clamp(-utils::MAX_STEERING, utils::MAX_STEERING);

            return steering;
        }
        FVectorT_traj ContextQMDP::GetSteeringByStanleyBatch(
            const FVectorT_traj &ref_point_xs, const FVectorT_traj &ref_point_ys, const FVectorT_traj &ref_point_thetas,
            const FVectorT_traj &future_curvatures, const FVectorT_traj &lateral_offsets, const FVectorT_traj &ego_xs,
            const FVectorT_traj &ego_ys, const FVectorT_traj &ego_vs, const FVectorT_traj &ego_thetas,
            const FVectorT_traj &ego_thetas_cos, const FVectorT_traj &ego_thetas_sin)
        {
            // Compute the front-wheel position
            FVectorT_traj ego_front_rear_xs = ego_xs + ego_thetas_cos * utils::FRONT_TO_CENTER;
            FVectorT_traj ego_front_rear_ys = ego_ys + ego_thetas_sin * utils::FRONT_TO_CENTER;

            // Compute the vector from the front wheel to the target point
            FVectorT_traj target_2_front_xs = ego_front_rear_xs - ref_point_xs;
            FVectorT_traj target_2_front_ys = ego_front_rear_ys - ref_point_ys;

            // Compute lateral error
            FVectorT_traj lateral_errors =
                target_2_front_xs * ego_thetas_sin - target_2_front_ys * ego_thetas_cos + lateral_offsets;

            // Compute heading error
            FVectorT_traj theta_fs = ref_point_thetas - ego_thetas;

            // Use different k values to handle curvature
            FVectorT_traj k_values = FVectorT_traj::select(future_curvatures > utils::STANLEY_CURVATURE_THRESHOLD,
                                                           FVectorT_traj::fill(utils::AGGRESIVE_STANLEY_K),
                                                           FVectorT_traj::fill(utils::CONSERVATIVE_STANLEY_K));

            // Use SIMD to compute the Stanley lateral control term in parallel
            FVectorT_traj numerator = k_values * lateral_errors;
            FVectorT_traj denominator = ego_vs.max(
                utils::STANLEY_MIN_SPEED_TRAJ); // Add a lower bound to speed to avoid excessive amplification

            // Use our newly implemented atan2_approx function
            FVectorT_traj lateral_control = numerator.atan2_approx(denominator);

            // Combine control terms
            FVectorT_traj steering = lateral_control + theta_fs;

            // Normalize angle - using SIMD parallel version
            steering = utils::NormalizeAngleSIMD<FVectorT_traj>(steering);

            // Clamp to maximum steering angle
            steering = steering.clamp(-utils::MAX_STEERING, utils::MAX_STEERING);

            return steering;
        }

        // *Note: current ego steering could be incorporated; the current version assumes steering = 0
        void ContextQMDP::UpdateStateBatch(FVectorT_qmdp &ego_xs, FVectorT_qmdp &ego_ys, FVectorT_qmdp &ego_vs,
                                           FVectorT_qmdp &ego_as, FVectorT_qmdp &ego_thetas,
                                           FVectorT_qmdp &ego_thetas_cos, FVectorT_qmdp &ego_thetas_sin,
                                           const FVectorT_qmdp &cal_accs, const FVectorT_qmdp &cal_steerings)
        {
            // Constant definitions
            constexpr float STEP_INTERVAL = utils::TIME_STEP;
            constexpr float WHEEL_BASE = utils::WHEEL_BASE;
            constexpr float MAX_YAW_RATE = utils::MAX_YAW_RATE; // Maximum yaw rate (rad/s), can be tuned as needed
            constexpr float REAR_TO_CENTER = utils::REAR_TO_CENTER;
            constexpr float MAX_VEL = utils::MAX_VEL;   // Maximum speed, can be tuned as needed
            constexpr float MAX_JERK = utils::MAX_JERK; // Jerk limit

            // Handle acceleration smoothing constraints
            FVectorT_qmdp delta_accs = cal_accs - ego_as; // Compare with accelerations from the previous step
            float         jerk_limits = MAX_JERK * STEP_INTERVAL;

            // Adjust acceleration based on jerk limit
            FVectorT_qmdp delta_acc_signs = delta_accs.sign();
            FVectorT_qmdp delta_acc_abs = delta_accs.abs();
            ego_as =
                FVectorT_qmdp::select(delta_acc_abs > jerk_limits, ego_as + jerk_limits * delta_acc_signs, cal_accs);

            // Compute average speed within the time step
            FVectorT_qmdp avg_speeds = (ego_vs + ego_as * (STEP_INTERVAL * 0.5f)).clamp(0.0f, MAX_VEL);

            // Limit steering angle to prevent oversteering
            FVectorT_qmdp max_steerings =
                FVectorT_qmdp::fill(WHEEL_BASE * MAX_YAW_RATE).atan2_approx(avg_speeds + 0.01f);
            FVectorT_qmdp limited_steerings = cal_steerings.clamp(-max_steerings, max_steerings);

            // Compute turning radius and yaw rate
            FVectorT_qmdp turning_radii = WHEEL_BASE / (limited_steerings.tan() + 0.0001f);
            FVectorT_qmdp yaw_rates = avg_speeds / turning_radii;

            // Compute heading change
            FVectorT_qmdp betas = yaw_rates * STEP_INTERVAL;

            // Compute rear-wheel position
            FVectorT_qmdp rear_pos_xs = ego_xs - ego_thetas_cos * REAR_TO_CENTER;
            FVectorT_qmdp rear_pos_ys = ego_ys - ego_thetas_sin * REAR_TO_CENTER;

            // Update heading
            ego_thetas = utils::NormalizeAngleSIMD<FVectorT_qmdp>(ego_thetas + betas);

            // Use beta to determine near-straight-line motion
            FVectorT_qmdp is_straight = betas.abs() < 1e-4f;

            // Curved-model update: use yaw_rate × R × (sin(theta+beta) - sin(theta)) (kept)
            FVectorT_qmdp delta_xs_curved = turning_radii * (ego_thetas.sin() - ego_thetas_sin);
            FVectorT_qmdp delta_ys_curved = turning_radii * (ego_thetas_cos - ego_thetas.cos());

            // Update heading cosine/sine
            ego_thetas_cos = ego_thetas.cos();
            ego_thetas_sin = ego_thetas.sin();

            // Straight-line model update: delta_x = v * cos(θ) * Δt, delta_y = v * sin(θ) * Δt
            FVectorT_qmdp delta_xs_linear = avg_speeds * ego_thetas_cos * STEP_INTERVAL;
            FVectorT_qmdp delta_ys_linear = avg_speeds * ego_thetas_sin * STEP_INTERVAL;

            // Use the more stable option to update the rear-wheel position
            rear_pos_xs = rear_pos_xs + FVectorT_qmdp::select(is_straight, delta_xs_linear, delta_xs_curved);
            rear_pos_ys = rear_pos_ys + FVectorT_qmdp::select(is_straight, delta_ys_linear, delta_ys_curved);

            // Update vehicle position
            ego_xs = rear_pos_xs + ego_thetas_cos * REAR_TO_CENTER;
            ego_ys = rear_pos_ys + ego_thetas_sin * REAR_TO_CENTER;

            // Update speed
            ego_vs = ego_vs + ego_as * STEP_INTERVAL;
            ego_vs = ego_vs.clamp(0.0f, MAX_VEL);
        }

        // *Note: current ego steering could be incorporated in; the current version assumes steering = 0
        void ContextQMDP::UpdateStateBatch(FVectorT_traj &ego_xs, FVectorT_traj &ego_ys, FVectorT_traj &ego_vs,
                                           FVectorT_traj &ego_as, FVectorT_traj &ego_thetas,
                                           FVectorT_traj &ego_thetas_cos, FVectorT_traj &ego_thetas_sin,
                                           const FVectorT_traj &cal_accs, const FVectorT_traj &cal_steerings,
                                           float time_step)
        {
            // Handle acceleration smoothing (jerk) limits
            FVectorT_traj delta_accs = cal_accs - ego_as; // Compare with previous acceleration
            float         jerk_limits = utils::MAX_JERK * time_step;

            // Adjust acceleration according to jerk limits
            FVectorT_traj delta_acc_signs = delta_accs.sign();
            FVectorT_traj delta_acc_abs = delta_accs.abs();
            ego_as =
                FVectorT_traj::select(delta_acc_abs > jerk_limits, ego_as + jerk_limits * delta_acc_signs, cal_accs);

            // Compute average speed within the time step
            FVectorT_traj avg_speeds = (ego_vs + ego_as * (time_step * 0.5f)).clamp(0.0f, utils::MAX_VEL);

            // Limit steering angle to prevent excessive steering
            FVectorT_traj max_steerings =
                FVectorT_traj::fill(utils::WHEEL_BASE * utils::MAX_YAW_RATE).atan2_approx(avg_speeds + 0.01f);

            FVectorT_traj limited_steerings = cal_steerings.max(-max_steerings).min(max_steerings);

            // Compute turning radius and yaw rate
            FVectorT_traj turning_radii = utils::WHEEL_BASE / (limited_steerings.tan() + 0.0001f);

            FVectorT_traj yaw_rates = avg_speeds / turning_radii;

            // Compute heading change
            FVectorT_traj betas = yaw_rates * time_step;

            // Compute rear-wheel position
            FVectorT_traj rear_pos_xs = ego_xs - ego_thetas_cos * utils::REAR_TO_CENTER;
            FVectorT_traj rear_pos_ys = ego_ys - ego_thetas_sin * utils::REAR_TO_CENTER;

            // Update heading
            ego_thetas = utils::NormalizeAngleSIMD<FVectorT_traj>(ego_thetas + betas);

            // Use beta to determine near-straight-line motion
            FVectorT_traj is_straight = betas.abs() < 1e-4f;

            // Curved-model update: use yaw_rate × R × (sin(theta+beta) - sin(theta)) (kept)
            FVectorT_traj delta_x_curved = turning_radii * (ego_thetas.sin() - ego_thetas_sin);
            FVectorT_traj delta_y_curved = turning_radii * (ego_thetas_cos - ego_thetas.cos());

            // Update heading cosine/sine
            ego_thetas_cos = ego_thetas.cos();
            ego_thetas_sin = ego_thetas.sin();

            // Straight-line model update: delta_x = v * cos(theta) * dt, delta_y = v * sin(theta) * dt
            FVectorT_traj delta_x_linear = avg_speeds * ego_thetas_cos * time_step;
            FVectorT_traj delta_y_linear = avg_speeds * ego_thetas_sin * time_step;

            // Use the more stable method to update rear wheel
            rear_pos_xs = rear_pos_xs + FVectorT_traj::select(is_straight, delta_x_linear, delta_x_curved);
            rear_pos_ys = rear_pos_ys + FVectorT_traj::select(is_straight, delta_y_linear, delta_y_curved);

            // Update vehicle position
            ego_xs = rear_pos_xs + ego_thetas_cos * utils::REAR_TO_CENTER;
            ego_ys = rear_pos_ys + ego_thetas_sin * utils::REAR_TO_CENTER;

            // Update speed
            ego_vs = ego_vs + ego_as * time_step;
            ego_vs = ego_vs.clamp(0.0f, utils::MAX_VEL);
        }

        void ContextQMDP::UpdateSpeedBatch(FVectorT_traj &ego_vs, FVectorT_traj &ego_avg_vs, FVectorT_traj &ego_as,
                                           const FVectorT_traj &cal_accs, float time_step)
        {
            // Handle acceleration smoothing (jerk) limits
            FVectorT_traj delta_accs = cal_accs - ego_as; // Compare with previous acceleration
            float         jerk_limits = utils::MAX_JERK * time_step;

            // Adjust acceleration according to jerk limits
            FVectorT_traj delta_acc_signs = delta_accs.sign();
            FVectorT_traj delta_acc_abs = delta_accs.abs();
            ego_as =
                FVectorT_traj::select(delta_acc_abs > jerk_limits, ego_as + jerk_limits * delta_acc_signs, cal_accs);

            // Update speed
            ego_avg_vs = (ego_vs + ego_as * (time_step * 0.5f)).clamp(0.0f, utils::MAX_VEL);
            ego_vs = (ego_vs + ego_as * time_step).clamp(0.0f, utils::MAX_VEL);
        }

        void ContextQMDP::UpdateHeadingBatch(FVectorT_traj &ego_xs, FVectorT_traj &ego_ys,
                                             FVectorT_traj &ego_avg_speeds, FVectorT_traj &ego_thetas,
                                             FVectorT_traj &ego_thetas_cos, FVectorT_traj &ego_thetas_sin,
                                             const FVectorT_traj &cal_steerings, float time_step)
        {
            // Limit steering angle to prevent excessive steering
            FVectorT_traj max_steerings =
                FVectorT_traj::fill(utils::WHEEL_BASE * utils::MAX_YAW_RATE).atan2_approx(ego_avg_speeds + 0.01f);

            FVectorT_traj limited_steerings = cal_steerings.max(-max_steerings).min(max_steerings);

            // Compute turning radius and yaw rate
            FVectorT_traj turning_radii = utils::WHEEL_BASE / (limited_steerings.tan() + 0.0001f);

            FVectorT_traj yaw_rates = ego_avg_speeds / turning_radii;

            // Compute heading change
            FVectorT_traj betas = yaw_rates * time_step;

            // Compute rear-wheel position
            FVectorT_traj rear_pos_xs = ego_xs - ego_thetas_cos * utils::REAR_TO_CENTER;
            FVectorT_traj rear_pos_ys = ego_ys - ego_thetas_sin * utils::REAR_TO_CENTER;

            // Update heading
            ego_thetas = utils::NormalizeAngleSIMD<FVectorT_traj>(ego_thetas + betas);

            // Use beta to determine near-straight-line motion
            FVectorT_traj is_straight = betas.abs() < 1e-4f;

            // Curved-model update: use yaw_rate × R × (sin(theta+beta) - sin(theta)) (kept)
            FVectorT_traj delta_x_curved = turning_radii * (ego_thetas.sin() - ego_thetas_sin);
            FVectorT_traj delta_y_curved = turning_radii * (ego_thetas_cos - ego_thetas.cos());

            // Update heading cosine/sine
            ego_thetas_cos = ego_thetas.cos();
            ego_thetas_sin = ego_thetas.sin();

            // Straight-line model update: delta_x = v * cos(theta) * dt, delta_y = v * sin(theta) * dt
            FVectorT_traj delta_x_linear = ego_avg_speeds * ego_thetas_cos * time_step;
            FVectorT_traj delta_y_linear = ego_avg_speeds * ego_thetas_sin * time_step;

            // Use the more stable method to update rear wheel
            rear_pos_xs = rear_pos_xs + FVectorT_traj::select(is_straight, delta_x_linear, delta_x_curved);
            rear_pos_ys = rear_pos_ys + FVectorT_traj::select(is_straight, delta_y_linear, delta_y_curved);

            // Update vehicle position
            ego_xs = rear_pos_xs + ego_thetas_cos * utils::REAR_TO_CENTER;
            ego_ys = rear_pos_ys + ego_thetas_sin * utils::REAR_TO_CENTER;
        }
    } // namespace planning
} // namespace vec_qmdp
