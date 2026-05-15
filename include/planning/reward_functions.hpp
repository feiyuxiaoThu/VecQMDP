/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file reward_functions.hpp
 * @brief Inline SIMD-templated cost/reward functions for scoring candidate trajectories.
 */

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <utils/aligned_allocator.hpp>
#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        using AlignedVectorFloat = utils::AlignedVectorFloat;
        using AlignedVectorInt = utils::AlignedVectorInt;
        using Path = utils::Path;

        // =====================================================================
        // Collision type classification (nuPlan metric)
        // =====================================================================

        enum class CollisionType
        {
            NO_COLLISION = 0,
            STOPPED_EGO_COLLISION = 1,
            STOPPED_TRACK_COLLISION = 2,
            ACTIVE_FRONT_COLLISION = 3,
            REAR_COLLISION = 4,
            ACTIVE_LATERAL_COLLISION = 5
        };

        // =====================================================================
        // QMDP reward / penalty functions — inline SIMD-template free functions
        // =====================================================================

        inline float GetCollisionPenalty(float /*ego_vs*/, bool has_collision)
        {
            return has_collision ? utils::CRASH_PENALTY : 0.0f;
        }

        template <typename T> inline T GetCollisionPenalty(const T & /*ego_vs*/, const T &collision_flags)
        {
            return collision_flags * utils::CRASH_PENALTY;
        }

        inline float GetMovementPenalty(float ego_v)
        {
            return std::min(ego_v - utils::MAX_VEL, 0.0f) / utils::MAX_VEL * utils::MOVEMENT_PENALTY;
        }

        template <typename T> inline T GetMovementPenalty(const T &ego_vs)
        {
            return (ego_vs - utils::MAX_VEL).min(0.0f) / utils::MAX_VEL * utils::MOVEMENT_PENALTY;
        }

        template <typename T> inline T GetMovementPenalty(const T &ego_vs, const T &active_flags)
        {
            return active_flags * (ego_vs - utils::MAX_VEL).min(0.0f) / utils::MAX_VEL * utils::MOVEMENT_PENALTY;
        }

        template <typename T> inline T GetCrossEvaluationMovementPenaltyBatch(const T &ego_vs)
        {
            T cross_evaluation_movement_penalty =
                T::select(ego_vs < utils::REWARD_SPEED_LOW, utils::CROSS_EVALUATION_MOVEMENT_PENALTY_HIGH,
                          T::select(ego_vs < utils::REWARD_SPEED_MID, utils::CROSS_EVALUATION_MOVEMENT_PENALTY_MID,
                                    utils::CROSS_EVALUATION_MOVEMENT_PENALTY_LOW));
            return (ego_vs - utils::MAX_VEL).min(0.0f) / utils::MAX_VEL * cross_evaluation_movement_penalty;
        }

        inline float GetGoalPenalty(bool low_speed_mode, float miss_goal_penalty, float ego_goal_diff_frenet_s)
        {
            if (utils::approaching_terminal_point)
            {
                float goal_penalty_factor =
                    low_speed_mode ? utils::GOAL_PENALTY_FACTOR_LOW : utils::GOAL_PENALTY_FACTOR_HIGH;
                return miss_goal_penalty * goal_penalty_factor *
                       ((ego_goal_diff_frenet_s < utils::GOAL_DISTANCE_THRESHOLD)
                            ? 1.0f
                            : utils::GOAL_DISTANCE_THRESHOLD /
                                  std::max(ego_goal_diff_frenet_s, utils::GOAL_DISTANCE_MIN_DENOMINATOR));
            }
            else
            {
                return miss_goal_penalty * utils::GOAL_PENALTY_FACTOR_LOW;
            }
        }

        template <typename T, typename U>
        inline T GetGoalPenalty(const std::vector<std::shared_ptr<Path>> &ego_paths, const T &ego_vs,
                                const U &target_path_idxs, const T &active_flags)
        {
            AlignedVectorFloat miss_goal_penalties(T::num_scalars, 0.0f);
            const auto         target_path_idxs_array = target_path_idxs.to_array();
            for (std::size_t i = 0; i < T::num_scalars; ++i)
            {
                miss_goal_penalties[i] = ego_paths[target_path_idxs_array[i]]->miss_goal_penalty_;
            }

            T miss_goal_penalties_vector = T(miss_goal_penalties.data());
            T miss_goal_penalties_factor = T::select(ego_vs > utils::REWARD_SPEED_HIGH, utils::GOAL_PENALTY_FACTOR_HIGH,
                                                     utils::GOAL_PENALTY_FACTOR_LOW);

            return active_flags * miss_goal_penalties_vector * miss_goal_penalties_factor;
        }

        template <typename T>
        inline T GetGoalPenalty(bool low_speed_mode, const T &miss_goal_penalty, const T &ego_goal_diff_frenet_ss,
                                const T &active_flags)
        {
            if (utils::approaching_terminal_point)
            {
                float goal_penalty_factor =
                    low_speed_mode ? utils::GOAL_PENALTY_FACTOR_LOW : utils::GOAL_PENALTY_FACTOR_HIGH;
                return active_flags * miss_goal_penalty * goal_penalty_factor *
                       T::select(ego_goal_diff_frenet_ss < utils::GOAL_DISTANCE_THRESHOLD, 1.0f,
                                 utils::GOAL_DISTANCE_THRESHOLD /
                                     ego_goal_diff_frenet_ss.max(utils::GOAL_DISTANCE_MIN_DENOMINATOR));
            }
            else
            {
                return active_flags * miss_goal_penalty * utils::GOAL_PENALTY_FACTOR_LOW;
            }
        }

        inline float GetSteeringPenalty(float speed, float steering_command)
        {
            float abs_steering = std::fabs(steering_command);
            return (abs_steering > utils::STEERING_THRESHOLD && speed > utils::REWARD_SPEED_LOW)
                       ? abs_steering * ((speed > utils::STEERING_PENALTY_SPEED_HIGH)
                                             ? utils::STEERING_PENALTY_WEIGHT_HIGH
                                             : utils::STEERING_PENALTY_WEIGHT_LOW)
                       : 0.0f;
        }

        template <typename T>
        inline T GetSteeringPenalty(const T &speeds, const T &steering_commands, const T &active_flags)
        {
            auto abs_steering = steering_commands.abs();
            return active_flags * T::select(abs_steering > utils::STEERING_THRESHOLD & speeds > utils::REWARD_SPEED_LOW,
                                            abs_steering * T::select(speeds > utils::STEERING_PENALTY_SPEED_HIGH,
                                                                     utils::STEERING_PENALTY_WEIGHT_HIGH,
                                                                     utils::STEERING_PENALTY_WEIGHT_LOW),
                                            0.0f);
        }

        inline float GetActionPenalty(int curr_path_idx, int target_path_idx)
        {
            return (curr_path_idx != target_path_idx) ? utils::ACTION_PENALTY : 0.0f;
        }

        template <typename T, typename U>
        inline T GetActionPenalty(const U &curr_path_idxs, const U &target_path_idxs, const T &active_flags)
        {
            return active_flags * (curr_path_idxs != target_path_idxs).template as<T>().abs() * utils::ACTION_PENALTY;
        }

        template <typename T> inline T GetLateralOffsetPenaltyBatch(const T &ego_vs, const T &lateral_offsets)
        {
            return T::select(lateral_offsets.abs() >= utils::LATERAL_OFFSET_HIGH_THRESHOLD,
                             T::select(ego_vs > utils::REWARD_SPEED_HIGH, utils::LATERAL_OFFSET_PENALTY_HIGH,
                                       utils::LATERAL_OFFSET_PENALTY_LOW),
                             T::select(lateral_offsets.abs() > 0.0f, utils::LATERAL_OFFSET_PENALTY_LOW, 0.0f));
        }

        /**
         * @brief Lateral offset penalty guiding the vehicle away from non-drivable area.
         * Linear penalty that encourages moving in the correct direction with minimal adjustment.
         */
        template <typename T, typename U>
        inline T GetLateralOffsetOnNonDrivableAreaPenaltyBatch(const T &lateral_offset,
                                                               const U &left_corners_on_non_drivable_area_mask,
                                                               const U &right_corners_on_non_drivable_area_mask)
        {
            // Left corners on non-drivable area: encourage moving right (negative lateral_offset).
            // Penalty increases linearly with positive lateral_offset.
            auto left_penalty = T::select(
                left_corners_on_non_drivable_area_mask.template as<T>(),
                lateral_offset * utils::LATERAL_OFFSET_GUIDANCE + utils::LATERAL_OFFSET_GUIDANCE_OFFSET, 0.0f);

            // Right corners on non-drivable area: encourage moving left (positive lateral_offset).
            // Penalty increases linearly with negative lateral_offset.
            auto right_penalty = T::select(
                right_corners_on_non_drivable_area_mask.template as<T>(),
                -lateral_offset * utils::LATERAL_OFFSET_GUIDANCE + utils::LATERAL_OFFSET_GUIDANCE_OFFSET, 0.0f);

            return left_penalty + right_penalty;
        }

        template <typename T, typename U>
        inline T GetOnNonRoutePenalty(float ego_v, const U &on_non_route_mask, const T &ego_path_curvatures_proposal)
        {
            if (ego_v <= utils::REWARD_SPEED_VERY_LOW)
                return T::fill(0.0f);

            float penalty_factor;
            if (ego_v <= utils::REWARD_SPEED_MID)
                penalty_factor = utils::ON_NON_ROUTE_PENALTY_LOW;
            else if (ego_v <= utils::REWARD_SPEED_HIGH)
                penalty_factor = utils::ON_NON_ROUTE_PENALTY_MID;
            else
                penalty_factor = utils::ON_NON_ROUTE_PENALTY_HIGH;

            return on_non_route_mask.template convert<T>().abs() * penalty_factor *
                   T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_HIGH_THRESHOLD,
                             utils::CURVATURE_PENALTY_HIGH_MULTIPLIER,
                             T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_MED_THRESHOLD,
                                       utils::CURVATURE_PENALTY_MED_MULTIPLIER, 1.0f));
        }

        template <typename T, typename U>
        inline T GetOnNonRoutePenalty(const T &ego_vs, const U &on_non_route_mask,
                                      const T &ego_path_curvatures_proposal)
        {
            // Increase penalty in high-curvature regions.
            return on_non_route_mask.template convert<T>().abs() *
                   T::select(ego_vs > utils::REWARD_SPEED_HIGH, utils::ON_NON_ROUTE_PENALTY_HIGH,
                             T::select(ego_vs > utils::REWARD_SPEED_MID, utils::ON_NON_ROUTE_PENALTY_MID,
                                       utils::ON_NON_ROUTE_PENALTY_LOW)) *
                   T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_HIGH_THRESHOLD,
                             utils::CURVATURE_PENALTY_HIGH_MULTIPLIER,
                             T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_MED_THRESHOLD,
                                       utils::CURVATURE_PENALTY_MED_MULTIPLIER, 1.0f));
        }

        template <typename T, typename U>
        inline T GetOnDifferentPathLanePenaltyBatch(float ego_v, const U &on_different_path_lane_mask,
                                                    const U &on_non_route_mask, const T &ego_path_curvatures_proposal)
        {
            // Increase penalty in high-curvature regions.
            // Only penalize on-route vehicles.
            if (ego_v <= utils::REWARD_SPEED_VERY_LOW)
                return T::fill(0.0f);

            float penalty_factor = (ego_v <= utils::REWARD_SPEED_MID) ? utils::ON_DIFFERENT_PATH_LANES_PENALTY_LOW
                                                                      : utils::ON_DIFFERENT_PATH_LANES_PENALTY_HIGH;

            return (on_different_path_lane_mask & ~on_non_route_mask).template convert<T>().abs() * penalty_factor *
                   T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_HIGH_THRESHOLD,
                             utils::CURVATURE_PENALTY_HIGH_MULTIPLIER,
                             T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_MED_THRESHOLD,
                                       utils::CURVATURE_PENALTY_MED_MULTIPLIER, 1.0f));
        }

        template <typename T, typename U>
        inline T GetOnDifferentPathLanePenaltyBatch(const T &ego_vs, const U &on_different_path_lane_mask,
                                                    const U &on_non_route_mask, const T &ego_path_curvatures_proposal)
        {
            // Increase penalty in high-curvature regions.
            // Only penalize on-route vehicles.
            return (on_different_path_lane_mask & ~on_non_route_mask).template convert<T>().abs() *
                   T::select(ego_vs > utils::REWARD_SPEED_MID, utils::ON_DIFFERENT_PATH_LANES_PENALTY_HIGH,
                             utils::ON_DIFFERENT_PATH_LANES_PENALTY_LOW) *
                   T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_HIGH_THRESHOLD,
                             utils::CURVATURE_PENALTY_HIGH_MULTIPLIER,
                             T::select(ego_path_curvatures_proposal > utils::CURVATURE_PENALTY_MED_THRESHOLD,
                                       utils::CURVATURE_PENALTY_MED_MULTIPLIER, 1.0f));
        }

        template <typename T, typename U> inline T GetNonDrivableAreaPenaltyBatch(const U &on_non_drivable_area_masks)
        {
            return T::select(on_non_drivable_area_masks.template as<T>(), utils::NON_ON_DRIVABLE_AREA_PENALTY_STAGE_1,
                             0.0f);
        }

        // =====================================================================
        // nuPlan metric reward / penalty functions
        // =====================================================================

        template <typename T, typename U>
        inline T GetDirectionCompliancePenaltyBatch(const T &ego_vs, const U &ego_oncoming_traffic,
                                                    const U &on_non_drivable_area, T &ego_oncoming_culmulative_vs)
        {
            ego_oncoming_culmulative_vs =
                T::select(ego_oncoming_traffic.template as<T>(), ego_oncoming_culmulative_vs + ego_vs, 0.0f);

            T violation_mask = (ego_oncoming_culmulative_vs >= utils::DRIVING_DIRECTION_VIOLATION_SPEED_THRESHOLD) &
                               (~on_non_drivable_area).template as<T>();

            T penalty = T::select(violation_mask, utils::DIR_VIOLATION_PENALTY, 0.0f);

            // Reset accumulator after a violation is registered.
            ego_oncoming_culmulative_vs = T::select(violation_mask, 0.0f, ego_oncoming_culmulative_vs);

            return penalty;
        }

        // =====================================================================
        // Geometry helper — line-rectangle intersection (used by collision type)
        // =====================================================================

        /**
         * @brief Tests whether the line segment [ego_left_front, ego_right_front] intersects the
         *        axis-aligned bounding rectangle of each exo vehicle (SIMD batch).
         * @return Per-lane bitmask: all bits set = intersection detected.
         */
        template <typename T, typename U>
        U checkIntersectionLineRectangleBatch(const T &ego_left_front_corner_xs, const T &ego_left_front_corner_ys,
                                              const T &ego_right_front_corner_xs, const T &ego_right_front_corner_ys,
                                              const T &exo_xs, const T &exo_ys, const T &exo_cos_thetas,
                                              const T &exo_sin_thetas, const T &exo_bb_extent_xs,
                                              const T &exo_bb_extent_ys, U active_flags)
        {
            // === Step 1: Transform endpoints to the rectangle's local coordinate frame ===
            T p1_rel_x = ego_left_front_corner_xs - exo_xs;
            T p1_rel_y = ego_left_front_corner_ys - exo_ys;
            T p2_rel_x = ego_right_front_corner_xs - exo_xs;
            T p2_rel_y = ego_right_front_corner_ys - exo_ys;

            // Inverse rotation: [cos(θ) sin(θ); -sin(θ) cos(θ)]
            T p1_local_x = exo_cos_thetas * p1_rel_x + exo_sin_thetas * p1_rel_y;
            T p1_local_y = -exo_sin_thetas * p1_rel_x + exo_cos_thetas * p1_rel_y;
            T p2_local_x = exo_cos_thetas * p2_rel_x + exo_sin_thetas * p2_rel_y;
            T p2_local_y = -exo_sin_thetas * p2_rel_x + exo_cos_thetas * p2_rel_y;

            // === Step 2: Endpoint-inside test ===
            // Note: extent_xs maps to local y axis, extent_ys maps to local x axis.
            T p1_inside = (p1_local_x.abs() <= exo_bb_extent_ys) & (p1_local_y.abs() <= exo_bb_extent_xs);
            T p2_inside = (p2_local_x.abs() <= exo_bb_extent_ys) & (p2_local_y.abs() <= exo_bb_extent_xs);

            U is_intersecting = (p1_inside | p2_inside).template as<U>();
            active_flags = active_flags & (~is_intersecting);
            if (active_flags.none())
                return is_intersecting;

            // === Step 3: SAT test for line vs AABB ===
            // 3.1 Check rectangle axes (AABB axes)
            T min_x = p1_local_x.min(p2_local_x);
            T max_x = p1_local_x.max(p2_local_x);
            T separated_x = (min_x > exo_bb_extent_ys) | (max_x < -exo_bb_extent_ys);

            T min_y = p1_local_y.min(p2_local_y);
            T max_y = p1_local_y.max(p2_local_y);
            T separated_y = (min_y > exo_bb_extent_xs) | (max_y < -exo_bb_extent_xs);

            U is_separated = (separated_x | separated_y).template as<U>();
            active_flags = active_flags & (~is_separated);
            if (active_flags.none())
                return is_intersecting;

            // 3.2 Check line-segment normal axis (cross-product axis)
            // Axis vector N = (-dy, dx); radius = extent_x*|dy| + extent_y*|dx|
            T dx = p2_local_x - p1_local_x;
            T dy = p2_local_y - p1_local_y;
            T projected_radius = exo_bb_extent_ys * dy.abs() + exo_bb_extent_xs * dx.abs();
            T distance_to_center = (p1_local_x * -dy + p1_local_y * dx).abs();
            T separated_cross = distance_to_center > projected_radius;
            is_separated = is_separated | separated_cross.template as<U>();

            return is_intersecting | ((~is_separated) & active_flags);
        }

        // =====================================================================
        // nuPlan collision classification and no-at-fault penalty
        // =====================================================================

        /**
         * @brief Classifies the type of each collision (stopped ego/track, front, rear, lateral).
         * @return Per-lane integer encoding of CollisionType.
         */
        template <typename T, typename U>
        U GetCollisionTypeBatch(const U &collision_flags, const T &ego_rear_xs, const T &ego_rear_ys,
                                const T &ego_left_front_corner_xs, const T &ego_left_front_corner_ys,
                                const T &ego_right_front_corner_xs, const T &ego_right_front_corner_ys, const T &ego_vs,
                                const T &ego_thetas, const T &ego_thetas_cos, const T &ego_thetas_sin, const T &exo_xs,
                                const T &exo_ys, const T &exo_vs, const T &exo_thetas, const T &exo_cos_thetas,
                                const T &exo_sin_thetas, const T &exo_bb_extent_xs, const T &exo_bb_extent_ys,
                                int iteration)
        {
            U collision_type = U::fill(0); // CollisionType::NO_COLLISION
            U active_flags = collision_flags;

            // Stopped-ego collision (skip on first iteration to avoid false positives)
            if (iteration > 0)
            {
                U zero_ego_speed_mask = (ego_vs <= utils::VELOCITY_ALMOST_ZERO_THRESHOLD).template as<U>();
                collision_type = U::select(active_flags & zero_ego_speed_mask,
                                           (int)CollisionType::STOPPED_EGO_COLLISION, collision_type);
                active_flags = active_flags & (~zero_ego_speed_mask);
                if (active_flags.none())
                    return collision_type;
            }

            // Stopped-track collision
            U zero_track_speed_mask = (exo_vs.abs() <= utils::VELOCITY_ALMOST_ZERO_THRESHOLD).template as<U>();
            collision_type = U::select(active_flags & zero_track_speed_mask,
                                       (int)CollisionType::STOPPED_TRACK_COLLISION, collision_type);
            active_flags = active_flags & (~zero_track_speed_mask);
            if (active_flags.none())
                return collision_type;

            // Rear collision when both ego and track are moving
            T rel_x = exo_xs - ego_rear_xs;
            T rel_y = exo_ys - ego_rear_ys;

            T pos_cond1 =
                utils::IsAngleWithinThreshold<T>(rel_x, rel_y, -ego_thetas_cos, -ego_thetas_sin, utils::PI_1_6);
            T head_cond1 = utils::IsAngleWithinThreshold<T>(exo_cos_thetas, exo_sin_thetas, ego_thetas_cos,
                                                            ego_thetas_sin, utils::COLLISION_REAR_HEADING_THRESHOLD);
            T pos_cond2 = utils::IsAngleWithinThreshold<T>(rel_x, rel_y, -ego_thetas_cos, -ego_thetas_sin,
                                                           utils::COLLISION_REAR_POSITION_THRESHOLD);
            T head_cond2 = utils::IsAngleWithinThreshold<T>(exo_cos_thetas, exo_sin_thetas, ego_thetas_cos,
                                                            ego_thetas_sin, utils::PI_1_6);

            U in_behind_mask = ((pos_cond1 & head_cond1) | (pos_cond2 & head_cond2)).template as<U>();
            collision_type =
                U::select(active_flags & in_behind_mask, (int)CollisionType::REAR_COLLISION, collision_type);
            active_flags = active_flags & (~in_behind_mask);
            if (active_flags.none())
                return collision_type;

            // Front collision: ego front edge intersects exo bounding box
            U front_rear_intersect_mask = checkIntersectionLineRectangleBatch(
                ego_left_front_corner_xs, ego_left_front_corner_ys, ego_right_front_corner_xs,
                ego_right_front_corner_ys, exo_xs, exo_ys, exo_cos_thetas, exo_sin_thetas, exo_bb_extent_xs,
                exo_bb_extent_ys, active_flags);

            collision_type = U::select(active_flags & front_rear_intersect_mask,
                                       (int)CollisionType::ACTIVE_FRONT_COLLISION, collision_type);
            active_flags = active_flags & (~front_rear_intersect_mask);

            // Remaining active collisions are lateral
            collision_type = U::select(active_flags, (int)CollisionType::ACTIVE_LATERAL_COLLISION, collision_type);

            return collision_type;
        }

        /**
         * @brief Computes the no-at-fault collision penalty (nuPlan metric).
         *        Also updates exo_inactive_flags for agents the ego is not at fault for.
         */
        template <typename T, typename U>
        T GetNoAtFaultCollisionPenaltyBatch(
            int scenario_offset, const U &original_collision_flags, const U &expanded_collision_flags,
            U &true_original_collision_flags, const U &on_multiple_lanes, const U &on_non_drivable_area,
            const T &ego_rear_xs, const T &ego_rear_ys, const T &ego_left_front_corner_xs,
            const T &ego_left_front_corner_ys, const T &ego_right_front_corner_xs, const T &ego_right_front_corner_ys,
            const T &ego_vs, const T &ego_thetas, const T &ego_thetas_cos, const T &ego_thetas_sin, const T &exo_xs,
            const T &exo_ys, const T &exo_vs, const T &exo_thetas, const T &exo_cos_thetas, const T &exo_sin_thetas,
            const T &exo_expanded_bb_extent_xs, const T &exo_expanded_bb_extent_ys, const T &collided_min_distances,
            const AlignedVectorInt                                &collided_agent_idxs,
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> &exo_inactive_flags, int iteration)
        {
            U collision_type = GetCollisionTypeBatch<T, U>(
                expanded_collision_flags, ego_rear_xs, ego_rear_ys, ego_left_front_corner_xs, ego_left_front_corner_ys,
                ego_right_front_corner_xs, ego_right_front_corner_ys, ego_vs, ego_thetas, ego_thetas_cos,
                ego_thetas_sin, exo_xs, exo_ys, exo_vs, exo_thetas, exo_cos_thetas, exo_sin_thetas,
                exo_expanded_bb_extent_xs, exo_expanded_bb_extent_ys, iteration);

            U collisions_at_stopped_track_or_active_front =
                (collision_type == (int)CollisionType::STOPPED_TRACK_COLLISION) |
                (collision_type == (int)CollisionType::ACTIVE_FRONT_COLLISION);

            U collision_at_lateral = (collision_type == (int)CollisionType::ACTIVE_LATERAL_COLLISION);
            U ego_in_multiple_lanes_or_nondrivable_area = (on_multiple_lanes | on_non_drivable_area);

            U first_iteration_flag = (iteration == 0) ? U::fill(0xFFFFFFFF) : U::fill(0);
            U true_expanded_collision_flags = collisions_at_stopped_track_or_active_front |
                                              (ego_in_multiple_lanes_or_nondrivable_area & collision_at_lateral);
            true_original_collision_flags =
                (original_collision_flags |
                 (first_iteration_flag & (ego_vs < utils::REWARD_SPEED_MID).template as<U>())) &
                true_expanded_collision_flags;
            U inactive_collision_flags = ~true_original_collision_flags & original_collision_flags;

            if (inactive_collision_flags.any())
            {
                auto inactive_collision_arr = inactive_collision_flags.to_array();
                for (size_t scenario_idx = 0; scenario_idx < T::num_scalars; ++scenario_idx)
                {
                    if (inactive_collision_arr[scenario_idx] && collided_agent_idxs[scenario_idx] != -1)
                    {
                        exo_inactive_flags[scenario_idx][collided_agent_idxs[scenario_idx]] = 0xFFFFFFFF;
                    }
                }
            }

            // Original (non-expanded) collisions incur full penalty.
            T original_penalty =
                T::select(true_original_collision_flags.template as<T>(), utils::CROSS_EVALUATION_PENALTY, 0.0f);

            // Short-horizon expanded collisions incur a distance-scaled partial penalty.
            if (iteration <= utils::NO_AT_FAULT_SHORT_HORIZON_ITERATIONS)
            {
                T dist_diff = (utils::NO_AT_FAULT_DISTANCE_THRESHOLD - collided_min_distances).max(0.0f);
                T distance_penalty = (utils::CROSS_EVALUATION_PENALTY * utils::NO_AT_FAULT_PENALTY_FACTOR) * dist_diff;
                T expanded_penalty =
                    T::select((true_expanded_collision_flags & (~true_original_collision_flags)).template as<T>(),
                              distance_penalty, 0.0f);
                return original_penalty + expanded_penalty;
            }
            else
            {
                return original_penalty;
            }
        }

    } // namespace planning
} // namespace vec_qmdp
