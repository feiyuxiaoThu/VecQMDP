/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <core/net_belief.hpp>
#include <unordered_set>

namespace vec_qmdp
{
    namespace core
    {
        NetBelief::NetBelief() { context_qmdp_ = std::make_shared<ContextQMDP>(); }

        NetBelief::~NetBelief() {}

        void NetBelief::updateEgoRefPaths(const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                          const std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths,
                                          const size_t                              ego_curr_ref_path_idx)
        {
            ego_ref_paths_ = ego_ref_paths;
            for (const auto &extra_ref_path : ego_extra_ref_paths)
            {
                if (extra_ref_path != nullptr)
                {
                    ego_ref_paths_.emplace_back(extra_ref_path);
                }
            }
            ego_curr_ref_path_idx_ = ego_curr_ref_path_idx;
        }

        void NetBelief::initializeBelief() {}

        void NetBelief::updateBelief(const py::object &pred_traj_x, const py::object &pred_traj_y,
                                     const py::object &pred_traj_v, const py::object &pred_traj_theta,
                                     const py::object &pred_traj_theta_cos, const py::object &pred_traj_theta_sin,
                                     const py::object &pred_prob)
        {
            // Extract numpy array types
            np::ndarray pred_traj_x_nd = py::extract<np::ndarray>(pred_traj_x);
            np::ndarray pred_traj_y_nd = py::extract<np::ndarray>(pred_traj_y);
            np::ndarray pred_traj_v_nd = py::extract<np::ndarray>(pred_traj_v);
            np::ndarray pred_traj_theta_nd = py::extract<np::ndarray>(pred_traj_theta);
            np::ndarray pred_traj_theta_cos_nd = py::extract<np::ndarray>(pred_traj_theta_cos);
            np::ndarray pred_traj_theta_sin_nd = py::extract<np::ndarray>(pred_traj_theta_sin);
            np::ndarray pred_prob_nd = py::extract<np::ndarray>(pred_prob);
            pred_traj_x_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_x_nd.get_data());
            pred_traj_y_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_y_nd.get_data());
            pred_traj_v_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_v_nd.get_data());
            pred_traj_theta_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_theta_nd.get_data());
            pred_traj_theta_cos_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_theta_cos_nd.get_data());
            pred_traj_theta_sin_nd_data_ptr_ = reinterpret_cast<float *>(pred_traj_theta_sin_nd.get_data());
            pred_prob_nd_data_ptr_ = reinterpret_cast<float *>(pred_prob_nd.get_data());
        }

        /**
         * Filter out discarded agents
         *
         * @param ego_x Ego vehicle x coordinate
         * @param ego_y Ego vehicle y coordinate
         * @param ego_theta Ego vehicle heading
         * @param static_motion Infer static motion (e.g., red light stop) from surrounding agents; if true, ego is
         * considered static. Defaults to true.
         */
        void NetBelief::filterDiscardedAgents(float ego_x, float ego_y, float ego_v, float ego_theta,
                                              bool &static_motion)
        {
            // If there are no exo vehicles or no reference paths, return immediately
            if (observed_exo_state_.num_vehicles_ == 0)
            {
                return;
            }

            float ego_theta_cos = std::cos(ego_theta);
            float ego_theta_sin = std::sin(ego_theta);

            // nearby agent (if there is not any nearby agent, then we can't infer whether it is static motion)
            bool has_nearby_agent = false;

            // Reset agent_valid_mask_ elements to false
            std::fill(observed_exo_state_.agent_valid_mask_.begin(), observed_exo_state_.agent_valid_mask_.end(),
                      false);

            utils::NOTICE_VEHICLE_IDXS.clear();

            const int batch_size = FloatVectorWidth; // SIMD vector width

            // 1. For each reference path, compute ego and exo Frenet coordinates
            std::vector<float> ego_frenet_s(ego_ref_paths_.size(), 0.0f);

            // Compute ego's Frenet coordinate on each reference path at the current timestep
            for (size_t p = 0; p < ego_ref_paths_.size(); ++p)
            {
                std::shared_ptr<Path> path = ego_ref_paths_[p];
                if (path == nullptr)
                {
                    continue;
                }

                // Calculate ego vehicle's nearest point on current path at current moment
                int ego_nearest_idx = path->NearestBatch(ego_x, ego_y, ego_theta);
                ego_frenet_s[p] = ego_nearest_idx * utils::PATH_POINT_INTERVAL;
            }

            // Determine if static motion should be used based on whether ego is at red light
            if (ego_ref_paths_[ego_curr_ref_path_idx_] &&
                ego_frenet_s[ego_curr_ref_path_idx_] > ego_ref_paths_[ego_curr_ref_path_idx_]->red_light_point_s_ &&
                ego_frenet_s[ego_curr_ref_path_idx_] <
                    ego_ref_paths_[ego_curr_ref_path_idx_]->red_light_point_s_ + utils::RED_LIGHT_PROXIMITY_DISTANCE)
            {
                static_motion = true;
                return;
            }

            // 2. Filter agents by Frenet coordinates (keep leading vehicles on the current reference path and vehicles
            // with lateral distance < 2.5 on any path)
            float min_lead_s = utils::MAX_VALUE;
            int   closest_lead_vehicle_idx = -1;
            float max_follow_s = -utils::MAX_VALUE;
            int   closest_follow_vehicle_idx = -1;

            // First pass: mark non-vehicle obstacles and find the closest leading vehicle
            std::unordered_map<int, float> curr_path_lead_vehicle_idx_map_frenet_ls;
            std::unordered_map<int, float> curr_path_follow_vehicle_idx_map_frenet_ls;
            for (size_t v = 0; v < observed_exo_state_.num_vehicles_; ++v)
            {
                // Check if the object is valid
                if (observed_exo_state_.obs_xs_[v] == 0 && observed_exo_state_.obs_ys_[v] == 0 &&
                        observed_exo_state_.obs_vs_[v] == 0 ||
                    collided_agents_tokens_.count(observed_exo_state_.tokens_[v]))
                {
                    continue;
                }

                float observed_v =
                    (std::fabs(utils::NormalizeAngle(ego_theta - observed_exo_state_.obs_thetas_[v])) > utils::PI_2_3)
                        ? -observed_exo_state_.obs_vs_[v]
                        : observed_exo_state_.obs_vs_[v];

                // Check each reference path
                // First check the current lane reference line
                if (ego_ref_paths_[ego_curr_ref_path_idx_] != nullptr)
                {
                    float frenet_s = observed_exo_state_.obs_frenet_s_[ego_curr_ref_path_idx_][v];
                    float abs_frenet_l = std::fabs(observed_exo_state_.obs_frenet_l_[ego_curr_ref_path_idx_][v]);

                    if (static_motion && observed_exo_state_.agent_vehicle_mask_[v] &&
                        abs_frenet_l <= utils::FILTER_ON_LANE_LATERAL_THRESHOLD &&
                        ego_frenet_s[ego_curr_ref_path_idx_] - frenet_s <=
                            utils::FILTER_STATIC_MOTION_LONGITUDINAL_AHEAD &&
                        ego_frenet_s[ego_curr_ref_path_idx_] - frenet_s > 0.0f)
                    {
                        has_nearby_agent = true;
                        if (observed_v > utils::VELOCITY_ALMOST_ZERO_THRESHOLD)
                            static_motion = false;
                    }

                    // For non-vehicle objects and stationary (or reversed) vehicles (e.g., parked on wide roads), allow
                    // looser lateral distance
                    if (!observed_exo_state_.agent_vehicle_mask_[v])
                    {
                        if (abs_frenet_l <= utils::FILTER_NON_VEHICLE_LATERAL_THRESHOLD)
                        {
                            observed_exo_state_.agent_valid_mask_[v] = true;
                        }
                    }
                    else if (observed_v <= utils::VELOCITY_ALMOST_ZERO_THRESHOLD &&
                             abs_frenet_l >= utils::FILTER_ON_LANE_LATERAL_THRESHOLD &&
                             abs_frenet_l <= utils::FILTER_STATIONARY_VEHICLE_LATERAL_MAX)
                    {
                        observed_exo_state_.agent_valid_mask_[v] = true;
                    }
                    else
                    {
                        // Consider lead vehicle
                        if (frenet_s > ego_frenet_s[ego_curr_ref_path_idx_])
                        {
                            if (abs_frenet_l < utils::FILTER_VEHICLE_LATERAL_THRESHOLD)
                            {
                                if (frenet_s < min_lead_s)
                                {
                                    min_lead_s = frenet_s;
                                    closest_lead_vehicle_idx = v;
                                }
                                curr_path_lead_vehicle_idx_map_frenet_ls[v] = abs_frenet_l;

                                // Patch: notice vehicles that appear far in Frenet but are actually nearby due to
                                // looped roads
                                if (frenet_s - ego_frenet_s[ego_curr_ref_path_idx_] >
                                        utils::NOTICE_VEHICLE_DISTANCE_FRENET_S &&
                                    utils::SquaredDistance(observed_exo_state_.obs_xs_[v],
                                                           observed_exo_state_.obs_ys_[v], ego_x,
                                                           ego_y) < utils::NOTICE_VEHICLE_DISTANCE_SQUARE &&
                                    std::fabs(utils::NormalizeAngle(observed_exo_state_.obs_thetas_[v] - ego_theta)) >
                                        utils::NOTICE_VEHICLE_HEADING_DIFF)
                                {
                                    utils::NOTICE_VEHICLE_IDXS.emplace_back(v);
                                }
                            }
                            else if (abs_frenet_l <
                                     utils::FILTER_WIDE_LATERAL_KEEP_THRESHOLD) // If other vehicle's lateral distance
                                                                                // is large and ahead of ego, keep it
                                                                                // directly (including opposite lane
                                                                                // vehicles)
                            {
                                observed_exo_state_.agent_valid_mask_[v] = true;
                            }
                        }

                        // Consider follow vehicle
                        if (abs_frenet_l < utils::FILTER_STATIONARY_VEHICLE_LATERAL_MAX &&
                            frenet_s < ego_frenet_s[ego_curr_ref_path_idx_] - utils::FILTER_FOLLOW_LONGITUDINAL_OFFSET)
                        {
                            if (abs_frenet_l < utils::FILTER_VEHICLE_LATERAL_THRESHOLD && frenet_s > max_follow_s &&
                                std::fabs(utils::NormalizeAngle(observed_exo_state_.obs_thetas_[v] - ego_theta)) >
                                    utils::PI_1_10)
                            {
                                max_follow_s = frenet_s;
                                closest_follow_vehicle_idx = v;
                            }

                            curr_path_follow_vehicle_idx_map_frenet_ls[v] = abs_frenet_l;
                        }
                    }
                }

                // Then check other lane reference lines
                for (size_t p = 0; p < ego_ref_paths_.size(); ++p)
                {
                    if (ego_ref_paths_[p] == nullptr || p == ego_curr_ref_path_idx_)
                    {
                        continue;
                    }

                    float frenet_s = observed_exo_state_.obs_frenet_s_[p][v];
                    float abs_frenet_l = std::fabs(observed_exo_state_.obs_frenet_l_[p][v]);

                    if (abs_frenet_l > utils::FILTER_OTHER_LANE_LATERAL_CUTOFF)
                    {
                        continue;
                    }

                    if (static_motion && observed_exo_state_.agent_vehicle_mask_[v] &&
                        abs_frenet_l <= utils::FILTER_ON_LANE_LATERAL_THRESHOLD &&
                        ego_frenet_s[p] - frenet_s <= utils::FILTER_STATIC_MOTION_LONGITUDINAL_AHEAD &&
                        ego_frenet_s[p] - frenet_s > utils::FILTER_STATIC_MOTION_LONGITUDINAL_BEHIND)
                    {
                        has_nearby_agent = true;
                        if (observed_v > utils::VELOCITY_ALMOST_ZERO_THRESHOLD)
                            static_motion = false;
                    }

                    // For non-vehicle objects, keep if conditions are met
                    if (!observed_exo_state_.agent_vehicle_mask_[v])
                    {
                        observed_exo_state_.agent_valid_mask_[v] = true;
                    }
                    // For vehicles on other paths, keep if lateral distance criteria are met
                    else if (observed_exo_state_.agent_vehicle_mask_[v])
                    {
                        // If vehicle is actually on adjacent lane
                        // Assuming curr_path_lead_vehicle_idx_map_frenet_ls and
                        // curr_path_follow_vehicle_idx_map_frenet_ls are of type std::map<int, float>
                        float threshold = utils::FILTER_ADJACENT_LANE_LATERAL_OFFSET;
                        bool  is_lead = curr_path_lead_vehicle_idx_map_frenet_ls.count(v);
                        bool  is_follow = curr_path_follow_vehicle_idx_map_frenet_ls.count(v);

                        bool is_valid = false;

                        if (!is_lead && !is_follow)
                        {
                            is_valid = true;
                        }
                        else if (is_lead)
                        {
                            is_valid = (abs_frenet_l < curr_path_lead_vehicle_idx_map_frenet_ls[v] + threshold);
                        }
                        else if (is_follow)
                        {
                            is_valid = (abs_frenet_l < curr_path_follow_vehicle_idx_map_frenet_ls[v] + threshold) &&
                                       abs_frenet_l !=
                                           curr_path_follow_vehicle_idx_map_frenet_ls[v]; // exclude the situation where
                                                                                          // the two reference lines
                                                                                          // share the same current path
                        }

                        if (is_valid)
                        {
                            observed_exo_state_.agent_valid_mask_[v] = true;
                        }
                    }
                }
            }

            if (!has_nearby_agent)
            {
                static_motion = false;
            }

            // Mark the closest leading vehicle on the current path
            if (closest_lead_vehicle_idx >= 0)
            {
                observed_exo_state_.agent_valid_mask_[closest_lead_vehicle_idx] = true;
            }

            // Mark the closest following vehicle on the current path
            if (closest_follow_vehicle_idx >= 0)
            {
                observed_exo_state_.agent_valid_mask_[closest_follow_vehicle_idx] = true;
            }

            // 3. Use MLSample to check for path intersections and keep vehicles with intersections
            // Note: the scenarios sampled here are used for vehicle filtering, max vehicle num should be
            // observed_max_num
            assert(ego_ref_paths_[ego_curr_ref_path_idx_] != nullptr);
            const std::shared_ptr<Path> ego_curr_ref_path = ego_ref_paths_[ego_curr_ref_path_idx_];
            float                       longitudinal_threshold =
                utils::FILTER_INTERSECTION_LONGITUDINAL_THRESHOLD; // Longitudinal safety distance threshold
            float lateral_threshold = utils::FILTER_INTERSECTION_LATERAL_THRESHOLD; // Lateral safety distance threshold

            for (size_t i = 0; i < observed_exo_state_.num_vehicles_; ++i)
            {
                // Skip already processed vehicles, vehicles to ignore, or static objects (e.g., cones)
                if (observed_exo_state_.agent_valid_mask_[i] || curr_path_lead_vehicle_idx_map_frenet_ls.count(i) ||
                    curr_path_follow_vehicle_idx_map_frenet_ls.count(i) || !observed_exo_state_.agent_dynamic_mask_[i])
                {
                    continue;
                }

                // Find the highest probability mode
                size_t ml_mode = 0;
                float  max_prob = 0.0f;

                for (size_t m = 0; m < utils::MAX_PRED_MODES; ++m)
                {
                    const size_t prob_idx = i * utils::MAX_PRED_MODES + m;
                    if (pred_prob_nd_data_ptr_[prob_idx] > max_prob)
                    {
                        max_prob = pred_prob_nd_data_ptr_[prob_idx];
                        ml_mode = m;
                    }
                }

                // Calculate source data starting index
                size_t src_start_idx =
                    i * utils::MAX_PRED_MODES * utils::DUMMY_TIME_STEPS + ml_mode * utils::DUMMY_TIME_STEPS;

                // Get the angular relationship between vehicle and path at t=0, determine search direction
                uint32_t pre_nearest_idx = observed_exo_state_.obs_nearest_idxs_[ego_curr_ref_path_idx_][i];
                float    agent_theta_t0 = pred_traj_theta_nd_data_ptr_[src_start_idx];
                float    path_theta_t0 = ego_curr_ref_path->thetas_[pre_nearest_idx];

                // Calculate angle difference and normalize to [-π, π]
                float theta_diff = std::fabs(utils::NormalizeAngle(agent_theta_t0 - path_theta_t0));

                // Determine search direction based on angle difference
                int step_direction = (theta_diff < utils::PI_1_2) ? 1 : -1;

                float exo_bb_extent_x = observed_exo_state_.obs_expanded_bb_extent_xs_[i];
                float exo_bb_extent_y = observed_exo_state_.obs_expanded_bb_extent_ys_[i];

                IVectorT_1 prev_nearest_idxs = IVectorT_1::fill(pre_nearest_idx);
                IVectorT_1 step_directions = IVectorT_1::fill(step_direction);
                for (size_t t_batch = 0; t_batch < utils::MAX_PRED_TIME_STEPS - 1; t_batch += batch_size)
                {
                    // Get position at current time step
                    uint32_t   current_v_batch = t_batch + src_start_idx;
                    FVectorT_1 current_xs = FVectorT_1::load_contiguous(pred_traj_x_nd_data_ptr_, current_v_batch);
                    FVectorT_1 current_ys = FVectorT_1::load_contiguous(pred_traj_y_nd_data_ptr_, current_v_batch);

                    ego_curr_ref_path->NearestBatch(current_xs, current_ys, prev_nearest_idxs, step_directions);

                    // Calculate Frenet coordinates at current time step
                    FVectorT_1 path_xs = FVectorT_1::gather(ego_curr_ref_path->GetXs().data(), prev_nearest_idxs);
                    FVectorT_1 path_ys = FVectorT_1::gather(ego_curr_ref_path->GetYs().data(), prev_nearest_idxs);

                    FVectorT_1 dx = current_xs - path_xs;
                    FVectorT_1 dy = current_ys - path_ys;

                    FVectorT_1 frenet_s = ego_curr_ref_path->GetFrenetS(prev_nearest_idxs);
                    FVectorT_1 frenet_l_abs = (dx * dx + dy * dy).sqrt();

                    // Calculate projection radius at current time step
                    FVectorT_1 current_thetas =
                        FVectorT_1::load_contiguous(pred_traj_theta_nd_data_ptr_, current_v_batch);
                    FVectorT_1 current_thetas_cos = current_thetas.cos();
                    FVectorT_1 current_thetas_sin = current_thetas.sin();
                    FVectorT_1 path_thetas = FVectorT_1::gather(ego_curr_ref_path->thetas_.data(), prev_nearest_idxs);
                    FVectorT_1 path_thetas_cos = path_thetas.cos();
                    FVectorT_1 path_thetas_sin = path_thetas.sin();

                    FVectorT_1 proj_radius_ls =
                        (-current_thetas_cos * path_thetas_sin + current_thetas_sin * path_thetas_cos).abs() *
                            exo_bb_extent_y +
                        (current_thetas_sin * path_thetas_sin + current_thetas_cos * path_thetas_cos).abs() *
                            exo_bb_extent_x;

                    // Check if current time step has intersection
                    FVectorT_1 intersection_mask =
                        (frenet_s > ego_frenet_s[ego_curr_ref_path_idx_]) &
                        (frenet_s - ego_frenet_s[ego_curr_ref_path_idx_] < longitudinal_threshold) &
                        (frenet_l_abs < proj_radius_ls + utils::EGO_BB_EXTENT_X);

                    // Mark intersecting vehicles; once there is an intersection, stop checking
                    if (intersection_mask.any())
                    {
                        observed_exo_state_.agent_valid_mask_[i] = true;
                        break;
                    }
                }

                // Process the last time step
                if (!observed_exo_state_.agent_valid_mask_[i])
                {
                    int last_prev_nearest_idx = prev_nearest_idxs[{0, 7}];

                    // Get the position at current time step
                    uint32_t current_v_batch = utils::MAX_PRED_TIME_STEPS - 1 + src_start_idx;
                    float    current_x = pred_traj_x_nd_data_ptr_[current_v_batch];
                    float    current_y = pred_traj_y_nd_data_ptr_[current_v_batch];

                    // Batch search for nearest point
                    ego_curr_ref_path->NearestSerial(current_x, current_y, last_prev_nearest_idx, step_direction);

                    // Calculate Frenet coordinates at current time step
                    float path_x = ego_curr_ref_path->GetX(last_prev_nearest_idx);
                    float path_y = ego_curr_ref_path->GetY(last_prev_nearest_idx);

                    float dx = current_x - path_x;
                    float dy = current_y - path_y;

                    float frenet_s = ego_curr_ref_path->GetFrenetS(last_prev_nearest_idx);
                    float frenet_l_abs = std::sqrt(dx * dx + dy * dy);

                    // Calculate projection radius at current time step
                    float current_theta = pred_traj_theta_nd_data_ptr_[current_v_batch];
                    float current_theta_cos = std::cos(current_theta);
                    float current_theta_sin = std::sin(current_theta);
                    float path_theta = ego_curr_ref_path->thetas_[last_prev_nearest_idx];
                    float path_theta_cos = std::cos(path_theta);
                    float path_theta_sin = std::sin(path_theta);

                    float proj_radius_l =
                        std::fabs(-current_theta_cos * path_theta_sin + current_theta_sin * path_theta_cos) *
                            exo_bb_extent_y +
                        std::fabs(current_theta_sin * path_theta_sin + current_theta_cos * path_theta_cos) *
                            exo_bb_extent_x;

                    // Check if current time step has intersection
                    if (frenet_s > ego_frenet_s[ego_curr_ref_path_idx_] &&
                        frenet_s - ego_frenet_s[ego_curr_ref_path_idx_] < longitudinal_threshold &&
                        frenet_l_abs < proj_radius_l + utils::EGO_BB_EXTENT_X)
                    {
                        observed_exo_state_.agent_valid_mask_[i] = true;
                    }
                }
            }

            // Update collided_agents_ (check if there are any collided agents, and if so, add them to collided_agents_)
            int cal_whole_times = observed_exo_state_.num_vehicles_ / batch_size;
            int cal_remaining_num = observed_exo_state_.num_vehicles_ % batch_size;

            // Validate batches
            for (int i = 0; i < cal_whole_times * batch_size; i += batch_size)
            {
                FVectorT_1 exo_xs = FVectorT_1::load_contiguous(observed_exo_state_.obs_xs_.data(), i);
                FVectorT_1 exo_ys = FVectorT_1::load_contiguous(observed_exo_state_.obs_ys_.data(), i);
                FVectorT_1 exo_thetas = FVectorT_1::load_contiguous(observed_exo_state_.obs_thetas_.data(), i);
                FVectorT_1 exo_bb_extent_xs =
                    FVectorT_1::load_contiguous(observed_exo_state_.obs_expanded_bb_extent_xs_.data(), i);
                FVectorT_1 exo_bb_extent_ys =
                    FVectorT_1::load_contiguous(observed_exo_state_.obs_expanded_bb_extent_ys_.data(), i);

                // Check for margin collision
                FVectorT_1 non_margin_collision_flag = context_qmdp_->checkSATCollisionBatch1(
                    ego_x, ego_y, ego_theta_cos, ego_theta_sin, exo_xs, exo_ys, exo_thetas.cos(), exo_thetas.sin(),
                    utils::EGO_BB_EXTENT_X, utils::EGO_BB_EXTENT_Y, exo_bb_extent_xs, exo_bb_extent_ys);

                if (!non_margin_collision_flag.all())
                {
                    for (int j = 0; j < batch_size; ++j)
                    {
                        if (!non_margin_collision_flag[{0, j}])
                        {
                            bool in_front_of_ego =
                                observed_exo_state_.obs_frenet_s_[ego_curr_ref_path_idx_][i + j] >
                                ego_frenet_s[ego_curr_ref_path_idx_] - utils::COLLISION_EGO_FRONT_OFFSET;
                            if (!in_front_of_ego ||
                                observed_exo_state_.obs_vs_[i + j] <= utils::CREEP_VELOCITY_THRESHOLD ||
                                ego_v >= utils::LOW_SPEED_THRESHOLD)
                            {
                                observed_exo_state_.obs_expanded_bb_extent_xs_[i + j] =
                                    observed_exo_state_.obs_original_bb_extent_xs_[i + j];
                                observed_exo_state_.obs_expanded_bb_extent_ys_[i + j] =
                                    observed_exo_state_.obs_original_bb_extent_ys_[i + j];
                            }
                        }
                    }

                    // Check for physical collision
                    exo_bb_extent_xs =
                        FVectorT_1::load_contiguous(observed_exo_state_.obs_original_bb_extent_xs_.data(), i);
                    exo_bb_extent_ys =
                        FVectorT_1::load_contiguous(observed_exo_state_.obs_original_bb_extent_ys_.data(), i);
                    FVectorT_1 non_physical_collision_flag = context_qmdp_->checkSATCollisionBatch1(
                        ego_x, ego_y, ego_theta_cos, ego_theta_sin, exo_xs, exo_ys, exo_thetas.cos(), exo_thetas.sin(),
                        utils::EGO_BB_EXTENT_X, utils::EGO_BB_EXTENT_Y, exo_bb_extent_xs, exo_bb_extent_ys);

                    if (!non_physical_collision_flag.all())
                    {
                        for (int j = 0; j < batch_size; ++j)
                        {
                            if (!non_physical_collision_flag[{0, j}])
                            {
                                collided_agents_tokens_.insert(observed_exo_state_.tokens_[i + j]);
                            }
                        }
                    }
                }
            }

            // Process remaining validations
            if (cal_remaining_num > 0)
            {
                int        begin_idx = cal_whole_times * batch_size;
                FVectorT_1 exo_xs = FVectorT_1::load_contiguous(observed_exo_state_.obs_xs_.data(), begin_idx);
                FVectorT_1 exo_ys = FVectorT_1::load_contiguous(observed_exo_state_.obs_ys_.data(), begin_idx);
                FVectorT_1 exo_thetas = FVectorT_1::load_contiguous(observed_exo_state_.obs_thetas_.data(), begin_idx);
                FVectorT_1 exo_bb_extent_xs =
                    FVectorT_1::load_contiguous(observed_exo_state_.obs_expanded_bb_extent_xs_.data(), begin_idx);
                FVectorT_1 exo_bb_extent_ys =
                    FVectorT_1::load_contiguous(observed_exo_state_.obs_expanded_bb_extent_ys_.data(), begin_idx);

                FVectorT_1 non_margin_collision_flag = context_qmdp_->checkSATCollisionBatch1(
                    ego_x, ego_y, ego_theta_cos, ego_theta_sin, exo_xs, exo_ys, exo_thetas.cos(), exo_thetas.sin(),
                    utils::EGO_BB_EXTENT_X, utils::EGO_BB_EXTENT_Y, exo_bb_extent_xs, exo_bb_extent_ys);

                if (!non_margin_collision_flag.all())
                {
                    for (int j = 0; j < cal_remaining_num; ++j)
                    {
                        if (!non_margin_collision_flag[{0, j}])
                        {
                            bool in_front_of_ego =
                                observed_exo_state_.obs_frenet_s_[ego_curr_ref_path_idx_][begin_idx + j] >
                                ego_frenet_s[ego_curr_ref_path_idx_] - utils::COLLISION_EGO_FRONT_OFFSET;

                            if (!in_front_of_ego ||
                                observed_exo_state_.obs_vs_[begin_idx + j] <= utils::CREEP_VELOCITY_THRESHOLD ||
                                ego_v >= utils::LOW_SPEED_THRESHOLD)
                            {
                                observed_exo_state_.obs_expanded_bb_extent_xs_[begin_idx + j] =
                                    observed_exo_state_.obs_original_bb_extent_xs_[begin_idx + j];
                                observed_exo_state_.obs_expanded_bb_extent_ys_[begin_idx + j] =
                                    observed_exo_state_.obs_original_bb_extent_ys_[begin_idx + j];
                            }
                        }
                    }

                    exo_bb_extent_xs =
                        FVectorT_1::load_contiguous(observed_exo_state_.obs_original_bb_extent_xs_.data(), begin_idx);
                    exo_bb_extent_ys =
                        FVectorT_1::load_contiguous(observed_exo_state_.obs_original_bb_extent_ys_.data(), begin_idx);
                    FVectorT_1 non_physical_collision_flag = context_qmdp_->checkSATCollisionBatch1(
                        ego_x, ego_y, ego_theta_cos, ego_theta_sin, exo_xs, exo_ys, exo_thetas.cos(), exo_thetas.sin(),
                        utils::EGO_BB_EXTENT_X, utils::EGO_BB_EXTENT_Y, exo_bb_extent_xs, exo_bb_extent_ys);

                    if (!non_physical_collision_flag.all())
                    {
                        for (int j = 0; j < cal_remaining_num; ++j)
                        {
                            if (!non_physical_collision_flag[{0, j}])
                            {
                                collided_agents_tokens_.insert(observed_exo_state_.tokens_[begin_idx + j]);
                            }
                        }
                    }
                }
            }

            // 6. Set valid_mask_ to false for all collided agents
            for (size_t i = 0; i < observed_exo_state_.num_vehicles_; ++i)
            {
                if (observed_exo_state_.agent_valid_mask_[i] &&
                    collided_agents_tokens_.count(observed_exo_state_.tokens_[i]))
                {
                    observed_exo_state_.agent_valid_mask_[i] = false;
                }
            }
        }
    } // namespace core
} // namespace vec_qmdp