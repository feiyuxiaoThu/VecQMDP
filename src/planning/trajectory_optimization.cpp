/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <planning/trajectory_optimization.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        TrajectoryOptimization::TrajectoryOptimization(int num_scenarios)
            : context_qmdp_(std::make_shared<ContextQMDP>()), tracker_(std::make_shared<Tracker>()),
              global_time_size_(utils::PROPOSAL_DUMMY_SIZE), step_time_size_(utils::STEP_TIME_LEN),
              vehicle_max_size_(utils::MAX_SIM_VEHICLES)
        {
            scenario_size_ = num_scenarios;

            // Calculate total data size
            exo_total_size_ = global_time_size_ * vehicle_max_size_;

            initializeDataStructures();
        }

        TrajectoryOptimization::TrajectoryOptimization(int                                       num_scenarios,
                                                       const std::vector<std::shared_ptr<Path>> &ref_paths,
                                                       const std::vector<std::shared_ptr<Path>> &extra_ref_paths,
                                                       size_t curr_ref_path_idx, size_t target_ref_path_idx)
            : context_qmdp_(std::make_shared<ContextQMDP>()), tracker_(std::make_shared<Tracker>()),
              global_time_size_(utils::PROPOSAL_DUMMY_SIZE), step_time_size_(utils::STEP_TIME_LEN),
              vehicle_max_size_(utils::MAX_SIM_VEHICLES), path_idx_(ref_paths[curr_ref_path_idx]->GetSize()),
              ref_paths_(ref_paths), extra_ref_paths_(extra_ref_paths), ego_curr_path_idx_(curr_ref_path_idx),
              ego_target_path_idx_(target_ref_path_idx)
        {
            scenario_size_ = num_scenarios;

            // Calculate total data size
            exo_total_size_ = global_time_size_ * vehicle_max_size_;

            initializeDataStructures();
        }

        TrajectoryOptimization::~TrajectoryOptimization() = default;

        void TrajectoryOptimization::UpdatePathIndex(const std::vector<std::shared_ptr<Path>> &ref_paths,
                                                     const std::vector<std::shared_ptr<Path>> &extra_ref_paths,
                                                     size_t curr_ref_path_idx, int best_action_idx,
                                                     int best_target_path_idx, float best_target_offset,
                                                     float best_value, int second_best_action_idx,
                                                     int second_best_target_path_idx, float second_best_target_offset,
                                                     float second_best_value, float curr_path_value)
        {
            path_idx_ = ref_paths[curr_ref_path_idx]->GetSize();
            ref_paths_ = ref_paths;
            extra_ref_paths_ = extra_ref_paths;
            ego_curr_path_idx_ = curr_ref_path_idx;
            ego_target_path_idx_ = best_target_path_idx;
            ego_second_target_path_idx_ = second_best_target_path_idx;
            best_target_offset_ = best_target_offset;
            second_best_target_offset_ = second_best_target_offset;
            best_action_value_ = best_value;
            second_best_action_value_ = second_best_value;
            curr_path_value_ = curr_path_value;
        }

        void TrajectoryOptimization::initializeDataStructures()
        {
            // Initialize exo vehicle data
            exo_xs_flat_.resize(scenario_size_ * exo_total_size_, utils::MAX_VALUE);
            exo_ys_flat_.resize(scenario_size_ * exo_total_size_, utils::MAX_VALUE);
            exo_vs_flat_.resize(scenario_size_ * exo_total_size_, 0.0f);
            exo_thetas_flat_.resize(scenario_size_ * exo_total_size_, 0.0f);
            exo_cos_thetas_flat_.resize(scenario_size_ * exo_total_size_, 0.0f);
            exo_sin_thetas_flat_.resize(scenario_size_ * exo_total_size_, 0.0f);
            exo_original_bb_extent_xs_.resize(vehicle_max_size_, 0.0f);
            exo_original_bb_extent_ys_.resize(vehicle_max_size_, 0.0f);
            exo_expanded_bb_extent_xs_.resize(vehicle_max_size_, 0.0f);
            exo_expanded_bb_extent_ys_.resize(vehicle_max_size_, 0.0f);
            exo_ss_flat_.resize(2, AlignedVectorFloat(scenario_size_ * exo_total_size_, 0.0f));
            exo_ls_flat_.resize(2, AlignedVectorFloat(scenario_size_ * exo_total_size_, 0.0f));
            exo_ls_projected_radius_flat_.resize(2, AlignedVectorFloat(scenario_size_ * exo_total_size_, 0.0f));
            exo_expanded_strtrees_.resize(2);

            for (int i = 0; i < 2; ++i)
            {
                size_t tree_count = scenario_size_ * global_time_size_;
                exo_expanded_strtrees_[i].reserve(tree_count);
                for (size_t j = 0; j < tree_count; ++j)
                {
                    exo_expanded_strtrees_[i].push_back(std::make_shared<STRtree>());
                }
            }

            // Initialize arrays for all offset values
            ego_xs_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_ys_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_thetas_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_vs_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_as_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_idm_as_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);
            ego_path_curvatures_proposal_.resize(utils::PROPOSAL_TRAJECTORY_SIZE);

            ego_xs_published_proposal_.resize(utils::PROPOSAL_PUBLISHED_TRAJECTORY_SIZE);
            ego_ys_published_proposal_.resize(utils::PROPOSAL_PUBLISHED_TRAJECTORY_SIZE);
            ego_thetas_published_proposal_.resize(utils::PROPOSAL_PUBLISHED_TRAJECTORY_SIZE);
            ego_vs_published_proposal_.resize(utils::PROPOSAL_PUBLISHED_TRAJECTORY_SIZE);
            ego_as_published_proposal_.resize(utils::PROPOSAL_PUBLISHED_TRAJECTORY_SIZE);

            ego_lateral_offsets_ = FVectorT_traj(utils::LATERAL_OFFSETS, true); // true: broadcast row-wise
            ego_find_exo_lateral_offsets_ = FVectorT_traj(utils::ACTUAL_LATERAL_OFFSETS,
                                                          true); // true: broadcast row-wise

            // Initialize importance weights
            importance_weights_.resize(scenario_size_);
            init_nearest_idxs_.resize(exo_total_size_, 0);
            target_nearest_idxs_.resize(exo_total_size_, 0);
        }

        bool TrajectoryOptimization::determineReferencePath(const std::shared_ptr<core::EgoState>  &ego_state,
                                                            const std::shared_ptr<core::NetBelief> &belief,
                                                            const std::shared_ptr<Path>            &target_path_ptr,
                                                            int                                     target_path_idx)
        {
            const auto &ego_x = ego_state->x();
            const auto &ego_y = ego_state->y();
            const auto &ego_v = ego_state->v();
            const auto &ego_theta = ego_state->theta();

            const auto &observed_exo_state = belief->observed_exo_state_;

            // Find nearest point on the target path
            size_t target_path_nearest_idx;
            target_path_nearest_idx = target_path_ptr->NearestBatch(ego_x, ego_y, ego_theta);

            const AlignedVectorFloat &target_path_exo_ss = observed_exo_state.obs_frenet_s_[target_path_idx];
            const AlignedVectorFloat &target_path_exo_ls = observed_exo_state.obs_frenet_l_[target_path_idx];

            std::pair<float, float> LC_lead_vehicles_close_to_ego_reference_path(utils::MAX_VALUE, 0.0f);
            std::pair<float, float> LC_following_vehicles_close_to_ego_reference_path(utils::MAX_VALUE, 0.0f);
            float                   ego_target_path_progress = target_path_nearest_idx * utils::PATH_POINT_INTERVAL;

            for (const auto &valid_agent_idx : observed_exo_state.valid_agent_idxs_)
            {
                float target_delta_exo_ss = target_path_exo_ss[valid_agent_idx] - ego_target_path_progress;
                float exo_bb_extent_x = observed_exo_state.obs_expanded_bb_extent_xs_[valid_agent_idx];
                float exo_bb_extent_y = observed_exo_state.obs_expanded_bb_extent_ys_[valid_agent_idx];

                // Find leading vehicle on the target path
                if (target_delta_exo_ss >= 0.0f &&
                    std::fabs(target_path_exo_ls[valid_agent_idx]) <= exo_bb_extent_x + utils::EGO_BB_EXTENT_X)
                {
                    float abs_delta_exo_ss = abs(target_delta_exo_ss) - exo_bb_extent_y - utils::EGO_BB_EXTENT_Y;

                    if (abs_delta_exo_ss < LC_lead_vehicles_close_to_ego_reference_path.first)
                    {
                        LC_lead_vehicles_close_to_ego_reference_path.first = abs_delta_exo_ss;
                        LC_lead_vehicles_close_to_ego_reference_path.second =
                            observed_exo_state.obs_vs_[valid_agent_idx];
                    }
                }
                // Find following vehicle on the target path
                else if (target_delta_exo_ss < 0.0f &&
                         std::fabs(target_path_exo_ls[valid_agent_idx]) <= exo_bb_extent_x + utils::EGO_BB_EXTENT_X)
                {
                    float abs_delta_exo_ss = abs(target_delta_exo_ss) - exo_bb_extent_y - utils::EGO_BB_EXTENT_Y;

                    if (abs_delta_exo_ss < LC_following_vehicles_close_to_ego_reference_path.first)
                    {
                        LC_following_vehicles_close_to_ego_reference_path.first = abs_delta_exo_ss;
                        LC_following_vehicles_close_to_ego_reference_path.second =
                            observed_exo_state.obs_vs_[valid_agent_idx];
                    }
                }
            }

            // Check whether lane change to target path is allowed
            bool LC_allowance = context_qmdp_->LCAllowanceCheckSerial(LC_lead_vehicles_close_to_ego_reference_path,
                                                                      LC_following_vehicles_close_to_ego_reference_path,
                                                                      ego_v, utils::MAX_VEL, true);

            return LC_allowance;
        }

        void TrajectoryOptimization::determinLateralOffsetSpace(const std::shared_ptr<core::EgoState>  &ego_state,
                                                                const std::shared_ptr<core::NetBelief> &belief,
                                                                const std::shared_ptr<utils::MapUtils> &map_utils_ptr,
                                                                bool &larget_left_offset_cared,
                                                                bool &larget_right_offset_cared)
        {
            // For LC, overly distant lane changes are unreasonable; e.g., when changing
            // left, the largest left lateral offset is disallowed
            if (ref_paths_[ego_curr_path_idx_]->GetRank() > ref_paths_[ego_target_path_idx_]->GetRank())
            {
                if (ego_state->v() > utils::LOW_SPEED_THRESHOLD)
                {
                    larget_left_offset_cared = false;
                }
            }
            else if (ref_paths_[ego_curr_path_idx_]->GetRank() < ref_paths_[ego_target_path_idx_]->GetRank())
            {
                if (ego_state->v() > utils::LOW_SPEED_THRESHOLD)
                {
                    larget_right_offset_cared = false;
                }
            }
            // For LF, only allow lateral offsets where a potential lane change is feasible
            else
            {
                // If the current lane is off-route, neither left nor right offset is allowed
                if (!map_utils_ptr->IsOnRoute(ego_state->edge_token()))
                {
                    larget_left_offset_cared = false;
                    larget_right_offset_cared = false;
                    return;
                }

                bool on_lane_connector = ego_state->edge_name() == "LANE_CONNECTOR";

                // *test left offset. it will be false if left path exists and the lane
                // changing is not possiblie
                int                   left_rank = ref_paths_[ego_target_path_idx_]->GetRank() - 1;
                std::shared_ptr<Path> left_path;
                int                   left_path_idx = -1;

                // find left path
                for (int i = 0; i < ref_paths_.size(); ++i)
                {
                    if (ref_paths_[i] && ref_paths_[i]->GetRank() == left_rank)
                    {
                        left_path = ref_paths_[i];
                        left_path_idx = i;
                    }
                }
                if (left_path == nullptr)
                {
                    for (int i = 0; i < extra_ref_paths_.size(); ++i)
                    {
                        if (extra_ref_paths_[i] && extra_ref_paths_[i]->GetRank() == left_rank)
                        {
                            left_path = extra_ref_paths_[i];
                            left_path_idx = i + ref_paths_.size();
                        }
                    }
                }

                if (left_path)
                {
                    // *if the agent has been on the lane connector, then it can use the
                    // largest offset
                    larget_left_offset_cared =
                        !left_path->not_on_route_ &&
                            determineReferencePath(ego_state, belief, left_path, left_path_idx) ||
                        (on_lane_connector && left_path->not_on_route_);
                }

                // *test right offset. it will be false if right path exists and the
                // lane changing is not possiblie
                int                   right_rank = ref_paths_[ego_target_path_idx_]->GetRank() + 1;
                std::shared_ptr<Path> right_path;
                int                   right_path_idx = -1;

                // find right path
                for (int i = 0; i < ref_paths_.size(); ++i)
                {
                    if (ref_paths_[i] && ref_paths_[i]->GetRank() == right_rank)
                    {
                        right_path = ref_paths_[i];
                        right_path_idx = i;
                    }
                }
                if (right_path == nullptr)
                {
                    for (int i = 0; i < extra_ref_paths_.size(); ++i)
                    {
                        if (extra_ref_paths_[i] && extra_ref_paths_[i]->GetRank() == right_rank)
                        {
                            right_path = extra_ref_paths_[i];
                            right_path_idx = i + ref_paths_.size();
                        }
                    }
                }

                if (right_path)
                {
                    // *if the agent has been on the lane connector, then it can use the largest offset
                    larget_right_offset_cared =
                        !right_path->not_on_route_ &&
                            determineReferencePath(ego_state, belief, right_path, right_path_idx) ||
                        (on_lane_connector && right_path->not_on_route_);
                }
            }
        }

        TrajectoryOptimization::TrajectoryModeInfo
        TrajectoryOptimization::determineTrajectoryMode(const std::shared_ptr<core::EgoState> &ego_state)
        {
            TrajectoryModeInfo mode_info;

            // Check whether a target path differs from current path
            bool has_target_path = ego_curr_path_idx_ != ego_target_path_idx_;

            // Determine trajectory mode: true = LF (Lane Following), false = LC (Lane
            // Change)
            mode_info.use_lf_mode =
                (ego_curr_path_idx_ == ego_target_path_idx_ ||
                 (ego_curr_path_idx_ == ego_second_target_path_idx_ && ego_initial_v_ < utils::LOW_SPEED_THRESHOLD &&
                  curr_path_value_ - best_action_value_ > utils::MODE_VALUE_DIFF_THRESHOLD));

            // Initialize distance and index fields
            mode_info.init_path_min_dist = 0.0f;
            mode_info.target_path_min_dist = 0.0f;
            mode_info.init_path_nearest_idx = 0;
            mode_info.target_path_nearest_idx = 0;

            if (mode_info.use_lf_mode)
            {
                // LF mode: use current path
                mode_info.ref_path_idx = ego_curr_path_idx_;
                mode_info.ref_path_nearest_idx = ref_paths_[mode_info.ref_path_idx]->NearestBatch(
                    ego_initial_x_, ego_initial_y_, ego_initial_theta_);
                mode_info.ref_path_idx_for_frenet = 0;

                // LF mode only requires processing the current path
                mode_info.need_target_path = false;
            }
            else
            {
                // LC mode: determine whether ego has already reached the target path
                mode_info.init_path_nearest_idx = ref_paths_[ego_curr_path_idx_]->NearestBatch(
                    ego_initial_x_, ego_initial_y_, ego_initial_theta_, mode_info.init_path_min_dist);
                mode_info.target_path_nearest_idx = ref_paths_[ego_target_path_idx_]->NearestBatch(
                    ego_initial_x_, ego_initial_y_, ego_initial_theta_, mode_info.target_path_min_dist);

                // Check if ego is already on the target path, also considering lateral
                // offset
                bool on_target_path = (mode_info.init_path_min_dist - mode_info.target_path_min_dist >
                                           utils::ON_TARGET_PATH_DISTANCE_MARGIN ||
                                       mode_info.target_path_min_dist < utils::ACTION_CHANGE_DISTANCE_THRESHOLD);

                if (on_target_path)
                {
                    // Already on target path, switch to LF mode
                    mode_info.use_lf_mode = true;
                    // If also on current path and second best still points to current
                    // path, use current path
                    if (mode_info.init_path_min_dist < utils::ACTION_CHANGE_DISTANCE_THRESHOLD &&
                        ego_curr_path_idx_ == ego_second_target_path_idx_)
                    {
                        mode_info.ref_path_idx = ego_curr_path_idx_;
                        mode_info.ref_path_nearest_idx = mode_info.init_path_nearest_idx;
                        mode_info.ref_path_idx_for_frenet = 0;
                        mode_info.need_target_path = false;
                    }
                    else
                    {
                        mode_info.ref_path_idx = ego_target_path_idx_;
                        mode_info.ref_path_nearest_idx = mode_info.target_path_nearest_idx;
                        mode_info.ref_path_idx_for_frenet = 1;
                        mode_info.need_target_path = true;
                    }
                }
                else
                {
                    // Not yet on target path, remain in LC mode
                    mode_info.use_lf_mode = false;
                    mode_info.ref_path_idx = ego_target_path_idx_;
                    mode_info.ref_path_nearest_idx = mode_info.target_path_nearest_idx;
                    mode_info.need_target_path = true;
                }
            }

            return mode_info;
        }

        // done
        void TrajectoryOptimization::importanceSampleScenarios(int                                     thread_id,
                                                               const std::shared_ptr<core::NetBelief> &belief,
                                                               bool                                    need_target_path)
        {
            // static thread_local std::mt19937 rng(utils::RANDOM_SEED);
            static thread_local std::mt19937                          rng(utils::RANDOM_SEED + thread_id);
            static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            // Retrieve observed exo state
            const auto &observed_exo_state = belief->GetObservedExoState();

            // Retrieve valid agent indices
            const auto &valid_agent_idxs = observed_exo_state.valid_agent_idxs_;

            {
                // Copy AABB extents
                memcpy(&exo_original_bb_extent_xs_[0], &observed_exo_state.valid_obs_original_bb_extent_xs_[0],
                       vehicle_max_size_ * sizeof(float));

                memcpy(&exo_original_bb_extent_ys_[0], &observed_exo_state.valid_obs_original_bb_extent_ys_[0],
                       vehicle_max_size_ * sizeof(float));

                memcpy(&exo_expanded_bb_extent_xs_[0], &observed_exo_state.valid_obs_expanded_bb_extent_xs_[0],
                       vehicle_max_size_ * sizeof(float));

                memcpy(&exo_expanded_bb_extent_ys_[0], &observed_exo_state.valid_obs_expanded_bb_extent_ys_[0],
                       vehicle_max_size_ * sizeof(float));
            }

            // Retrieve prediction trajectory data pointers
            const float *const &pred_traj_x_nd_data_ptr = belief->pred_traj_x_nd_data_ptr_;
            const float *const &pred_traj_y_nd_data_ptr = belief->pred_traj_y_nd_data_ptr_;
            const float *const &pred_traj_v_nd_data_ptr = belief->pred_traj_v_nd_data_ptr_;
            const float *const &pred_traj_theta_nd_data_ptr = belief->pred_traj_theta_nd_data_ptr_;
            const float *const &pred_traj_theta_cos_nd_data_ptr = belief->pred_traj_theta_cos_nd_data_ptr_;
            const float *const &pred_traj_theta_sin_nd_data_ptr = belief->pred_traj_theta_sin_nd_data_ptr_;
            const float *const &pred_prob_nd_data_ptr = belief->pred_prob_nd_data_ptr_;

            // Use the passed parameter to decide whether to process the target path
            // bool has_target_path = ego_curr_path_idx_ != ego_target_path_idx_;

            // Pre-process pred_prob_nd_data_ptr to obtain truncated proposal
            // probabilities
            const float threshold = utils::IMPORTANCE_SAMPLE_PROB_THRESHOLD;

            // Sampleable mode indices per agent
            std::vector<std::vector<size_t>> truncated_modes(exo_valid_num_); // [agent] -> list of mode idxs

            for (size_t i = 0; i < exo_valid_num_; ++i)
            {
                size_t               agent_idx = valid_agent_idxs[i];
                std::vector<size_t> &mode_list = truncated_modes[i];
                for (size_t mode = 0; mode < utils::MAX_PRED_MODES; ++mode)
                {
                    size_t mode_idx = agent_idx * utils::MAX_PRED_MODES + mode;
                    float  prob = pred_prob_nd_data_ptr[mode_idx];
                    if (prob >= threshold)
                    {
                        mode_list.emplace_back(mode);
                    }
                }
            }

            // Importance-sample data for each scenario
            for (size_t s = 0; s < utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD; ++s)
            {
                double importance_prob = 1.0f;
                double actual_prob = 1.0f;

                // Importance-sample each agent's mode from belief for the current
                // scenario
                for (size_t i = 0; i < exo_valid_num_; ++i)
                {
                    size_t      agent_idx = valid_agent_idxs[i];
                    std::string token = observed_exo_state.tokens_[agent_idx];

                    // Retrieve the truncated mode list for this agent
                    const std::vector<size_t> &valid_modes = truncated_modes[i];
                    size_t                     num_valid_modes = valid_modes.size();

            // Guard against divide-by-zero: if no valid modes exist, fall back
            // to default handling
            if (num_valid_modes == 0)
            {
                // std::cout << "ERROR: num_valid_modes == 0" << std::endl;
                // When all mode probabilities are below threshold, use the
                // highest-probability mode as fallback
                float max_prob = 0.0f;
                size_t best_mode = 0;
                for (size_t mode = 0; mode < utils::MAX_PRED_MODES; ++mode)
                {
                    size_t mode_idx = agent_idx * utils::MAX_PRED_MODES + mode;
                    float prob = pred_prob_nd_data_ptr[mode_idx];
                    if (prob > max_prob)
                    {
                        max_prob = prob;
                        best_mode = mode;
                    }
                }

                        // Use the best mode and assign a reasonable weight
                        size_t sampled_mode = best_mode;
                        importance_prob *= max_prob; // use actual probability as importance weight
                        actual_prob *= max_prob;

                        // Data copy logic remains the same
                        size_t src_start_idx = agent_idx * utils::MAX_PRED_MODES * utils::DUMMY_TIME_STEPS +
                                               sampled_mode * utils::DUMMY_TIME_STEPS;

                        size_t dst_start_idx = getExoIdxVehicleTime(s, i, 0);

                        {
                            memcpy(&exo_xs_flat_[dst_start_idx], &pred_traj_x_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            memcpy(&exo_ys_flat_[dst_start_idx], &pred_traj_y_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            memcpy(&exo_vs_flat_[dst_start_idx], &pred_traj_v_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            memcpy(&exo_thetas_flat_[dst_start_idx], &pred_traj_theta_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            memcpy(&exo_cos_thetas_flat_[dst_start_idx],
                                   &pred_traj_theta_cos_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            memcpy(&exo_sin_thetas_flat_[dst_start_idx],
                                   &pred_traj_theta_sin_nd_data_ptr[src_start_idx],
                                   utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                            init_nearest_idxs_[i] = observed_exo_state.obs_nearest_idxs_[ego_curr_path_idx_][agent_idx];

                            if (need_target_path)
                            {
                                target_nearest_idxs_[i] =
                                    observed_exo_state.obs_nearest_idxs_[ego_target_path_idx_][agent_idx];
                            }
                        }

                        continue; // Skip normal sampling flow
                    }

                    // Uniformly sample one mode from the valid set
                    std::uniform_int_distribution<size_t> int_dist(0, num_valid_modes - 1);
                    size_t                                rand_idx = int_dist(rng);
                    size_t                                sampled_mode = valid_modes[rand_idx];

                    // Update importance probability
                    importance_prob *= 1.0 / num_valid_modes;

                    // Update actual probability
                    const size_t prob_idx = agent_idx * utils::MAX_PRED_MODES + sampled_mode;
                    double       mode_prob = pred_prob_nd_data_ptr[prob_idx];
                    actual_prob *= mode_prob;

                    // Compute source data start index
                    size_t src_start_idx = agent_idx * utils::MAX_PRED_MODES * utils::DUMMY_TIME_STEPS +
                                           sampled_mode * utils::DUMMY_TIME_STEPS;

                    // Compute destination data start index
                    size_t dst_start_idx = getExoIdxVehicleTime(s, i, 0);

                    {
                        // Copy x-coordinates for all time steps at once
                        memcpy(&exo_xs_flat_[dst_start_idx], &pred_traj_x_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy y-coordinates for all time steps at once
                        memcpy(&exo_ys_flat_[dst_start_idx], &pred_traj_y_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy velocities for all time steps at once
                        memcpy(&exo_vs_flat_[dst_start_idx], &pred_traj_v_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy heading angles for all time steps at once
                        memcpy(&exo_thetas_flat_[dst_start_idx], &pred_traj_theta_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy cosine of heading angles for all time steps at once
                        memcpy(&exo_cos_thetas_flat_[dst_start_idx], &pred_traj_theta_cos_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy sine of heading angles for all time steps at once
                        memcpy(&exo_sin_thetas_flat_[dst_start_idx], &pred_traj_theta_sin_nd_data_ptr[src_start_idx],
                               utils::PROPOSAL_TRAJECTORY_SIZE * sizeof(float));

                        // Copy nearest point indices on the current path
                        init_nearest_idxs_[i] = observed_exo_state.obs_nearest_idxs_[ego_curr_path_idx_][agent_idx];

                        if (need_target_path)
                        {
                            target_nearest_idxs_[i] =
                                observed_exo_state.obs_nearest_idxs_[ego_target_path_idx_][agent_idx];
                        }
                    }
                }

                {
                    // Compute Frenet coordinates
                    core::ExoStates::GetFrenetPointsBatch(
                        thread_id, ref_paths_[ego_curr_path_idx_], init_nearest_idxs_, exo_thetas_flat_, exo_xs_flat_,
                        exo_ys_flat_, exo_vs_flat_, exo_ss_flat_[0], exo_ls_flat_[0], s * exo_total_size_,
                        vehicle_max_size_, utils::PROPOSAL_TRAJECTORY_SIZE, global_time_size_, exo_valid_num_);

                    if (need_target_path)
                    {
                        core::ExoStates::GetFrenetPointsBatch(
                            ref_paths_[ego_target_path_idx_], target_nearest_idxs_, exo_thetas_flat_, exo_xs_flat_,
                            exo_ys_flat_, exo_ss_flat_[1], exo_ls_flat_[1], s * exo_total_size_, vehicle_max_size_,
                            utils::PROPOSAL_TRAJECTORY_SIZE, global_time_size_, exo_valid_num_);
                    }
                }

                {
                    // Build STR-trees
                    core::ExoStates::buildSTRtreesFrenetBatch(
                        thread_id, ref_paths_[ego_curr_path_idx_], exo_cos_thetas_flat_, exo_sin_thetas_flat_,
                        exo_original_bb_extent_xs_, exo_original_bb_extent_ys_, exo_expanded_bb_extent_xs_,
                        exo_expanded_bb_extent_ys_, exo_ss_flat_[0], exo_ls_flat_[0], exo_ls_projected_radius_flat_[0],
                        init_nearest_idxs_,
                        // exo_original_strtrees_[0],
                        exo_expanded_strtrees_[0], s * exo_total_size_, s * global_time_size_, vehicle_max_size_,
                        utils::PROPOSAL_TRAJECTORY_SIZE, global_time_size_, exo_valid_num_);

                    if (need_target_path)
                    {
                        core::ExoStates::buildSTRtreesFrenetBatch(
                            thread_id, ref_paths_[ego_target_path_idx_], exo_cos_thetas_flat_, exo_sin_thetas_flat_,
                            exo_original_bb_extent_xs_, exo_original_bb_extent_ys_, exo_expanded_bb_extent_xs_,
                            exo_expanded_bb_extent_ys_, exo_ss_flat_[1], exo_ls_flat_[1],
                            exo_ls_projected_radius_flat_[1], target_nearest_idxs_, exo_expanded_strtrees_[1],
                            s * exo_total_size_, s * global_time_size_, vehicle_max_size_,
                            utils::PROPOSAL_TRAJECTORY_SIZE, global_time_size_, exo_valid_num_);
                    }
                }

                // Compute importance weights
                // Guard against divide-by-zero or infinity: ensure importance_prob is a
                // finite positive number
                if (importance_prob <= 0.0 || !std::isfinite(importance_prob))
                {
                    // If importance_prob is invalid, set weight to 1.0 (uniform weight)
                    importance_weights_[s] = 1.0;
                }
                else
                {
                    importance_weights_[s] = actual_prob / importance_prob;
                }

                // Additional check: ensure the final weight is a finite number
                if (!std::isfinite(importance_weights_[s]))
                {
                    importance_weights_[s] = 1.0;
                }
            }
        }

        void TrajectoryOptimization::generateProposalTrajectory(const TrajectoryModeInfo &mode_info)
        {
            // Extract all required information from mode_info
            bool   use_lf_mode = mode_info.use_lf_mode;
            size_t ref_path_idx = mode_info.ref_path_idx;
            int    ref_path_nearest_idx = mode_info.ref_path_nearest_idx;
            int    ref_path_idx_for_frenet = mode_info.ref_path_idx_for_frenet;

            // Variables needed for LC mode
            float init_path_min_dist = mode_info.init_path_min_dist;
            float target_path_min_dist = mode_info.target_path_min_dist;
            int   init_path_nearest_idx = mode_info.init_path_nearest_idx;
            int   target_path_nearest_idx = mode_info.target_path_nearest_idx;

            std::fill(exo_active_flags_arr_.begin(), exo_active_flags_arr_.begin() + exo_valid_num_, 0xFFFFFFFF);
            std::fill(exo_active_flags_arr_.begin() + exo_valid_num_, exo_active_flags_arr_.end(), 0);

            FVectorT_12 exo_active_flags = IVectorT_12(exo_active_flags_arr_.data()).template as<FVectorT_12>();

            if (use_lf_mode)
            {
                // LF mode: invoke generateProposalTrajectoryLFBatch once
                context_qmdp_->generateProposalTrajectoryLFBatch(
                    FVectorT_traj::num_scalars, ref_paths_[ref_path_idx]->GetSize(), ego_initial_x_, ego_initial_y_,
                    ego_initial_theta_, ego_initial_v_, ego_initial_a_, ego_lateral_offsets_ + best_target_offset_,
                    ego_find_exo_lateral_offsets_ + best_target_offset_, ref_paths_[ref_path_idx], ref_path_nearest_idx,
                    exo_xs_flat_, exo_ys_flat_, exo_ss_flat_[ref_path_idx_for_frenet],
                    exo_ls_flat_[ref_path_idx_for_frenet], exo_ls_projected_radius_flat_[ref_path_idx_for_frenet],
                    exo_vs_flat_, exo_cos_thetas_flat_, exo_sin_thetas_flat_, exo_expanded_bb_extent_xs_,
                    exo_expanded_bb_extent_ys_, exo_expanded_strtrees_[ref_path_idx_for_frenet], exo_active_flags,
                    ego_xs_proposal_, ego_ys_proposal_, ego_thetas_proposal_, ego_vs_proposal_, ego_as_proposal_,
                    ego_idm_as_proposal_, ego_path_curvatures_proposal_, ego_xs_published_proposal_,
                    ego_ys_published_proposal_, ego_thetas_published_proposal_, ego_vs_published_proposal_,
                    ego_as_published_proposal_);
            }
            else
            {
                // LC mode: invoke generateProposalTrajectoryLCBatch once
                // init_path_nearest_idx and target_path_nearest_idx have already been
                // computed above
                bool turn_left = ego_curr_path_idx_ > ego_target_path_idx_;

                // Check whether ego is currently performing a lane change
                float init_path_theta = ref_paths_[ego_curr_path_idx_]->thetas_[init_path_nearest_idx];
                float init_path_l = ref_paths_[ego_curr_path_idx_]->PointOnLeft(
                                        ego_initial_x_, ego_initial_y_, init_path_theta, init_path_nearest_idx) *
                                    init_path_min_dist;
                float heading_diff = std::fabs(utils::NormalizeAngle(init_path_theta - ego_initial_theta_));
                bool  immediate_assumed_at_target_path =
                    (ego_initial_v_ < utils::LOW_SPEED_THRESHOLD) ||
                    (turn_left && (heading_diff > utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD ||
                                   init_path_l > utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD)) ||
                    (!turn_left && (heading_diff > utils::LC_COMPLETED_ALLOWANCE_HEADING_DIFF_THRESHOLD ||
                                    init_path_l < -utils::LC_COMPLETED_ALLOWANCE_LATERAL_OFFSET_THRESHOLD));

                context_qmdp_->generateProposalTrajectoryLCBatch(
                    FVectorT_traj::num_scalars, ref_paths_[ego_target_path_idx_]->GetSize(), ego_initial_x_,
                    ego_initial_y_, ego_initial_theta_, ego_initial_v_, ego_initial_a_, turn_left,
                    ego_lateral_offsets_ + best_target_offset_, ego_find_exo_lateral_offsets_ + best_target_offset_,
                    immediate_assumed_at_target_path, ref_paths_[ego_curr_path_idx_], ref_paths_[ego_target_path_idx_],
                    IVectorT_traj::fill(0), IVectorT_traj::fill(init_path_nearest_idx),
                    IVectorT_traj::fill(target_path_nearest_idx), exo_xs_flat_, exo_ys_flat_, exo_ss_flat_,
                    exo_ls_flat_, exo_ls_projected_radius_flat_, exo_vs_flat_, exo_cos_thetas_flat_,
                    exo_sin_thetas_flat_, exo_expanded_bb_extent_xs_, exo_expanded_bb_extent_ys_,
                    exo_expanded_strtrees_, exo_active_flags, ego_xs_proposal_, ego_ys_proposal_, ego_thetas_proposal_,
                    ego_vs_proposal_, ego_as_proposal_, ego_idm_as_proposal_, ego_xs_published_proposal_,
                    ego_ys_published_proposal_, ego_thetas_published_proposal_, ego_vs_published_proposal_,
                    ego_as_published_proposal_);
            }
        }

        void TrajectoryOptimization::trackProposalTrajectoryCppBatch()
        {
            ego_xs_traj_cpp_.clear();
            ego_ys_traj_cpp_.clear();
            ego_vs_traj_cpp_.clear();
            ego_as_traj_cpp_.clear();
            ego_thetas_traj_cpp_.clear();

            tracker_->trackTrajectoryWithWorkspace(
                ego_xs_proposal_, ego_ys_proposal_, ego_vs_proposal_, ego_as_proposal_, ego_thetas_proposal_,
                ego_xs_traj_cpp_, ego_ys_traj_cpp_, ego_vs_traj_cpp_, ego_as_traj_cpp_, ego_thetas_traj_cpp_,
                ego_initial_x_, ego_initial_y_, ego_initial_v_, ego_current_a_, ego_initial_theta_,
                ego_initial_steering_angle_, ego_initial_steering_rate_);
        }

        std::pair<size_t, float>
        TrajectoryOptimization::crossScenarioEvaluation(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                                        const std::shared_ptr<core::NetBelief>     &belief,
                                                        const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
                                                        const std::shared_ptr<utils::MapUtils>     &map_utils_ptr)
        {
            std::pair<size_t, float> result;

            auto &ego_xs_traj = ego_xs_traj_cpp_;
            auto &ego_ys_traj = ego_ys_traj_cpp_;
            auto &ego_vs_traj = ego_vs_traj_cpp_;
            auto &ego_as_traj = ego_as_traj_cpp_;
            auto &ego_thetas_traj = ego_thetas_traj_cpp_;

            IVectorT_traj collided_timesteps = IVectorT_traj::fill(100);

            std::vector<int> ttc_timesteps(utils::PROPOSAL_BATCH_SIZE, 100);

            {
                FVectorT_traj values = context_qmdp_->crossScenarioEvaluationBatch(
                    thread_id, importance_weights_, ego_lateral_offsets_ + best_target_offset_, ego_initial_v_,
                    ego_xs_proposal_, ego_ys_proposal_, ego_vs_proposal_, ego_as_proposal_, ego_xs_traj, ego_ys_traj,
                    ego_vs_traj, ego_as_traj, ego_thetas_traj, ego_path_curvatures_proposal_,
                    ref_paths_[ego_curr_path_idx_], exo_xs_flat_, exo_ys_flat_, exo_vs_flat_, exo_ss_flat_[0],
                    exo_ls_flat_[0], exo_thetas_flat_, exo_cos_thetas_flat_, exo_sin_thetas_flat_,
                    exo_original_bb_extent_xs_, exo_original_bb_extent_ys_, exo_expanded_bb_extent_xs_,
                    exo_expanded_bb_extent_ys_, exo_expanded_strtrees_[0], occupancy_map, collided_timesteps,
                    ttc_timesteps);
                values_array = values.to_array();
            }

            // Determine whether the max left/right offsets are feasible
            bool larget_left_offset_cared = true;
            bool larget_right_offset_cared = true;

            {
                // 1. Determine based on feasibility of lane changes to adjacent lanes
                determinLateralOffsetSpace(ego_state, belief, map_utils_ptr, larget_left_offset_cared,
                                           larget_right_offset_cared);
            }

            float  global_max_value = std::numeric_limits<float>::lowest();
            size_t global_max_idx = 0;

            float  constrained_max_value = std::numeric_limits<float>::lowest();
            size_t constrained_max_idx = 0;

            float  second_edge_max_value = std::numeric_limits<float>::lowest();
            size_t second_edge_max_idx = 0;

            constexpr int second_edge_offset_idx_left_begin = utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD * 1;
            constexpr int second_edge_offset_idx_left_end = utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD * 2 - 1;
            constexpr int second_edge_offset_idx_right_begin = utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD * 7;
            constexpr int second_edge_offset_idx_right_end = utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD * 8 - 1;

            for (size_t i = 0; i < utils::PROPOSAL_BATCH_SIZE; ++i)
            {
                // Update global maximum
                if (values_array[i] > global_max_value)
                {
                    global_max_value = values_array[i];
                    global_max_idx = i;
                }

                // Track the second-edge maximum value
                if ((i >= second_edge_offset_idx_left_begin && i <= second_edge_offset_idx_left_end) ||
                    (i >= second_edge_offset_idx_right_begin && i <= second_edge_offset_idx_right_end))
                {
                    if (values_array[i] > second_edge_max_value)
                    {
                        second_edge_max_value = values_array[i];
                        second_edge_max_idx = i;
                    }
                }

                // Update constrained maximum when constraints are satisfied
                if ((!larget_left_offset_cared && i >= second_edge_offset_idx_right_begin) ||
                    (!larget_right_offset_cared && i <= second_edge_offset_idx_left_end))
                {
                    continue;
                }

                if (values_array[i] > constrained_max_value)
                {
                    constrained_max_value = values_array[i];
                    constrained_max_idx = i;
                }
            }

            // Evaluate the gap between global and constrained maxima
            if (global_max_value - constrained_max_value > utils::CROSS_EVAL_COLLISION_GAP_THRESHOLD)
            {
                result = std::make_pair(global_max_idx, global_max_value);
            }
            else if (second_edge_max_value - constrained_max_value > utils::CROSS_EVAL_SPEED_GAP_THRESHOLD)
            {
                result = std::make_pair(second_edge_max_idx, second_edge_max_value);
            }
            else
            {
                result = std::make_pair(constrained_max_idx, constrained_max_value);
            }

            time_to_infraction_ =
                collided_timesteps[utils::div_mod<size_t>(result.first, utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD)];

            return result;
        }

        namespace
        {

            // Compute step_move_len for low-speed / gentle braking scenarios.
            // Shared by the hard-brake trajectory branch and getStaticTrajectory.
            // Velocity thresholds (ascending):
            //   VELOCITY_SLIGHTLY_NEGATIVE (-0.05) < VELOCITY_ALMOST_ZERO (0.05)
            //   < NEAR_STOP (0.5) < LOW_SPEED (3.0)
            float computeGentleBrakeStepMove(float velocity, float acceleration)
            {
                if (velocity <= utils::LOW_SPEED_THRESHOLD || acceleration < utils::MAX_DEC)
                {
                    if (velocity > utils::NEAR_STOP_VELOCITY_THRESHOLD)
                        return utils::STEP_MOVE_LIGHT_BRAKE;
                    if (velocity > utils::VELOCITY_ALMOST_ZERO_THRESHOLD)
                        return utils::STEP_MOVE_SLIGHT_BRAKE;
                    if (velocity > utils::VELOCITY_SLIGHTLY_NEGATIVE_THRESHOLD)
                        return 0.0f;
                    return utils::STEP_MOVE_SLIGHT_FORWARD;
                }
                return utils::STEP_MOVE_LIGHT_FORWARD;
            }

            // Compute PD-controller correcting step move for emergency braking.
            // High-speed branch: proportional-only with velocity correction factor.
            // Low-speed branch: PD control with both proportional and derivative gains.
            float computePDCorrectionStepMove(float current_velocity, float current_acceleration)
            {
                float dt_error = -current_acceleration;
                float correcting_velocity;

                if (current_velocity > utils::PD_VELOCITY_SWITCH_THRESHOLD)
                {
                    float u_t = utils::PD_KP_HIGH_SPEED * (-current_velocity);
                    float correction = std::clamp(u_t, utils::PD_MAX_DEC_CLAMP, utils::PD_MAX_ACC_CLAMP);
                    correcting_velocity = utils::PD_VELOCITY_CORRECTION_FACTOR * (current_velocity + correction);
                }
                else
                {
                    float u_t = utils::PD_KP_LOW_SPEED * (-current_velocity) + utils::PD_KD_LOW_SPEED * dt_error;
                    correcting_velocity = std::clamp(u_t, utils::PD_MAX_DEC_CLAMP, utils::PD_MAX_ACC_CLAMP);
                }

                return correcting_velocity * utils::TIME_STEP;
            }

            // Compute step_move_len for TTC-based emergency braking (non-critical case).
            // Uses PD-computed step_move_len as a floor via std::max for higher speeds.
            // Velocity thresholds (ascending):
            //   VELOCITY_SLIGHTLY_NEGATIVE (-0.05) < VELOCITY_ALMOST_ZERO (0.05)
            //   < CREEP (0.3) < SLOW (1.0) < LOW_SPEED (3.0)
            float computeTTCBrakeStepMove(float velocity, float current_acceleration, float pd_step_move_len)
            {
                if (velocity < utils::VELOCITY_SLIGHTLY_NEGATIVE_THRESHOLD)
                    return utils::STEP_MOVE_SLIGHT_FORWARD;
                if (velocity < utils::VELOCITY_ALMOST_ZERO_THRESHOLD)
                    return 0.0f;
                if (velocity < utils::CREEP_VELOCITY_THRESHOLD)
                    return utils::STEP_MOVE_SLIGHT_BRAKE;
                if (velocity < utils::LOW_SPEED_THRESHOLD || current_acceleration < utils::MAX_DEC)
                {
                    if (velocity < utils::SLOW_VELOCITY_THRESHOLD)
                        return std::max(pd_step_move_len, utils::STEP_MOVE_LIGHT_BRAKE);
                    return std::max(pd_step_move_len, utils::STEP_MOVE_MODERATE_BRAKE);
                }
                return std::max(pd_step_move_len, utils::STEP_MOVE_HARD_BRAKE);
            }

        } // anonymous namespace

        float TrajectoryOptimization::checkAndGenerateEmergencyBrake(const std::shared_ptr<core::EgoState> &ego_state,
                                                                     float lateral_offset, float first_acc,
                                                                     float second_acc) const
        {
            float current_acceleration = ego_state->a();
            float current_velocity = ego_state->v();
            float next_velocity = current_velocity + current_acceleration * utils::PUBLISHED_PROPOSAL_TIME_STEP;

            // Detect hard-brake trajectory pattern
            bool trajectory_hard_brake = (first_acc < utils::HARD_BRAKE_DEC + utils::HARD_BRAKE_DEC_TOLERANCE ||
                                          current_velocity < utils::VELOCITY_NEAR_ZERO_THRESHOLD) &&
                                         second_acc < utils::HARD_BRAKE_DEC + utils::HARD_BRAKE_DEC_TOLERANCE &&
                                         std::fabs(lateral_offset) < utils::HARD_BRAKE_LATERAL_OFFSET_THRESHOLD;

            // Case 1: Imminent collision (time-to-infraction within TTC thresholds)
            if (time_to_infraction_ <= utils::EMERGENCY_TTC_CRITICAL ||
                (time_to_infraction_ <= utils::EMERGENCY_TTC_EXTENDED &&
                 current_velocity <= utils::EMERGENCY_BRAKE_MAX_EGO_SPEED))
            {
                float pd_step = computePDCorrectionStepMove(current_velocity, current_acceleration);

                // Critical collision at high speed: apply maximum braking immediately
                if (time_to_infraction_ <= utils::EMERGENCY_TTC_CRITICAL &&
                    current_velocity > utils::LOW_SPEED_THRESHOLD)
                {
                    std::cout << "step_move_len: " << utils::STEP_MOVE_HARD_BRAKE << std::endl;
                    return utils::STEP_MOVE_HARD_BRAKE;
                }

                float step_move_len = computeTTCBrakeStepMove(next_velocity, current_acceleration, pd_step);
                std::cout << "step_move_len: " << step_move_len << std::endl;
                return step_move_len;
            }

            // Case 2: Hard-braking trajectory at low speed or extreme deceleration
            if (trajectory_hard_brake && (current_velocity <= utils::HARD_BRAKE_VELOCITY_THRESHOLD ||
                                          current_acceleration < utils::HARD_BRAKE_ACC_THRESHOLD))
            {
                float step_move_len = computeGentleBrakeStepMove(next_velocity, current_acceleration);
                std::cout << "step_move_len: " << step_move_len << std::endl;
                return step_move_len;
            }

            // No emergency braking needed
            std::cout << "step_move_len: " << utils::EMERGENCY_BRAKE_INACTIVE_SENTINEL << std::endl;
            return utils::EMERGENCY_BRAKE_INACTIVE_SENTINEL;
        }

        std::vector<std::vector<float>>
        TrajectoryOptimization::getOptimizedTrajectory(const std::shared_ptr<core::EgoState> &ego_state,
                                                       const size_t                          &best_scenario) const
        {
            auto pair_idx = utils::div_mod<size_t>(best_scenario, utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD);

            auto proposal_idx = pair_idx;

            float best_lateral_offset = ego_lateral_offsets_[proposal_idx] + best_target_offset_;

            // Compute interpolated trajectory size (original points * 2 - 1)
            size_t                          original_size = ego_xs_proposal_.size();
            size_t                          interpolated_size = ego_xs_published_proposal_.size();
            std::vector<std::vector<float>> optimized_trajectory(interpolated_size);

            // Check whether emergency braking should be applied
            float step_move_len =
                checkAndGenerateEmergencyBrake(ego_state, best_lateral_offset, ego_idm_as_proposal_[1][proposal_idx],
                                               ego_idm_as_proposal_[2][proposal_idx]);
            if (step_move_len != utils::EMERGENCY_BRAKE_INACTIVE_SENTINEL)
            {
                float moving_len_x = step_move_len * std::cos(ego_state->theta());
                float moving_len_y = step_move_len * std::sin(ego_state->theta());
                for (int i = 0; i < interpolated_size; ++i)
                {
                    optimized_trajectory[i] = {ego_state->x() + moving_len_x * i, ego_state->y() + moving_len_y * i,
                                               ego_state->theta(), 0.0f, 0.0f};
                }
            }
            else
            {
                // Compute total trajectory length
                float trajectory_length = 0.0f;
                for (size_t i = 0; i < original_size - 1; ++i)
                {
                    trajectory_length += std::sqrt(
                        std::pow(ego_xs_proposal_[i][proposal_idx] - ego_xs_proposal_[i + 1][proposal_idx], 2) +
                        std::pow(ego_ys_proposal_[i][proposal_idx] - ego_ys_proposal_[i + 1][proposal_idx], 2));
                }

                std::cout << "trajectory_length: " << trajectory_length << std::endl;

                if (trajectory_length < utils::MIN_TRAJECTORY_LENGTH &&
                    ego_state->v() < utils::NEAR_STOP_VELOCITY_THRESHOLD)
                {
                    return getStaticTrajectory(ego_state);
                }

                // Convert to output format
                for (size_t i = 0; i < interpolated_size; ++i)
                {
                    optimized_trajectory[i] = {
                        ego_xs_published_proposal_[i][proposal_idx], ego_ys_published_proposal_[i][proposal_idx],
                        ego_thetas_published_proposal_[i][proposal_idx], ego_vs_published_proposal_[i][proposal_idx],
                        ego_as_published_proposal_[i][proposal_idx]};
                }
            }

            return optimized_trajectory;
        }

        std::vector<std::vector<float>>
        TrajectoryOptimization::getOptimizedTrackedTrajectory(const std::shared_ptr<core::EgoState> &ego_state,
                                                              const size_t &best_scenario) const
        {
            const auto pair_idx = utils::div_mod<size_t>(best_scenario, utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD);

            // Compute trajectory size
            const size_t                    original_size = ego_xs_traj_cpp_.size();
            std::vector<std::vector<float>> tracked_optimized_trajectory(original_size);

            // Process original trajectory points
            for (size_t i = 0; i < original_size; ++i)
            {
                tracked_optimized_trajectory[i] = {ego_xs_traj_cpp_[i][pair_idx], ego_ys_traj_cpp_[i][pair_idx],
                                                   ego_thetas_traj_cpp_[i][pair_idx], ego_vs_traj_cpp_[i][pair_idx],
                                                   ego_as_traj_cpp_[i][pair_idx]};
            }

            return tracked_optimized_trajectory;
        }

        std::vector<std::vector<float>>
        TrajectoryOptimization::getStaticTrajectory(const std::shared_ptr<core::EgoState> &ego_state) const
        {
            // Compute interpolated trajectory size (original points * 2 - 1)
            const size_t                    original_size = utils::PROPOSAL_TRAJECTORY_SIZE;
            const size_t                    interpolated_size = original_size * 2 - 1;
            std::vector<std::vector<float>> static_trajectory(interpolated_size);

            float step_move_len = computeGentleBrakeStepMove(ego_state->v(), ego_state->a());

            std::cout << "step_move_len: " << step_move_len << std::endl;

            // Compute displacement step for emergency braking
            float moving_len_x = step_move_len * std::cos(ego_state->theta());
            float moving_len_y = step_move_len * std::sin(ego_state->theta());
            for (int i = 0; i < interpolated_size; ++i)
            {
                static_trajectory[i] = {ego_state->x() + moving_len_x * i, ego_state->y() + moving_len_y * i,
                                        ego_state->theta(), 0.0f, 0.0f};
            }

            return static_trajectory;
        }

        // Step 1: Generate Proposals only
        void TrajectoryOptimization::optimizeTrajectoryStep1_Generate(
            int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
            const std::shared_ptr<core::NetBelief> &belief, const std::shared_ptr<utils::MapUtils> &map_utils_ptr)
        {
            // Retrieve ego vehicle state
            ego_initial_x_ = ego_state->x();
            ego_initial_y_ = ego_state->y();
            ego_initial_v_ = ego_state->v();
            ego_initial_a_ = ego_state->a();
            ego_initial_theta_ = ego_state->theta();
            ego_current_a_ = ego_state->a();
            ego_initial_steering_angle_ = ego_state->steering_angle();
            ego_initial_steering_rate_ = ego_state->steering_rate();

            // Update exo vehicle count
            const auto &valid_agent_idxs = belief->GetObservedExoState().valid_agent_idxs_;
            exo_valid_num_ = std::min(static_cast<uint32_t>(valid_agent_idxs.size()), utils::MAX_SIM_VEHICLES);
            context_qmdp_->UpdateExoSize(exo_valid_num_);

            // 1. Determine trajectory mode
            TrajectoryModeInfo mode_info = determineTrajectoryMode(ego_state);

            // 2. Sample scenarios
            importanceSampleScenarios(thread_id, belief, mode_info.need_target_path);

            // 3. Generate proposals
            generateProposalTrajectory(mode_info);
        }

        // Step 2: Evaluate (assumes tracking has been completed externally and
        // injected)
        std::pair<size_t, float> TrajectoryOptimization::optimizeTrajectoryStep2_Evaluate(
            int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
            const std::shared_ptr<core::NetBelief> &belief, const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
            const std::shared_ptr<utils::MapUtils> &map_utils_ptr)
        {
            // Perform C++ tracking if required
            trackProposalTrajectoryCppBatch();

            // Perform evaluation
            return crossScenarioEvaluation(thread_id, ego_state, belief, occupancy_map, map_utils_ptr);
        }
    } // namespace planning
} // namespace vec_qmdp
