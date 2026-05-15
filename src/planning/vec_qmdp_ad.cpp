/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

/**
 * @file vec_qmdp_ad.cpp
 * @brief Autonomous-driving instantiation of VecQMDP — VecQMDP_AD implementation.
 *
 * Contains all AD-specific logic:
 *   - State buffer initialisation / reset.
 *   - Scenario sampling from NetBelief (mode sampling + Frenet projection).
 *   - SIMD-batched node expansion via ContextQMDP::StepBatch.
 *   - Full beliefTreeSearch() pipeline.
 *   - AD action-transition computation (computeNodeCandidateActions).
 */
#include <algorithm>
#include <cmath>
#include <cstring> // memcpy
#include <limits>
#include <planning/vec_qmdp_ad.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        // -----------------------------------------------------------------------
        // Construction / Initialisation
        // -----------------------------------------------------------------------

        VecQMDP_AD::VecQMDP_AD(int num_scenarios_per_worker, size_t num_threads)
            : VecQMDP(num_scenarios_per_worker,
                      utils::NUM_ACTIONS, // branching factor for the AD action space
                      utils::TREE_HEIGHT) // planning depth
        {
            // Initialise AD-specific early-termination defaults from params.hpp.
            // These can be overridden at any time via the inherited setters.
            setDiscountFactor(utils::AD_DEFAULT_DISCOUNT_FACTOR);
            setExplorationConstant(utils::AD_DEFAULT_EXPLORATION_CONSTANT);
            setMaxBackupLambda(utils::AD_DEFAULT_MAX_BACKUP_LAMBDA);
            setDepthSyncLambda(utils::AD_DEFAULT_DEPTH_SYNC_LAMBDA);
            setEarlyTermMinExpandCalls(utils::EARLY_TERM_MIN_EXPAND_CALLS);
            setEarlyTermCheckInterval(utils::EARLY_TERM_CHECK_INTERVAL);
            setEarlyTermStableIterations(utils::EARLY_TERM_STABLE_ITERATIONS);
            setEarlyTermMinBestVisits(utils::EARLY_TERM_MIN_BEST_ACTION_VISITS);
            setEarlyTermQChangeThreshold(utils::EARLY_TERM_Q_CHANGE_THRESHOLD);
            setPrunedThreshold(utils::PRUNED_THRESHOLD);
            setMaxPlanningTimeMs(utils::MAX_PLANNING_TIME);

            // AD-specific dimension info
            global_time_size_ = utils::DUMMY_TIME_STEPS;
            step_time_size_ = utils::STEP_TIME_LEN;
            vehicle_size_ = utils::MAX_SIM_VEHICLES;
            path_num_ = utils::NUM_REF_PATH;
            exo_total_size_ = utils::DUMMY_TIME_STEPS * utils::MAX_SIM_VEHICLES;
            vehicle_valid_num_ = 0;

            initializeDataStructures();
            context_qmdp_ = std::make_shared<ContextQMDP>();

            // Spawn worker threads after all domain state is ready.
            // num_threads > 1: creates N worker VecQMDP_AD instances + thread pool.
            // num_threads == 1: no workers, no pool (single-threaded path).
            initParallelInfrastructure(num_threads);
        }

        void VecQMDP_AD::initializeDataStructures()
        {
            // Ego state buffers (one entry per tree node across all scenarios)
            ego_xs_flat_.resize(node_size_, 0.0f);
            ego_ys_flat_.resize(node_size_, 0.0f);
            ego_vs_flat_.resize(node_size_, 0.0f);
            ego_as_flat_.resize(node_size_, 0.0f);
            ego_thetas_flat_.resize(node_size_, 0.0f);
            ego_curr_ref_path_idxs_.resize(node_size_, 0);
            ego_curr_lateral_offsets_.resize(node_size_, 0.0f);

            // Exo agent state buffers — layout: [scenario][vehicle][time]
            const uint32_t scenario_elements = utils::TOTAL_SCNEARIO_ELEMENTS_PARALLEL;
            exo_xs_flat_.resize(scenario_elements, utils::MAX_VALUE);
            exo_ys_flat_.resize(scenario_elements, utils::MAX_VALUE);
            exo_vs_flat_.resize(scenario_elements, 0.0f);
            exo_thetas_flat_.resize(scenario_elements, 0.0f);
            exo_cos_thetas_flat_.resize(scenario_elements, 0.0f);
            exo_sin_thetas_flat_.resize(scenario_elements, 0.0f);
            exo_bb_extent_xs_.resize(vehicle_size_, 0.0f);
            exo_bb_extent_ys_.resize(vehicle_size_, 0.0f);

            exo_ss_flat_.resize(path_num_);
            exo_ls_flat_.resize(path_num_);
            exo_ls_projected_radius_flat_.resize(path_num_);
            for (int i = 0; i < static_cast<int>(path_num_); ++i)
            {
                exo_ss_flat_[i].resize(scenario_elements, 0.0f);
                exo_ls_flat_[i].resize(scenario_elements, 0.0f);
                exo_ls_projected_radius_flat_[i].resize(scenario_elements, 0.0f);
            }

            // STRtrees: one per (reference path, scenario × time)
            exo_strtrees_.resize(path_num_);
            for (int i = 0; i < static_cast<int>(path_num_); ++i)
            {
                size_t tree_count = static_cast<size_t>(global_time_size_) * scenario_size_;
                exo_strtrees_[i].reserve(tree_count);
                for (size_t j = 0; j < tree_count; ++j)
                    exo_strtrees_[i].push_back(std::make_shared<STRtree>());
            }

            // Per-node exo inactive flags
            node_exo_inactive_flags_.resize(node_size_, std::array<int, utils::MAX_SIM_VEHICLES>());
        }

        void VecQMDP_AD::resetDataStructures()
        {
            // Reset generic tree state
            resetTreeStructures();

            // AD-specific: candidate-action caches are already cleared by
            // resetTreeStructures().  No additional AD-specific reset needed here;
            // exo buffers are fully overwritten each round by sampleScenarios().
        }

        void VecQMDP_AD::updateEgoInitialState(const std::shared_ptr<core::EgoState> &ego_state)
        {
            initial_ego_x_ = ego_state->x();
            initial_ego_y_ = ego_state->y();
            initial_ego_v_ = ego_state->v();
            initial_ego_a_ = ego_state->a();
            initial_ego_theta_ = ego_state->theta();
        }

        // -----------------------------------------------------------------------
        // Action-transition computation (VecQMDP hook)
        // -----------------------------------------------------------------------

        std::vector<uint32_t> VecQMDP_AD::computeNodeCandidateActions(uint32_t node_idx)
        {
            const uint32_t current_action = encodeAction(
                ego_curr_ref_path_idxs_[node_idx], convertOffsetToIndexRobust(ego_curr_lateral_offsets_[node_idx]));

            const auto &transitions = STATIC_TRANSITION_TABLE[current_action];

            std::vector<uint32_t> actions;
            actions.reserve(transitions.size());
            for (const auto &t : transitions)
            {
                // Only add if the target path is available (not null)
                if (t.target_path_idx < ego_ref_paths_.size() && ego_ref_paths_[t.target_path_idx] != nullptr)
                    actions.push_back(t.target_action_idx);
            }
            return actions;
        }

        // -----------------------------------------------------------------------
        // Action decomposition (VecQMDP hook)
        // -----------------------------------------------------------------------

        void VecQMDP_AD::decomposeAction(uint32_t action_idx, int &path_idx_out, float &offset_out) const
        {
            path_idx_out = static_cast<int>(utils::ACTION_TO_PATH[action_idx]);
            offset_out = utils::PATH_OFFSETS_FLOAT[utils::ACTION_TO_OFFSET[action_idx]];
        }

        void VecQMDP_AD::decomposeActionBatch(const IVectorT_qmdp &action_idxs, IVectorT_qmdp &path_idxs_out,
                                              FVectorT_qmdp &offsets_out) const
        {
            AlignedVectorInt   tmp_paths(scenario_size_, 0);
            AlignedVectorFloat tmp_offsets(scenario_size_, 0.0f);
            auto               action_arr = action_idxs.to_array();
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                int   path;
                float offset;
                decomposeAction(static_cast<uint32_t>(action_arr[s]), path, offset);
                tmp_paths[s] = path;
                tmp_offsets[s] = offset;
            }
            path_idxs_out = IVectorT_qmdp(tmp_paths.data());
            offsets_out = FVectorT_qmdp(tmp_offsets.data());
        }

        // -----------------------------------------------------------------------
        // Scenario sampling
        // -----------------------------------------------------------------------

        void VecQMDP_AD::sampleScenarios(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                         const std::shared_ptr<core::NetBelief> &belief, const size_t &curr_path_idx)
        {
            static thread_local std::mt19937                          rng(utils::RANDOM_SEED + thread_id);
            static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            const float ego_x = ego_state->x();
            const float ego_y = ego_state->y();
            const float ego_v = ego_state->v();
            const float ego_theta = ego_state->theta();

            // Seed every scenario root node with the current ego state
            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                uint32_t root = s * tree_node_size_;
                ego_xs_flat_[root] = ego_x;
                ego_ys_flat_[root] = ego_y;
                ego_vs_flat_[root] = ego_v;
                ego_thetas_flat_[root] = ego_theta;
                ego_curr_ref_path_idxs_[root] = static_cast<int>(curr_path_idx);
                ego_curr_lateral_offsets_[root] = 0.0f;
                // Encode as (path, zero-offset); index 1 = zero offset
                node_curr_action_idxs_[root] = static_cast<int>(encodeAction(curr_path_idx, 1));
            }

            // Observed exo-agent state from belief
            const core::ObservedExoState &obs = belief->GetObservedExoState();
            const auto                   &valid_agent_idxs = obs.valid_agent_idxs_;
            size_t valid_num = std::min(static_cast<uint32_t>(obs.num_vehicles_), utils::MAX_SIM_VEHICLES);
            vehicle_valid_num_ = static_cast<uint32_t>(valid_num);
            context_qmdp_->UpdateExoSize(valid_num);

            if (valid_num == 0)
                return;

            memcpy(exo_bb_extent_xs_.data(), obs.valid_obs_original_bb_extent_xs_.data(), valid_num * sizeof(float));
            memcpy(exo_bb_extent_ys_.data(), obs.valid_obs_original_bb_extent_ys_.data(), valid_num * sizeof(float));

            // Predicted trajectory data pointers from belief
            const float *pred_x = belief->pred_traj_x_nd_data_ptr_;
            const float *pred_y = belief->pred_traj_y_nd_data_ptr_;
            const float *pred_v = belief->pred_traj_v_nd_data_ptr_;
            const float *pred_theta = belief->pred_traj_theta_nd_data_ptr_;
            const float *pred_cos = belief->pred_traj_theta_cos_nd_data_ptr_;
            const float *pred_sin = belief->pred_traj_theta_sin_nd_data_ptr_;
            const float *pred_prob = belief->pred_prob_nd_data_ptr_;

            for (uint32_t s = 0; s < scenario_size_; ++s)
            {
                // Reset exo inactive flags at the root of this scenario tree
                auto &inactive = node_exo_inactive_flags_[getNodeIdx(0, s)];
                std::fill(inactive.begin(), inactive.begin() + valid_num, 0);
                std::fill(inactive.begin() + valid_num, inactive.end(), 0xFFFFFFFF);

                // Nearest-index scratch buffer for Frenet computation
                std::vector<AlignedVectorInt> nearest_idxs(path_num_, AlignedVectorInt(exo_total_size_, 0));

                // Sample one mode per agent and copy its trajectory into the buffers
                for (size_t i = 0; i < valid_num; ++i)
                {
                    size_t agent = valid_agent_idxs[i];

                    // Weighted-categorical mode sampling
                    float  rand_val = dist(rng);
                    float  cum_prob = 0.0f;
                    size_t sampled_mode = 0;
                    for (size_t m = 0; m < utils::MAX_PRED_MODES; ++m)
                    {
                        cum_prob += pred_prob[agent * utils::MAX_PRED_MODES + m];
                        if (rand_val <= cum_prob)
                        {
                            sampled_mode = m;
                            break;
                        }
                    }

                    size_t src = agent * utils::MAX_PRED_MODES * utils::DUMMY_TIME_STEPS +
                                 sampled_mode * utils::DUMMY_TIME_STEPS;
                    size_t dst = getExoIdxVehicleTime(s, i, 0);

                    memcpy(&exo_xs_flat_[dst], &pred_x[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));
                    memcpy(&exo_ys_flat_[dst], &pred_y[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));
                    memcpy(&exo_vs_flat_[dst], &pred_v[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));
                    memcpy(&exo_thetas_flat_[dst], &pred_theta[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));
                    memcpy(&exo_cos_thetas_flat_[dst], &pred_cos[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));
                    memcpy(&exo_sin_thetas_flat_[dst], &pred_sin[src], utils::MAX_PRED_TIME_STEPS * sizeof(float));

                    for (size_t p = 0; p < path_num_; ++p)
                        nearest_idxs[p][i] = obs.obs_nearest_idxs_[p][agent];
                }

                // Compute Frenet (s, l) coordinates for all agents in this scenario
                core::ExoStates::GetFrenetPointsBatch(ego_ref_paths_, curr_path_idx, nearest_idxs, exo_thetas_flat_,
                                                      exo_xs_flat_, exo_ys_flat_, exo_vs_flat_, exo_ss_flat_,
                                                      exo_ls_flat_, s * exo_total_size_, vehicle_size_,
                                                      utils::MAX_PRED_TIME_STEPS, global_time_size_, valid_num);

                // Build STRtrees for fast collision broadphase
                core::ExoStates::buildSTRtreesFrenetBatch(
                    ego_ref_paths_, exo_cos_thetas_flat_, exo_sin_thetas_flat_, exo_bb_extent_xs_, exo_bb_extent_ys_,
                    exo_ss_flat_, exo_ls_flat_, exo_ls_projected_radius_flat_, nearest_idxs, exo_strtrees_,
                    s * exo_total_size_, s * global_time_size_, vehicle_size_, utils::MAX_PRED_TIME_STEPS,
                    global_time_size_, valid_num);
            }
        }

        // -----------------------------------------------------------------------
        // Node expansion (SIMD-batched)
        // -----------------------------------------------------------------------

        void VecQMDP_AD::expandNodesBatch(const AlignedVectorInt &node_idxs, IVectorT_qmdp &target_action_idxs,
                                          IVectorT_qmdp &target_path_idxs, FVectorT_qmdp &target_offsets,
                                          const int & /*expand_nodes_calls*/)
        {
            int batch = static_cast<int>(node_idxs.size());
            if (batch == 0)
                return;

            IVectorT_qmdp node_idxs_vec(node_idxs.data());

            // Gather node state
            IVectorT_qmdp active_flags, init_path_idxs, stepped_times, depths;
            FVectorT_qmdp xs, ys, thetas, vs, as, init_offsets, rewards, step_rollout_rewards;
            std::vector<std::array<int, utils::MAX_SIM_VEHICLES>> exo_inactive(batch);

            active_flags = IVectorT_qmdp::gather(node_active_flags_.data(), node_idxs_vec);
            depths = IVectorT_qmdp::gather(node_depth_.data(), node_idxs_vec);
            active_flags = active_flags & (depths < static_cast<int>(tree_height_));

            xs = FVectorT_qmdp::gather(ego_xs_flat_.data(), node_idxs_vec);
            ys = FVectorT_qmdp::gather(ego_ys_flat_.data(), node_idxs_vec);
            thetas = FVectorT_qmdp::gather(ego_thetas_flat_.data(), node_idxs_vec);
            vs = FVectorT_qmdp::gather(ego_vs_flat_.data(), node_idxs_vec);
            as = FVectorT_qmdp::gather(ego_as_flat_.data(), node_idxs_vec);
            init_path_idxs = IVectorT_qmdp::gather(ego_curr_ref_path_idxs_.data(), node_idxs_vec);
            init_offsets = FVectorT_qmdp::gather(ego_curr_lateral_offsets_.data(), node_idxs_vec);
            stepped_times = depths * static_cast<int>(step_time_size_);
            rewards = FVectorT_qmdp::fill(0.0f);
            step_rollout_rewards = FVectorT_qmdp::fill(0.0f);

            for (int i = 0; i < batch; ++i)
                exo_inactive[i] = node_exo_inactive_flags_[node_idxs[i]];

            // Simulate one step for the entire batch
            context_qmdp_->StepBatch(
                batch, utils::PATH_SIZE, initial_ego_v_ < utils::EGO_LOW_SPEED_THRESHOLD, stepped_times, xs, ys, thetas,
                vs, as, init_path_idxs, init_offsets, ego_ref_paths_, exo_xs_flat_, exo_ys_flat_, exo_ss_flat_,
                exo_ls_flat_, exo_ls_projected_radius_flat_, exo_vs_flat_, exo_cos_thetas_flat_, exo_sin_thetas_flat_,
                exo_bb_extent_xs_, exo_bb_extent_ys_, exo_strtrees_, target_path_idxs, target_offsets, rewards,
                step_rollout_rewards, step_time_size_, active_flags, exo_inactive);

            // Compute child indices
            IVectorT_qmdp scenario_idxs = node_idxs_vec / static_cast<int32_t>(tree_node_size_);
            IVectorT_qmdp relative_idxs = node_idxs_vec % static_cast<int32_t>(tree_node_size_);
            IVectorT_qmdp child_idxs = relative_idxs * static_cast<int32_t>(num_actions_) + 1 + target_action_idxs;
            IVectorT_qmdp global_child_idxs = child_idxs + scenario_idxs * static_cast<int32_t>(tree_node_size_);
            IVectorT_qmdp valid_child_mask = child_idxs < static_cast<int32_t>(tree_node_size_);

            // Write child state / trigger immediate back-propagation on boundary or collision
            for (size_t i = 0; i < IVectorT_qmdp::num_scalars; ++i)
            {
                const auto idx = utils::div_mod<size_t>(i, vector_with);

                if (!valid_child_mask[idx])
                {
                    // Child beyond tree boundary: treat step reward as rollout value
                    backPropagate(relative_idxs[idx], scenario_idxs[idx], rewards[idx] + step_rollout_rewards[idx]);
                }
                else
                {
                    const auto gci = global_child_idxs[idx];
                    ego_xs_flat_[gci] = xs[idx];
                    ego_ys_flat_[gci] = ys[idx];
                    ego_vs_flat_[gci] = vs[idx];
                    ego_as_flat_[gci] = as[idx];
                    ego_thetas_flat_[gci] = thetas[idx];
                    ego_curr_ref_path_idxs_[gci] = init_path_idxs[idx];
                    ego_curr_lateral_offsets_[gci] = init_offsets[idx];
                    node_active_flags_[gci] = active_flags[idx];
                    node_exo_inactive_flags_[gci] = exo_inactive[i];
                    // Also update the generic action-index cache used by explore
                    node_curr_action_idxs_[gci] = target_action_idxs[idx];

                    node_rewards_[gci] = rewards[idx];

                    if (~active_flags[idx]) // collision occurred
                    {
                        backPropagate(child_idxs[idx], scenario_idxs[idx], step_rollout_rewards[idx]);
                        backupDepth(child_idxs[idx], scenario_idxs[idx]);
                    }
                }
            }

            // ---- Rollout phase ----
            IVectorT_qmdp rollout_len =
                IVectorT_qmdp::gather(node_rollout_len_.data(), node_idxs_vec) * static_cast<int>(step_time_size_);
            FVectorT_qmdp ro_rewards = FVectorT_qmdp::fill(0.0f);
            FVectorT_qmdp ro_rollout = FVectorT_qmdp::fill(0.0f);
            IVectorT_qmdp ro_flags = active_flags & valid_child_mask;

            if (ro_flags.none())
                return;

            IVectorT_qmdp ro_flags_save = ro_flags;
            int           max_ro = rollout_len.hmax();

            while (max_ro > 0)
            {
                ro_flags = ro_flags & (rollout_len > 0);
                if (ro_flags.none())
                    break;

                context_qmdp_->StepBatch(
                    batch, utils::PATH_SIZE, initial_ego_v_ < utils::EGO_LOW_SPEED_THRESHOLD, stepped_times, xs, ys,
                    thetas, vs, as, init_path_idxs, init_offsets, ego_ref_paths_, exo_xs_flat_, exo_ys_flat_,
                    exo_ss_flat_, exo_ls_flat_, exo_ls_projected_radius_flat_, exo_vs_flat_, exo_cos_thetas_flat_,
                    exo_sin_thetas_flat_, exo_bb_extent_xs_, exo_bb_extent_ys_, exo_strtrees_, target_path_idxs,
                    target_offsets, ro_rewards, ro_rollout, step_time_size_, ro_flags, exo_inactive);

                rollout_len = rollout_len - static_cast<int>(step_time_size_);
                max_ro -= static_cast<int>(step_time_size_);
            }

            // Back-propagate rollout results
            for (size_t i = 0; i < IVectorT_qmdp::num_scalars; ++i)
            {
                const auto idx = utils::div_mod<size_t>(i, vector_with);
                if (ro_flags_save[idx])
                {
                    backPropagate(child_idxs[idx], scenario_idxs[idx], ro_rewards[idx] + ro_rollout[idx]);
                    backupDepth(child_idxs[idx], scenario_idxs[idx]);
                }
            }
        }

        // -----------------------------------------------------------------------
        // BeliefTreeSearch search entry point
        // -----------------------------------------------------------------------

        double VecQMDP_AD::beliefTreeSearch(int thread_id, const std::shared_ptr<core::EgoState> &ego_state,
                                            const std::shared_ptr<core::NetBelief>   &belief,
                                            const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                            const size_t                             &curr_path_idx)
        {
            ego_ref_paths_ = ego_ref_paths;
            resetDataStructures();
            updateEgoInitialState(ego_state);
            sampleScenarios(thread_id, ego_state, belief, curr_path_idx);

            auto start = std::chrono::steady_clock::now();

            IVectorT_qmdp target_action_idxs;
            IVectorT_qmdp target_path_idxs;
            FVectorT_qmdp target_offsets;

            int expand_calls = 0;
            int iteration_count = 0;

            while (true)
            {
                auto elapsed =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count();
                if (elapsed >= static_cast<long long>(max_planning_time_ms_))
                    break;

                if (expand_calls >= static_cast<int>(early_term_min_expand_calls_) && iteration_count > 0 &&
                    iteration_count % static_cast<int>(early_term_check_interval_) == 0)
                {
                    updateConvergenceTracking();
                    if (checkEarlyTermination())
                        break;
                }
                ++iteration_count;

                AlignedVectorInt nodes = exploreNodesVote(target_action_idxs);
                decomposeActionBatch(target_action_idxs, target_path_idxs, target_offsets);
                expandNodesBatch(nodes, target_action_idxs, target_path_idxs, target_offsets, expand_calls);
                ++expand_calls;
            }

            return static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
                    .count());
        }

        // -----------------------------------------------------------------------
        // Parallel search entry point
        // -----------------------------------------------------------------------

        std::shared_ptr<VecQMDP> VecQMDP_AD::makeWorkerInstance() const
        {
            // Create a single-threaded worker with the same per-thread scenario count.
            return std::make_shared<VecQMDP_AD>(static_cast<int>(scenario_size_), /*num_threads=*/1);
        }

        void VecQMDP_AD::parallelBeliefTreeSearch(const std::shared_ptr<core::EgoState>    &ego_state,
                                                  const std::shared_ptr<core::NetBelief>   &belief,
                                                  const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                                  const size_t                             &curr_path_idx)
        {
            dispatchParallelSearch(
                [&](VecQMDP *worker, size_t thread_idx) -> BeliefTreeSearchThreadResult
                {
                    auto *ad_worker = static_cast<VecQMDP_AD *>(worker);
                    ad_worker->beliefTreeSearch(static_cast<int>(thread_idx), ego_state, belief, ego_ref_paths,
                                                curr_path_idx);

                    // Collect first-level action Q-values from this worker's scenarios.
                    BeliefTreeSearchThreadResult result;
                    result.action_sum_values.assign(utils::NUM_ACTIONS, 0.0f);
                    result.action_counts.assign(utils::NUM_ACTIONS, 0);

                    for (uint32_t a = 0; a < utils::NUM_ACTIONS; ++a)
                    {
                        const uint32_t child_idx = a + 1; // child of root
                        float          sum_val = 0.0f;
                        int            valid_cnt = 0;

                        for (uint32_t s = 0; s < ad_worker->scenario_size_; ++s)
                        {
                            const uint32_t gci = ad_worker->getNodeIdx(child_idx, s);
                            if (ad_worker->getNodeCount(gci) > 0)
                            {
                                sum_val += ad_worker->getQNodeValue(gci);
                                ++valid_cnt;
                            }
                        }
                        result.action_sum_values[a] = sum_val;
                        result.action_counts[a] = valid_cnt;
                    }
                    return result;
                });

            // Compute global Q statistics used by downstream modules.
            float  max_q = -std::numeric_limits<float>::max();
            double sum_q = 0.0;
            int    valid_action_n = 0;

            for (uint32_t a = 0; a < utils::NUM_ACTIONS; ++a)
            {
                const float q = aggregated_action_values_[a];
                if (q > utils::ACTION_VALUE_INITIAL_MIN)
                {
                    if (q > max_q)
                        max_q = q;
                    sum_q += q;
                    ++valid_action_n;
                }
            }
            utils::max_q_value = (valid_action_n > 0) ? max_q : 0.0f;
            utils::weighted_q_value = (valid_action_n > 0) ? static_cast<float>(sum_q / valid_action_n) : 0.0f;
        }

        // -----------------------------------------------------------------------
        // Second-level action aggregation
        // -----------------------------------------------------------------------

        void VecQMDP_AD::getSecondBestAction(int first_best_action_idx, int &second_best_action_idx,
                                             int &second_best_target_path_idx, float &second_best_target_offset,
                                             float &second_best_value) const
        {
            // Guard: if the first-level search produced no valid results, return defaults.
            bool has_valid = false;
            for (float v : aggregated_action_values_)
                if (std::abs(v) > utils::ACTION_VALUE_ZERO_THRESHOLD)
                {
                    has_valid = true;
                    break;
                }

            if (!has_valid)
            {
                second_best_action_idx = first_best_action_idx;
                second_best_target_path_idx = utils::ACTION_TO_PATH[second_best_action_idx];
                second_best_target_offset = utils::PATH_OFFSETS_FLOAT[utils::ACTION_TO_OFFSET[second_best_action_idx]];
                second_best_value = 0.0f;
                return;
            }

            // Build the set of workers to aggregate over.
            // In single-threaded mode worker_instances_ is empty; use 'this'.
            std::vector<const VecQMDP_AD *> workers;
            if (worker_instances_.empty())
            {
                workers.push_back(this);
            }
            else
            {
                workers.reserve(worker_instances_.size());
                for (const auto &w : worker_instances_)
                    workers.push_back(static_cast<const VecQMDP_AD *>(w.get()));
            }

            const uint32_t first_action_child_base =
                utils::NUM_ACTIONS * (static_cast<uint32_t>(first_best_action_idx) + 1);

            std::vector<float> total_second_values(utils::NUM_ACTIONS, 0.0f);
            std::vector<int>   total_valid_counts(utils::NUM_ACTIONS, 0);

            for (const auto *w : workers)
            {
                for (uint32_t a = 0; a < utils::NUM_ACTIONS; ++a)
                {
                    const uint32_t child_idx = first_action_child_base + a + 1;

                    for (uint32_t s = 0; s < w->scenario_size_; ++s)
                    {
                        const uint32_t gci = w->getNodeIdx(child_idx, s);
                        if (w->getNodeCount(gci) > 0)
                        {
                            total_second_values[a] += w->getQNodeValue(gci);
                            ++total_valid_counts[a];
                        }
                    }
                }
            }

            second_best_value = utils::ACTION_VALUE_INITIAL_MIN;
            for (int i = 0; i < static_cast<int>(utils::NUM_ACTIONS); ++i)
            {
                const float avg = (total_valid_counts[i] > 0)
                                      ? total_second_values[i] / static_cast<float>(total_valid_counts[i])
                                      : utils::ACTION_VALUE_INITIAL_MIN;
                if (avg > second_best_value)
                {
                    second_best_value = avg;
                    second_best_action_idx = i;
                }
            }

            second_best_target_path_idx = utils::ACTION_TO_PATH[second_best_action_idx];
            second_best_target_offset = utils::PATH_OFFSETS_FLOAT[utils::ACTION_TO_OFFSET[second_best_action_idx]];
        }

    } // namespace planning
} // namespace vec_qmdp
