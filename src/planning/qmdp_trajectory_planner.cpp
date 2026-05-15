/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <algorithm>
#include <future>
#include <planning/qmdp_trajectory_planner.hpp>

#include <stdexcept>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h> // For GIL management
#include <numpy/arrayobject.h>

namespace vec_qmdp
{
    namespace planning
    {
        QMDPTrajectoryPlanner::QMDPTrajectoryPlanner(size_t num_threads, std::shared_ptr<utils::MapUtils> map_utils_ptr)
            : map_utils_ptr_(map_utils_ptr), num_threads_(num_threads), last_best_action_idx_(-1),
              last_best_target_path_idx_(-1), best_trajectory_thread_idx_(0), best_trajectory_scenario_idx_(0)
        {
            // Single VecQMDP_AD that manages its own internal thread pool.
            qmdp_instance_ = std::make_shared<VecQMDP_AD>(utils::NUM_SCENARIOS_PER_THREAD, num_threads);

            // Separate thread pool for trajectory optimisation (orthogonal concern).
            traj_opt_thread_pool_ = std::make_shared<utils::ThreadPool>(num_threads);

            trajectory_opt_instances_.reserve(num_threads);
            for (size_t i = 0; i < num_threads; ++i)
                trajectory_opt_instances_.emplace_back(
                    std::make_shared<TrajectoryOptimization>(utils::NUM_SCENARIOS_TRAJ_OPT_PER_THREAD));
        }

        QMDPTrajectoryPlanner::~QMDPTrajectoryPlanner() {}

        // -----------------------------------------------------------------------
        // Parallel BeliefTreeSearch search
        // -----------------------------------------------------------------------

        void QMDPTrajectoryPlanner::parallelBeliefTreeSearch(const std::shared_ptr<core::EgoState>  &ego_state,
                                                             const std::shared_ptr<core::NetBelief> &belief,
                                                             std::vector<std::shared_ptr<Path>>      ego_ref_paths,
                                                             const size_t                           &curr_path_idx)
        {
            // Remove reference paths that are not on the route (except the current
            // one).
            for (int i = 0; i < static_cast<int>(ego_ref_paths.size()); ++i)
            {
                if (i != static_cast<int>(curr_path_idx) && ego_ref_paths[i] && ego_ref_paths[i]->not_on_route_)
                {
                    ego_ref_paths[i] = nullptr;
                }
            }

            // Delegate to VecQMDP_AD; results are stored in qmdp_instance_.
            qmdp_instance_->parallelBeliefTreeSearch(ego_state, belief, ego_ref_paths, curr_path_idx);
        }

        // -----------------------------------------------------------------------
        // Best-action selection
        // -----------------------------------------------------------------------

        void QMDPTrajectoryPlanner::getBestAction(const std::shared_ptr<core::EgoState> &ego_state,
                                                  const size_t &curr_path_idx, int &best_action_idx,
                                                  int &best_target_path_idx, float &best_target_offset,
                                                  float &best_value)
        {
            const auto &action_values = qmdp_instance_->getAggregatedActionValues();

            bool has_valid_values = false;
            for (float v : action_values)
                if (v > utils::ACTION_VALUE_INITIAL_MIN + 1e-6f)
                {
                    has_valid_values = true;
                    break;
                }

            if (!has_valid_values)
            {
                std::cout << "[Warning] All action values are 0.0. Returning current "
                             "path index."
                          << std::endl;
                best_action_idx = static_cast<int>(curr_path_idx) * utils::NUM_LATERAL_OFFSETS + 1;
                best_target_path_idx = static_cast<int>(curr_path_idx);
                best_target_offset = 0.0f;
                best_value = 0.0f;
                return;
            }

            best_action_idx = 0;
            best_value = utils::ACTION_VALUE_INITIAL_MIN;

            for (int i = 0; i < static_cast<int>(action_values.size()); ++i)
            {
                float value = action_values[i] +
                              ((i == static_cast<int>(curr_path_idx)) ? utils::CURRENT_PATH_PREFERENCE_BONUS : 0.0f);

                if (value > best_value)
                {
                    best_action_idx = i;
                    best_target_path_idx = utils::ACTION_TO_PATH[i];
                    best_target_offset = utils::PATH_OFFSETS_FLOAT[utils::ACTION_TO_OFFSET[i]];
                    best_value = value;
                }
            }

            // If all actions collided, fall back to the previous or default action.
            if (best_value < utils::ALL_COLLIDED_VALUE_THRESHOLD && curr_path_idx != static_cast<size_t>(-1))
            {
                const int default_action = static_cast<int>(curr_path_idx) * utils::NUM_LATERAL_OFFSETS + 1;

                if (last_best_action_idx_ != -1 &&
                    action_values[last_best_action_idx_] - action_values[default_action] >
                        utils::LAST_ACTION_PREFERENCE_MARGIN)
                {
                    best_action_idx = last_best_action_idx_;
                    best_target_path_idx = utils::ACTION_TO_PATH[best_action_idx];
                    best_target_offset = utils::PATH_OFFSETS_FLOAT[utils::ACTION_TO_OFFSET[best_action_idx]];
                }
                else
                {
                    best_action_idx = default_action;
                    best_target_path_idx = static_cast<int>(curr_path_idx);
                    best_target_offset = 0.0f;
                }
            }

            last_best_action_idx_ = best_action_idx;
            last_best_target_path_idx_ = best_target_path_idx;
        }

        void QMDPTrajectoryPlanner::resetLastBestAction()
        {
            last_best_action_idx_ = -1;
            last_best_target_path_idx_ = -1;
        }

        // -----------------------------------------------------------------------
        // Parallel trajectory optimisation
        // -----------------------------------------------------------------------
        void QMDPTrajectoryPlanner::parallelOptimizeTrajectory(
            const std::shared_ptr<core::EgoState> &ego_state, const std::shared_ptr<core::NetBelief> &belief,
            std::vector<std::shared_ptr<Path>> ego_ref_paths, std::vector<std::shared_ptr<Path>> ego_extra_ref_paths,
            const std::shared_ptr<utils::OccupancyMap> &occupancy_map, size_t curr_path_idx, float curr_path_value,
            int best_action_idx, int best_target_path_idx, float best_target_offset, float best_value,
            int second_best_action_idx, int second_best_target_path_idx, float second_best_target_offset,
            float second_best_value)
        {
            std::vector<std::future<std::pair<size_t, float>>> futures;
            futures.reserve(num_threads_);

            size_t best_scenario_idx = 0;

            {
                // CRITICAL: Release GIL before spawning worker threads
                PyThreadState *_save = PyEval_SaveThread();

                for (size_t i = 0; i < num_threads_; ++i)
                {
                    futures.emplace_back(traj_opt_thread_pool_->enqueue(
                        [=, this]()
                        {
                            auto &traj_opt = trajectory_opt_instances_[i];

                            // ---- Phase 1: Parallel proposal generation ----
                            traj_opt->UpdatePathIndex(ego_ref_paths, ego_extra_ref_paths, curr_path_idx,
                                                      best_action_idx, best_target_path_idx, best_target_offset,
                                                      best_value, second_best_action_idx, second_best_target_path_idx,
                                                      second_best_target_offset, second_best_value, curr_path_value);

                            traj_opt->optimizeTrajectoryStep1_Generate(i, ego_state, belief, map_utils_ptr_);

                            // ---- Phase 2: Parallel evaluation ----
                            return traj_opt->optimizeTrajectoryStep2_Evaluate(i, ego_state, belief, occupancy_map,
                                                                              map_utils_ptr_);
                        }));
                }

                // ---- Phase 3: Aggregate results ----
                float best_eval_value = -std::numeric_limits<float>::max();
                for (size_t i = 0; i < num_threads_; ++i)
                {
                    auto [scenario_idx, value] = futures[i].get();
                    if (value > best_eval_value)
                    {
                        best_eval_value = value;
                        best_scenario_idx = scenario_idx;
                        best_trajectory_thread_idx_ = i;
                    }
                }
                PyEval_RestoreThread(_save);
            }

            auto &final_inst = trajectory_opt_instances_[best_trajectory_thread_idx_];
            best_trajectory_ = final_inst->getOptimizedTrajectory(ego_state, best_scenario_idx);
            best_tracked_trajectory_ = final_inst->getOptimizedTrackedTrajectory(ego_state, best_scenario_idx);
        }

        // -----------------------------------------------------------------------
        // Top-level planning entry point
        // -----------------------------------------------------------------------

        // Data flow: Python numpy array → NetBelief::updateBelief → NetBelief member
        // variables → NetBelief::Sample → ExoStates → VecQMDP_AD::sampleScenarios →
        // VecQMDP member variables
        int QMDPTrajectoryPlanner::planTrajectory(const std::shared_ptr<core::EgoState>      &ego_state,
                                                  const std::shared_ptr<core::NetBelief>     &belief,
                                                  std::vector<std::shared_ptr<Path>>          ego_ref_paths,
                                                  std::vector<std::shared_ptr<Path>>          ego_extra_ref_paths,
                                                  const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
                                                  size_t curr_path_idx, bool static_motion)
        {
            std::cout << "Begin planning with ego state: " << ego_state->x() << ", " << ego_state->y() << ", "
                      << ego_state->v() << ", " << ego_state->a() << ", " << ego_state->theta() << std::endl;

            if (static_motion)
            {
                std::cout << "static motion" << std::endl;
                best_trajectory_ = trajectory_opt_instances_[0]->getStaticTrajectory(ego_state);
                best_tracked_trajectory_ = best_trajectory_;
                resetLastBestAction();
                return last_best_action_idx_;
            }

            // Run parallel BeliefTreeSearch; aggregated Q-values land inside qmdp_instance_.
            parallelBeliefTreeSearch(ego_state, belief, ego_ref_paths, curr_path_idx);

            const auto &action_values = qmdp_instance_->getAggregatedActionValues();
            float       curr_path_value = action_values[curr_path_idx * utils::NUM_LATERAL_OFFSETS + 1];

            int   best_action_idx = 0;
            int   second_best_action_idx = 0;
            int   best_target_path_idx = 0;
            int   second_best_target_path_idx = 0;
            float best_target_offset = 0.0f;
            float second_best_target_offset = 0.0f;
            float best_value = 0.0f, second_best_value = 0.0f;

            getBestAction(ego_state, curr_path_idx, best_action_idx, best_target_path_idx, best_target_offset,
                          best_value);

            // Second-best action is now computed inside VecQMDP_AD.
            qmdp_instance_->getSecondBestAction(best_action_idx, second_best_action_idx, second_best_target_path_idx,
                                                second_best_target_offset, second_best_value);

            parallelOptimizeTrajectory(ego_state, belief, ego_ref_paths, ego_extra_ref_paths, occupancy_map,
                                       curr_path_idx, curr_path_value, best_action_idx, best_target_path_idx,
                                       best_target_offset, best_value, second_best_action_idx,
                                       second_best_target_path_idx, second_best_target_offset, second_best_value);

            return last_best_action_idx_;
        }

    } // namespace planning
} // namespace vec_qmdp
