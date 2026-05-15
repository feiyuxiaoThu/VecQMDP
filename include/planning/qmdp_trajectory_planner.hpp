/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file qmdp_trajectory_planner.hpp
 * @brief Core orchestration layer for belief-aware autonomous driving motion planning.
 */

#pragma once

#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <fstream>
#include <memory>
#include <planning/trajectory_optimization.hpp>
#include <planning/vec_qmdp_ad.hpp>
#include <utils/global_utils.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace planning
    {
        namespace py = boost::python;
        namespace np = boost::python::numpy;

        class QMDPTrajectoryPlanner
        {
          public:
            QMDPTrajectoryPlanner(size_t                           num_threads = std::thread::hardware_concurrency(),
                                  std::shared_ptr<utils::MapUtils> map_utils_ptr = nullptr);
            virtual ~QMDPTrajectoryPlanner();

            /// Run parallel BeliefTreeSearch search across all worker threads inside qmdp_instance_.
            /// Aggregated action Q-values are stored inside qmdp_instance_ and can be
            /// read via getBestAction() / getSecondBestAction().
            void parallelBeliefTreeSearch(const std::shared_ptr<core::EgoState>  &ego_state,
                                          const std::shared_ptr<core::NetBelief> &belief,
                                          std::vector<std::shared_ptr<Path>>      ego_ref_paths,
                                          const size_t                           &curr_path_idx);

            void parallelOptimizeTrajectory(const std::shared_ptr<core::EgoState>      &ego_state,
                                            const std::shared_ptr<core::NetBelief>     &belief,
                                            std::vector<std::shared_ptr<Path>>          ego_ref_paths,
                                            std::vector<std::shared_ptr<Path>>          ego_extra_ref_paths,
                                            const std::shared_ptr<utils::OccupancyMap> &occupancy_map,
                                            size_t curr_path_idx, float curr_path_value, int best_action_idx,
                                            int best_target_path_idx, float best_target_offset, float best_value,
                                            int second_best_action_idx, int second_best_target_path_idx,
                                            float second_best_target_offset, float second_best_value);

            /// Select the best first-level action from the aggregated Q-values.
            void getBestAction(const std::shared_ptr<core::EgoState> &ego_state, const size_t &curr_path_idx,
                               int &best_action_idx, int &best_target_path_idx, float &best_target_offset,
                               float &best_value);

            void resetLastBestAction();

            /// Execute parallel BeliefTreeSearch search + trajectory optimization, return best action.
            int planTrajectory(const std::shared_ptr<core::EgoState>      &ego_state,
                               const std::shared_ptr<core::NetBelief>     &belief,
                               std::vector<std::shared_ptr<Path>>          ego_ref_paths,
                               std::vector<std::shared_ptr<Path>>          ego_extra_ref_paths,
                               const std::shared_ptr<utils::OccupancyMap> &occupancy_map, size_t curr_path_idx,
                               bool static_motion);

            inline size_t getNumThreads() const { return std::max<size_t>(1, num_threads_); }

            inline std::shared_ptr<std::vector<std::vector<float>>> getBestTrajectory() const
            {
                return std::make_shared<std::vector<std::vector<float>>>(best_trajectory_);
            }

            inline std::shared_ptr<std::vector<std::vector<float>>> getBestTrackedCppTrajectory() const
            {
                return std::make_shared<std::vector<std::vector<float>>>(best_tracked_trajectory_);
            }

          private:
            std::shared_ptr<utils::MapUtils> map_utils_ptr_;

            /// Single QMDP planner that owns its own internal thread pool and workers.
            std::shared_ptr<VecQMDP_AD> qmdp_instance_;

            /// Thread pool used exclusively for parallel trajectory optimisation.
            std::shared_ptr<utils::ThreadPool> traj_opt_thread_pool_;

            /// One trajectory optimisation instance per trajectory-opt worker thread.
            std::vector<std::shared_ptr<TrajectoryOptimization>> trajectory_opt_instances_;

            size_t num_threads_;

            int last_best_action_idx_;
            int last_best_target_path_idx_;

            mutable std::vector<std::vector<float>> best_trajectory_;
            mutable std::vector<std::vector<float>> best_tracked_trajectory_;

            mutable size_t best_trajectory_thread_idx_;
            mutable size_t best_trajectory_scenario_idx_;
        };

    } // namespace planning
} // namespace vec_qmdp