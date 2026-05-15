/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file net_belief.hpp
 * @brief Manages environmental uncertainty and agent belief filtering for autonomous driving scenarios.
 */

#pragma once

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <core/state.hpp>
#include <memory>
#include <planning/context_qmdp.hpp>
#include <utils/path_utils.hpp>
#include <vector>

namespace vec_qmdp
{
    namespace core
    {
        namespace py = boost::python;
        namespace np = boost::python::numpy;

        class NetBelief
        {
            using ContextQMDP = planning::ContextQMDP;

          public:
            NetBelief();
            ~NetBelief();

            void updateEgoRefPaths(const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                   const std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths,
                                   const size_t                              ego_curr_ref_path_idx);

            void initializeBelief();

            void updateBelief(const py::object &pred_traj_x, const py::object &pred_traj_y,
                              const py::object &pred_traj_v, const py::object &pred_traj_theta,
                              const py::object &pred_traj_theta_cos, const py::object &pred_traj_theta_sin,
                              const py::object &pred_prob);

            inline ObservedExoState &GetObservedExoState() { return observed_exo_state_; }

            /**
             * Keep only the most relevant agent trajectories for intent sampling.
             * - First, only keep agents that lie on reference paths.
             * - For agents on the current reference path: only keep the leading vehicle closest to ego; trailing
             * vehicles are ignored.
             * - Finally, keep agents that have a future path intersection at t=0.
             * Also consider static obstacles located ahead of the ego vehicle (on the current lane or adjacent lanes).
             */
            void filterDiscardedAgents(float ego_x, float ego_y, float ego_v, float ego_theta, bool &static_motion);

          public:
            std::shared_ptr<ContextQMDP>       context_qmdp_;
            ObservedExoState                   observed_exo_state_;
            std::vector<std::shared_ptr<Path>> ego_ref_paths_;
            size_t                             ego_curr_ref_path_idx_;
            std::unordered_set<std::string>    collided_agents_tokens_;

            // Prediction trajectory arrays, fixed size: [utils::MAX_OBS_VEHICLES][MAX_PRED_MODES][DUMMY_TIME_STEPS]
            float *pred_traj_x_nd_data_ptr_;         // [agent_num, mode_num, time_step]
            float *pred_traj_y_nd_data_ptr_;         // [agent_num, mode_num, time_step]
            float *pred_traj_v_nd_data_ptr_;         // [agent_num, mode_num, time_step]
            float *pred_traj_theta_nd_data_ptr_;     // [agent_num, mode_num, time_step]
            float *pred_traj_theta_cos_nd_data_ptr_; // [agent_num, mode_num, time_step]
            float *pred_traj_theta_sin_nd_data_ptr_; // [agent_num, mode_num, time_step]
            float *pred_prob_nd_data_ptr_;           // [agent_num, mode_num]
        };

    } // namespace core
} // namespace vec_qmdp