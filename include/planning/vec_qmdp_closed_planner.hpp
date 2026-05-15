/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file vec_qmdp_closed_planner.hpp
 * @brief Python-facing Boost.Python wrapper exposing the VecQMDP planner.
 */

#pragma once

#include <core/net_belief.hpp>
#include <core/state.hpp>
#include <map>
#include <memory>
#include <planning/context_qmdp.hpp>
#include <planning/qmdp_trajectory_planner.hpp>
#include <string>
#include <utils/map_utils.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>

namespace vec_qmdp
{
    namespace planning
    {
        namespace py = boost::python;

        class VecQMDPClosedPlanner
        {
            using Path = utils::Path;
            using STRtree = collision::STRtree;
            // using AABB = STRtree::AABB;
          public:
            VecQMDPClosedPlanner();
            ~VecQMDPClosedPlanner();

            // Initialize
            bool OnInit();

            // Callback handlers
            void LocationCallback(const py::object &location_info);
            void ExoAgentsCallback(float ego_x, float ego_y, float ego_v, float ego_theta, const py::dict &agents_info,
                                   bool &static_motion);
            void BeliefCallback(const py::object &pred_traj_x, const py::object &pred_traj_y,
                                const py::object &pred_traj_v, const py::object &pred_traj_theta,
                                const py::object &pred_traj_theta_cos, const py::object &pred_traj_theta_sin,
                                const py::object &pred_prob);
            void MapCallback(py::object &map_info);

            // Planning interface
            py::tuple MakePlanning(py::object &ego_info, py::dict &exo_info, py::object &map_info,
                                   py::object &pred_traj_x, py::object &pred_traj_y, py::object &pred_traj_v,
                                   py::object &pred_traj_theta, py::object &pred_traj_theta_cos,
                                   py::object &pred_traj_theta_sin, py::object &pred_prob, int iteration,
                                   float offset_x, float offset_y, std::string scenario_token = "");

          private:
            // Ego state
            std::shared_ptr<core::EgoState>    ego_state_;
            std::vector<std::shared_ptr<Path>> ego_ref_paths_;
            std::vector<std::shared_ptr<Path>> ego_extra_ref_paths_;
            int                                curr_ref_path_idx_;

            // Exogenous belief (other agents)
            std::shared_ptr<core::NetBelief> exo_belief_;

            // Trajectory planner
            std::shared_ptr<QMDPTrajectoryPlanner> trajectory_planner_;

            // Map utilities
            std::shared_ptr<utils::MapUtils> map_utils_ptr_;

            // Occupancy map
            std::shared_ptr<utils::OccupancyMap> occupancy_map_;

            int current_iteration_;
        };
    } // namespace planning
} // namespace vec_qmdp