/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <planning/vec_qmdp_closed_planner.hpp>
#include <sstream>

// Ensure all required Boost.Python headers are included
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

        // Create pickle support for VecQMDPClosedPlanner
        struct VecQMDPClosedPlanner_pickle_suite : py::pickle_suite
        {
            static py::tuple getstate(const VecQMDPClosedPlanner &p)
            {
                // Construct a Python tuple containing all necessary state elements
                // This is a minimal example; a real application should save more state
                return py::make_tuple(0); // Can be extended as needed
            }

            static void setstate(VecQMDPClosedPlanner &p, py::tuple state)
            {
                // Restore object state from 'state'
                // This is a minimal example; a real application may need to restore more fields
            }
        };

        VecQMDPClosedPlanner::VecQMDPClosedPlanner() { this->OnInit(); }

        VecQMDPClosedPlanner::~VecQMDPClosedPlanner() = default;

        bool VecQMDPClosedPlanner::OnInit()
        {
            // Initialize map utilities
            map_utils_ptr_ = std::make_shared<utils::MapUtils>();

            // Initialize occupancy map
            occupancy_map_ = std::make_shared<utils::OccupancyMap>(map_utils_ptr_);

            // Initialize trajectory planner
            trajectory_planner_ = std::make_shared<QMDPTrajectoryPlanner>(utils::NUM_THREADS, map_utils_ptr_);

            // Initialize NetBelief
            exo_belief_ = std::make_shared<core::NetBelief>();

            // Initialize EgoState
            ego_state_ = std::make_shared<core::EgoState>();

            // Initialize ego reference paths container
            ego_ref_paths_.resize(3, nullptr);

            // Fix random seed for determinism
            std::srand(utils::RANDOM_SEED);

            return true;
        }

        void VecQMDPClosedPlanner::LocationCallback(const py::object &ego_info)
        {
            // LOG_DS << "LocationCallback";
            ego_state_->UpdateEgoState(ego_info);
        }

        void VecQMDPClosedPlanner::ExoAgentsCallback(float ego_x, float ego_y, float ego_v, float ego_theta,
                                                     const py::dict &agents_info, bool &static_motion)
        {
            exo_belief_->updateEgoRefPaths(ego_ref_paths_, ego_extra_ref_paths_, curr_ref_path_idx_);
            exo_belief_->GetObservedExoState().updateObservedExoState(agents_info, ego_ref_paths_,
                                                                      ego_extra_ref_paths_);
            exo_belief_->filterDiscardedAgents(ego_x, ego_y, ego_v, ego_theta, static_motion);
            exo_belief_->GetObservedExoState().updateValidExoState();
        }

        void VecQMDPClosedPlanner::BeliefCallback(const py::object &pred_traj_x, const py::object &pred_traj_y,
                                                  const py::object &pred_traj_v, const py::object &pred_traj_theta,
                                                  const py::object &pred_traj_theta_cos,
                                                  const py::object &pred_traj_theta_sin, const py::object &pred_prob)
        {
            exo_belief_->updateBelief(pred_traj_x, pred_traj_y, pred_traj_v, pred_traj_theta, pred_traj_theta_cos,
                                      pred_traj_theta_sin, pred_prob);
        }

        void VecQMDPClosedPlanner::MapCallback(py::object &map_info)
        {
            map_utils_ptr_->getMapUtilsModule();
            map_utils_ptr_->UpdateMapUtilsObject(map_info);
            map_utils_ptr_->Reset();
            map_utils_ptr_->UpdateGlobalRouteEdgeIds();

            occupancy_map_->UpdateMapUtilsObject(map_info);
        }

        py::tuple VecQMDPClosedPlanner::MakePlanning(py::object &ego_info, py::dict &exo_info, py::object &map_info,
                                                     py::object &pred_traj_x, py::object &pred_traj_y,
                                                     py::object &pred_traj_v, py::object &pred_traj_theta,
                                                     py::object &pred_traj_theta_cos, py::object &pred_traj_theta_sin,
                                                     py::object &pred_prob, int iteration, float offset_x,
                                                     float offset_y, std::string scenario_token)
        {
            utils::iteration = iteration;
            utils::scenario_token = scenario_token;

            // Update vehicle and environment information
            this->LocationCallback(ego_info);

            // Ego vehicle traveled distance
            static float moved_len = 0.0f;

            // Ego vehicle reference line edge ids
            static std::vector<std::string> refline_edge_ids;
            static std::vector<std::string> refline_edge_names;

            // Whether ego is close to a junction on the current lane
            static bool is_close_to_junction = false;

            // Ego's last executed action
            static int last_best_action = -1;

            // If this is the first iteration, initialize the map and ego vehicle information
            if (iteration == 0)
            {
                moved_len = 0.0f;

                utils::offset_x = offset_x;
                utils::offset_y = offset_y;

                this->MapCallback(map_info);
                std::string              edge_token;
                std::string              edge_name;
                std::vector<std::string> candidate_edge_token;

                map_utils_ptr_->UpdateEgoStartingLaneId(ego_state_->x(), ego_state_->y(), ego_state_->theta(),
                                                        edge_token, edge_name, candidate_edge_token);
                ego_state_->edge_token_ = edge_token;
                ego_state_->edge_name_ = edge_name;

                // Update ego map information
                map_utils_ptr_->UpdateEgoEdgeDirectly(edge_token, edge_name);

                // Get reference lines (constitute the action space)
                refline_edge_ids.clear();
                refline_edge_names.clear();
                map_utils_ptr_->GetEgoRefPaths(ego_state_->x(), ego_state_->y(), ego_state_->v(), curr_ref_path_idx_,
                                               ego_ref_paths_, ego_extra_ref_paths_, refline_edge_ids,
                                               refline_edge_names, "", edge_token, edge_name, candidate_edge_token,
                                               is_close_to_junction);

                occupancy_map_->UpdateDrivableMapObjects(
                    ego_state_->x(), ego_state_->y(), ego_state_->v(), ego_state_->theta(), edge_token, edge_name,
                    refline_edge_ids, refline_edge_names,
                    utils::OCCUPANCY_MAP_RADIUS_OFFSET + utils::VELOCITY_TO_DISTANCE_TIME_HORIZON * ego_state_->v());

                // Initialize belief
                BeliefCallback(pred_traj_x, pred_traj_y, pred_traj_v, pred_traj_theta, pred_traj_theta_cos,
                               pred_traj_theta_sin, pred_prob);
            }
            else
            {
                // Clear traffic light information
                // LOG_DS << "Reset red light info";
                map_utils_ptr_->ResetRedLightInfo();

                std::string prev_edge_token = ego_state_->edge_token();
                std::string prev_edge_name = ego_state_->edge_name();

                // Update ego map information
                // LOG_DS << "Update ego map info";
                std::string              edge_token = prev_edge_token;
                std::string              edge_name = prev_edge_name;
                std::vector<std::string> candidate_edge_tokens;
                map_utils_ptr_->UpdateEgoEdge(
                    ego_state_->x(), ego_state_->y(), edge_token, edge_name,
                    (last_best_action == -1
                         ? std::vector<std::string>()
                         : ego_ref_paths_[utils::ACTION_TO_PATH[last_best_action]]->comprised_ref_path_ids_),
                    candidate_edge_tokens);
                ego_state_->edge_token_ = edge_token;
                ego_state_->edge_name_ = edge_name;

                // Update ego vehicle traveled distance
                moved_len += ego_state_->v() * 0.1f;

                // Update reference paths (and the current reference path index)
                // When the ego vehicle's edge changes, reference paths need to be updated
                // (note: when transitioning from lane->lane connector or lane connector->lane connector,
                //  update ref path if the current refline index changes)
                // Also update reference paths when on a lane and approaching a junction with multiple successors
                if (prev_edge_token != edge_token)
                {
                    // Reset close_to_junction when edge changes
                    is_close_to_junction = false;

                    if (edge_name == "LANE")
                    {
                        // LOG_DS << "Update ego ref paths";
                        refline_edge_ids.clear();
                        refline_edge_names.clear();
                        map_utils_ptr_->GetEgoRefPaths(
                            ego_state_->x(), ego_state_->y(), ego_state_->v(), curr_ref_path_idx_, ego_ref_paths_,
                            ego_extra_ref_paths_, refline_edge_ids, refline_edge_names, prev_edge_token, edge_token,
                            edge_name, std::vector<std::string>(), is_close_to_junction);
                    }
                    else if (edge_name == "LANE_CONNECTOR")
                    {
                        // Update curr_ref_path_idx_
                        map_utils_ptr_->UpdateEgoCurrRefPathIdx(ego_ref_paths_, edge_token, candidate_edge_tokens,
                                                                curr_ref_path_idx_);

                        // Update ego_ref_paths_ and ego_extra_ref_paths_
                        map_utils_ptr_->UpdateEgoRefPaths(ego_ref_paths_, ego_extra_ref_paths_, curr_ref_path_idx_,
                                                          refline_edge_ids, refline_edge_names, edge_token, edge_name);

                        // Update traffic light information for reference paths
                        map_utils_ptr_->UpdateEgoRefPathsTrafficInfo(ego_ref_paths_);
                    }

                    map_utils_ptr_->UpdateEgoEdgeDirectly(edge_token, edge_name);

                    // Reset last best action due to reference-line changes
                    trajectory_planner_->resetLastBestAction();
                }
                else
                {
                    // Check whether to update reference paths when close to a junction
                    if (edge_name == "LANE" && !is_close_to_junction)
                    {
                        float dist_to_junction =
                            map_utils_ptr_->DistanceToEndOfEdge(ego_state_->x(), ego_state_->y(), edge_token);
                        if (dist_to_junction <= utils::LEAST_DIST2JUNCTION)
                        {
                            // Check whether the current edge has multiple successors
                            const auto &successors = map_utils_ptr_->GetRouteSuccessorIdsById(edge_token, edge_name);
                            if (successors.size() > 1)
                            {
                                // std::cout << "Update ego ref paths because close to junction with multiple
                                // successors" << std::endl;
                                is_close_to_junction = true;

                                refline_edge_ids.clear();
                                refline_edge_names.clear();
                                map_utils_ptr_->GetEgoRefPaths(ego_state_->x(), ego_state_->y(), ego_state_->v(),
                                                               curr_ref_path_idx_, ego_ref_paths_, ego_extra_ref_paths_,
                                                               refline_edge_ids, refline_edge_names, prev_edge_token,
                                                               edge_token, edge_name, std::vector<std::string>(),
                                                               is_close_to_junction);

                                // Reset last best action due to reference-line changes
                                trajectory_planner_->resetLastBestAction();
                            }
                        }
                    }

                    // LOG_DS << "Update ego ref paths traffic info";

                    // Keep reference path the same but update traffic light info
                    map_utils_ptr_->UpdateEgoRefPathsTrafficInfo(ego_ref_paths_);
                }

                if (prev_edge_name != edge_name || moved_len > utils::DRIVABLE_MAP_UPDATE_DISTANCE)
                {
                    // Update drivable area
                    occupancy_map_->UpdateDrivableMapObjects(
                        ego_state_->x(), ego_state_->y(), ego_state_->v(), ego_state_->theta(), edge_token, edge_name,
                        refline_edge_ids, refline_edge_names,
                        utils::OCCUPANCY_MAP_RADIUS_OFFSET +
                            utils::VELOCITY_TO_DISTANCE_TIME_HORIZON * ego_state_->v());
                    moved_len = 0.0f;
                }
            }

            bool static_motion =
                (iteration <= utils::STATIC_MOTION_MAX_ITERATION) && ego_state_->IsCloseToJunction(map_utils_ptr_);

            // Update other agents' information
            this->ExoAgentsCallback(ego_state_->x(), ego_state_->y(), ego_state_->v(), ego_state_->theta(), exo_info,
                                    static_motion);

            // Run planning
            last_best_action =
                trajectory_planner_->planTrajectory(ego_state_, exo_belief_, ego_ref_paths_, ego_extra_ref_paths_,
                                                    occupancy_map_, curr_ref_path_idx_, static_motion);

            // Retrieve planned trajectory
            auto trajectory = trajectory_planner_->getBestTrajectory();

            auto tracked_cpp_trajectory = trajectory_planner_->getBestTrackedCppTrajectory();

#if ENABLE_PERFORMANCE_MONITORING
            utils::printPerformanceData();
            utils::printPerformanceTiming();
            utils::resetPerformanceTiming();
#endif

            return py::make_tuple(trajectory, 0, tracked_cpp_trajectory, tracked_cpp_trajectory // generated trajectory
            );
        }
    } // namespace planning
} // namespace vec_qmdp

// Export module to Python
BOOST_PYTHON_MODULE(vec_qmdp_closed_planner)
{
    using namespace boost::python;
    using namespace boost::python::numpy;

    boost::python::numpy::initialize();

    // Create bindings for trajectory point containers
    class_<std::vector<float>>("FloatList").def(vector_indexing_suite<std::vector<float>, true>());

    class_<std::vector<std::vector<float>>, std::shared_ptr<std::vector<std::vector<float>>>>("TrajectoryPoints")
        .def(vector_indexing_suite<std::vector<std::vector<float>>, true>());

    class_<std::vector<std::string>>("StringList").def(vector_indexing_suite<std::vector<std::string>, true>());

    // Export VecQMDPClosedPlanner class and its methods
    class_<vec_qmdp::planning::VecQMDPClosedPlanner>("VecQMDPClosedPlanner")
        .def("MakePlanning", &vec_qmdp::planning::VecQMDPClosedPlanner::MakePlanning)
        .def_pickle(vec_qmdp::planning::VecQMDPClosedPlanner_pickle_suite());
}
