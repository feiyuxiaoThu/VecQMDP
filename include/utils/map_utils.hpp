/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file map_utils.hpp
 * @brief HD-map interface (Python/Boost.Python bridge), lane topology caching, and reference-path construction.
 */

#pragma once

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include <utils/global_utils.hpp>
#include <utils/math_utils.hpp>
#include <utils/params.hpp>
#include <utils/path_utils.hpp>

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <mutex>

namespace vec_qmdp
{
    namespace utils
    {
        namespace py = boost::python;
        namespace np = boost::python::numpy;

        // RAII wrapper for Python GIL (Global Interpreter Lock)
        // This ensures thread-safe access to Python objects in multi-threaded C++ code
        class ScopedGILAcquire
        {
          public:
            ScopedGILAcquire() : gstate_(PyGILState_Ensure()) {}
            ~ScopedGILAcquire() { PyGILState_Release(gstate_); }

            // Prevent copying
            ScopedGILAcquire(const ScopedGILAcquire &) = delete;
            ScopedGILAcquire &operator=(const ScopedGILAcquire &) = delete;

          private:
            PyGILState_STATE gstate_;
        };

        // RAII wrapper for releasing GIL before spawning threads
        // Use this when the main thread (which holds GIL) needs to spawn worker threads
        class ScopedGILRelease
        {
          public:
            ScopedGILRelease() : tstate_(PyEval_SaveThread()) {}
            ~ScopedGILRelease() { PyEval_RestoreThread(tstate_); }

            // Prevent copying
            ScopedGILRelease(const ScopedGILRelease &) = delete;
            ScopedGILRelease &operator=(const ScopedGILRelease &) = delete;

          private:
            PyThreadState *tstate_;
        };

        class MapUtils
        {
          private:
            // Mutex to protect cache data structures in multi-threaded access
            mutable std::mutex cache_mutex_;

          public:
            MapUtils();
            ~MapUtils();

            inline void getMapUtilsModule()
            {
                map_utils_py = py::import("python_planner.utils.map_utils_py");
                SemanticMapLayerPy = map_utils_py.attr("SemanticMapLayer");
                Point2DPy = map_utils_py.attr("Point2D");
                PointPy = map_utils_py.attr("Point");
            }

            inline void UpdateMapUtilsObject(const py::object &map_info) { MapUtilsObject = map_info; }

            inline void ResetRedLightInfo() { LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT.clear(); }

            inline void ResetPath() { LANE_ID_MAP_SMOOTH_PATH.clear(); }

            inline void Reset()
            {
                LANE_ID_MAP_SUCCESSORS.clear();
                LANE_ID_MAP_ROUTE_SUCCESSORS.clear();
                LANE_ID_MAP_PREDECESSORS.clear();
                LANE_ID_MAP_ROUTE_PREDECESSORS.clear();
                LANE_ID_MAP_LEFT_NEIGHBOR.clear();
                LANE_ID_MAP_RIGHT_NEIGHBOR.clear();
                LANE_ID_MAP_MISSED_GOAL_NUM.clear();
                LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM.clear();
                LANE_ID_MAP_IS_TERMINATED.clear();
                LANE_ID_MAP_EDGE_TYPE.clear();
                LANE_ID_MAP_ROADBLOCK_ID.clear();
                LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT.clear();
                LANE_ID_MAP_HAS_PARENT_RELATION.clear();
                ResetRedLightInfo();
                ResetPath();
            }

            inline void UpdateGlobalRouteEdgeIds()
            {
                py::list global_routes_edge_ids =
                    py::extract<py::list>(py::list(MapUtilsObject.attr("_route_lane_dict").attr("keys")()));

                for (int i = 0; i < len(global_routes_edge_ids); ++i)
                {
                    std::string edge_id = py::extract<std::string>(global_routes_edge_ids[i]);
                    // std::cout << "global_routes_edge_ids[" << i << "]: " << edge_id << std::endl;
                    GLOBAL_ROUTE_EDGE_ID[edge_id] = true;
                }
            }

            inline bool IsOnRoute(const std::string &edge_id) { return GLOBAL_ROUTE_EDGE_ID.count(edge_id); }

            inline boost::python::object GetObjectById(std::string object_id, std::string layer_name)
            {
                ScopedGILAcquire      gil_lock; // Acquire GIL for thread-safe Python access
                boost::python::object layer_name_py = SemanticMapLayerPy.attr(layer_name.c_str());
                auto                  result = MapUtilsObject.attr("GetObjectById")(object_id, layer_name_py);
                return result;
            }

            /**
             * edge name should be "LANE" or "LANE_CONNECTOR"
             */
            inline bool IsPointInLaneById(const std::string &lane_id, float x, float y, std::string edge_name = "LANE")
            {
                if (lane_id.empty())
                    return false;

                py::object point_py = Point2DPy(x, y);
                py::object in_lane_py = MapUtilsObject.attr("IsPointInLaneById")(lane_id, point_py, edge_name);
                return py::extract<bool>(in_lane_py);
            }

            inline py::object GetLaneOrConnectorById(const std::string &edge_id, const std::string &edge_name = "")
            {
                py::object edge_object;
                if (edge_name != "")
                    edge_object = GetObjectById(edge_id, edge_name);
                else
                {
                    edge_object = GetObjectById(edge_id, "LANE");
                    if (edge_object.is_none())
                        edge_object = GetObjectById(edge_id, "LANE_CONNECTOR");
                }
                return edge_object;
            }

            inline py::object GetLaneOrConnectorAndNameById(const std::string &edge_id, std::string &edge_name)
            {
                py::object edge_object;
                if (edge_name != "")
                    edge_object = GetObjectById(edge_id, edge_name);
                else
                {
                    edge_object = GetObjectById(edge_id, "LANE");

                    if (edge_object.is_none())
                    {
                        edge_name = "LANE_CONNECTOR";
                        edge_object = GetObjectById(edge_id, "LANE_CONNECTOR");
                    }
                    else
                    {
                        edge_name = "LANE";
                    }
                }
                return edge_object;
            }

            inline void GetEdgeIdByPosition(float x, float y, std::vector<std::string> &edge_ids,
                                            std::string &edge_name)
            {
                py::object point_py = Point2DPy(x, y);
                py::object layer_name_py = SemanticMapLayerPy.attr("LANE");
                py::list   edge_id_py =
                    py::extract<py::list>(MapUtilsObject.attr("GetAllMapObjectsIds")(point_py, layer_name_py));

                if (len(edge_id_py) > 0)
                {
                    edge_name = "LANE";
                    edge_ids.emplace_back(py::extract<std::string>(edge_id_py[0]));
                    return;
                }

                // Check whether the point is on a lane connector
                layer_name_py = SemanticMapLayerPy.attr("LANE_CONNECTOR");
                edge_id_py = py::extract<py::list>(MapUtilsObject.attr("GetAllMapObjectsIds")(point_py, layer_name_py));

                if (len(edge_id_py) > 0)
                {
                    edge_name = "LANE_CONNECTOR";
                    for (int i = 0; i < len(edge_id_py); ++i)
                    {
                        edge_ids.emplace_back(py::extract<std::string>(edge_id_py[i]));
                    }
                    return;
                }
            }

            inline std::string GetEdgeTypeById(const std::string &lane_id)
            {
                if (LANE_ID_MAP_EDGE_TYPE.count(lane_id))
                    return LANE_ID_MAP_EDGE_TYPE[lane_id];

                py::object object_name_py = MapUtilsObject.attr("GetEdgeTypeById")(lane_id);
                LANE_ID_MAP_EDGE_TYPE[lane_id] = py::extract<std::string>(object_name_py);

                return LANE_ID_MAP_EDGE_TYPE[lane_id];
            }

            inline float MissGoalNum(const std::string &edge_id)
            {
                if (LANE_ID_MAP_MISSED_GOAL_NUM.count(edge_id))
                    return LANE_ID_MAP_MISSED_GOAL_NUM[edge_id];

                int num = py::extract<int>(MapUtilsObject.attr("PenaltyOnMissGoalLaneId")(edge_id));

                LANE_ID_MAP_MISSED_GOAL_NUM[edge_id] = num;
                return num;
            }

            inline float ActualMissGoalNum(const std::string &edge_id)
            {
                if (LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM.count(edge_id))
                    return LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM[edge_id];

                int num = py::extract<int>(MapUtilsObject.attr("ActualPenaltyOnMissGoalLaneId")(edge_id));
                if (num == utils::MISS_GOAL_TERMINATED_PENALTY)
                {
                    LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM[edge_id] = utils::MISS_GOAL_TERMINATED_PENALTY;
                    LANE_ID_MAP_IS_TERMINATED.insert(edge_id);
                }
                else if (num > utils::MISS_GOAL_HIGH_PENALTY_OFFSET)
                {
                    LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM[edge_id] = num - utils::MISS_GOAL_HIGH_PENALTY_OFFSET;
                    LANE_ID_MAP_IS_TERMINATED.insert(edge_id);
                }
                else
                    LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM[edge_id] = num;

                return LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM[edge_id];
            }

            inline bool IsTerminated(const std::string &edge_id) { return LANE_ID_MAP_IS_TERMINATED.count(edge_id); }

            inline std::vector<std::string> GetPredecessorIdsById(std::string edge_id, std::string edge_name = "")
            {
                if (LANE_ID_MAP_PREDECESSORS.count(edge_id))
                    return LANE_ID_MAP_PREDECESSORS[edge_id];

                std::vector<std::string> predecessor_ids;

                py::object edge_object = GetLaneOrConnectorById(edge_id, edge_name);

                py::list predecessor_ids_py =
                    py::extract<py::list>(MapUtilsObject.attr("GetPredecessorIds")(edge_object));

                for (int i = 0; i < len(predecessor_ids_py); ++i)
                {
                    predecessor_ids.emplace_back(py::extract<std::string>(predecessor_ids_py[i]));
                }

                LANE_ID_MAP_PREDECESSORS[edge_id] = std::move(predecessor_ids);
                return LANE_ID_MAP_PREDECESSORS[edge_id];
            }

            /**
             * Successors are sorted by miss-goal penalty; successors that lead to a dead end are filtered out.
             */
            inline std::vector<std::string> GetSuccessorIdsById(std::string edge_id, std::string edge_name = "")
            {
                if (edge_id.empty())
                    return std::vector<std::string>();

                // First check: read from cache with lock
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    if (LANE_ID_MAP_SUCCESSORS.count(edge_id))
                    {
                        return LANE_ID_MAP_SUCCESSORS[edge_id];
                    }
                }

                // Cache miss: query Python (with GIL)
                ScopedGILAcquire         gil_lock;
                std::vector<std::string> successor_ids;

                py::object edge_object = GetLaneOrConnectorById(edge_id, edge_name);

                py::list successor_ids_py = py::extract<py::list>(MapUtilsObject.attr("GetSuccessorIds")(edge_object));

                for (int i = 0; i < len(successor_ids_py); ++i)
                {
                    successor_ids.emplace_back(py::extract<std::string>(successor_ids_py[i]));
                }
                // Double-check and write to cache with lock
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    // Double-check: another thread might have populated it
                    if (!LANE_ID_MAP_SUCCESSORS.count(edge_id))
                    {
                        LANE_ID_MAP_SUCCESSORS[edge_id] = successor_ids;
                    }
                    return LANE_ID_MAP_SUCCESSORS[edge_id];
                }
            }

            inline bool HasParentRelation(std::string edge_id_1, std::string edge_id_2)
            {

                // First check: read from cache with lock
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto                        key = std::make_pair(edge_id_1, edge_id_2);
                    if (LANE_ID_MAP_HAS_PARENT_RELATION.count(key))
                    {
                        return LANE_ID_MAP_HAS_PARENT_RELATION[key];
                    }
                }

                // Cache miss: compute relationship (may call Python via GetSuccessorIdsById)
                auto successor_ids_1 = GetSuccessorIdsById(edge_id_1);

                bool has_relation = false;
                if (std::find(successor_ids_1.begin(), successor_ids_1.end(), edge_id_2) != successor_ids_1.end())
                {
                    has_relation = true;
                }
                else
                {
                    auto successor_ids_2 = GetSuccessorIdsById(edge_id_2);
                    if (std::find(successor_ids_2.begin(), successor_ids_2.end(), edge_id_1) != successor_ids_2.end())
                    {
                        has_relation = true;
                    }
                }

                // Write to cache with lock
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto                        key1 = std::make_pair(edge_id_1, edge_id_2);
                    auto                        key2 = std::make_pair(edge_id_2, edge_id_1);
                    LANE_ID_MAP_HAS_PARENT_RELATION[key1] = has_relation;
                    LANE_ID_MAP_HAS_PARENT_RELATION[key2] = has_relation;
                }

                return has_relation;
            }

            inline bool IsDeadEnd(std::string edge_id)
            {
                std::vector<std::string> successor_ids = GetSuccessorIdsById(edge_id);
                return (successor_ids.size() == 0) ? true : false;
            }

            // make sure the input edge object is lane instead of lane connector
            inline std::pair<std::string, std::string> GetLeftAndRightNeighborId(const std::string &edge_id)
            {
                auto      edge_object = GetLaneOrConnectorById(edge_id);
                py::tuple left_and_right_id =
                    py::extract<py::tuple>(MapUtilsObject.attr("GetLeftAndRightNeighborId")(edge_object));

                std::string left_neighbor_id = py::extract<std::string>(left_and_right_id[0]);
                std::string right_neighbor_id = py::extract<std::string>(left_and_right_id[1]);
                LANE_ID_MAP_LEFT_NEIGHBOR[edge_id] = left_neighbor_id;
                LANE_ID_MAP_RIGHT_NEIGHBOR[edge_id] = right_neighbor_id;

                return std::make_pair(left_neighbor_id, right_neighbor_id);
            }

            inline std::string GetLeftNeighborIdById(const std::string &edge_id)
            {
                if (edge_id.empty())
                    return "";
                if (LANE_ID_MAP_LEFT_NEIGHBOR.count(edge_id))
                    return LANE_ID_MAP_LEFT_NEIGHBOR[edge_id];

                return GetLeftAndRightNeighborId(edge_id).first;
            }

            inline std::string GetRightNeighborIdById(const std::string &edge_id)
            {
                if (edge_id.empty())
                    return "";
                if (LANE_ID_MAP_RIGHT_NEIGHBOR.count(edge_id))
                    return LANE_ID_MAP_RIGHT_NEIGHBOR[edge_id];

                return GetLeftAndRightNeighborId(edge_id).second;
            }

            inline bool IsLaneLeftNeigbor(const std::string &edge_id, const std::string &neighbor_id)
            {
                if (edge_id.empty())
                    return false;

                auto left_neighbor_id = GetLeftNeighborIdById(edge_id);
                while (left_neighbor_id != "")
                {
                    if (left_neighbor_id == neighbor_id)
                        return true;
                    left_neighbor_id = GetLeftNeighborIdById(left_neighbor_id);
                }

                return false;
            }

            inline int ComputeLeftDepth(const std::string &edge_id)
            {
                if (edge_id.empty())
                    return 10;

                int         depth = 0;
                std::string curr = edge_id;
                while (true)
                {
                    auto left = GetLeftNeighborIdById(curr);
                    if (left.empty())
                        break;
                    ++depth;
                    curr = left;
                }
                return depth;
            }

            /*
             * Check whether edge_id has a red traffic light
             * @param edge_id must be a lane_connector
             */
            inline bool HasRedTrafficLight(const std::string &edge_id)
            {
                // // for replay log
                // return false;

                if (edge_id.empty())
                    return false;

                if (LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT.count(edge_id))
                    return LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT[edge_id];

                py::object edge_object = GetLaneOrConnectorById(edge_id, "LANE_CONNECTOR");
                if (edge_object.is_none())
                    return false;

                py::object has_red_py = MapUtilsObject.attr("HasRedTrafficLight")(edge_object);
                bool       has_red = py::extract<bool>(has_red_py);
                LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT[edge_id] = has_red;
                return has_red;
            }

            inline std::shared_ptr<Path> GetRefLine(const std::string &edge_id, const std::string &edge_name = "")
            {
                if (edge_id.empty())
                    return nullptr;

                if (LANE_ID_MAP_SMOOTH_PATH.count(edge_id))
                    return LANE_ID_MAP_SMOOTH_PATH[edge_id];
                else
                {
                    std::string tmp_name = edge_name;
                    auto        edge_object = GetLaneOrConnectorAndNameById(edge_id, tmp_name);
                    auto        ref_line = MapUtilsObject.attr("GetRefLine")(edge_object, PATH_POINT_INTERVAL);

                    py::list xs = py::extract<py::list>(ref_line[0]);
                    py::list ys = py::extract<py::list>(ref_line[1]);
                    py::list thetas = py::extract<py::list>(ref_line[2]);
                    py::list kappas = py::extract<py::list>(ref_line[4]);
                    float    weighted_avg_r_squared = py::extract<float>(ref_line[7]);

                    int                   size = len(xs);
                    std::shared_ptr<Path> path = std::make_shared<Path>();
                    path->lane_id_ = edge_id;
                    path->lane_name_ = tmp_name;
                    path->is_straight_ = py::extract<bool>(ref_line[6]);
                    path->path_len_ = size * PATH_POINT_INTERVAL;
                    path->weighted_avg_r_squared_ = weighted_avg_r_squared;

                    path->resize(size);
                    for (int i = 0; i < size; ++i)
                    {
                        path->xs_[i] = static_cast<float>(py::extract<double>(xs[i]));
                        path->ys_[i] = static_cast<float>(py::extract<double>(ys[i]));
                        path->thetas_[i] = static_cast<float>(py::extract<double>(thetas[i]));
                        path->kappas_[i] = static_cast<float>(py::extract<double>(kappas[i]));
                    }

                    // path->speed_limit_ = utils::MAX_VEL;
                    py::object speed_limit_py = edge_object.attr("speed_limit_mps");
                    if (speed_limit_py.is_none())
                    {
                        path->speed_limit_ = utils::MAX_VEL;
                    }
                    else
                    {
                        path->speed_limit_ = py::extract<float>(speed_limit_py);
                    }

                    // Compute the maximum curvature for each segment
                    int segment_size = size / CURVATURE_SEGMENT_LENGTH_SIZE;
                    for (int i = 0; i < segment_size - 1; ++i)
                    {
                        float max_signed_curvature = 0.0f;
                        float max_curvature = 0.0f;
                        for (int j = i * CURVATURE_SEGMENT_LENGTH_SIZE; j < (i + 1) * CURVATURE_SEGMENT_LENGTH_SIZE;
                             ++j)
                        {
                            float abs_kappa = std::abs(path->kappas_[j]);
                            max_signed_curvature =
                                (abs_kappa > max_signed_curvature) ? path->kappas_[j] : max_signed_curvature;
                            max_curvature = std::max(max_curvature, abs_kappa);
                        }
                        path->max_signed_curvature_vec_.emplace_back(max_signed_curvature);
                        path->max_curvature_vec_.emplace_back(max_curvature);
                        path->min_desired_speed_vec_.emplace_back(
                            std::min(path->speed_limit_, utils::CalInterpolatedMaxSpeed(max_curvature)));
                        path->max_curvature_idx_vec_.emplace_back(i * CURVATURE_SEGMENT_LENGTH_SIZE +
                                                                  CURVATURE_SEGMENT_LENGTH_SIZE - 1);
                    }

                    // The remaining part forms a final segment
                    float max_signed_curvature = 0.0f;
                    float remaining_max_curvature = 0.0f;
                    for (int j = std::max(0, (segment_size - 1) * CURVATURE_SEGMENT_LENGTH_SIZE); j < size; ++j)
                    {
                        float abs_kappa = std::abs(path->kappas_[j]);
                        max_signed_curvature =
                            (abs_kappa > max_signed_curvature) ? path->kappas_[j] : max_signed_curvature;
                        remaining_max_curvature = std::max(remaining_max_curvature, std::abs(path->kappas_[j]));
                    }

                    path->max_signed_curvature_vec_.emplace_back(max_signed_curvature);
                    path->max_curvature_vec_.emplace_back(remaining_max_curvature);
                    path->min_desired_speed_vec_.emplace_back(
                        std::min(path->speed_limit_, utils::CalInterpolatedMaxSpeed(remaining_max_curvature)));
                    path->max_curvature_idx_vec_.emplace_back(size - 1);

                    path->miss_goal_penalty_ = ActualMissGoalNum(edge_id);

                    LANE_ID_MAP_SMOOTH_PATH[edge_id] = path;

                    return path;
                }
            }

            inline std::string GetRoadBlockIdById(const std::string &edge_id)
            {
                if (LANE_ID_MAP_ROADBLOCK_ID.count(edge_id))
                    return LANE_ID_MAP_ROADBLOCK_ID[edge_id];

                py::object edge_object = GetLaneOrConnectorById(edge_id);
                py::object road_block_id_py = edge_object.attr("get_roadblock_id");

                LANE_ID_MAP_ROADBLOCK_ID[edge_id] = py::extract<std::string>(road_block_id_py);

                return LANE_ID_MAP_ROADBLOCK_ID[edge_id];
            }

            inline std::vector<std::string> GetRoutePredecessorIdsById(std::string edge_id, std::string edge_name = "")
            {
                if (edge_id.empty())
                    return std::vector<std::string>();

                if (LANE_ID_MAP_ROUTE_PREDECESSORS.count(edge_id))
                    return LANE_ID_MAP_ROUTE_PREDECESSORS[edge_id];

                std::vector<std::string> predecessor_ids = GetPredecessorIdsById(edge_id, edge_name);

                std::vector<std::string> route_predecessor_ids;
                for (const auto &id : predecessor_ids)
                {
                    if (GLOBAL_ROUTE_EDGE_ID.count(id))
                        route_predecessor_ids.emplace_back(id);
                }

                // If all predecessors would be filtered out, return predecessor_ids directly
                if (route_predecessor_ids.size() == 0)
                {
                    route_predecessor_ids = predecessor_ids;
                }

                LANE_ID_MAP_ROUTE_PREDECESSORS[edge_id] = std::move(route_predecessor_ids);
                return LANE_ID_MAP_ROUTE_PREDECESSORS[edge_id];
            }

            inline std::vector<std::string> GetRouteSuccessorIdsById(std::string edge_id, std::string edge_name = "")
            {
                if (edge_id.empty())
                    return std::vector<std::string>();

                if (LANE_ID_MAP_ROUTE_SUCCESSORS.count(edge_id))
                    return LANE_ID_MAP_ROUTE_SUCCESSORS[edge_id];

                std::vector<std::string> successor_ids = GetSuccessorIdsById(edge_id, edge_name);

                // Sort successor_ids by ActualMissGoalNum, smaller values first
                std::sort(successor_ids.begin(), successor_ids.end(), [this](const std::string &a, const std::string &b)
                          { return ActualMissGoalNum(a) < ActualMissGoalNum(b); });

                // Filter out lanes not on the route and lanes that lead to a dead end
                std::vector<std::string> route_successor_ids;
                for (const auto &id : successor_ids)
                {
                    if (GLOBAL_ROUTE_EDGE_ID.count(id))
                    {
                        std::vector<std::string> next_successor_ids = GetSuccessorIdsById(id);
                        if (next_successor_ids.size() == 0)
                            continue;

                        std::string next_successor_id = next_successor_ids[0];
                        auto        path = GetRefLine(next_successor_id);
                        if (IsDeadEnd(next_successor_id) && path->GetPathLen() <= utils::DEAD_END_MIN_PATH_LENGTH)
                            continue;

                        route_successor_ids.emplace_back(id);
                    }
                }

                // If all successors are filtered out, return the original successor_ids
                if (route_successor_ids.size() == 0)
                {
                    route_successor_ids = successor_ids;
                }

                LANE_ID_MAP_ROUTE_SUCCESSORS[edge_id] = std::move(route_successor_ids);
                return LANE_ID_MAP_ROUTE_SUCCESSORS[edge_id];
            }

            inline float DistanceToEndOfEdge(float x, float y, const std::string &edge_id)
            {
                return Distance(x, y, GetRefLine(edge_id)->GetXs().back(), GetRefLine(edge_id)->GetYs().back()) -
                       EGO_BB_EXTENT_Y;
            }

            inline std::vector<std::string> QueryNearestLaneIds(float x, float y, const double &heading,
                                                                const double &radius = 2.0,
                                                                const double &heading_error = M_PI_4, int number = 1)
            {
                py::object point_py = Point2DPy(x, y);
                py::list   nearest_lanes_ids = py::extract<py::list>(
                    MapUtilsObject.attr("QueryNearestLaneIds")(point_py, heading, radius, heading_error, number));
                std::vector<std::string> nearest_lane_ids;
                for (int i = 0; i < len(nearest_lanes_ids); ++i)
                {
                    nearest_lane_ids.emplace_back(py::extract<std::string>(nearest_lanes_ids[i]));
                }

                return nearest_lane_ids;
            }

            inline void UpdateEgoStartingLaneId(float x, float y, float theta, std::string &edge_token,
                                                std::string &edge_name, std::vector<std::string> &candidate_edge_token)
            {
                py::object point_py = PointPy(x, y);
                py::list   starting_lanes_ids =
                    py::extract<py::list>(MapUtilsObject.attr("GetStartingLaneIds")(point_py));

                assert(len(starting_lanes_ids) > 0);

                for (int i = 0; i < len(starting_lanes_ids); ++i)
                {
                    candidate_edge_token.emplace_back(py::extract<std::string>(starting_lanes_ids[i]));
                }

                edge_token = candidate_edge_token[0];
                edge_name = GetEdgeTypeById(candidate_edge_token[0]);

                if (edge_name == "LANE")
                {
                    edge_token = candidate_edge_token[0];
                    std::shared_ptr<Path> path = GetRefLine(edge_token, "LANE");

                    // If the ego vehicle is on a lane and its heading differs significantly from the lane heading,
                    // re-query nearest lanes
                    size_t nearest_idx = path->NearestBatch(x, y, theta);
                    float  heading_difference = std::fabs(NormalizeAngle(theta - path->GetTheta(nearest_idx)));
                    if (heading_difference > M_PI_2)
                    {
                        std::vector<std::string> nearest_lane_ids =
                            QueryNearestLaneIds(x, y, theta, utils::NEAREST_LANE_QUERY_RADIUS);
                        if (nearest_lane_ids.size() > 0)
                        {
                            edge_token = nearest_lane_ids[0];
                        }
                    }
                }
            }

            inline void UpdateEgoEdgeDirectly(const std::string &ego_edge_id, const std::string &ego_edge_name)
            {
                MapUtilsObject.attr("UpdateEgoEdgeDirectly")(ego_edge_id, ego_edge_name);
            }

            void UpdateEgoEdge(float x, float y, std::string &prev_edge_id, std::string &prev_edge_name,
                               const std::vector<std::string> &target_path_edge_tokens,
                               std::vector<std::string>       &candidate_edge_tokens)
            {
                if (prev_edge_id.empty() || IsPointInLaneById(prev_edge_id, x, y, prev_edge_name))
                {
                    return;
                }

                std::vector<std::string> edge_ids;
                std::string              edge_name;
                GetEdgeIdByPosition(x, y, edge_ids, edge_name);
                if (edge_ids.size() == 0)
                {
                    return;
                }

                const auto &route_successor_ids = GetRouteSuccessorIdsById(prev_edge_id, prev_edge_name);
                if (edge_name == "LANE")
                {
                    // If the updated edge is on the route or is a left/right neighbor of the route, update ego_edge_id
                    // and ego_edge_name
                    if (route_successor_ids.size() == 0 || route_successor_ids[0] == edge_ids[0] ||
                        GetLeftNeighborIdById(prev_edge_id) == edge_ids[0] ||
                        GetRightNeighborIdById(prev_edge_id) == edge_ids[0])
                    {
                        prev_edge_id = edge_ids[0];
                        prev_edge_name = edge_name;
                    }
                }
                else
                {
                    // If currently on a lane connector, ensure the updated position corresponds to a route lane
                    // connector
                    if (prev_edge_name == "LANE")
                    {
                        for (const auto &edge_id : edge_ids)
                        {
                            if (std::find(route_successor_ids.begin(), route_successor_ids.end(), edge_id) !=
                                route_successor_ids.end())
                            {
                                prev_edge_id = edge_id;
                                prev_edge_name = edge_name;
                                candidate_edge_tokens.emplace_back(edge_id);
                            }
                        }

                        // If the vehicle moved directly from a Lane to a successor in a neighboring lane
                        if (candidate_edge_tokens.empty())
                        {
                            for (const auto &edge_id : edge_ids)
                            {
                                if (std::find(target_path_edge_tokens.begin(), target_path_edge_tokens.end(),
                                              edge_id) != target_path_edge_tokens.end())
                                {
                                    prev_edge_id = edge_id;
                                    prev_edge_name = edge_name;
                                    candidate_edge_tokens.emplace_back(edge_id);
                                    break;
                                }
                            }
                        }

                        // If still empty, set it to the first found edge
                        if (candidate_edge_tokens.empty())
                        {
                            prev_edge_id = edge_ids[0];
                            prev_edge_name = edge_name;
                            candidate_edge_tokens.emplace_back(edge_ids[0]);
                        }
                    }
                    else
                    {
                        for (const auto &edge_id : edge_ids)
                        {
                            if (GLOBAL_ROUTE_EDGE_ID.count(edge_id))
                            {
                                prev_edge_id = edge_id;
                                prev_edge_name = edge_name;
                                candidate_edge_tokens.emplace_back(edge_id);
                            }
                        }
                    }
                }
            }

            inline void UpdateEgoCurrRefPathIdx(const std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                                const std::string                        &curr_edge_id,
                                                const std::vector<std::string>           &candidate_edge_tokens,
                                                int                                      &curr_ref_path_idx)
            {
                // first check if the curr_ref_path_idx still holds
                for (int i = 0; i < candidate_edge_tokens.size(); ++i)
                {
                    if (std::find(ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_.begin(),
                                  ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_.end(),
                                  candidate_edge_tokens[i]) !=
                        ego_ref_paths[curr_ref_path_idx]->comprised_ref_path_ids_.end())
                    {
                        return;
                    }
                }

                for (int i = 0; i < ego_ref_paths.size(); ++i)
                {
                    if (ego_ref_paths[i] == nullptr)
                    {
                        continue;
                    }

                    for (int j = 0; j < ego_ref_paths[i]->comprised_ref_path_ids_.size(); ++j)
                    {
                        if (ego_ref_paths[i]->comprised_ref_path_ids_[j] == curr_edge_id)
                        {
                            curr_ref_path_idx = i;
                            return;
                        }
                    }
                }
            }

            /**
             * Create an extended Path by extending the specified edge forward and backward.
             * @param edge_id ID of the current edge
             * @param successor_id specified successor ID
             * @param lookback_points_size number of points to extend backward
             * @param lookahead_points_size number of points to extend forward
             * @return shared pointer to the extended Path
             */
            std::shared_ptr<Path> ExtendPath(const std::string &edge_id, const std::string &successor_id,
                                             std::vector<std::string> &refline_edge_ids,
                                             std::vector<std::string> &refline_edge_names, int lookback_points_size,
                                             int lookahead_points_size, bool for_lane);

            /**
             * Prepare reference paths for the ego vehicle
             */
            void GetEgoRefPaths(float x, float y, float v, int &curr_ref_path_idx,
                                std::vector<std::shared_ptr<Path>> &ref_paths,
                                std::vector<std::shared_ptr<Path>> &extra_ref_paths,
                                std::vector<std::string>           &refline_edge_ids,
                                std::vector<std::string> &refline_edge_names, const std::string &last_edge_id,
                                const std::string &curr_edge_id, const std::string &edge_name,
                                const std::vector<std::string> &candidate_edge_token, bool &is_close_to_junction);

            /**
             * Update ego reference paths. Specifically, when the vehicle is on a lane connector,
             * update left/right neighbor reference paths as needed.
             */
            void UpdateEgoRefPaths(std::vector<std::shared_ptr<Path>> &ego_ref_paths,
                                   std::vector<std::shared_ptr<Path>> &ego_extra_ref_paths, int &curr_ref_path_idx,
                                   std::vector<std::string> &refline_edge_ids,
                                   std::vector<std::string> &refline_edge_names, const std::string &curr_edge_id,
                                   const std::string &edge_name);

            void UpdateEgoRefPathsTrafficInfo(const std::vector<std::shared_ptr<Path>> &ego_ref_paths);

          public:
            std::unordered_map<std::string, std::vector<std::string>> LANE_ID_MAP_SUCCESSORS;
            std::unordered_map<std::string, std::vector<std::string>> LANE_ID_MAP_ROUTE_SUCCESSORS;
            std::unordered_map<std::string, std::vector<std::string>> LANE_ID_MAP_PREDECESSORS;
            std::unordered_map<std::string, std::vector<std::string>> LANE_ID_MAP_ROUTE_PREDECESSORS;
            std::unordered_map<std::string, std::string>              LANE_ID_MAP_LEFT_NEIGHBOR;
            std::unordered_map<std::string, std::string>              LANE_ID_MAP_RIGHT_NEIGHBOR;
            std::unordered_map<std::string, std::shared_ptr<Path>>    LANE_ID_MAP_SMOOTH_PATH;
            std::unordered_map<std::string, float>                    LANE_ID_MAP_MISSED_GOAL_NUM;
            std::unordered_map<std::string, float>                    LANE_ID_MAP_ACTUAL_MISSED_GOAL_NUM;
            std::set<std::string>                                     LANE_ID_MAP_IS_TERMINATED;
            std::unordered_map<std::string, std::string>              LANE_ID_MAP_EDGE_TYPE;
            std::unordered_map<std::string, std::string>              LANE_ID_MAP_ROADBLOCK_ID;
            std::unordered_map<std::string, bool>                     LANE_CONNECTOR_ID_MAP_RED_TRAFFIC_LIGHT;
            std::unordered_map<std::string, bool>                     GLOBAL_ROUTE_EDGE_ID;
            std::unordered_map<std::pair<std::string, std::string>, bool, utils::pair_hash>
                LANE_ID_MAP_HAS_PARENT_RELATION;

            py::object MapUtilsObject;
            py::object map_utils_py;
            py::object SemanticMapLayerPy;
            py::object Point2DPy;
            py::object PointPy;
        };
    } // namespace utils
} // namespace vec_qmdp