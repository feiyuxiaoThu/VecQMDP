/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file occupancy_map.hpp
 * @brief Drivable-area polygon containment checks via GEOS PreparedPolygon.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utils/map_utils.hpp>
#include <vector>

#include <geos/geom/Coordinate.h>
#include <geos/geom/CoordinateSequence.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/LinearRing.h>
#include <geos/geom/Polygon.h>
#include <geos/geom/prep/PreparedPolygon.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/object.hpp>

#include <vamp/vector.hh>

namespace vec_qmdp
{
    namespace utils
    {
        namespace py = boost::python;

        enum class PolygonCategory
        {
            NON_DRIVABLE_AREA = 0,
            ROUTE_EDGES = 1,
            NOT_ON_ROUTE_ROADBLOCKS_AND_CARPARKS = 2,
            INTERSECTIONS = 3
        };

        class OccupancyMap
        {
          public:
            OccupancyMap(const std::shared_ptr<MapUtils> &map_utils_ptr);
            ~OccupancyMap();

            /**
             * Initialize the python module and objects
             */
            void Initialize();

            /**
             * Update map utils object reference
             */
            void UpdateMapUtilsObject(const py::object &map_utils_obj);

            /**
             * Update drivable map objects by calling python GetDrivableMapObjects
             * @param ego_x Current ego vehicle x position
             * @param ego_y Current ego vehicle y position
             * @param ego_heading Current ego vehicle heading
             * @param new_ego_edge_id Current ego edge id
             * @param new_ego_edge_name Current ego edge name ("LANE" or "LANE_CONNECTOR")
             * @param map_radius Radius for map object query
             */
            void UpdateDrivableMapObjects(float ego_x, float ego_y, float ego_v, float ego_heading,
                                          const std::string &new_ego_edge_id, const std::string &new_ego_edge_name,
                                          const std::vector<std::string> &refline_edge_ids,
                                          const std::vector<std::string> &refline_edge_names,
                                          float map_radius = utils::OCCUPANCY_MAP_RADIUS_OFFSET);

            void ContainsPointsInDrivableAreaBatch(
                const FVectorT_traj &center_xs, const FVectorT_traj &center_ys, const FVectorT_traj &rear_axle_xs,
                const FVectorT_traj &rear_axle_ys, const FVectorT_traj *corners_xs, const FVectorT_traj *corners_ys,
                IVectorT_traj &on_non_drivable_area_mask, IVectorT_traj &left_corners_on_non_drivable_area_mask,
                IVectorT_traj &right_corners_on_non_drivable_area_mask, IVectorT_traj &on_coming_traffic_mask,
                IVectorT_traj &on_intersection_mask, IVectorT_traj &on_multiple_lane_mask,
                IVectorT_traj &on_different_path_lanes_mask, IVectorT_traj &on_non_route_mask) const;

            /**
             * Check if a point is contained within any of the route edge polygons using SIMD
             * @param point_x x coordinate
             * @param point_y y coordinate
             * @return true if point is contained in any route edge polygon
             */
            bool ContainsPointInRouteEdges(float point_x, float point_y) const;

            /**
             * Check if a point is contained within any roadblock/carpark polygons using SIMD
             * @param point_x x coordinate
             * @param point_y y coordinate
             * @return true if point is contained in any roadblock/carpark polygon
             */
            bool ContainsPointInRoadblocksAndCarparks(float point_x, float point_y) const;

            /**
             * Check if a point is contained within any intersection polygon using SIMD
             * @param point_x x coordinate
             * @param point_y y coordinate
             * @return true if point is contained in any intersection polygon
             */
            bool ContainsPointInIntersections(float point_x, float point_y) const;

            /**
             * Batch check if points are contained within route edge polygons
             * @param point_x Array of x coordinates
             * @param point_y Array of y coordinates
             * @param num_points Number of points to check
             * @param results Output array of results
             */
            void ContainsPointsInRouteEdges(const float *point_x, const float *point_y, int num_points,
                                            bool *results) const;

            /**
             * Clear all polygon maps
             */
            void Clear();

            /**
             * Get number of polygons in each category
             */
            std::tuple<size_t, size_t, size_t> GetPolygonCounts() const;

            /**
             * Get access to route edges polygons for coordinate extraction
             */
            const std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &
            GetRouteEdges() const
            {
                return route_edges_;
            }

            /**
             * Get access to roadblocks and carparks polygons for coordinate extraction
             */
            const std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &
            GetRoadblocksAndCarparks() const
            {
                return not_on_route_roadblocks_and_carparks_;
            }

            /**
             * Get access to intersections polygons for coordinate extraction
             */
            const std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &
            GetIntersections() const
            {
                return intersections_;
            }

          private:
            /**
             * Create PreparedPolygon from coordinate list
             * @param coords List of coordinate pairs [(x1,y1), (x2,y2), ...]
             * @return Shared pointer to created PreparedPolygon
             */
            std::shared_ptr<const geos::geom::prep::PreparedPolygon>
            CreatePreparedPolygon(const std::vector<std::vector<float>> &coords);

            /**
             * Process inserted tokens and coordinates to create new polygons
             * @param inserted_tokens List of token ids to insert
             * @param inserted_coords List of coordinate arrays for each token
             * @param polygon_map Target map to insert polygons
             */
            void ProcessInsertedPolygons(
                const std::vector<std::string>                     &inserted_tokens,
                const std::vector<std::vector<std::vector<float>>> &inserted_coords,
                std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &polygon_map);

            /**
             * Process deleted tokens to remove polygons
             * @param deleted_tokens List of token ids to delete
             * @param polygon_map Target map to remove polygons from
             */
            void ProcessDeletedPolygons(
                const std::vector<std::string> &deleted_tokens,
                std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &polygon_map);

            /**
             * Get polygon map by category index
             * @param category Category index (0: route_edges, 1: roadblocks_and_carparks, 2: intersections)
             * @return Pointer to the corresponding polygon map
             */
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> *
            GetPolygonMapByCategory(int category);

          private:
            // shared map utility
            std::shared_ptr<MapUtils> map_utils_ptr_;

            // Three polygon maps for different object types
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> route_edges_;
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>>
                not_on_route_roadblocks_and_carparks_;
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> intersections_;

            // GEOS geometry factory
            geos::geom::GeometryFactory::Ptr geom_factory_;

            // Python objects for boost binding
            py::object map_utils_object_;
            py::object map_utils_py_;
            py::object Point2DPy_;

            // Initialization flag
            bool initialized_;
        };

    } // namespace utils
} // namespace vec_qmdp