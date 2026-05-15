/*
 * Copyright (c) 2026 VecQMDP Contributors.
 * All rights reserved.
 */

#include <utils/global_utils.hpp>
#include <utils/occupancy_map.hpp>

#include <geos/geom/prep/PreparedGeometryFactory.h>
#include <geos/operation/buffer/BufferOp.h>

#include <cassert>
#include <iostream>

namespace vec_qmdp
{
    namespace utils
    {
        OccupancyMap::OccupancyMap(const std::shared_ptr<utils::MapUtils> &map_utils_ptr)
            : initialized_(false), map_utils_ptr_(map_utils_ptr)
        {
            // Initialize GEOS geometry factory
            geom_factory_ = geos::geom::GeometryFactory::create();
            Initialize();
        }

        OccupancyMap::~OccupancyMap() { Clear(); }

        void OccupancyMap::Initialize()
        {
            if (initialized_)
                return;

            try
            {
                // Import the map utils python module
                map_utils_py_ = py::import("python_planner.utils.map_utils_py");
                Point2DPy_ = map_utils_py_.attr("Point2D");

                initialized_ = true;
                std::cout << "OccupancyMap: Python module initialized successfully" << std::endl;
            }
            catch (const py::error_already_set &e)
            {
                PyErr_Print(); // Print Python exception
                std::cerr << "OccupancyMap: Failed to initialize Python module" << std::endl;
                initialized_ = false;
            }
        }

        void OccupancyMap::UpdateMapUtilsObject(const py::object &map_utils_obj) { map_utils_object_ = map_utils_obj; }

        void OccupancyMap::UpdateDrivableMapObjects(float ego_x, float ego_y, float ego_v, float ego_heading,
                                                    const std::string              &new_ego_edge_id,
                                                    const std::string              &new_ego_edge_name,
                                                    const std::vector<std::string> &refline_edge_ids,
                                                    const std::vector<std::string> &refline_edge_names,
                                                    float                           map_radius)
        {
            if (!initialized_)
            {
                std::cerr << "OccupancyMap: Not initialized, cannot update drivable map objects" << std::endl;
                return;
            }

            if (map_utils_object_.is_none())
            {
                std::cerr << "OccupancyMap: Map utils object not set" << std::endl;
                return;
            }

            try
            {
                // std::cout << "[OccupancyMap] Calling Python GetDrivableMapObjects (ASan disabled for this call)" << std::endl;
                // std::cout.flush();
                
                // Call Python GetDrivableMapObjects function
                // Note: This calls into Python, which then calls back into nuplan C++ code
                // In ASan mode, this cross-language call chain is extremely slow or hangs
                // So we disable ASan checking for this specific function
                py::object result = map_utils_object_.attr("GetDrivableMapObjects")(
                    ego_x, ego_y, ego_v, ego_heading, new_ego_edge_id, new_ego_edge_name, refline_edge_ids,
                    refline_edge_names, map_radius);

                std::cout << "[OccupancyMap] Python call completed successfully" << std::endl;
                std::cout.flush();

                // Extract the three returned lists
                py::tuple result_tuple = py::extract<py::tuple>(result);
                py::list  inserted_tokens_py = py::extract<py::list>(result_tuple[0]);
                py::list  inserted_coords_py = py::extract<py::list>(result_tuple[1]);
                py::list  deleted_tokens_py = py::extract<py::list>(result_tuple[2]);

                // Process each category: route_edges (0), roadblocks_and_carparks (1), intersections (2)
                for (int category = 0; category < 3; ++category)
                {
                    // Extract inserted tokens and coordinates for this category
                    py::list category_inserted_tokens = py::extract<py::list>(inserted_tokens_py[category]);
                    py::list category_inserted_coords = py::extract<py::list>(inserted_coords_py[category]);
                    py::list category_deleted_tokens = py::extract<py::list>(deleted_tokens_py[category]);

                    // Convert to C++ containers
                    std::vector<std::string>                     inserted_tokens;
                    std::vector<std::vector<std::vector<float>>> inserted_coords;
                    std::vector<std::string>                     deleted_tokens;

                    // Process inserted tokens
                    for (int i = 0; i < len(category_inserted_tokens); ++i)
                    {
                        std::string token = py::extract<std::string>(category_inserted_tokens[i]);
                        inserted_tokens.push_back(token);

                        // Extract coordinates for this token
                        py::list token_coords = py::extract<py::list>(category_inserted_coords[i]);
                        std::vector<std::vector<float>> coords;

                        for (int j = 0; j < len(token_coords); ++j)
                        {
                            py::list           coord_pair = py::extract<py::list>(token_coords[j]);
                            std::vector<float> coord = {py::extract<float>(coord_pair[0]),
                                                        py::extract<float>(coord_pair[1])};
                            coords.push_back(coord);
                        }
                        inserted_coords.push_back(coords);
                    }

                    // Process deleted tokens
                    for (int i = 0; i < len(category_deleted_tokens); ++i)
                    {
                        std::string token = py::extract<std::string>(category_deleted_tokens[i]);
                        deleted_tokens.push_back(token);
                    }

                    // Update the appropriate polygon map
                    auto *target_map = GetPolygonMapByCategory(category);
                    if (target_map)
                    {
                        ProcessDeletedPolygons(deleted_tokens, *target_map);
                        ProcessInsertedPolygons(inserted_tokens, inserted_coords, *target_map);
                    }
                }
            }
            catch (const py::error_already_set &e)
            {
                PyErr_Print(); // Print Python exception
                std::cerr << "OccupancyMap: Error calling GetDrivableMapObjects" << std::endl;
            }
        }

        bool has_common_element(const std::vector<std::string> &v0, const std::vector<std::string> &v1,
                                const std::vector<std::string> &v2, const std::vector<std::string> &v3,
                                std::string &out1, std::string &out2)
        {
            bool found_two_different_elements = false;
            for (const auto &a : v0)
            {
                for (const auto &b : v1)
                {
                    if (a != b)
                    {
                        if (!found_two_different_elements)
                        {
                            out1 = a;
                            out2 = b;
                            found_two_different_elements = true;
                        }
                        continue;
                    }
                    for (const auto &c : v2)
                    {
                        if (a != c)
                        {
                            if (!found_two_different_elements)
                            {
                                out1 = a;
                                out2 = c;
                                found_two_different_elements = true;
                            }
                            continue;
                        }
                        for (const auto &d : v3)
                        {
                            if (a == d)
                            {
                                return true; // Found common element
                            }
                            else if (!found_two_different_elements)
                            {
                                out1 = a;
                                out2 = d;
                                found_two_different_elements = true;
                            }
                        }
                    }
                }
            }
            return false; // No common elements
        }

        void OccupancyMap::ContainsPointsInDrivableAreaBatch(
            const FVectorT_traj &center_xs, const FVectorT_traj &center_ys, const FVectorT_traj &rear_axle_xs,
            const FVectorT_traj &rear_axle_ys, const FVectorT_traj *corners_xs, const FVectorT_traj *corners_ys,
            IVectorT_traj &on_non_drivable_area_mask, IVectorT_traj &left_corners_on_non_drivable_area_mask,
            IVectorT_traj &right_corners_on_non_drivable_area_mask, IVectorT_traj &on_coming_traffic_mask,
            IVectorT_traj &on_intersection_mask, IVectorT_traj &on_multiple_lane_mask,
            IVectorT_traj &on_different_path_lanes_mask, IVectorT_traj &on_non_route_mask) const
        {
            constexpr int corner_category_num = 6;

            static thread_local utils::OccupancyMapWorkspace ws;
            ws.reset();

            auto &corners_area_ids = ws.corners_area_ids;
            auto &corners_area_on_drivable_area_mask = ws.corners_area_on_drivable_area_mask;
            auto &on_non_drivable_area_mask_v = ws.on_non_drivable_area_mask_v;
            auto &left_corners_on_non_drivable_area_mask_v = ws.left_corners_on_non_drivable_area_mask_v;
            auto &right_corners_on_non_drivable_area_mask_v = ws.right_corners_on_non_drivable_area_mask_v;
            auto &on_coming_traffic_mask_v = ws.on_coming_traffic_mask_v;
            auto &on_intersection_mask_v = ws.on_intersection_mask_v;
            auto &on_multiple_lane_mask_v = ws.on_multiple_lane_mask_v;
            auto &on_different_path_lanes_mask_v = ws.on_different_path_lanes_mask_v;
            auto &on_non_route_mask_v = ws.on_non_route_mask_v;

            auto &min_ys = ws.min_ys;
            auto &max_ys = ws.max_ys;
            auto &different_lateral_offset_min_ys = ws.different_lateral_offset_min_ys;
            auto &different_lateral_offset_max_ys = ws.different_lateral_offset_max_ys;
            auto &corners_xs_arr = ws.corners_xs_arr;
            auto &corners_ys_arr = ws.corners_ys_arr;
            auto &center_xs_arr = ws.center_xs_arr;
            auto &center_ys_arr = ws.center_ys_arr;
            auto &rear_axle_xs_arr = ws.rear_axle_xs_arr;
            auto &rear_axle_ys_arr = ws.rear_axle_ys_arr;

            const auto &VECTOR_SIZE = ws.VECTOR_SIZE;
            auto       &buffer_index = ws.buffer_index;
            auto       &vec_index = ws.vec_index;
            auto       &simd_segments_x1_buffer = ws.simd_segments_x1_buffer;
            auto       &simd_segments_y1_buffer = ws.simd_segments_y1_buffer;
            auto       &simd_segments_x2_buffer = ws.simd_segments_x2_buffer;
            auto       &simd_segments_y2_buffer = ws.simd_segments_y2_buffer;
            auto       &simd_valid_mask_buffer = ws.simd_valid_mask_buffer;
            auto       &simd_segments_x1_vec = ws.simd_segments_x1_vec;
            auto       &simd_segments_y1_vec = ws.simd_segments_y1_vec;
            auto       &simd_segments_x2_vec = ws.simd_segments_x2_vec;
            auto       &simd_segments_y2_vec = ws.simd_segments_y2_vec;
            auto       &simd_valid_mask_vec = ws.simd_valid_mask_vec;

            center_xs_arr = center_xs.to_array();
            center_ys_arr = center_ys.to_array();
            rear_axle_xs_arr = rear_axle_xs.to_array();
            rear_axle_ys_arr = rear_axle_ys.to_array();

            float batch_min_x = corners_xs[0].hmin();
            float batch_max_x = corners_xs[0].hmax();

            for (int i = 1; i < corner_category_num; ++i)
            {
                batch_min_x = std::min(batch_min_x, corners_xs[i].hmin());
                batch_max_x = std::max(batch_max_x, corners_xs[i].hmax());
            }

            float batch_min_y = corners_ys[0].hmin();
            float batch_max_y = corners_ys[0].hmax();

            for (int i = 0; i < corner_category_num; ++i)
            {
                min_ys[i] = corners_ys[i].hmin();
                max_ys[i] = corners_ys[i].hmax();

                if (i > 0)
                {
                    batch_min_y = std::min(batch_min_y, min_ys[i]);
                    batch_max_y = std::max(batch_max_y, max_ys[i]);
                }

                if (std::fabs(max_ys[i] - min_ys[i]) >= utils::DIFFERENCE_YAXIS_OFFSET_THRESHOLD)
                {
                    for (int j = 0; j < FVectorT_traj::num_rows; ++j)
                    {
                        different_lateral_offset_min_ys[i][j] = corners_ys[i].row(j).hmin();
                        different_lateral_offset_max_ys[i][j] = corners_ys[i].row(j).hmax();
                    }
                }
                corners_xs_arr[i] = corners_xs[i].to_array();
                corners_ys_arr[i] = corners_ys[i].to_array();
            }

            int count = 0;
            // on route drivable area (lanes and lane connectors)
            for (const auto &[polygon_id, polygon] : route_edges_)
            {
                const auto *env = polygon->getGeometry().getEnvelopeInternal();
                if (batch_max_x < env->getMinX() || batch_min_x > env->getMaxX() || batch_max_y < env->getMinY() ||
                    batch_min_y > env->getMaxY())
                {
                    continue;
                }

                bool print_debug = false;
                polygon->containsPointsInRouteEdgesBatchOptimized(
                    center_xs, center_ys, corners_xs, corners_ys, center_xs_arr, center_ys_arr, corners_xs_arr,
                    corners_ys_arr, min_ys, max_ys, different_lateral_offset_min_ys, different_lateral_offset_max_ys,
                    corner_category_num, polygon_id, corners_area_ids, corners_area_on_drivable_area_mask,
                    on_coming_traffic_mask_v, VECTOR_SIZE, buffer_index, vec_index, simd_segments_x1_buffer,
                    simd_segments_y1_buffer, simd_segments_x2_buffer, simd_segments_y2_buffer, simd_valid_mask_buffer,
                    simd_segments_x1_vec, simd_segments_y1_vec, simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec, print_debug);
            }

            count = 0;
            // non-on-route drivable area (roadblocks and carparks)
            for (const auto &[polygon_id, polygon] : not_on_route_roadblocks_and_carparks_)
            {
                const auto *env = polygon->getGeometry().getEnvelopeInternal();
                if (batch_max_x < env->getMinX() || batch_min_x > env->getMaxX() || batch_max_y < env->getMinY() ||
                    batch_min_y > env->getMaxY())
                {
                    continue;
                }

                polygon->containsPointsInNotOnRouteBatchOptimized(
                    corners_xs, corners_ys, corners_xs_arr, corners_ys_arr, min_ys, max_ys,
                    different_lateral_offset_min_ys, different_lateral_offset_max_ys, corner_category_num,
                    corners_area_on_drivable_area_mask, VECTOR_SIZE, buffer_index, vec_index, simd_segments_x1_buffer,
                    simd_segments_y1_buffer, simd_segments_x2_buffer, simd_segments_y2_buffer, simd_valid_mask_buffer,
                    simd_segments_x1_vec, simd_segments_y1_vec, simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec);
            }

            count = 0;
            // intersections
            for (const auto &[polygon_id, polygon] : intersections_)
            {
                const auto *env = polygon->getGeometry().getEnvelopeInternal();
                if (batch_max_x < env->getMinX() || batch_min_x > env->getMaxX() || batch_max_y < env->getMinY() ||
                    batch_min_y > env->getMaxY())
                {
                    continue;
                }

                polygon->containsPointsInIntersectionBatchOptimized(
                    rear_axle_xs, rear_axle_ys, corners_xs, corners_ys, rear_axle_xs_arr, rear_axle_ys_arr,
                    corners_xs_arr, corners_ys_arr, min_ys, max_ys, different_lateral_offset_min_ys,
                    different_lateral_offset_max_ys, corner_category_num, corners_area_on_drivable_area_mask,
                    on_intersection_mask_v, VECTOR_SIZE, buffer_index, vec_index, simd_segments_x1_buffer,
                    simd_segments_y1_buffer, simd_segments_x2_buffer, simd_segments_y2_buffer, simd_valid_mask_buffer,
                    simd_segments_x1_vec, simd_segments_y1_vec, simd_segments_x2_vec, simd_segments_y2_vec,
                    simd_valid_mask_vec);
            }

            for (int i = 0; i < FVectorT_traj::num_scalars; ++i)
            {

                if (corners_area_on_drivable_area_mask[0][i] && corners_area_on_drivable_area_mask[2][i] &&
                    corners_area_on_drivable_area_mask[4][i])
                {
                    left_corners_on_non_drivable_area_mask_v[i] = 0;
                }

                if (corners_area_on_drivable_area_mask[1][i] && corners_area_on_drivable_area_mask[3][i] &&
                    corners_area_on_drivable_area_mask[5][i])
                {
                    right_corners_on_non_drivable_area_mask_v[i] = 0;
                }

                // if either left or right corners are on non-drivable area, the car is on non-drivable area
                if (left_corners_on_non_drivable_area_mask_v[i] || right_corners_on_non_drivable_area_mask_v[i])
                {
                    on_non_drivable_area_mask_v[i] = 0xFFFFFFFF;
                    on_non_route_mask_v[i] = 0xFFFFFFFF;
                    continue;
                }

                if (corners_area_ids[0][i].empty() || corners_area_ids[1][i].empty() ||
                    corners_area_ids[2][i].empty() || corners_area_ids[3][i].empty())
                {
                    on_non_route_mask_v[i] = 0xFFFFFFFF;
                    on_multiple_lane_mask_v[i] = 0xFFFFFFFF;
                    on_different_path_lanes_mask_v[i] = 0xFFFFFFFF;
                }
                // if all corners are not on the same lane, then the car is on multiple lanes
                else
                {
                    std::string common1, common2;

                    bool has_comment =
                        has_common_element(corners_area_ids[0][i], corners_area_ids[1][i], corners_area_ids[2][i],
                                           corners_area_ids[3][i], common1, common2);

                    if (!has_comment)
                    {
                        on_multiple_lane_mask_v[i] = 0xFFFFFFFF;
                        if (!map_utils_ptr_->HasParentRelation(common1, common2))
                        {
                            on_different_path_lanes_mask_v[i] = 0xFFFFFFFF;
                        }
                    }
                }
            }

            on_non_drivable_area_mask = IVectorT_traj(on_non_drivable_area_mask_v.data());
            left_corners_on_non_drivable_area_mask = IVectorT_traj(left_corners_on_non_drivable_area_mask_v.data());
            right_corners_on_non_drivable_area_mask = IVectorT_traj(right_corners_on_non_drivable_area_mask_v.data());
            on_coming_traffic_mask = IVectorT_traj(on_coming_traffic_mask_v.data());
            on_intersection_mask = IVectorT_traj(on_intersection_mask_v.data());
            on_multiple_lane_mask = IVectorT_traj(on_multiple_lane_mask_v.data());
            on_different_path_lanes_mask = IVectorT_traj(on_different_path_lanes_mask_v.data());
            on_non_route_mask = IVectorT_traj(on_non_route_mask_v.data());
        }

        bool OccupancyMap::ContainsPointInRouteEdges(float point_x, float point_y) const
        {
            auto point = geom_factory_->createPoint(geos::geom::Coordinate(point_x, point_y));
            for (const auto &[id, polygon] : route_edges_)
            {
                if (polygon->contains(point.get()))
                {
                    return true;
                }
            }
            return false;
        }

        bool OccupancyMap::ContainsPointInRoadblocksAndCarparks(float point_x, float point_y) const
        {
            auto point = geom_factory_->createPoint(geos::geom::Coordinate(point_x, point_y));
            for (const auto &[id, polygon] : not_on_route_roadblocks_and_carparks_)
            {
                if (polygon->contains(point.get()))
                {
                    return true;
                }
            }
            return false;
        }

        bool OccupancyMap::ContainsPointInIntersections(float point_x, float point_y) const
        {
            auto point = geom_factory_->createPoint(geos::geom::Coordinate(point_x, point_y));
            for (const auto &[id, polygon] : intersections_)
            {
                if (polygon->contains(point.get()))
                {
                    return true;
                }
            }
            return false;
        }

        void OccupancyMap::ContainsPointsInRouteEdges(const float *point_x, const float *point_y, int num_points,
                                                      bool *results) const
        {
            // Initialize all results to false
            for (int i = 0; i < num_points; ++i)
            {
                results[i] = false;
            }

            // Check each polygon against all points
            for (const auto &[id, polygon] : route_edges_)
            {
                for (int i = 0; i < num_points; ++i)
                {
                    if (!results[i])
                    {
                        auto point = geom_factory_->createPoint(geos::geom::Coordinate(point_x[i], point_y[i]));
                        if (polygon->contains(point.get()))
                        {
                            results[i] = true;
                        }
                    }
                }
            }
        }

        void OccupancyMap::Clear()
        {
            route_edges_.clear();
            not_on_route_roadblocks_and_carparks_.clear();
            intersections_.clear();
        }

        std::tuple<size_t, size_t, size_t> OccupancyMap::GetPolygonCounts() const
        {
            return std::make_tuple(route_edges_.size(), not_on_route_roadblocks_and_carparks_.size(),
                                   intersections_.size());
        }

        std::unique_ptr<geos::geom::CoordinateSequence>
        createCoordinateSequence(const std::vector<std::vector<float>> &coords)
        {
            std::unique_ptr<geos::geom::CoordinateSequence> coord_seq =
                std::make_unique<geos::geom::CoordinateSequence>();

            for (const auto &p : coords)
            {
                coord_seq->add(p[0], p[1]);
            }

            return coord_seq;
        }

        std::shared_ptr<const geos::geom::prep::PreparedPolygon>
        OccupancyMap::CreatePreparedPolygon(const std::vector<std::vector<float>> &coords)
        {
            try
            {
                if (coords.size() < 3)
                {
                    std::cerr << "OccupancyMap: Invalid polygon coordinates (less than 3 points)" << std::endl;
                    return nullptr;
                }

                // Create coordinate sequence
                auto coord_seq = createCoordinateSequence(coords);

                // Ensure the polygon is closed
                if (!coord_seq->getAt(0).equals2D(coord_seq->getAt(coord_seq->size() - 1)))
                {
                    coord_seq->add(coord_seq->getAt(0));
                }

                // Create linear ring
                auto linear_ring = geom_factory_->createLinearRing(std::move(coord_seq));

                // Create polygon
                auto polygon = geom_factory_->createPolygon(std::move(linear_ring));

                // Create prepared polygon for fast containment queries using factory
                auto prepared_geom = geos::geom::prep::PreparedGeometryFactory::prepare(polygon.get());

                // Cast to PreparedPolygon
                auto prepared_polygon = dynamic_cast<const geos::geom::prep::PreparedPolygon *>(prepared_geom.get());
                if (prepared_polygon)
                {
                    // Keep the geometry alive by storing it with the prepared geometry
                    return std::shared_ptr<const geos::geom::prep::PreparedPolygon>(
                        prepared_polygon,
                        [polygon = std::move(polygon),
                         prepared_geom = std::move(prepared_geom)](const geos::geom::prep::PreparedPolygon *)
                        {
                            // Custom deleter that keeps both polygon and prepared_geom alive
                        });
                }

                return nullptr;
            }
            catch (const std::exception &e)
            {
                std::cerr << "OccupancyMap: Error creating prepared polygon: " << e.what() << std::endl;
                return nullptr;
            }
        }

        void OccupancyMap::ProcessInsertedPolygons(
            const std::vector<std::string>                                                            &inserted_tokens,
            const std::vector<std::vector<std::vector<float>>>                                        &inserted_coords,
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &polygon_map)
        {
            assert(inserted_tokens.size() == inserted_coords.size());

            for (size_t i = 0; i < inserted_tokens.size(); ++i)
            {
                const std::string &token = inserted_tokens[i];
                const auto        &coords = inserted_coords[i];

                auto prepared_polygon = CreatePreparedPolygon(coords);
                if (prepared_polygon)
                {
                    polygon_map[token] = prepared_polygon;
                }
                else
                {
                    std::cerr << "OccupancyMap: Failed to create polygon for token: " << token << std::endl;
                }
            }
        }

        void OccupancyMap::ProcessDeletedPolygons(
            const std::vector<std::string>                                                            &deleted_tokens,
            std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> &polygon_map)
        {
            for (const std::string &token : deleted_tokens)
            {
                auto it = polygon_map.find(token);
                if (it != polygon_map.end())
                {
                    polygon_map.erase(it);
                    // std::cout << "OccupancyMap: Deleted polygon for token: " << token << std::endl;
                }
            }
        }

        std::unordered_map<std::string, std::shared_ptr<const geos::geom::prep::PreparedPolygon>> *
        OccupancyMap::GetPolygonMapByCategory(int category)
        {
            switch (category)
            {
            case 0:
                return &route_edges_;
            case 1:
                return &not_on_route_roadblocks_and_carparks_;
            case 2:
                return &intersections_;
            default:
                std::cerr << "OccupancyMap: Invalid category: " << category << std::endl;
                return nullptr;
            }
        }
    } // namespace utils
} // namespace vec_qmdp