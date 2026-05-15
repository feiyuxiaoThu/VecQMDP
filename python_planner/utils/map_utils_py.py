# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import Point
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, Point2D
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    PolygonMapObject,
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
    PolylineMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType, TrafficLightStatusData

from nuplan.common.maps.abstract_map import AbstractMap

from python_planner.utils.bfs_roadblock import *
from python_planner.utils.route_utils import *
from python_planner.utils.math_utils import *

# Any map object that can appear in a spatial query result
MapObject = Union[Lane, LaneConnector, RoadBlockGraphEdgeMapObject,
                    PolygonMapObject, Intersection, StopLine]

# Lane-like objects that have baseline paths and connectivity
LaneObject = Union[Lane, LaneConnector, LaneGraphEdgeMapObject]

# Layers used for drivable-area queries around the ego vehicle
DRIVABLE_MAP_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.INTERSECTION,
    SemanticMapLayer.CARPARK_AREA,
]

class MapUtils:
    """HD-map query interface wrapping NuPlan's AbstractMap.

    Provides route repair, goal-miss penalty computation, drivable-area retrieval,
    reference-line interpolation, traffic-light status tracking, and lane-level
    spatial queries. Instantiated once per planning episode and shared with the
    C++ planner via pybind (MapUtilsObject).
    """

    def __init__(self,
                 map_api: AbstractMap,
                 mission_goal: StateSE2,
                 roadblock_ids: List[str],
                 route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
                 route_lane_dict: Dict[str, LaneGraphEdgeMapObject]) -> None:
        """
        Initialize MapUtils.

        Args:
            map_api: nuPlan HD-map API instance.
            mission_goal: target pose (x, y, heading) for the planning episode.
            roadblock_ids: ordered list of roadblock ids along the route.
            route_roadblock_dict: mapping from roadblock id to its map object.
            route_lane_dict: mapping from lane/lane-connector id to its map object.
        """
        self._map_api = map_api
        self._mission_goal = mission_goal

        # Goal location metadata
        self._goal_off_road = False
        self._goal_edge_id = ""
        self._goal_block_id = ""

        # Route information
        self._start_car_roadblock = None
        self._route_roadblock_dict = route_roadblock_dict
        self._roadblock_ids = roadblock_ids
        self._route_lane_dict = route_lane_dict

        # Goal-miss penalty bookkeeping
        self._route_goal_dict = {}
        self._lane_missed_goal_num_dict = {}
        self._actual_lane_missed_goal_num_dict = {}
        self._start_nearest_lane_connectors = []

        # Rasterization bounds (populated later)
        self._center_x = 0
        self._center_y = 0
        self._min_x = 0
        self._min_y = 0
        self._max_x = 0
        self._max_y = 0
        self._resolution = 0
        self._x_dimension = 0
        self._y_dimension = 0
        self._offset_x = 0
        self._offset_y = 0

        # Polygon construction tracking
        self._constructed_route_roadblock_ids = set()
        # [on-route lanes/connectors, off-route roadblocks, intersections/carpark areas]
        self._constructed_drivable_map_objects_ids = [set(), set(), set()]
        self._roadblock_cache = {}
        self._checked_toched_roadblock_pairs = {}

        # Current ego lane/roadblock tracking
        self._ego_edge_id = ""
        self._ego_edge_name = ""
        self._ego_roadblock_id = ""

    def are_roadblocks_geometrically_adjacent(
        self,
        roadblock_1: RoadBlockGraphEdgeMapObject,
        roadblock_2: RoadBlockGraphEdgeMapObject,
        distance_threshold: float = 5.0,
        intersection_threshold: float = 0.1
    ) -> bool:
        """
        Determine whether two NuPlanRoadBlocks are geometrically adjacent.

        Args:
            roadblock_1: First roadblock.
            roadblock_2: Second roadblock.
            distance_threshold: Distance threshold in meters beyond which two roadblocks are considered non-adjacent.
            intersection_threshold: Intersection area threshold in square meters above which two roadblocks are considered overlapping.

        Returns:
            Tuple[bool, str]: (is_adjacent, adjacency_type_description)
            Adjacency types: 'intersecting', 'touching', 'close', 'not_adjacent'
        """
        if roadblock_1.id == roadblock_2.id:
            return True

        try:
            poly1 = roadblock_1.polygon
            poly2 = roadblock_2.polygon

            # Check whether the distance is small enough
            distance = poly1.distance(poly2)
            if distance <= distance_threshold:
                return True

            # Check whether the two roadblocks touch (shared boundary, no overlap)
            if poly1.touches(poly2):
                return True

            return False

        except Exception as e:
            print(f"Error in geometric adjacency check: {e}")
            return False

    def get_cached_roadblock(self, roadblock_id: str):
        """
        Retrieve a roadblock map object, using a cache to avoid redundant lookups.

        Args:
            roadblock_id: id of the roadblock or roadblock connector.

        Returns:
            RoadBlockGraphEdgeMapObject: the cached roadblock object, or None if not found.
        """
        if roadblock_id not in self._roadblock_cache:
            block = self._map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            if block is None:
                block = self._map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._roadblock_cache[roadblock_id] = block

        return self._roadblock_cache[roadblock_id]

    def GetAdjacentRoadBlocks(self, on_route_roadblock_ids: List[str], candidate_roadblock: RoadBlockGraphEdgeMapObject):
        """
        Check if a candidate roadblock is geometrically adjacent to any on-route roadblock.

        Args:
            on_route_roadblock_ids: list of roadblock ids along the route.
            candidate_roadblock: the off-route roadblock to test against.

        Returns:
            bool: True if the candidate is adjacent to at least one on-route roadblock.
        """
        candidate_id = candidate_roadblock.id
        for on_route_roadblock_id in on_route_roadblock_ids:
            roadblock_pair = (candidate_id, on_route_roadblock_id)
            if roadblock_pair not in self._checked_toched_roadblock_pairs:
                roadblock = self.get_cached_roadblock(on_route_roadblock_id)
                self._checked_toched_roadblock_pairs[roadblock_pair] = self.are_roadblocks_geometrically_adjacent(roadblock, candidate_roadblock)
            if self._checked_toched_roadblock_pairs[roadblock_pair]:
                return True
            
        return False
            
    def GetDrivableMapObjects(self, ego_x: float, ego_y: float, ego_v: float, ego_heading: float, new_ego_edge_id: str, new_ego_edge_name: str, refline_edge_ids: List[str], refline_edge_names: List[str], map_radius: float = 50.0) -> Tuple[List[List[str]], List[List[List[Tuple[float, float]]]], List[List[str]]]:
        """
        Query drivable-area map objects near the ego vehicle and compute incremental insert/delete sets.

        Args:
            ego_x: ego x position in local coordinates.
            ego_y: ego y position in local coordinates.
            ego_v: ego velocity in m/s.
            ego_heading: ego heading in radians.
            new_ego_edge_id: id of the lane/connector the ego currently occupies.
            new_ego_edge_name: human-readable name of that lane/connector.
            refline_edge_ids: ordered edge ids along the current reference line.
            refline_edge_names: corresponding edge names.
            map_radius: spatial query radius in meters.

        Returns:
            Tuple[List[List[str]], List[List[List[Tuple[float, float]]]], List[List[str]]]:
                (inserted_tokens, inserted_coords, deleted_tokens) grouped by layer
                [on-route edges, off-route roadblocks/carparks, intersections].
        """
        ego_pos = Point(ego_x + self._offset_x, ego_y + self._offset_y)
        drivable_map_objects = self._map_api.get_proximal_map_objects(ego_pos, map_radius, DRIVABLE_MAP_LAYERS)

        route_roadblock_ids = set() # IDs of roadblocks that are on the route
        constructed_object_ids = [set(), set(), set()] # IDs of already-constructed objects

        inserted_tokens = [[], [], []] # IDs of objects to insert (edge, roadblock and carpark, intersection)
        inserted_coords = [[], [], []] # Coordinates of objects to insert
        deleted_tokens = [[], [], []] # IDs of objects to delete

        # Process roadblocks on the route
        curr_edge_obj = self._map_api.get_map_object(new_ego_edge_id, SemanticMapLayer.LANE if new_ego_edge_name == "LANE" else SemanticMapLayer.LANE_CONNECTOR)
        new_roadblock_id = curr_edge_obj.get_roadblock_id()
        curr_roadblock_idx = -1
        
        # Find the index of the current roadblock within the route
        for idx, roadblock_id in enumerate(self._roadblock_ids):
            if roadblock_id == new_roadblock_id:
                curr_roadblock_idx = idx
                break
            
        if curr_roadblock_idx != -1:
            # Add the preceding roadblock
            if self._ego_roadblock_id == "" and curr_roadblock_idx != 0:
                self._ego_roadblock_id = self._roadblock_ids[curr_roadblock_idx - 1]
            
            if self._ego_roadblock_id:
                route_roadblock_ids.add(self._ego_roadblock_id)

            # Add the current roadblock and all subsequent route roadblocks
            extended_radius = max(20.0 * ego_v, 120.0)
            for idx, roadblock_id in enumerate(self._roadblock_ids[curr_roadblock_idx:]):
                if roadblock_id in self._constructed_route_roadblock_ids:
                    route_roadblock_ids.add(roadblock_id)
                    continue

                block = self.get_cached_roadblock(roadblock_id)
                if idx <= 1 or (block is not None and block.polygon.centroid.distance(ego_pos) < extended_radius):
                    route_roadblock_ids.add(roadblock_id)
                else:
                    break

        # Process lanes and lane connectors on the route
        for roadblock_id in route_roadblock_ids:
            block = self.get_cached_roadblock(roadblock_id)
            if block is None:
                continue
                
            if roadblock_id not in self._constructed_route_roadblock_ids:
                for edge in block.interior_edges:
                    inserted_tokens[0].append(edge.id)
                    translated_coords = [
                        [x - self._offset_x, y - self._offset_y]
                        for x, y in edge.polygon.exterior.coords
                    ]
                    inserted_coords[0].append(translated_coords)
                    constructed_object_ids[0].add(edge.id)
            else:
                for edge in block.interior_edges:
                    constructed_object_ids[0].add(edge.id)

        # Process lanes and lane connectors on the reference line
        for idx, edge_id in enumerate(refline_edge_ids):
            if edge_id in constructed_object_ids[0]:
                continue
            
            edge = self._map_api.get_map_object(edge_id, SemanticMapLayer.LANE if refline_edge_names[idx] == "LANE" else SemanticMapLayer.LANE_CONNECTOR)
            inserted_tokens[0].append(edge_id)
            translated_coords = [
                [x - self._offset_x, y - self._offset_y]
                for x, y in edge.polygon.exterior.coords
            ]
            inserted_coords[0].append(translated_coords)
            constructed_object_ids[0].add(edge_id)

        # Generic function to process drivable objects that are off-route
        def process_drivable_objects(objects, layer_name, layer_index):
            for obj in objects:
                obj_pos = (obj.polygon.centroid.x - self._offset_x, obj.polygon.centroid.y - self._offset_y)
                
                # ROADBLOCK requires an additional check to see whether it is on-route
                if layer_name == "ROADBLOCK" and obj.id in self._roadblock_ids:
                    continue
                abs_normalized_angle = abs(normalize_angle(ego_heading - math.atan2((obj_pos[1] - ego_y), (obj_pos[0] - ego_x))))
                if InFront(abs_normalized_angle, 60.0 / 180 * np.pi) or (layer_name == "ROADBLOCK" and self.GetAdjacentRoadBlocks(route_roadblock_ids, obj)) or (layer_name == "INTERSECTION" and InBehind(abs_normalized_angle, 60.0 / 180 * np.pi)):
                    constructed_object_ids[layer_index].add(obj.id)
                    if obj.id not in self._constructed_drivable_map_objects_ids[layer_index]:
                        inserted_tokens[layer_index].append(obj.id)
                        # Translate the exterior coordinate points of the polygon
                        translated_coords = [
                            [x - self._offset_x, y - self._offset_y]
                            for x, y in obj.polygon.exterior.coords
                        ]
                        inserted_coords[layer_index].append(translated_coords)

        # Process all drivable map object types
        process_drivable_objects(drivable_map_objects[SemanticMapLayer.ROADBLOCK], "ROADBLOCK", 1)
        process_drivable_objects(drivable_map_objects[SemanticMapLayer.CARPARK_AREA], "CARPARK_AREA", 1)
        process_drivable_objects(drivable_map_objects[SemanticMapLayer.INTERSECTION], "INTERSECTION", 2)

        # Update state
        self._ego_roadblock_id = new_roadblock_id

        # Identify objects that need to be removed
        for layer_index in range(3):
            for object_id in self._constructed_drivable_map_objects_ids[layer_index]:
                if object_id not in constructed_object_ids[layer_index]:
                    deleted_tokens[layer_index].append(object_id)
        
        self._constructed_route_roadblock_ids = route_roadblock_ids
        self._constructed_drivable_map_objects_ids = constructed_object_ids

        return inserted_tokens, inserted_coords, deleted_tokens

    def UpdateEgoEdgeDirectly(self, ego_edge_id: str, ego_edge_name: str) -> None:
        """
        Directly set the ego lane/connector without spatial lookup.

        Args:
            ego_edge_id: id of the lane or lane connector.
            ego_edge_name: human-readable name of the lane or lane connector.
        """
        self._ego_edge_id = ego_edge_id
        self._ego_edge_name = ego_edge_name

    def UpdateOffset(self, offset_x: float, offset_y: float) -> None:
        """
        Set the coordinate offset between local and map frames.

        Args:
            offset_x: x-axis offset in meters.
            offset_y: y-axis offset in meters.
        """
        self._offset_x = offset_x
        self._offset_y = offset_y

    def UpdateTrafficStatus(self, traffic_light_data : Optional[List[TrafficLightStatusData]]) -> None:
        """
        Rebuild the traffic-light status lookup from the latest observation.

        Args:
            traffic_light_data: list of traffic light observations, or None if unavailable.
        """
        self._traffic_light_status = {}
        self._traffic_light_status[TrafficLightStatusType.RED] = []
        self._traffic_light_status[TrafficLightStatusType.GREEN] = []
        self._traffic_light_status[TrafficLightStatusType.YELLOW] = []
        if traffic_light_data is not None:
            for data in traffic_light_data:
                if data.status != TrafficLightStatusType.UNKNOWN:
                    self._traffic_light_status[data.status].append(str(data.lane_connector_id))

    def GetObjectById(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        """
        Retrieve a single map object by its id and semantic layer.

        Args:
            object_id: id of the map object.
            layer: semantic map layer to query.

        Returns:
            Optional[MapObject]: the map object, or None if not found.
        """
        return self._map_api.get_map_object(object_id, layer)

    def GetOneMapObject(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        """
        Return the first map object at a given point and layer.

        Args:
            point: query point in map coordinates.
            layer: semantic map layer to query.

        Returns:
            Optional[MapObject]: the first matching object, or None if none found.
        """
        all_objects = self._map_api.get_all_map_objects(point, layer)

        return all_objects[0] if len(all_objects) > 0 else None

    def GetAllMapObjectsIds(self, point: Point2D, layer: SemanticMapLayer, has_offset: bool = True) -> List[str]:
        """
        Return ids of all map objects at a given point. Called by C++ with offset-adjusted coordinates.

        Args:
            point: query point (in local coordinates if has_offset is True).
            layer: semantic map layer to query.
            has_offset: whether to apply the stored coordinate offset before querying.

        Returns:
            List[str]: ids of all matching map objects.
        """
        if has_offset:
            point = Point(point.x + self._offset_x, point.y + self._offset_y)

        return [obj.id for obj in self._map_api.get_all_map_objects(point, layer)]

    def GetEdgeTypeById(self, lane_id: str) -> str:
        """
        Determine the edge type for a given lane id.

        Args:
            lane_id: id of the lane or lane connector; empty string for unknown.

        Returns:
            str: one of 'EDGE', 'LANE', or 'LANE_CONNECTOR'.
        """
        if lane_id == "":
            return "EDGE"
        elif self._map_api.get_map_object(lane_id, SemanticMapLayer.LANE) is not None:
            return "LANE"
        else:
            return "LANE_CONNECTOR"

    def GetStartingLaneIds(self, point: Point2D) -> List[str]:
        """
        Find candidate starting lane ids for the ego at a given point. Called by C++ with offset-adjusted coordinates.

        Args:
            point: ego position in local coordinates (offset is applied internally).

        Returns:
            List[str]: ids of lanes containing the point, or the nearest lane if none contain it.
        """
        point = Point(point.x + self._offset_x, point.y + self._offset_y)
        
        closest_distance = np.inf
        candidate_lanes = []
        candidate_lane_ids = []
        
        # first find whether the car is on the route
        for edge in self._route_lane_dict.values():
            if edge.contains_point(point):
                candidate_lanes.append(edge)
                candidate_lane_ids.append(edge.id)

            distance = edge.polygon.distance(point)
            if distance < closest_distance:
                nearest_starting_lane = edge
                closest_distance = distance
        
        # if the car is offroad, the nearest start roadblock is found
        if len(candidate_lanes) == 0 and self._start_car_roadblock:
            for edge in self._start_car_roadblock.interior_edges:
                if edge.contains_point(point):
                    candidate_lanes.append(edge)
                else:
                    distance = edge.polygon.distance(point)
                    if distance < closest_distance:
                        nearest_starting_lane = edge
                        closest_distance = distance
                                
        if len(candidate_lanes) > 0:
            return [lane.id for lane in candidate_lanes]
        else:
            return [nearest_starting_lane.id]

    def GetRefLine(self, edge: LaneObject, path_step: float, degree = 3) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """
        Compute a smoothed reference line from a lane edge via spline interpolation.

        Args:
            edge: lane or lane connector whose baseline path is used.
            path_step: resampling step size in meters along the spline.
            degree: spline degree for UnivariateSpline fitting.

        Returns:
            Tuple[List[float], ...]: (xs, ys, headings, curvatures, left_widths, right_widths)
                in local coordinates.
        """
        refline_path = edge.baseline_path.discrete_path[:-1]

        xs = [point.x - self._offset_x for point in refline_path]
        ys = [point.y - self._offset_y for point in refline_path]

        xs = np.array(xs)
        ys = np.array(ys)
        
        all_straight_line = True

        def line_r_squared(xs, ys):
            # Fitting linear regression model
            model = LinearRegression()
            X = xs.reshape(-1, 1)
            model.fit(X, ys)
            r_squared = model.score(X, ys)
            return r_squared
        
        def process_segment(xs, ys, path_step, degree, distance_offset, all_straight_line_):
            
            # Check if the line is nearly a straight line
            r_squared = line_r_squared(xs, ys)
            if r_squared > 0.99:
                degree = 1
            
            if degree > 1:
                all_straight_line_ = False
            
            # Step 1: Calculate cumulative distance along original points for interpolation
            distances_orig = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
            cumulative_distances_orig = np.cumsum(distances_orig)
            cumulative_distances_orig = np.insert(cumulative_distances_orig, 0, 0)
            
            # Step 2: Interpolate original points to equal spacing
            desired_distances = np.arange(0, cumulative_distances_orig[-1], path_step)
            x_interp = np.interp(desired_distances, cumulative_distances_orig, xs)
            y_interp = np.interp(desired_distances, cumulative_distances_orig, ys)
            
            if degree == 1:
                t = np.arange(len(xs)).reshape(-1, 1)

                # Fit x(t) and y(t) separately
                model_x = LinearRegression().fit(t, xs)
                model_y = LinearRegression().fit(t, ys)

                dx_dt = model_x.coef_[0]
                dy_dt = model_y.coef_[0]

                # Derive heading from the tangent vector
                heading = np.full_like(x_interp, np.arctan2(dy_dt, dx_dt))

                # Curvature is always 0 for straight segments
                curvature = np.zeros_like(x_interp)
                dkappa = np.zeros_like(x_interp)
            else:
                # Step 3: Fit a CubicSpline using arc length
                # Use a smoothing spline; tune parameter s for better results
                spl_x = UnivariateSpline(cumulative_distances_orig, xs, s=0.1)
                spl_y = UnivariateSpline(cumulative_distances_orig, ys, s=0.1)
                
                # Compute derivatives
                dx = spl_x(desired_distances, 1)
                dy = spl_y(desired_distances, 1)
                ddx = spl_x(desired_distances, 2)
                ddy = spl_y(desired_distances, 2)

                # Compute heading and curvature
                heading = np.arctan2(dy, dx)
                curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
                curvature[np.isnan(curvature)] = 0.0

                # Compute curvature rate-of-change dkappa
                dkappa = np.diff(curvature) / np.diff(desired_distances)
                dkappa = np.append(dkappa, dkappa[-1] if len(dkappa) > 0 else 0.0)

            return (list(x_interp), 
                    list(y_interp), 
                    list(heading), 
                    list(desired_distances + distance_offset), 
                    list(curvature), 
                    list(dkappa), 
                    all_straight_line_,
                    r_squared)


        total_length = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        segment_length = 25.0
        num_segments = int(np.floor(total_length / segment_length))

        all_x_poly_new, all_y_poly_new, all_heading, all_distances, all_curvature, all_dkappa, all_r_squared = [], [], [], [], [], [], []
        segment_lengths = []  # Store the length of each segment
        segment_r_squareds = []  # Store the R-squared value of each segment

        if num_segments == 0:
            if len(xs) > 1:
                x_poly_new, y_poly_new, heading, distances, curvature, dkappa, all_straight_line, r_squared = process_segment(xs, ys, path_step, degree, 0, all_straight_line)
                all_x_poly_new.extend(x_poly_new)
                all_y_poly_new.extend(y_poly_new)
                all_heading.extend(heading)
                all_distances.extend(distances)
                all_curvature.extend(curvature)
                all_dkappa.extend(dkappa)
                segment_lengths.append(len(x_poly_new))
                segment_r_squareds.append(r_squared)
        else:
            for i in range(num_segments):
                start_index = np.searchsorted(np.cumsum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)), i * segment_length)
                if i == num_segments - 1:  # Include remaining parts in the last segment
                    end_index = len(xs)
                else:
                    end_index = np.searchsorted(np.cumsum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)), (i + 1) * segment_length)

                segment_xs = xs[start_index:end_index]
                segment_ys = ys[start_index:end_index]

                if len(segment_xs) > 1:
                    x_poly_new, y_poly_new, heading, distances, curvature, dkappa, all_straight_line, r_squared = process_segment(segment_xs, segment_ys, path_step, degree, i * segment_length, all_straight_line)
                    all_x_poly_new.extend(x_poly_new)
                    all_y_poly_new.extend(y_poly_new)
                    all_heading.extend(heading)
                    all_distances.extend(distances)
                    all_curvature.extend(curvature)
                    all_dkappa.extend(dkappa)
                    segment_lengths.append(len(x_poly_new))
                    segment_r_squareds.append(r_squared)

        # Compute the segment-length-weighted average R-squared
        weighted_avg_r_squared = 0.0
        total_length = sum(segment_lengths)
        if total_length > 0:
            weighted_avg_r_squared = sum(r_sq * length for r_sq, length in zip(segment_r_squareds, segment_lengths)) / total_length
            
        # If segment_lengths is shorter than 15, set all_curvature to 0
        if total_length < 15:
            all_curvature = [0.0] * len(all_curvature)
            all_dkappa = [0.0] * len(all_dkappa)

        return (all_x_poly_new, 
                all_y_poly_new, 
                all_heading, 
                all_distances, 
                all_curvature, 
                all_dkappa, 
                all_straight_line,
                weighted_avg_r_squared)
    
    def IsPointInLaneById(self, lane_id: str, point: Point2D, is_lane: str, has_offset: bool = True) -> bool:
        """
        Check whether a point lies inside a lane or lane connector. Called by C++ with offset-adjusted coordinates.

        Args:
            lane_id: id of the lane or lane connector.
            point: query point (in local coordinates if has_offset is True).
            is_lane: 'LANE' to query the lane layer, otherwise the lane-connector layer.
            has_offset: whether to apply the stored coordinate offset before querying.

        Returns:
            bool: True if the point is contained within the lane polygon.
        """
        if is_lane == "LANE":
            lane = self._map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
        else:
            lane = self._map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)

        if lane is None:
            return False

        if has_offset:
            point = Point2D(point.x + self._offset_x, point.y + self._offset_y)
        if lane.contains_point(point):
            return True
        else:
            return False

    def GetLeftAndRightNeighborId(self, edge: LaneObject) -> str:
        """
        Get the ids of the left and right adjacent lanes.

        Args:
            edge: lane or lane connector to query.

        Returns:
            Tuple[str, str]: (left_lane_id, right_lane_id); empty string if no neighbor.
        """
        left_lane, right_lane = edge.adjacent_edges
        left_lane_id = left_lane.id if left_lane is not None else ""
        right_lane_id = right_lane.id if right_lane is not None else ""
        
        return (left_lane_id, right_lane_id)

    def GetSuccessorIds(self, edge: LaneObject) -> List[str]:
        """
        Get the ids of all outgoing (successor) edges.

        Args:
            edge: lane or lane connector to query.

        Returns:
            List[str]: ids of successor edges.
        """
        return [successor.id for successor in edge.outgoing_edges]

    def GetPredecessorIds(self, edge: LaneObject) -> List[str]:
        """
        Get the ids of all incoming (predecessor) edges.

        Args:
            edge: lane or lane connector to query.

        Returns:
            List[str]: ids of predecessor edges.
        """
        return [predecessor.id for predecessor in edge.incoming_edges]

    def HasRedTrafficLight(self, edge: LaneObject):
        """
        Check whether a lane connector is controlled by a red traffic light.

        Args:
            edge: lane connector to check.

        Returns:
            bool: True if the edge has traffic lights and its current status is red.
        """
        if edge.has_traffic_lights() and edge.id in self._traffic_light_status[TrafficLightStatusType.RED]:
            return True
        else:
            return False

    def QueryNearestLaneLink(self, point: Point2D, heading: float, radius: float = 2.0, heading_error: float = np.pi / 4.0, num: int = 3) -> List[LaneConnector]:
        """
        Find nearby lane connectors whose heading aligns with the query heading.

        Args:
            point: query point in map coordinates.
            heading: query heading in radians.
            radius: spatial search radius in meters.
            heading_error: maximum heading deviation in radians.
            num: maximum number of lane connectors to return.

        Returns:
            List[LaneConnector]: up to `num` lane connectors sorted by distance.
        """
        lane_link_dict = self._map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE_CONNECTOR])
        lane_links = lane_link_dict[SemanticMapLayer.LANE_CONNECTOR]
        
        def IsHeadingAlongRefLine(point: Point2D, heading: float, ref_line: PolylineMapObject, heading_error: float) -> Tuple[bool, float]:
            nearest_pose = ref_line.get_nearest_pose_from_position(point)
            return close_angle(nearest_pose.heading, heading, heading_error), ((nearest_pose.x - point.x) ** 2 + (nearest_pose.y - point.y) ** 2)
        
        nearest_lane_links = []
        distances = []
        for lane_link in lane_links:
            if len(nearest_lane_links) >= 10:
                break
            along, distance_squared = IsHeadingAlongRefLine(point, heading, lane_link.baseline_path, heading_error)
            if along:
                nearest_lane_links.append(lane_link)
                distances.append(distance_squared)

        if len(distances) == 0:
            return []
        else:
            zipped_lists = list(zip(distances, nearest_lane_links))
            zipped_lists.sort(key=lambda x: x[0])
            distances_sorted, nearest_lane_links_sorted = zip(*zipped_lists)
            distances_sorted = list(distances_sorted)
            nearest_lane_links_sorted = list(nearest_lane_links_sorted)
            return nearest_lane_links_sorted[:num]

    def QueryNearestLaneIds(self, point: Point2D, heading: float, radius: float = 2.0, heading_error: float = np.pi / 4.0, num: int = 3) -> List[str]:
        """
        Find nearby lane ids whose heading aligns with the query heading.

        Args:
            point: query point in map coordinates.
            heading: query heading in radians.
            radius: spatial search radius in meters.
            heading_error: maximum heading deviation in radians.
            num: maximum number of lane ids to return.

        Returns:
            List[str]: up to `num` lane ids sorted by distance.
        """
        lane_dict = self._map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE])
        lanes = lane_dict[SemanticMapLayer.LANE]
        
        def IsHeadingAlongRefLine(point: Point2D, heading: float, ref_line: PolylineMapObject, heading_error: float) -> Tuple[bool, float]:
            nearest_pose = ref_line.get_nearest_pose_from_position(point)        
            return close_angle(nearest_pose.heading, heading, heading_error), ((nearest_pose.x - point.x) ** 2 + (nearest_pose.y - point.y) ** 2)
        
        nearest_lanes = []
        distances = []
        for lane in lanes:
            if len(nearest_lanes) >= 10:
                break
            along, distance_squared = IsHeadingAlongRefLine(point, heading, lane.baseline_path, heading_error)
            if along:
                nearest_lanes.append(lane)
                distances.append(distance_squared)

        if len(distances) == 0:
            return []
        else:
            zipped_lists = list(zip(distances, nearest_lanes))
            zipped_lists.sort(key=lambda x: x[0])
            distances_sorted, nearest_lanes_sorted = zip(*zipped_lists)
            distances_sorted = list(distances_sorted)
            nearest_lanes_sorted = list(nearest_lanes_sorted)
            return [lane.id for lane in nearest_lanes_sorted[:num]]

    def LoadRouteDicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Load roadblock and lane dictionaries of the target route from the map API.

        Args:
            route_roadblock_ids: ordered list of on-route roadblock ids.
        """
        # remove repeated ids while remaining order in list
        self._roadblock_ids = route_roadblock_ids
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        route_roadblock_dict = {}
        route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                route_lane_dict[lane.id] = lane
        self._route_roadblock_dict, self._route_lane_dict = (route_roadblock_dict, route_lane_dict)
        
    def RepairRouteStart(self, route_roadblock_ids: List[str], ego_state: EgoState) -> List[str]:
        """
        Repair the route head so it begins at the ego's current roadblock.

        Args:
            route_roadblock_ids: ordered list of on-route roadblock ids.
            ego_state: current ego state.

        Returns:
            List[str]: repaired route roadblock ids with start segment prepended if needed.
        """
        # Get the closet roadblock and candidate roadblocks for the given ego state

        # 1. Get the road block id of the start point
        road_block_obj = self.GetOneMapObject(ego_state.car_footprint.center.point, SemanticMapLayer.ROADBLOCK)
        start_block_id = ""
        if not road_block_obj:
            # if the car is on the junction
            self._start_nearest_lane_connectors = self.QueryNearestLaneLink(ego_state.car_footprint.center.point, ego_state.car_footprint.center.heading, radius=5.5)
            if len(self._start_nearest_lane_connectors) == 0:
                starting_block, starting_block_candidates = get_current_roadblock_candidates(
                    ego_state, self._map_api, route_roadblock_ids
                )
                self._start_car_roadblock = starting_block if starting_block else starting_block_candidates[0]
                start_block_id = self._start_car_roadblock.id
            else:
                for nearest_lane_connector in self._start_nearest_lane_connectors:
                    if nearest_lane_connector.get_roadblock_id() in route_roadblock_ids:
                        start_block_id = nearest_lane_connector.get_roadblock_id()
                        break
                if start_block_id == "":
                    start_block_id = self._start_nearest_lane_connectors[0].get_roadblock_id()
                self._start_car_roadblock = self._start_nearest_lane_connectors[0].parent
        else:
            start_block_id = road_block_obj.id
            self._start_car_roadblock = road_block_obj

        # 2. Repair the route if the start point is not on the route
        start_block_on_route = False
        if start_block_id in route_roadblock_ids:
            start_block_on_route = True
    
        if not start_block_on_route:
            # backward searching for the goal roadblock
            for i in range(len(route_roadblock_ids)):
                graph_search = BreadthFirstSearchRoadBlock(route_roadblock_ids[i], self._map_api, forward_search=False)
                (path, path_id), path_found = graph_search.search(start_block_id, max_depth=15)
                if path_found:
                    route_roadblock_ids[:i] = path_id[:-1]
                    break

        return route_roadblock_ids
        
    def RepairRouteGoal(self, route_roadblock_ids: List[str]) -> List[str]:
        """
        Repair the route tail so it ends at the goal's roadblock.

        Args:
            route_roadblock_ids: ordered list of on-route roadblock ids.

        Returns:
            List[str]: repaired route roadblock ids with goal segment appended if needed.
        """

        # 1. Get the road block id of the goal point
        road_block_obj = self.GetOneMapObject(self._mission_goal.point, SemanticMapLayer.ROADBLOCK)
        if not road_block_obj:
            nearest_lane_connectors = self.QueryNearestLaneLink(self._mission_goal.point, self._mission_goal.heading, radius=3)
            if len(nearest_lane_connectors) == 0:
                self._goal_off_road = True
                self._goal_block_id = route_roadblock_ids[-1]
            else:
                possible_goal_block_ids = []
                for edge in nearest_lane_connectors:
                    possible_goal_block_ids.append(edge.get_roadblock_id())
                for roadblock_id in route_roadblock_ids:
                    if roadblock_id in possible_goal_block_ids:
                        self._goal_block_id = roadblock_id
                        break
                if self._goal_block_id == "":
                    self._goal_block_id = possible_goal_block_ids[0]
        else:
            self._goal_block_id = road_block_obj.id
        
        # 2. Repair the route if the goal is not on the route
        if self._goal_block_id != route_roadblock_ids[-1]:
            # forward searching for the goal roadblock
            graph_search = BreadthFirstSearchRoadBlock(route_roadblock_ids[-1], self._map_api, forward_search=True)
            (path, path_id), path_found = graph_search.search(self._goal_block_id, max_depth=30)
            if path_found:
                route_roadblock_ids += path_id[1:]
                
        return route_roadblock_ids

    def RepairRouteIntermediate(self, route_roadblock_ids: List[str]) -> List[str]:
        """
        Fill gaps between consecutive roadblocks that are not directly connected.

        Args:
            route_roadblock_ids: ordered list of on-route roadblock ids.

        Returns:
            List[str]: repaired route roadblock ids with intermediate segments inserted and consecutive duplicates removed.
        """
        roadblocks_to_append = {}
        for i in range(len(route_roadblock_ids) - 1):
            i_route_roadblock_obj = self.GetObjectById(route_roadblock_ids[i + 1], SemanticMapLayer.ROADBLOCK)
            if not i_route_roadblock_obj:
                i_route_roadblock_obj = self.GetObjectById(route_roadblock_ids[i + 1], SemanticMapLayer.ROADBLOCK_CONNECTOR)
            next_incoming_block_ids = [
                _roadblock.id for _roadblock in i_route_roadblock_obj.incoming_edges
            ]
            is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

            if is_incoming:
                continue

            graph_search = BreadthFirstSearchRoadBlock(
                route_roadblock_ids[i], self._map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[i + 1], max_depth=5
            )

            if path_found and path and len(path) >= 3:
                path, path_id = path[1:-1], path_id[1:-1]
                roadblocks_to_append[i] = (path, path_id)

        # append missing intermediate roadblocks
        offset = 1
        for i, (path, path_id) in roadblocks_to_append.items():
            route_roadblock_ids[i + offset : i + offset] = path_id
            offset += len(path)

        # remove consecutive duplicates
        deduped = []
        for rid in route_roadblock_ids:
            if not deduped or deduped[-1] != rid:
                deduped.append(rid)

        return deduped

    def UpdateGoalMissNum(self):
        """
        Compute missed-goal cost for each lane/lane-connector, used for lane selection in path planning.

        The cost has two variants:
        - Actual missed-goal cost: based on lateral distance (lane count) to the goal lane.
        - Virtual missed-goal cost: actual cost plus dead-end penalties (+5 if a successor is off-route,
          propagated backward to predecessors).

        Procedure:
        1. Locate the lane edge containing the goal position.
        2. Set zero cost for the target lane; assign increasing costs to lateral neighbors.
        3. Work backward segment by segment from the goal to propagate costs.
        4. Add off-route successor penalties to produce the virtual cost.

        Updates:
            self._lane_missed_goal_num_dict: {edge_id: int} virtual missed-goal cost.
            self._actual_lane_missed_goal_num_dict: {edge_id: int} actual missed-goal cost.
        """
        # Reset the missed-goal cost dictionaries
        self._lane_missed_goal_num_dict = {}
        self._actual_lane_missed_goal_num_dict = {}
        
        # Special case: if the goal is outside the road network, set all segment costs to 0
        if self._goal_off_road:
            for roadblock_id in self._roadblock_ids:
                for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                    self._lane_missed_goal_num_dict[edge.id] = 0
                    self._actual_lane_missed_goal_num_dict[edge.id] = 0
            return
        
        # 1. Locate the lane edge containing the goal position
        if self._mission_goal:
            # First try to locate the goal within normal lanes
            edge_obj = self.GetOneMapObject(self._mission_goal.point, SemanticMapLayer.LANE)
            if not edge_obj:
                # If not found in normal lanes, search lane connectors
                obj_list = self._map_api.get_all_map_objects(self._mission_goal.point, SemanticMapLayer.LANE_CONNECTOR)
                if obj_list:
                    # Goal is on a lane connector
                    on_lane = False

                    # Select the lane connector belonging to the last road segment
                    possible_edge_obj = [obj for obj in obj_list if obj.get_roadblock_id() == self._roadblock_ids[-1]]
                    edge_obj = possible_edge_obj[0]
                    self._goal_edge_id = edge_obj.id
            else:
                # Goal is on a normal lane
                on_lane = True
                self._goal_edge_id = edge_obj.id
        else:
            edge_obj = None
        self._goal_block_id = self._roadblock_ids[-1]
        
        # 2. Update the missed-goal cost for the last road segment
        # Validate that the goal is on the last road segment; otherwise the input is invalid
        goal_misplaced = False
        if edge_obj.get_roadblock_id() != self._goal_block_id:
            goal_misplaced = True
        
        if edge_obj and not goal_misplaced:
            # Set zero missed-goal cost for the target lane
            self._lane_missed_goal_num_dict[edge_obj.id] = 0
            self._actual_lane_missed_goal_num_dict[edge_obj.id] = 0
            if on_lane:
                # If the goal is on a normal lane, assign incrementally increasing costs to adjacent lanes
                lateral_lane_num = 0
                left_neighbor, right_neighbor = edge_obj.adjacent_edges
                while left_neighbor is not None or right_neighbor is not None:
                    lateral_lane_num += 1
                    if left_neighbor:
                        self._lane_missed_goal_num_dict[left_neighbor.id] = lateral_lane_num
                        self._actual_lane_missed_goal_num_dict[left_neighbor.id] = lateral_lane_num
                        left_neighbor, _ = left_neighbor.adjacent_edges
                    if right_neighbor:
                        self._lane_missed_goal_num_dict[right_neighbor.id] = lateral_lane_num
                        self._actual_lane_missed_goal_num_dict[right_neighbor.id] = lateral_lane_num
                        _, right_neighbor = right_neighbor.adjacent_edges
            else:
                # If the goal is on a lane connector, set zero cost for all candidate target lane connectors
                for edge in possible_edge_obj:
                    self._lane_missed_goal_num_dict[edge.id] = 0
                    self._actual_lane_missed_goal_num_dict[edge.id] = 0
        else:
            # If the goal position is missing, set all lane costs in the last segment to zero
            for edge in self._route_roadblock_dict[self._roadblock_ids[-1]].interior_edges:
                self._lane_missed_goal_num_dict[edge.id] = 0
                self._actual_lane_missed_goal_num_dict[edge.id] = 0

        def UpdateMissGoalNumOnRoad(min_goal_lanes_, min_miss_num_, for_actual = False):
            """
            Update missed-goal cost values for a road segment.
            
            Starting from the lane with the minimum missed-goal cost, expand to adjacent lanes on both sides and assign incrementally increasing costs.
            Example: 0, -, -, -, -, 0 -> 0, 1, 2, 2, 1, 0
            
            Args:
                min_goal_lanes_: List of lanes with the minimum missed-goal cost.
                min_miss_num_: The minimum missed-goal cost value.
                for_actual: Whether to update the actual cost (True) or the virtual cost (False).
            """
            min_goal_lane_ids_ = [edge.id for edge in min_goal_lanes_]
            curr_block_edge_id_miss_num_dict_ = {}
            
            # Expand outward from each minimum-cost lane in both directions
            for edge in min_goal_lanes_:
                lateral_lane_num = min_miss_num_
                if edge.id in curr_block_edge_id_miss_num_dict_:
                    curr_block_edge_id_miss_num_dict_[edge.id].append(lateral_lane_num)
                else:
                    curr_block_edge_id_miss_num_dict_[edge.id] = [lateral_lane_num]
                
                # Assign incrementally increasing costs to left and right adjacent lanes
                left_neighbor, right_neighbor = edge.adjacent_edges
                while left_neighbor is not None or right_neighbor is not None:
                    lateral_lane_num += 1
                    if left_neighbor:
                        if left_neighbor.id in curr_block_edge_id_miss_num_dict_:
                            curr_block_edge_id_miss_num_dict_[left_neighbor.id].append(lateral_lane_num)
                        else:
                            curr_block_edge_id_miss_num_dict_[left_neighbor.id] = [lateral_lane_num]
                        left_neighbor, _ = left_neighbor.adjacent_edges
                    if right_neighbor:
                        if right_neighbor.id in curr_block_edge_id_miss_num_dict_:
                            curr_block_edge_id_miss_num_dict_[right_neighbor.id].append(lateral_lane_num)
                        else:
                            curr_block_edge_id_miss_num_dict_[right_neighbor.id] = [lateral_lane_num]
                        _, right_neighbor = right_neighbor.adjacent_edges
                        
            # Update the missed-goal cost dict, taking the minimum value
            for edge_id in curr_block_edge_id_miss_num_dict_.keys():
                if for_actual:
                    self._actual_lane_missed_goal_num_dict[edge_id] = min(curr_block_edge_id_miss_num_dict_[edge_id])
                else:
                    if edge_id not in self._lane_missed_goal_num_dict:
                        # If this is a newly encountered lane, add a virtual penalty of 5
                        self._lane_missed_goal_num_dict[edge_id] = min(curr_block_edge_id_miss_num_dict_[edge_id]) + 5
                    else:
                        self._lane_missed_goal_num_dict[edge_id] = min(curr_block_edge_id_miss_num_dict_[edge_id])

        # 3. Compute missed-goal costs segment by segment, working backward from the goal
        # Iterate over segments in reverse order (starting from the second-to-last)
        last_roadblock_id = ""
        for i, roadblock_id in enumerate(self._roadblock_ids[-2::-1]):
            if roadblock_id == last_roadblock_id:
                continue
            last_roadblock_id = roadblock_id
            on_lane = not on_lane  # Alternate between processing lanes and lane connectors
            
            # Skip the goal segment (avoids issues with circular routes)
            if roadblock_id == self._goal_block_id:
                continue
                
            road_min_missed_num = np.inf
            actual_road_min_missed_num = np.inf
            
            # Compute missed-goal cost for each lane in the current segment
            for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                successor_ids = [edge_.id for edge_ in edge.outgoing_edges]
                
                # Retrieve missed-goal costs for all successor lanes
                missed_nums        = [self._lane_missed_goal_num_dict[successor_id] if successor_id in self._lane_missed_goal_num_dict 
                                      else np.inf for successor_id in successor_ids]
                actual_missed_nums = [self._actual_lane_missed_goal_num_dict[successor_id] if successor_id in self._actual_lane_missed_goal_num_dict 
                                      else np.inf for successor_id in successor_ids]
                
                # Take the minimum missed-goal cost
                min_missed_num         = min(missed_nums) if len(missed_nums) > 0 else np.inf
                actual_min_missed_nums =  min(actual_missed_nums) if len(actual_missed_nums) > 0 else np.inf
                
                if min_missed_num == np.inf:
                    continue
                else:
                    # Update the missed-goal cost for the current lane
                    self._lane_missed_goal_num_dict[edge.id] = min_missed_num
                    self._actual_lane_missed_goal_num_dict[edge.id] = actual_min_missed_nums
                    road_min_missed_num = min(min_missed_num, road_min_missed_num)
                    actual_road_min_missed_num = min(actual_min_missed_nums, actual_road_min_missed_num)

            # For lane-type segments, also compute laterally adjacent lane costs
            if on_lane:
                # Find the lane with the minimum missed-goal cost
                min_goal_lanes = []
                for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                    if edge.id in self._lane_missed_goal_num_dict and self._lane_missed_goal_num_dict[edge.id] == road_min_missed_num:
                        min_goal_lanes.append(edge)
                UpdateMissGoalNumOnRoad(min_goal_lanes, road_min_missed_num)

                # Handle actual missed-goal costs in the same way
                actual_min_goal_lanes = []
                for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                    if edge.id in self._actual_lane_missed_goal_num_dict and self._actual_lane_missed_goal_num_dict[edge.id] == actual_road_min_missed_num:
                        actual_min_goal_lanes.append(edge)
                UpdateMissGoalNumOnRoad(actual_min_goal_lanes, actual_road_min_missed_num, True)

        # 4. Adjust costs for dead-end lanes; subtract the minimum missed-goal cost within each roadblock from all of its edges
        # This block adds a high penalty to lanes that have no on-route successor lanes
        for i, roadblock_id in enumerate(self._roadblock_ids[-2::-1]):
            if roadblock_id == self._goal_block_id:
                continue
            road_min_missed_num = np.inf
            min_actual_miss_num = np.inf
            for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                if edge.id not in self._actual_lane_missed_goal_num_dict:
                    self._actual_lane_missed_goal_num_dict[edge.id] = 5
                    
                successor_in_route = False
                for edge_ in edge.outgoing_edges:
                    if edge_.id in self._route_lane_dict.keys():
                        successor_in_route = True
                        break
                if not successor_in_route:
                    self._actual_lane_missed_goal_num_dict[edge.id] += 5
                
                # Save the minimum actual missed-goal cost
                min_actual_miss_num = min(min_actual_miss_num, self._actual_lane_missed_goal_num_dict[edge.id])

            for edge in self._route_roadblock_dict[roadblock_id].interior_edges:
                self._actual_lane_missed_goal_num_dict[edge.id] -= min_actual_miss_num

    def PenaltyOnMissGoalLaneId(self, edge_id: str) -> int:
        """
        Look up the virtual missed-goal penalty for a lane edge (includes dead-end penalties).

        Args:
            edge_id: id of the lane or lane connector.

        Returns:
            int: virtual missed-goal cost, or 10 if the edge is not on route.
        """
        if edge_id in self._lane_missed_goal_num_dict:
            return self._lane_missed_goal_num_dict[edge_id]
        else:
            return 10

    def ActualPenaltyOnMissGoalLaneId(self, edge_id: str) -> int:
        """
        Look up the actual missed-goal penalty for a lane edge (lateral distance only).

        Args:
            edge_id: id of the lane or lane connector.

        Returns:
            int: actual missed-goal cost, or 10 if the edge is not on route.
        """
        if edge_id in self._actual_lane_missed_goal_num_dict:
            return self._actual_lane_missed_goal_num_dict[edge_id]
        else:
            return 10
