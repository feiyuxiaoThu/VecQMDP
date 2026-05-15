# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Build QCNet input features from NuPlan agent states and HD map data,
including agent historical trajectories and map polygon/topology tensors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import *
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from python_planner.utils.agent_info import Agent


class QCNetFeatureBuilder:
    """
    Convert NuPlan simulation agent states and HD map into QCNet input feature tensors.

    Responsibilities:
      1. Agent feature construction — pack surrounding agents' historical trajectories
         (position, heading, velocity) into (num_agents, num_steps, dim) tensors with
         valid_mask and predict_mask.
      2. Map feature construction — query the NuPlan map API to extract lane, connector,
         and crosswalk geometry and topology, producing polygon-level and point-level
         features along with edge indices.

    The output feature dicts are consumed directly by QCNet forward inference.
    """

    def __init__(self,
                 dim: int=2,
                 radius: float=50,
                 num_historical_steps: int=20,
                 history_horizon: float = 2,
                 num_future_steps: int=80,
                 future_horizon: float = 8,
                 sample_interval: float=0.1,
                 ) -> None:
        """
        Initialize QCNetFeatureBuilder with temporal and spatial configuration.

        Args:
            dim: spatial dimension of positions and velocities (2 or 3)
            radius: map query radius in meters
            num_historical_steps: number of past timesteps (including current)
            history_horizon: duration of history window in seconds
            num_future_steps: number of future timesteps to predict
            future_horizon: duration of future window in seconds
            sample_interval: time interval between steps in seconds
        """
        super().__init__()

        self.dim = dim
        self.radius = radius
        self.num_historical_steps = num_historical_steps # history + current
        self.num_future_steps = num_future_steps
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.num_steps = self.num_historical_steps + self.num_future_steps + 1  # past + current + future
        self.current_center = StateSE2(0, 0, 0)
        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.TRAFFIC_CONE,
        ]
        self.input_objects_types = self.interested_objects_types + self.static_objects_types
        self.input_objects_types_value = [obj_type.value for obj_type in self.input_objects_types]
        self._valid_agent_types: Set[int] = {
            obj_type.value for obj_type in self.input_objects_types
        }
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK
        ]
        self._point_types = ['CENTERLINE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'CROSSWALK']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
        self._LEFT_BOUNDARY_TYPE = self._point_types.index('LEFT_BOUNDARY')
        self._RIGHT_BOUNDARY_TYPE = self._point_types.index('RIGHT_BOUNDARY')
        self._POINT_TYPE_CENTER = self._point_types.index('CENTERLINE')
        self._POINT_TYPE_CROSSWALK = self._point_types.index('CROSSWALK')
        self._SIDE_LEFT = self._point_sides.index('LEFT')
        self._SIDE_RIGHT = self._point_sides.index('RIGHT')
        self._SIDE_CENTER = self._point_sides.index('CENTER')
        self._POLY_TYPE_LANE = self.polygon_types.index(SemanticMapLayer.LANE)
        self._POLY_TYPE_CONNECTOR = self.polygon_types.index(SemanticMapLayer.LANE_CONNECTOR)
        self._POLY_TYPE_CROSSWALK = self.polygon_types.index(SemanticMapLayer.CROSSWALK)
        self._POLY_PRED = self._polygon_to_polygon_types.index('PRED')
        self._POLY_SUCC = self._polygon_to_polygon_types.index('SUCC')
        self._POLY_LEFT = self._polygon_to_polygon_types.index('LEFT')
        self._POLY_RIGHT = self._polygon_to_polygon_types.index('RIGHT')

    def _get_valid_agent_tokens(self, exo_agents: Dict[str, Agent]) -> List[str]:
        """
        Filter exo_agents to tokens whose object type is in the configured input types.

        Args:
            exo_agents: mapping from agent token to Agent object

        Returns:
            List[str]: tokens of agents with a valid (configured) object type
        """
        agents_token = []
        for agent in exo_agents.values():
            if agent.type in self.input_objects_types_value:
                agents_token.append(agent.token)

        return agents_token

    @torch.jit.unused
    def _build_agent_feature_from_simulation_inference(self, exo_agents: Dict[str, Agent], offset_x: float, offset_y: float, valid_agents_token: List[str]) -> Dict[str, Any]:
        """
        Build agent feature tensors using vectorized NumPy for online inference.

        Applies a coordinate offset to align agent positions with the ego-centric
        frame. No future observations are available; predict_mask covers only
        future timestep slots.

        Args:
            exo_agents: mapping from agent token to Agent object (contains history arrays)
            offset_x: x-coordinate offset to apply to all agent positions
            offset_y: y-coordinate offset to apply to all agent positions
            valid_agents_token: ordered list of tokens to include in the output

        Returns:
            Dict[str, Any]: agent feature dict with keys:
                num_nodes, av_index (-1), valid_mask, predict_mask,
                id, type, category, position, heading, velocity
        """
        num_agents = len(valid_agents_token)
        T = self.num_steps
        H = self.num_historical_steps + 1
        dim = self.dim

        if num_agents == 0:
            return {
                "num_nodes": 0,
                "av_index": -1,
                "valid_mask": torch.zeros(0, T, dtype=torch.bool),
                "predict_mask": torch.zeros(0, T, dtype=torch.bool),
                "id": valid_agents_token,
                "type": torch.zeros(0, dtype=torch.uint8),
                "category": torch.zeros(0, dtype=torch.uint8),
                "position": torch.zeros(0, T, dim, dtype=torch.float32),
                "heading": torch.zeros(0, T, dtype=torch.float32),
                "velocity": torch.zeros(0, T, dim, dtype=torch.float32),
            }

        agents = [exo_agents[token] for token in valid_agents_token]
        hist_valid = np.asarray([a._history_valid for a in agents], dtype=np.bool_)

        invalid_mask = ~hist_valid

        position = np.zeros((num_agents, T, dim), dtype=np.float32)
        heading = np.zeros((num_agents, T), dtype=np.float32)
        velocity = np.zeros((num_agents, T, dim), dtype=np.float32)

        pos_x_view = position[:, :H, 0]
        pos_x_view[:] = np.asarray([a._history_x for a in agents], dtype=np.float32)
        pos_x_view += offset_x
        pos_x_view[invalid_mask] = 0.0

        pos_y_view = position[:, :H, 1]
        pos_y_view[:] = np.asarray([a._history_y for a in agents], dtype=np.float32)
        pos_y_view += offset_y
        pos_y_view[invalid_mask] = 0.0

        heading_view = heading[:, :H]
        heading_view[:] = np.asarray([a._history_headings for a in agents], dtype=np.float32)

        vel_x_view = velocity[:, :H, 0]
        vel_x_view[:] = np.asarray([a._history_velocity_x for a in agents], dtype=np.float32)

        vel_y_view = velocity[:, :H, 1]
        vel_y_view[:] = np.asarray([a._history_velocity_y for a in agents], dtype=np.float32)

        valid_mask = np.zeros((num_agents, T), dtype=np.bool_)
        valid_mask[:, :H] = hist_valid
        valid_mask[:, 1:H] &= valid_mask[:, :H-1]
        valid_mask[:, 0] = False

        predict_mask = np.zeros((num_agents, T), dtype=np.bool_)
        predict_mask[:, H:] = True

        agent_type = np.fromiter((a.type for a in agents), dtype=np.uint8, count=num_agents)

        return {
            "num_nodes": num_agents,
            "av_index": -1,  # ego vehicle not included
            "valid_mask": torch.from_numpy(valid_mask),
            "predict_mask": torch.from_numpy(predict_mask),
            "id": valid_agents_token,
            "type": torch.from_numpy(agent_type),
            "category": torch.zeros(num_agents, dtype=torch.uint8),
            "position": torch.from_numpy(position),
            "heading": torch.from_numpy(heading),
            "velocity": torch.from_numpy(velocity),
        }

    def _build_map_feature_infer(
        self,
        current_center,
        map_api: 'AbstractMap',
        device,
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        """
        Build map feature dict: lane polygons, boundary points, edges

        Args:
            current_center: query center position (StateSE2 or compatible)
            map_api: nuPlan AbstractMap for spatial queries
            device: torch device

        Returns:
            Dict[str | Tuple, Any]: map feature dict with keys:
                'map_polygon', 'map_point',
                ('map_point', 'to', 'map_polygon'),
                ('map_polygon', 'to', 'map_polygon')
        """

        # Fetch and truncate objects
        effective_radius = min(self.radius, 80.0)  # cap maximum radius
        map_objects = map_api.get_proximal_map_objects(
            current_center,
            effective_radius,
            [
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.CROSSWALK,
            ]
        )
        lane_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )

        # Limit lane count, prioritize lanes closer to current position
        cx, cy = current_center.x, current_center.y
        max_lanes = 80  # maximum number of lanes
        if len(lane_objects) > max_lanes:
            n = len(lane_objects)
            distances = np.empty(n, dtype=np.float64)
            inf_val = float('inf')
            for i, lane in enumerate(lane_objects):
                centerline = lane.baseline_path.discrete_path
                if centerline:
                    mid = centerline[len(centerline) // 2]
                    dx = mid.x - cx
                    dy = mid.y - cy
                    distances[i] = dx * dx + dy * dy
                else:
                    distances[i] = inf_val
            sorted_indices = np.argpartition(distances, max_lanes)[:max_lanes]
            lane_objects = [lane_objects[i] for i in sorted_indices]

        # Limit crosswalk count to at most 10
        crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]
        max_crosswalks = 10
        if len(crosswalk_objects) > max_crosswalks:
            n_cw = len(crosswalk_objects)
            cw_distances = np.empty(n_cw, dtype=np.float64)
            for i, crosswalk in enumerate(crosswalk_objects):
                cw_center = crosswalk.polygon.centroid
                dx = cw_center.x - cx
                dy = cw_center.y - cy
                cw_distances[i] = dx * dx + dy * dy
            sorted_indices = np.argpartition(cw_distances, max_crosswalks)[:max_crosswalks]
            crosswalk_objects = [crosswalk_objects[i] for i in sorted_indices]

        # Global pre-definition and pre-allocation
        id_to_idx = {obj.id: idx for idx, obj in enumerate(lane_objects + crosswalk_objects)}

        num_lane = sum(1 for obj in lane_objects if isinstance(obj, Lane))
        num_crosswalks = len(crosswalk_objects)
        num_polygons = len(lane_objects) + num_crosswalks * 2

        LEFT_BOUNDARY_TYPE   = self._LEFT_BOUNDARY_TYPE
        RIGHT_BOUNDARY_TYPE  = self._RIGHT_BOUNDARY_TYPE
        POINT_TYPE_CENTER    = self._POINT_TYPE_CENTER
        SIDE_LEFT   = self._SIDE_LEFT
        SIDE_RIGHT  = self._SIDE_RIGHT
        SIDE_CENTER = self._SIDE_CENTER
        POLY_TYPE_LANE      = self._POLY_TYPE_LANE
        POLY_TYPE_CONNECTOR = self._POLY_TYPE_CONNECTOR
        POLY_TYPE_CROSSWALK = self._POLY_TYPE_CROSSWALK
        POLY_PRED  = self._POLY_PRED
        POLY_SUCC  = self._POLY_SUCC
        POLY_LEFT  = self._POLY_LEFT
        POLY_RIGHT = self._POLY_RIGHT

        polygon_position_np = np.zeros((num_polygons, self.dim), dtype=np.float32)
        polygon_orientation_np = np.zeros(num_polygons, dtype=np.float32)
        polygon_type_np = np.zeros(num_polygons, dtype=np.uint8)

        # Accumulated point positions and direction vectors across all lane segments
        global_point_pos = []
        global_point_vecs = []

        type_vals, type_lens = [], []
        side_vals, side_lens = [], []
        num_points_list = [0] * num_polygons

        max_connections_per_lane = 10
        edge_src, edge_dst, edge_types = [], [], []
        get_idx = id_to_idx.get
        dim = self.dim

        # Process lanes and lane connectors (polygon/point data + edges in one pass)
        for lane_segment in lane_objects:
            segment_idx = id_to_idx[lane_segment.id]
            is_lane = isinstance(lane_segment, Lane)
            centerline = lane_segment.baseline_path.discrete_path

            sampled_centerline = centerline[::8]
            center_pts = np.stack([p.array for p in sampled_centerline])

            polygon_position_np[segment_idx, :dim] = center_pts[0]
            polygon_orientation_np[segment_idx] = sampled_centerline[0].heading
            polygon_type_np[segment_idx] = POLY_TYPE_LANE if segment_idx < num_lane else POLY_TYPE_CONNECTOR

            left_coords = lane_segment.left_boundary.linestring.coords
            right_coords = lane_segment.right_boundary.linestring.coords

            left_pts = np.array(left_coords[::20], dtype=np.float32)
            right_pts = np.array(right_coords[::20], dtype=np.float32)

            if left_pts.shape[0] < 2:
                left_pts = np.array(left_coords[:2], dtype=np.float32)
            if right_pts.shape[0] < 2:
                right_pts = np.array(right_coords[:2], dtype=np.float32)

            left_vec = left_pts[1:] - left_pts[:-1]
            right_vec = right_pts[1:] - right_pts[:-1]
            center_vec = center_pts[1:] - center_pts[:-1]

            global_point_pos.extend([left_pts[:-1], right_pts[:-1], center_pts[:-1]])
            global_point_vecs.extend([left_vec, right_vec, center_vec])

            n_l, n_r, n_c = left_vec.shape[0], right_vec.shape[0], center_vec.shape[0]
            num_points_list[segment_idx] = n_l + n_r + n_c

            type_vals.extend([LEFT_BOUNDARY_TYPE, RIGHT_BOUNDARY_TYPE, POINT_TYPE_CENTER])
            type_lens.extend([n_l, n_r, n_c])
            side_vals.extend([SIDE_LEFT, SIDE_RIGHT, SIDE_CENTER])
            side_lens.extend([n_l, n_r, n_c])

            pred_inds = [get_idx(p.id) for p in lane_segment.incoming_edges]
            pred_inds = [i for i in pred_inds if i is not None]
            if len(pred_inds) > max_connections_per_lane:
                pred_inds = pred_inds[:max_connections_per_lane]
            n_pred = len(pred_inds)
            if n_pred:
                edge_src.extend(pred_inds)
                edge_dst.extend([segment_idx] * n_pred)
                edge_types.extend([POLY_PRED] * n_pred)

            succ_inds = [get_idx(s.id) for s in lane_segment.outgoing_edges]
            succ_inds = [i for i in succ_inds if i is not None]
            if len(succ_inds) > max_connections_per_lane:
                succ_inds = succ_inds[:max_connections_per_lane]
            n_succ = len(succ_inds)
            if n_succ:
                edge_src.extend(succ_inds)
                edge_dst.extend([segment_idx] * n_succ)
                edge_types.extend([POLY_SUCC] * n_succ)

            if is_lane:
                left_nb, right_nb = lane_segment.adjacent_edges
                if left_nb:
                    left_nb_idx = get_idx(left_nb.id)
                    if left_nb_idx is not None:
                        edge_src.append(left_nb_idx)
                        edge_dst.append(segment_idx)
                        edge_types.append(POLY_LEFT)
                if right_nb:
                    right_nb_idx = get_idx(right_nb.id)
                    if right_nb_idx is not None:
                        edge_src.append(right_nb_idx)
                        edge_dst.append(segment_idx)
                        edge_types.append(POLY_RIGHT)

        # Due to training computational resource constraints, a simplified crosswalk processing method is used here (only setting centroid position and type, without generating point-level features).
        # For full crosswalk processing needs left/right boundaries, centerline, bidirectional point-level features
        self._process_crosswalks_simple(
            crosswalk_objects, id_to_idx, num_crosswalks,
            POLY_TYPE_CROSSWALK, polygon_position_np, polygon_type_np)

        polygon_position = torch.as_tensor(polygon_position_np, device=device)
        polygon_orientation = torch.as_tensor(polygon_orientation_np, device=device)
        polygon_type = torch.as_tensor(polygon_type_np, device=device)

        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8, device=device)
        polygon_is_intersection[num_lane:] = 1

        num_points_tensor = torch.tensor(num_points_list, dtype=torch.long, device=device)
        total_points = num_points_tensor.sum().item()

        if total_points > 0:
            flat_point_pos_np = np.concatenate(global_point_pos, axis=0)
            flat_vecs_np = np.concatenate(global_point_vecs, axis=0)

            flat_point_pos = torch.as_tensor(flat_point_pos_np, device=device)
            flat_vecs = torch.as_tensor(flat_vecs_np, device=device)

            flat_point_orientation = torch.atan2(flat_vecs[:, 1], flat_vecs[:, 0])
            flat_point_magnitude = torch.norm(flat_vecs[:, :2], p=2, dim=-1)

            flat_point_type = torch.as_tensor(
                np.repeat(np.array(type_vals, dtype=np.uint8), np.array(type_lens, dtype=np.intp)),
                device=device,
            )
            flat_point_side = torch.as_tensor(
                np.repeat(np.array(side_vals, dtype=np.uint8), np.array(side_lens, dtype=np.intp)),
                device=device,
            )
        else:
            flat_point_pos = torch.empty((0, self.dim), dtype=torch.float32, device=device)
            flat_point_orientation = torch.empty(0, dtype=torch.float32, device=device)
            flat_point_magnitude = torch.empty(0, dtype=torch.float32, device=device)
            flat_point_type = torch.empty(0, dtype=torch.uint8, device=device)
            flat_point_side = torch.empty(0, dtype=torch.uint8, device=device)

        point_to_polygon_edge_index = torch.stack([
            torch.arange(total_points, dtype=torch.long, device=device),
            torch.arange(num_polygons, dtype=torch.long, device=device).repeat_interleave(num_points_tensor)
        ], dim=0)

        if edge_src:
            polygon_to_polygon_edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
            polygon_to_polygon_type = torch.tensor(edge_types, dtype=torch.uint8, device=device)

            max_edges = 100000
            if polygon_to_polygon_edge_index.shape[1] > max_edges:
                perm = torch.randperm(polygon_to_polygon_edge_index.shape[1], device=device)[:max_edges]
                polygon_to_polygon_edge_index = polygon_to_polygon_edge_index[:, perm]
                polygon_to_polygon_type = polygon_to_polygon_type[perm]
        else:
            polygon_to_polygon_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            polygon_to_polygon_type = torch.zeros(0, dtype=torch.uint8, device=device)

        # Assemble final dict
        map_data = {
            'map_polygon': {
                'num_nodes': num_polygons,
                'position': polygon_position,
                'orientation': polygon_orientation,
                'type': polygon_type,
                'is_intersection': polygon_is_intersection
            },
            'map_point': {
                'num_nodes': total_points,
                'position': flat_point_pos.float(),
                'orientation': flat_point_orientation.float(),
                'magnitude': flat_point_magnitude.float(),
                'type': flat_point_type,
                'side': flat_point_side
            },
            ('map_point', 'to', 'map_polygon'): {
                'edge_index': point_to_polygon_edge_index
            },
            ('map_polygon', 'to', 'map_polygon'): {
                'edge_index': polygon_to_polygon_edge_index,
                'type': polygon_to_polygon_type
            },
        }

        if self.dim == 3:
            polygon_height = torch.zeros(num_polygons, dtype=torch.float32, device=device)
            map_data['map_polygon']['height'] = polygon_height
            map_data['map_point']['height'] = torch.zeros(total_points, dtype=torch.float32, device=device) if total_points > 0 else torch.empty(0, dtype=torch.float32, device=device)

        return map_data
    
    # below are helper functions
    def _process_crosswalks_simple(self, crosswalk_objects, id_to_idx, num_crosswalks,
                                   poly_type_crosswalk, polygon_position_np, polygon_type_np):
        """
        Simplified crosswalk processing: only set polygon-level position (centroid)
        and type. No point-level features are generated.
        """
        if crosswalk_objects:
            cw_centroids = np.array(
                [[cw.polygon.centroid.x, cw.polygon.centroid.y] for cw in crosswalk_objects],
                dtype=np.float32,
            )
            fwd_indices = [id_to_idx[cw.id] for cw in crosswalk_objects]
            bwd_indices = [i + num_crosswalks for i in fwd_indices]
            polygon_position_np[fwd_indices, :2] = cw_centroids
            polygon_position_np[bwd_indices, :2] = cw_centroids
            polygon_type_np[fwd_indices] = poly_type_crosswalk
            polygon_type_np[bwd_indices] = poly_type_crosswalk
