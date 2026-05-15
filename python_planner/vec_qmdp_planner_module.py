# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vectorized QMDP planner for autonomous driving in NuPlan.

Combines QCNet-based multi-modal trajectory prediction with QMDP-based
decision-making to produce safe ego trajectories. Bridges the NuPlan
simulation interface with the C++ VecQMDPClosedPlanner backend.
"""

# Standard library
import gc
import heapq
import math
import warnings
from typing import List, Tuple, Type

# Third-party
import hydra
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

# NuPlan
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Local
import python_planner.vec_qmdp_closed_planner as vec_qmdp_planner
from python_planner.qcpredictor import (
    QCNET_AGENT_TYPES,
    QCNetPredictorInference,
    STATIC_AGENT_TYPES,
)
from python_planner.utils.agent_info import *
from python_planner.utils.map_utils_py import *
from python_planner.utils.math_utils import *
from python_planner.utils.route_utils import *


class VecQMDPPlanner(AbstractPlanner, QCNetPredictorInference):
    """
    Vectorized QMDP planner for autonomous driving in NuPlan.

    Combines QCNet-based trajectory prediction with QMDP-based planning to generate
    safe ego trajectories. At each iteration the planner:
      1. Updates ego and surrounding agent states from NuPlan observations.
      2. Runs multi-modal trajectory prediction (QCNet) for nearby agents.
      3. Solves the QMDP planning problem via the C++ VecQMDPClosedPlanner backend.
      4. Returns a smooth, interpolated ego trajectory.

    Inherits from:
        AbstractPlanner  — NuPlan planner interface (initialize / compute_planner_trajectory).
        QCNetPredictorInference — QCNet model loading and inference utilities.
    """

    def __init__(
        self,
        horizon_seconds: float,
        sampling_time: float,
        scenario,
        use_saved_predictions: bool = False,
        use_pedestrian_simple_prediction: bool = True,
        use_trajectory_optimization: bool = True,
    ):
        """
        Constructor for VecQMDPPlanner.

        Args:
            horizon_seconds: [s] time horizon being run.
            sampling_time: [s] sampling timestep.
            scenario: NuPlan scenario object.
            use_saved_predictions: If True, load predictions from disk instead of running the network.
            use_pedestrian_simple_prediction: If True, use constant-velocity linear prediction for pedestrians.
            use_trajectory_optimization: If True, resample trajectory positions using smoothed velocity.
        """
        
        # Fix seed for reproducible neural network outputs
        self.set_random_seed(seed=42)

        self._horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self._sampling_time = TimePoint(int(sampling_time * 1e6))
        self._map_radius = 60.0
        self._vehicle = get_pacifica_parameters()
        self._ego_agent = Agent()
        self._exo_agents = {}
        self._prev_exo_agents = {}  # Snapshot of exo agents from the previous frame
        self._scenario = scenario
        self._time_delta = 0.1  # seconds between consecutive frames
        self._max_obs_exo_num = 192

        # Prediction trajectory parameters
        self._max_num_modes = 6
        self._dummy_timesteps = 48
        self._result_timesteps = 41
        self._max_history_frames = 21

        # Prediction source flags
        self._use_saved_predictions = use_saved_predictions
        self._use_pedestrian_simple_prediction = use_pedestrian_simple_prediction
        self._use_trajectory_optimization = use_trajectory_optimization

        # Ego-vehicle origin at frame 0; all coordinates are stored relative to this
        self._offset_x = 0.0
        self._offset_y = 0.0

        # Per-agent history cache. Schema per token:
        # {
        #   'x':          [x0, x1, ...],
        #   'y':          [y0, y1, ...],
        #   'heading':    [h0, h1, ...],
        #   'velocity_x': [vx0, vx1, ...],
        #   'velocity_y': [vy0, vy1, ...],
        #   'timestamp':  [t0,  t1,  ...],
        #   'type':       int,
        #   'valid':      [bool, ...],  # False when agent has disappeared
        # }
        self._agents_history_cache = {}

    def __del__(self):
        """
        Destructor — releases QCNet model and GPU resources.
        """
        
        self.cleanup_resources()

    def cleanup_resources(self):
        """
        Explicitly release GPU tensors, model weights, and aligned buffers.
        """

        import sys

        try:
            if sys.meta_path is None:
                print("Python is shutting down, skipping cleanup")
                return

            # Release QCNet model (only loaded when not using saved predictions)
            if hasattr(self, '_predict_model') and self._predict_model is not None:
                print("Cleaning up QCNet model", flush=True)
                if hasattr(self._predict_model, 'cpu'):
                    self._predict_model.cpu()
                del self._predict_model
                self._predict_model = None

            # Release other GPU tensors and resources
            if hasattr(self, '_feature_builder'):
                self._feature_builder = None
            if hasattr(self, '_data_for_pred'):
                self._data_for_pred = None
            if hasattr(self, '_device'):
                self._device = None

            # Release pre-allocated aligned buffers
            if hasattr(self, 'pred_traj_x'):
                for attr_name in ['pred_traj_x', 'pred_traj_y', 'pred_traj_v', 'pred_traj_theta',
                                 'pred_traj_theta_cos', 'pred_traj_theta_sin', 'pred_pi']:
                    if hasattr(self, attr_name):
                        arr = getattr(self, attr_name)
                        if hasattr(arr, 'ctypes') and arr.ctypes.data in self._alignment_buffers:
                            del self._alignment_buffers[arr.ctypes.data]
                        setattr(self, attr_name, None)
            gc.collect()

            if not getattr(self, '_use_saved_predictions', True) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            print(f"Error during resource cleanup in VecQMDPPlanner: {e}", flush=True)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Initialize the planner with map API, route, and pre-allocated buffers.

        Args:
            initialization: PlannerInitialization object containing map API,
                            mission goal, and route roadblock IDs.
        """

        self._iteration = 0
        self._map_api = initialization.map_api
        self._map_utils = MapUtils(self._map_api, initialization.mission_goal, initialization.route_roadblock_ids, {}, {})
        self._planner = vec_qmdp_planner.VecQMDPClosedPlanner()

        # # For replaylog
        # self._planner.EnableLogging()
        # self._planner.SetSaveIteration(140)

        self.getQCNet()
        self._initialization = initialization

        # Pre-allocate prediction trajectory arrays for up to _max_obs_exo_num agents
        self.pred_traj_x = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_traj_y = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_traj_v = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_traj_theta = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_traj_theta_cos = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_traj_theta_sin = self.aligned_array((self._max_obs_exo_num, self._max_num_modes, self._dummy_timesteps), dtype=np.float32)
        self.pred_pi = self.aligned_array((self._max_obs_exo_num, self._max_num_modes), dtype=np.float32)

        # Pre-allocate speed smoothing buffer
        self._speeds_buf = np.zeros(
            (self._max_obs_exo_num, self._max_num_modes, self._result_timesteps), dtype=np.float64)

        # Pre-allocate heading/speed rollout buffers.
        # Note: theta here is vehicle heading, not steering angle; values are continuous (not discretized).
        self._buf_theta = np.zeros(
            (self._max_obs_exo_num, self._max_num_modes, self._result_timesteps), dtype=np.float64)
        self._buf_v = np.zeros(
            (self._max_obs_exo_num, self._max_num_modes, self._result_timesteps), dtype=np.float64)

        # Pre-compute constant index arrays to avoid per-frame recomputation
        self._orig_indices_pred = np.minimum(2 * np.arange(1, self._result_timesteps) - 1, 79).astype(np.int64)
        self._prev_indices_pred = (self._orig_indices_pred - 1).astype(np.int64)
        self._valid_target_indices_pred = np.arange(1, self._result_timesteps)
        self._time_deltas_pred = 0.2 * np.arange(1, self._result_timesteps)

        # GPU index tensors (lazy-initialized on first inference)
        self._orig_indices_gpu = None
        self._prev_indices_gpu = None

    def name(self) -> str:
        """
        Return the name of the planner.

        Returns:
            str: Class name of the planner
        """

        return self.__class__.__name__
    
    @property
    def requires_scenario(self) -> bool:
        """
        Indicate that this planner requires the scenario object at construction time.

        Returns:
            bool: Always True
        """
        return True

    def observation_type(self) -> Type[Observation]:
        """
        Return the observation type expected by the planner.

        Returns:
            Type[Observation]: DetectionsTracks
        """

        return DetectionsTracks  # type: ignore

    def update_ego_info(self, ego_state: EgoState) -> None:
        """
        Update ego vehicle state fields from the current EgoState.

        Args:
            ego_state: Current ego state containing pose, velocity, and acceleration
        """

        # The ego pose is in the global frame while velocities are in the local frame.
        agent = self._ego_agent
        dynamic_state = ego_state._dynamic_car_state
        center = ego_state.center
        point = center.point

        agent.timestamp = ego_state.time_point.time_us
        agent.X = point.x - self._offset_x
        agent.Y = point.y - self._offset_y
        agent.heading = center.heading
        agent.steering = ego_state.tire_steering_angle
        agent.steering_rate = dynamic_state.tire_steering_rate
        agent.yaw_rate = dynamic_state.angular_velocity

        velocity_vec = dynamic_state.rear_axle_velocity_2d
        agent.velocity = velocity_vec
        agent.speed = velocity_vec.magnitude()

        acc_vec = dynamic_state.rear_axle_acceleration_2d
        agent.acceleration = acc_vec
        agent.longitudinal_acc = acc_vec.x
        agent.lateral_acc = acc_vec.y

    def update_exo_info(self, ego_state: EgoState, observation: Observation, iteration: int) -> None:
        """
        Update information of other agents in the scene.

        Args:
            ego_state: Current ego state
            observation: Current observation containing tracked objects
            iteration: Current simulation iteration index
        """

        # Local variable & function binding
        exo_agents = self._exo_agents
        history_cache = self._agents_history_cache
        offset_x = self._offset_x
        offset_y = self._offset_y
        time_delta = self._time_delta
        plan_hz = 1.0 / time_delta
        max_obs_num = self._max_obs_exo_num
        max_hist_frames = self._max_history_frames
        ego_center = ego_state.center
        ego_x_center = ego_center.point.x
        ego_y_center = ego_center.point.y
        ego_x_offset = ego_x_center - offset_x
        ego_y_offset = ego_y_center - offset_y
        ego_heading = ego_center.heading
        ego_speed = self._ego_agent.speed
        cos, sin, atan2, sqrt = math.cos, math.sin, math.atan2, math.sqrt
        norm_angle = normalize_angle_scalar
        abs_val = abs

        PI = math.pi
        PI_6 = PI / 6.0
        PI_3 = PI / 3.0
        PI_2 = PI / 2.0
        TWO_PI_3 = 2.0 * PI / 3.0
        FIVE_PI_6 = 5.0 * PI / 6.0

        TYPE_EGO = TrackedObjectType.EGO
        TYPE_PED = int(TrackedObjectType.PEDESTRIAN)
        TYPE_BIKE = int(TrackedObjectType.BICYCLE)

        # Distance filtering (squared to avoid sqrt)
        max_dist_sq = max(self._map_radius, 8.0 * ego_speed) ** 2
        candidates = []
        cand_append = candidates.append

        for obj in observation.tracked_objects:
            if obj.tracked_object_type == TYPE_EGO:
                continue
            p = obj.center.point
            dx = p.x - ego_x_center
            dy = p.y - ego_y_center
            dist_sq = dx * dx + dy * dy
            if dist_sq <= max_dist_sq:
                cand_append((dist_sq, obj))

        # Keep nearest max_obs_num agents
        top_k_objects = heapq.nsmallest(max_obs_num, candidates)
        current_exo_agents = {}

        for _, obj in top_k_objects:
            token = obj.track_token

            # Retrieve existing agent or create new one
            if token in exo_agents:
                agent = exo_agents[token]
            else:
                agent = Agent()
                agent.token = token
                # Back-fill history from cache for early iterations
                if iteration < max_hist_frames and token in history_cache:
                    h_data = history_cache[token]
                    end_idx = max_hist_frames - iteration
                    hx = h_data['x']; hy = h_data['y']; hh = h_data['heading']
                    hvx = h_data['velocity_x']; hvy = h_data['velocity_y']
                    hts = h_data['timestamp']; hv = h_data['valid']
                    ax = agent._history_x; ay = agent._history_y; ah = agent._history_headings
                    avx = agent._history_velocity_x; avy = agent._history_velocity_y
                    ats = agent._history_timestamps; av = agent._history_valid
                    for _j in range(end_idx):
                        _k = iteration + _j
                        ax[_j] = hx[_k]; ay[_j] = hy[_k]; ah[_j] = hh[_k]
                        avx[_j] = hvx[_k]; avy[_j] = hvy[_k]
                        ats[_j] = hts[_k]; av[_j] = hv[_k]

            # Position & type
            obj_center = obj.center
            obj_type = obj.tracked_object_type
            agent.timestamp = obj.metadata.timestamp_us
            agent.type = int(obj_type) if obj_type in QCNET_AGENT_TYPES else 6
            px = obj_center.point.x - offset_x
            py = obj_center.point.y - offset_y
            agent._x = px
            agent._y = py

            # Bounding box corners (unrolled — always 4 corners)
            c0, c1, c2, c3 = obj.box.all_corners()
            if agent._coords is None:
                agent._coords = [Point2D(0.0, 0.0), Point2D(0.0, 0.0),
                                  Point2D(0.0, 0.0), Point2D(0.0, 0.0)]
            coords = agent._coords
            coords[0].x = c0.x - offset_x; coords[0].y = c0.y - offset_y
            coords[1].x = c1.x - offset_x; coords[1].y = c1.y - offset_y
            coords[2].x = c2.x - offset_x; coords[2].y = c2.y - offset_y
            coords[3].x = c3.x - offset_x; coords[3].y = c3.y - offset_y

            # Kinematics
            heading = obj_center.heading
            if obj_type in AGENT_TYPES:
                velocity = obj.velocity
                vx, vy = velocity.x, velocity.y
                if vx == 0 and vy == 0:
                    track_heading = heading
                else:
                    vel_angle = atan2(vy, vx)
                    track_heading = heading if abs_val(norm_angle(heading - vel_angle)) < TWO_PI_3 else norm_angle(heading + PI)

                final_speed = sqrt(vx * vx + vy * vy)
                h_valid = agent._history_valid
                if h_valid and h_valid[-1]:
                    dx_h = px - agent._history_x[-1]
                    dy_h = py - agent._history_y[-1]
                    calc_speed = sqrt(dx_h * dx_h + dy_h * dy_h) * plan_hz
                    # Fuse: prefer position-derived speed when diff is in [1.0, 2.0) m/s
                    if calc_speed > 0.1:
                        speed_diff = abs_val(calc_speed - final_speed)
                        if 1.0 < speed_diff < 2.0:
                            final_speed = calc_speed
                agent.dynamic = True
            else:
                track_heading = norm_angle(heading + PI_2)
                final_speed = 0.0
                agent.dynamic = False

            cos_h = cos(track_heading)
            sin_h = sin(track_heading)
            agent._velocity_x = final_speed * cos_h
            agent._velocity_y = final_speed * sin_h
            agent.speed = final_speed
            agent._heading = track_heading

            agent._acceleration_x, agent._acceleration_y = 0.0, 0.0

            # Relative heading & position w.r.t. ego
            rel_heading = abs_val(norm_angle(track_heading - ego_heading))
            if rel_heading < PI_3:
                agent.heading_wrt_ego = RelativeHeading.PARALLEL
            elif rel_heading < TWO_PI_3:
                agent.heading_wrt_ego = RelativeHeading.NORMAL
            else:
                agent.heading_wrt_ego = RelativeHeading.INVERSE

            rel_pos_angle = abs_val(norm_angle(atan2(py - ego_y_offset, px - ego_x_offset) - ego_heading))
            if rel_pos_angle < PI_6:
                agent.position_wrt_ego = RelativePosition.FRONT
            elif rel_pos_angle < FIVE_PI_6:
                agent.position_wrt_ego = RelativePosition.LATERAL
            else:
                agent.position_wrt_ego = RelativePosition.BEHIND

            # Extent & safety margins
            agent.calculate_extent(cos_h, sin_h)

            a_type = agent.type
            a_speed = agent.speed

            if a_type == TYPE_PED:
                margin_x = margin_y = 0.5 if a_speed > 1.5 else (0.3 if a_speed > 1.0 else 0.2)
            elif a_type == TYPE_BIKE or a_type in STATIC_AGENT_TYPES:
                margin_x, margin_y = 0.2, 0.2
            elif rel_heading >= TWO_PI_3:
                margin_x, margin_y = 0.3, 0.3
            elif a_speed <= 2.0:
                margin_x, margin_y = (0.5, 0.3) if (rel_heading < PI_6 and a_speed >= 0.5) else (0.3, 0.3)
            else:
                margin_x, margin_y = (0.4, 0.4) if agent.is_a_truck() else (0.3, 0.3)

            agent.enlarge_extent(margin_x, margin_y)
            agent.add_to_history()
            current_exo_agents[token] = agent

        self._exo_agents = current_exo_agents

    def compute_planner_trajectory(self, current_input: PlannerInput) -> Tuple[AbstractTrajectory, List, AbstractTrajectory]:
        """
        Compute the planned trajectory for the current iteration.

        Args:
            current_input: Current planning input containing ego state, observations, and traffic data.

        Returns:
            InterpolatedTrajectory: Planned ego trajectory at 0.1s intervals.
        """

        print(f"--------------Iteration: {current_input.iteration.index} {self._scenario.token}---------------------", flush=True)

        # Extract current state
        history = current_input.history
        ego_state, observation = history.current_state

        # Initialize offset from ego origin on the first frame
        if current_input.iteration.index == 0:
            self._offset_x = ego_state.center.point.x
            self._offset_y = ego_state.center.point.y
            self._map_utils.UpdateOffset(self._offset_x, self._offset_y)
            self._initialize_agents_history_cache(current_input)
            
        self.update_ego_info(ego_state)
        self.update_exo_info(ego_state, observation, current_input.iteration.index)

        # make prediction
        if not self._use_saved_predictions:
            valid_agent_token = self._feature_builder._get_valid_agent_tokens(self._exo_agents)
            if len(valid_agent_token) > 0:
                self._data_for_pred["agent"] = self._feature_builder._build_agent_feature_from_simulation_inference(self._exo_agents, self._offset_x, self._offset_y, valid_agent_token)
                self._data_for_pred.update(self._feature_builder._build_map_feature_infer(ego_state.center, self._map_api, self._device)) # map features do not need to be rebuilt every frame

        valid_agent_token = self._get_valid_agent_tokens() if self._use_saved_predictions else valid_agent_token
        if len(valid_agent_token) > 0:
            (agents_token, num_agents) = self.make_prediction(current_input.iteration.index)

        # Update traffic light status
        self._map_utils.UpdateTrafficStatus(current_input.traffic_light_data)
        
        # On the first frame, repair route (start, goal, intermediate) and load route dicts
        if current_input.iteration.index == 0:
            route_roadblock_ids = self._map_utils.RepairRouteStart(self._map_utils._roadblock_ids, ego_state)
            route_roadblock_ids = self._map_utils.RepairRouteGoal(route_roadblock_ids)
            route_roadblock_ids = self._map_utils.RepairRouteIntermediate(route_roadblock_ids)
            self._map_utils.LoadRouteDicts(route_roadblock_ids)
            self._map_utils.UpdateGoalMissNum()


        # Call the planner to generate a trajectory
        planned_trajectory, _, tracked_py_trajectory, tracked_cpp_trajectory = self._planner.MakePlanning(
            self._ego_agent,
            self._exo_agents,
            self._map_utils,
            self.pred_traj_x,
            self.pred_traj_y,
            self.pred_traj_v,
            self.pred_traj_theta,
            self.pred_traj_theta_cos,
            self.pred_traj_theta_sin,
            self.pred_pi,
            current_input.iteration.index,
            self._offset_x,
            self._offset_y,
            self._scenario.token,
        )

        # Convert planner output to EgoState sequence and wrap as interpolated trajectory
        start_timestamp = self._ego_agent.timestamp
        trajectory_points = self._build_trajectory(planned_trajectory, start_timestamp)

        return InterpolatedTrajectory(trajectory_points)

    def _build_trajectory(self, trajectory, start_timestamp):
        """
        Convert a raw trajectory list into a list of EgoState objects.

        Args:
            trajectory: List of [x, y, heading, v, a] points in the local (offset) frame
            start_timestamp: Start timestamp in microseconds

        Returns:
            List[EgoState]: EgoState sequence at 0.1s intervals
        """

        if not trajectory:
            return []

        _build_state = EgoState.build_from_rear_axle
        _SE2 = StateSE2
        _Vec2 = StateVector2D
        _Time = TimePoint
        _cos = math.cos
        _sin = math.sin
        _vehicle = self._vehicle
        _rac = _vehicle.rear_axle_to_center
        _ox = self._offset_x
        _oy = self._offset_y
        _DT_US = 100000  # 0.1s in microseconds

        count = len(trajectory)
        result = [None] * count

        for i in range(count):
            pt = trajectory[i]
            h = pt[2]
            x = pt[0] - _rac * _cos(h) + _ox
            y = pt[1] - _rac * _sin(h) + _oy

            result[i] = _build_state(
                rear_axle_pose=_SE2(x, y, h),
                rear_axle_velocity_2d=_Vec2(pt[3], 0),
                rear_axle_acceleration_2d=_Vec2(pt[4], 0),
                tire_steering_angle=0.0,
                time_point=_Time(start_timestamp + i * _DT_US),
                vehicle_parameters=_vehicle,
                angular_vel=0.0,
            )

        return result

    def _initialize_agents_history_cache(self, current_input):
        """
        Pre-populate the history cache from the initial observation buffer.

        Args:
            current_input: PlannerInput at iteration 0, containing historical observations
        """
        history_observations = current_input.history.observations[-22:-1]
        self._max_history_frames = len(history_observations)

        agent_history_frames_count = {}

        for obs_idx, hist_obs in enumerate(history_observations):
            for hist_obj in hist_obs.tracked_objects:

                if hist_obj.tracked_object_type == TrackedObjectType.EGO:
                    continue

                token = hist_obj.track_token

                if token not in self._agents_history_cache:
                    self._agents_history_cache[token] = {
                        'x': [0.0] * self._max_history_frames,
                        'y': [0.0] * self._max_history_frames,
                        'heading': [0.0] * self._max_history_frames,
                        'velocity_x': [0.0] * self._max_history_frames,
                        'velocity_y': [0.0] * self._max_history_frames,
                        'timestamp': [0] * self._max_history_frames,
                        'valid': [False] * self._max_history_frames,
                        'type': int(hist_obj.tracked_object_type) if hist_obj.tracked_object_type in QCNET_AGENT_TYPES else 6 # 6: GENERIC_OBJECT
                    }
                    agent_history_frames_count[token] = 0

                hist_x = hist_obj.center.point.x - self._offset_x
                hist_y = hist_obj.center.point.y - self._offset_y
                hist_heading = hist_obj.center.heading
                hist_velocity_x = hist_obj.velocity.x
                hist_velocity_y = hist_obj.velocity.y

                frame_idx = obs_idx
                self._agents_history_cache[token]['x'][frame_idx] = hist_x
                self._agents_history_cache[token]['y'][frame_idx] = hist_y
                self._agents_history_cache[token]['heading'][frame_idx] = hist_heading
                self._agents_history_cache[token]['velocity_x'][frame_idx] = hist_velocity_x
                self._agents_history_cache[token]['velocity_y'][frame_idx] = hist_velocity_y
                self._agents_history_cache[token]['timestamp'][frame_idx] = hist_obj.metadata.timestamp_us
                self._agents_history_cache[token]['valid'][frame_idx] = True

                agent_history_frames_count[token] += 1
                        