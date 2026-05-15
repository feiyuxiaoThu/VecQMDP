# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import deque
from python_planner.utils.route_utils import *
from python_planner.utils.math_utils import *
from python_planner.utils.map_utils_py import *
from nuplan.common.actor_state.state_representation import StateVector2D
from enum import Enum


class RelativeHeading(Enum):
    PARALLEL = 1
    NORMAL = 2
    INVERSE = 3


class RelativePosition(Enum):
    FRONT = 1
    LATERAL = 2
    BEHIND = 3


class Agent:
    """Represents a traffic participant (vehicle, pedestrian, cyclist, etc.)
    with full kinematic state, bounding-box geometry, and fixed-length history buffers."""
    __slots__ = (
        '_token', '_out_of_region', '_timestamp', '_type',
        '_x', '_y', '_heading', '_yaw_rate',
        '_steering', '_steering_rate',
        '_velocity_x', '_velocity_y', '_speed',
        '_acceleration_x', '_acceleration_y',
        '_coords', '_dynamic',
        '_heading_wrt_ego', '_position_wrt_ego', '_roadblock_id',
        '_bb_extent_x', '_bb_extent_y',
        '_extentX', '_extentY', '_extentX_expanded', '_extentY_expanded',
        '_longitudinal_acc', '_lateral_acc',
        '_predictions', '_prediction_probabilities', '_predicted_trajectory',
        '_max_history_length',
        '_history_x', '_history_y', '_history_headings',
        '_history_velocity_x', '_history_velocity_y',
        '_history_timestamps', '_history_valid',
    )

    def __init__(self):
        """
        Initialize an Agent with all kinematic fields set to zero/None
        and fixed-length history buffers pre-allocated.
        """
        self._token = "none"
        self._out_of_region = False
        self._timestamp = 0
        self._type = 3
        self._x = 0.0
        self._y = 0.0
        self._heading = 0
        self._yaw_rate = 0
        self._steering = 0
        self._steering_rate = 0
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._speed = 0
        self._acceleration_x = 0.0
        self._acceleration_y = 0.0
        self._coords = None
        self._dynamic = False
        self._heading_wrt_ego = None
        self._position_wrt_ego = None
        self._roadblock_id = None
        self._bb_extent_x = 0
        self._bb_extent_y = 0
        self._longitudinal_acc = 0.0
        self._lateral_acc = 0.0
        self._extentX = 0.0
        self._extentY = 0.0
        self._extentX_expanded = 0.0
        self._extentY_expanded = 0.0
        self._predictions = None
        self._prediction_probabilities = None
        self._predicted_trajectory = None

        # initialize history storage
        self._max_history_length = 21  # max history length (2 s at 0.1 s step) + current frame

        # initialize fixed-length history deques (O(1) append + auto-trim)
        _ml = self._max_history_length
        self._history_x = deque([0.0] * _ml, maxlen=_ml)
        self._history_y = deque([0.0] * _ml, maxlen=_ml)
        self._history_headings = deque([0.0] * _ml, maxlen=_ml)
        self._history_velocity_x = deque([0.0] * _ml, maxlen=_ml)
        self._history_velocity_y = deque([0.0] * _ml, maxlen=_ml)
        self._history_timestamps = deque([0] * _ml, maxlen=_ml)
        self._history_valid = deque([False] * _ml, maxlen=_ml)

    # basic attributes
    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, value: str):
        self._token = value

    @property
    def out_of_region(self) -> bool:
        return self._out_of_region

    @out_of_region.setter
    def out_of_region(self, value: bool):
        self._out_of_region = value

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int):
        self._timestamp = value

    @property
    def type(self) -> int:
        return self._type

    @type.setter
    def type(self, value: int):
        self._type = value

    # position attributes
    @property
    def position(self) -> Point2D:
        """Get position (for compatibility)."""
        return Point2D(self._x, self._y)

    @position.setter
    def position(self, value: Point2D):
        """Set position."""
        self._x = value.x
        self._y = value.y

    @property
    def X(self) -> float:
        return self._x

    @X.setter
    def X(self, value: float):
        self._x = value

    @property
    def Y(self) -> float:
        return self._y

    @Y.setter
    def Y(self, value: float):
        self._y = value

    # heading attributes
    @property
    def heading(self) -> float:
        return self._heading

    @heading.setter
    def heading(self, value: float):
        self._heading = value

    @property
    def yaw_rate(self) -> float:
        return self._yaw_rate

    @yaw_rate.setter
    def yaw_rate(self, value: float):
        self._yaw_rate = value

    # steering attributes
    @property
    def steering(self) -> float:
        return self._steering

    @steering.setter
    def steering(self, value: float):
        self._steering = value

    @property
    def steering_rate(self) -> float:
        return self._steering_rate

    @steering_rate.setter
    def steering_rate(self, value: float):
        self._steering_rate = value

    # velocity attributes
    @property
    def velocity(self) -> StateVector2D:
        return StateVector2D(self._velocity_x, self._velocity_y)

    @velocity.setter
    def velocity(self, value: StateVector2D):
        self._velocity_x = value.x
        self._velocity_y = value.y

    @property
    def velocityX(self) -> float:
        return self._velocity_x

    @property
    def velocityY(self) -> float:
        return self._velocity_y

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = value

    # acceleration attributes
    @property
    def acceleration(self) -> StateVector2D:
        return StateVector2D(self._acceleration_x, self._acceleration_y)

    @acceleration.setter
    def acceleration(self, value: StateVector2D):
        self._acceleration_x = value.x
        self._acceleration_y = value.y

    @property
    def accX(self) -> float:
        return self._acceleration_x

    @property
    def accY(self) -> float:
        return self._acceleration_y

    # longitudinal / lateral acceleration attributes
    @property
    def longitudinal_acc(self) -> float:
        return self._longitudinal_acc

    @longitudinal_acc.setter
    def longitudinal_acc(self, value: float):
        self._longitudinal_acc = value

    @property
    def lateral_acc(self) -> float:
        return self._lateral_acc

    @lateral_acc.setter
    def lateral_acc(self, value: float):
        self._lateral_acc = value

    # corner coordinates attribute
    @property
    def coords(self) -> List[Point2D]:
        return self._coords

    @coords.setter
    def coords(self, value: List[Point2D]):
        self._coords = value

    # extent attributes
    @property
    def extentX(self) -> float:
        return self._extentX

    @extentX.setter
    def extentX(self, value: float):
        self._extentX = value

    @property
    def extentY(self) -> float:
        return self._extentY

    @extentY.setter
    def extentY(self, value: float):
        self._extentY = value

    @property
    def extentX_expanded(self) -> float:
        return self._extentX_expanded

    @extentX_expanded.setter
    def extentX_expanded(self, value: float):
        self._extentX_expanded = value

    @property
    def extentY_expanded(self) -> float:
        return self._extentY_expanded

    @extentY_expanded.setter
    def extentY_expanded(self, value: float):
        self._extentY_expanded = value

    def calculate_extent(self, cos_h=None, sin_h=None):
        """
        Compute half-extents of the bounding box in the agent's local frame.

        Projects each corner coordinate onto the agent's longitudinal and lateral
        axes and stores the maximum projection as extentX (lateral half-width)
        and extentY (longitudinal half-length).
        """
        x, y = self._x, self._y
        c = cos_h if cos_h is not None else math.cos(self._heading)
        s = sin_h if sin_h is not None else math.sin(self._heading)
        p0, p1, p2, p3 = self._coords
        dx0 = p0.x - x; dy0 = p0.y - y
        dx1 = p1.x - x; dy1 = p1.y - y
        dx2 = p2.x - x; dy2 = p2.y - y
        dx3 = p3.x - x; dy3 = p3.y - y
        self._extentX = max(dy0 * c - dx0 * s, dy1 * c - dx1 * s, dy2 * c - dx2 * s, dy3 * c - dx3 * s)
        self._extentY = max(dx0 * c + dy0 * s, dx1 * c + dy1 * s, dx2 * c + dy2 * s, dx3 * c + dy3 * s)

    def enlarge_extent(self, offset_x, offset_y):
        """
        Expand the bounding-box half-extents by fixed offsets.

        Args:
            offset_x: additional lateral margin to add to extentX.
            offset_y: additional longitudinal margin to add to extentY.
        """
        self._extentX_expanded = self._extentX + offset_x
        self._extentY_expanded = self._extentY + offset_y

    # dynamic / relational attributes
    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, value: bool):
        self._dynamic = value

    @property
    def heading_wrt_ego(self) -> RelativeHeading:
        return self._heading_wrt_ego

    @heading_wrt_ego.setter
    def heading_wrt_ego(self, value: RelativeHeading):
        self._heading_wrt_ego = value

    @property
    def position_wrt_ego(self) -> RelativePosition:
        return self._position_wrt_ego

    @position_wrt_ego.setter
    def position_wrt_ego(self, value: RelativePosition):
        self._position_wrt_ego = value

    @property
    def roadblock_id(self) -> str:
        return self._roadblock_id

    @roadblock_id.setter
    def roadblock_id(self, value: str):
        self._roadblock_id = value

    # prediction attributes
    @property
    def predictions(self):
        """Get trajectory predictions for all modes."""
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        """Set trajectory predictions for all modes."""
        self._predictions = value

    @property
    def prediction_probabilities(self):
        """Get probabilities for each prediction mode."""
        return self._prediction_probabilities

    @prediction_probabilities.setter
    def prediction_probabilities(self, value):
        """Set probabilities for each prediction mode."""
        self._prediction_probabilities = value

    @property
    def prediction(self):
        """Get the most likely trajectory prediction."""
        return self._predicted_trajectory

    @prediction.setter
    def prediction(self, value):
        """Set the most likely trajectory prediction."""
        self._predicted_trajectory = value

    def set_all_predictions(self, trajectories, probabilities):
        """Set all trajectory predictions with their probabilities and auto-select the most likely one.

        Args:
            trajectories: trajectory predictions for all modes [num_modes, num_timesteps, 2]
            probabilities: probability per mode [num_modes]
        """
        self._predictions = trajectories
        self._prediction_probabilities = probabilities
        if trajectories is not None and probabilities is not None:
            import numpy as np
            best_mode = np.argmax(probabilities)
            self._predicted_trajectory = trajectories[best_mode]

    # history properties
    @property
    def history_positions(self) -> List[Point2D]:
        """Get list of historical positions (for compatibility)."""
        return [Point2D(x, y) for x, y in zip(self._history_x, self._history_y)]

    @property
    def history_headings(self) -> List[float]:
        """Get list of historical headings."""
        return self._history_headings

    @property
    def history_velocities(self) -> List[StateVector2D]:
        """Get list of historical velocities (for compatibility)."""
        return [StateVector2D(vx, vy) for vx, vy in zip(self._history_velocity_x, self._history_velocity_y)]

    @property
    def history_timestamps(self) -> List[int]:
        """Get list of historical timestamps."""
        return self._history_timestamps

    @property
    def history_valid(self) -> List[bool]:
        """Get list of historical validity flags."""
        return self._history_valid

    def set_history_at_index(self, index, x, y, heading, velocity_x, velocity_y, timestamp, valid=True):
        """Set history values at the given index.

        Args:
            index: history index
            x, y: position coordinates
            heading: heading angle
            velocity_x, velocity_y: velocity components
            timestamp: timestamp
            valid: whether the entry is valid
        """
        if 0 <= index < self._max_history_length:
            self._history_x[index] = x
            self._history_y[index] = y
            self._history_headings[index] = heading
            self._history_velocity_x[index] = velocity_x
            self._history_velocity_y[index] = velocity_y
            self._history_timestamps[index] = timestamp
            self._history_valid[index] = valid

    def add_to_history(self):
        """
        Append the current kinematic state to the history buffers.

        If the buffer exceeds max_history_length the oldest entry is dropped,
        keeping the buffers at a fixed size.
        """
        self._history_x.append(self._x)
        self._history_y.append(self._y)
        self._history_headings.append(self._heading)
        self._history_velocity_x.append(self._velocity_x)
        self._history_velocity_y.append(self._velocity_y)
        self._history_timestamps.append(self._timestamp)
        self._history_valid.append(True)

    def clear_history(self):
        _ml = self._max_history_length
        self._history_x = deque([0.0] * _ml, maxlen=_ml)
        self._history_y = deque([0.0] * _ml, maxlen=_ml)
        self._history_headings = deque([0.0] * _ml, maxlen=_ml)
        self._history_velocity_x = deque([0.0] * _ml, maxlen=_ml)
        self._history_velocity_y = deque([0.0] * _ml, maxlen=_ml)
        self._history_timestamps = deque([0] * _ml, maxlen=_ml)
        self._history_valid = deque([False] * _ml, maxlen=_ml)

    def is_a_truck(self) -> bool:
        """
        Determine whether this agent is a large truck based on bounding-box size.

        Returns:
            bool: True if extentX > 1.3 m and extentY > 4.0 m.
        """
        return self._extentX > 1.3 and self._extentY > 4.0
