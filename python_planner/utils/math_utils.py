# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from typing import Union

_PI = math.pi
_2_PI = 2 * math.pi


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Map an angle to the range [-π, π].

    Args:
        angle: input angle in radians; scalar float or numpy array.

    Returns:
        Union[float, np.ndarray]: angle wrapped to [-π, π].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_angle_scalar(angle: float) -> float:
    """
    Scalar-only version of normalize_angle using math.fmod (faster for single floats).

    Args:
        angle: input angle in radians.

    Returns:
        float: angle wrapped to [-π, π].
    """
    return (angle + _PI) % _2_PI - _PI

def close_angle(angle1: float, angle2: float, threshold: float) -> bool:
    """
    Check whether two angles are within a given angular distance of each other.

    Args:
        angle1: first angle in radians.
        angle2: second angle in radians.
        threshold: maximum allowed angular difference in radians.

    Returns:
        bool: True if |normalize(angle1 - angle2)| < threshold.
    """
    angle_difference = normalize_angle(angle1 - angle2)  # in [-pi, pi)
    return abs(angle_difference) < threshold

def InFront(abs_normalized_angle: float, threshold: float = 45 / 180 * np.pi) -> bool:
    """
    Check whether a pre-computed absolute normalized angle falls within the forward cone.

    Args:
        abs_normalized_angle: absolute value of the heading-relative angle in radians.
        threshold: half-angle of the forward cone in radians (default 45°).

    Returns:
        bool: True if abs_normalized_angle < threshold.
    """
    return abs_normalized_angle < threshold

def InBehind(abs_normalized_angle: float, threshold: float = 45 / 180 * np.pi) -> bool:
    """
    Check whether a pre-computed absolute normalized angle falls outside the forward cone.

    Args:
        abs_normalized_angle: absolute value of the heading-relative angle in radians.
        threshold: half-angle of the forward cone in radians (default 45°).

    Returns:
        bool: True if abs_normalized_angle > threshold.
    """
    return abs_normalized_angle > threshold


def smooth_heading_batch(headings: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Vectorized version of smooth_heading_sequence supporting arbitrary leading dimensions.

    Applies selective jitter filtering along the last axis (time axis): a time step
    is corrected only when its angular difference from both neighbours exceeds threshold.

    Args:
        headings: angle array of shape [..., T], where the last dimension is time.
        threshold: angular difference threshold in radians.

    Returns:
        Smoothed angle array with the same shape.
    """
    if headings.shape[-1] <= 2:
        return headings.copy()

    smoothed = headings.copy()
    prev = smoothed[..., :-2]      # [..., T-2]  (indices 0..T-3)
    curr = headings[..., 1:-1]     # [..., T-2]  (indices 1..T-2)
    nxt = headings[..., 2:]        # [..., T-2]  (indices 2..T-1)

    # wrap-aware angular differences
    diff_prev = (curr - prev + _PI) % _2_PI - _PI
    diff_next = (curr - nxt + _PI) % _2_PI - _PI

    # mask for isolated jitter spikes (both neighbours exceed threshold)
    jitter_mask = (np.abs(diff_prev) > threshold) & (np.abs(diff_next) > threshold)

    if np.any(jitter_mask):
        # compute average of the two neighbours
        sin_avg = (np.sin(prev) + np.sin(nxt)) * 0.5
        cos_avg = (np.cos(prev) + np.cos(nxt)) * 0.5
        avg = np.arctan2(sin_avg, cos_avg)
        smoothed[..., 1:-1] = np.where(jitter_mask, avg, curr)

    # normalize to [-π, π]
    smoothed = (smoothed + _PI) % _2_PI - _PI
    return smoothed
