# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
QCNet-based trajectory prediction inference, including model loading,
multi-modal trajectory generation, pedestrian velocity adjustment,
and prediction data persistence.
"""
import os
import math
import pickle
import ctypes
import random
from typing import List, Dict, Tuple, Set

import numpy as np
from scipy.ndimage import uniform_filter1d

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from python_planner.qcpredictor.qcnet import QCNet
from python_planner.qcpredictor.feature_builders.qcnet_feature_builder import QCNetFeatureBuilder
from python_planner.utils.math_utils import *
from python_planner.utils.agent_info import *

# Constants
QCNET_AGENT_TYPES: Set[TrackedObjectType] = {
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.BARRIER,
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.GENERIC_OBJECT,
}

STATIC_AGENT_TYPES: Set[int] = {
    int(TrackedObjectType.CZONE_SIGN),
    int(TrackedObjectType.BARRIER),
    int(TrackedObjectType.TRAFFIC_CONE),
    int(TrackedObjectType.GENERIC_OBJECT),
}

home_path = os.path.expanduser("~")
MODEL_CHECKPOINT_PATH = os.path.join(
    home_path, 
    "VecQMDP/python_planner/qcpredictor/weights/qcnet.ckpt"
)

# Prediction base class — provides all prediction methods for VecQMDPPlanner
class QCNetPredictorInference:
    """
    Base class that provides trajectory-prediction methods for VecQMDPPlanner.

    Provides QCNet model loading, multi-modal trajectory inference,
    pedestrian velocity adjustment, and prediction data save/load.
    Inherited by VecQMDPPlanner via multiple inheritance.
    """

    _alignment_buffers = {}
    _seed_set = False

    @staticmethod
    def set_random_seed(seed: int = 42):
        """
        Set random seed for reproducibility across all backends.

        Args:
            seed: Random seed value, default is 42
        """

        if QCNetPredictorInference._seed_set:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        QCNetPredictorInference._seed_set = True
        print(f"Random seed set to: {seed}", flush=True)

    @classmethod
    def aligned_array(cls, shape, dtype=np.float32, alignment=32):
        """
        Allocate a NumPy array backed by aligned memory via ctypes.

        Args:
            shape: Shape of the array
            dtype: Data type, default is np.float32
            alignment: Memory alignment in bytes, default is 32

        Returns:
            np.ndarray: Aligned NumPy array
        """

        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize

        raw = ctypes.create_string_buffer(nbytes + alignment)
        start = ctypes.addressof(raw)
        offset = (alignment - (start % alignment)) % alignment
        ptr = start + offset

        buf_type = ctypes.c_char * nbytes
        buf = buf_type.from_address(ptr)

        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)

        arr_id = arr.ctypes.data
        cls._alignment_buffers[arr_id] = raw

        return arr

    @staticmethod
    def _get_ego_xy(ego_agent):
        """
        Get the x, y coordinates of the ego vehicle

        Args:
            ego_agent: ego agent object

        Returns:
            Tuple[float, float]: (ego_x, ego_y)
        """

        return ego_agent.X, ego_agent.Y

    @staticmethod
    def _adjust_pedestrian_velocity(v0, x0, y0, ego_agent, pedestrian_agent, print_=False):
        """
        Adjust pedestrian prediction speed based on crossing direction relative to ego.

        Rules:
        1. Crossing in front + already past the far side (>6m lateral) -> keep speed
        2. Crossing in front + in transition zone (2-6m lateral) -> moderately increase speed
        3. Otherwise -> reduce speed

        Args:
            v0: Initial predicted speed of the pedestrian
            x0: Pedestrian x position in world frame
            y0: Pedestrian y position in world frame
            ego_agent: Ego agent object
            pedestrian_agent: Pedestrian agent object
            print_: Enable debug printing, default is False

        Returns:
            float: Adjusted pedestrian speed
        """

        if v0 < 0.2:
            return v0

        ego_x, ego_y = QCNetPredictorInference._get_ego_xy(ego_agent)
        ego_heading = ego_agent.heading

        sin_h = math.sin(ego_heading)
        cos_h = math.cos(ego_heading)

        dx = x0 - ego_x
        dy = y0 - ego_y

        # Lateral offset in ego frame (left positive)
        lateral_left_pos = -dx * sin_h + dy * cos_h

        ped_vx = pedestrian_agent._velocity_x
        ped_vy = pedestrian_agent._velocity_y
        ped_v_lateral = -ped_vx * sin_h + ped_vy * cos_h

        if ped_v_lateral < -0.1:  # moving_left_to_right
            if lateral_left_pos > 6.0:
                return v0
            elif lateral_left_pos > 2.0:
                return 2.5 if 1.5 * v0 > 2.5 else 1.5 * v0
            else:
                return 0.8 * v0

        elif ped_v_lateral > 0.1:  # moving_right_to_left
            if lateral_left_pos < -6.0:
                return v0
            elif lateral_left_pos < -2.0:
                return 2.5 if 1.5 * v0 > 2.5 else 1.5 * v0
            else:
                return 0.8 * v0

        return v0

    def getQCNet(self):
        """
        Load and initialize the QCNet prediction model onto the GPU.

        If _use_saved_predictions is True, model loading is skipped and
        all model-related attributes are set to None.
        """

        if not self._use_saved_predictions:
            self._predict_model = QCNet.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
            self._predict_model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._predict_model = self._predict_model.to(self._device)
            self._feature_builder = QCNetFeatureBuilder()
            self._data_for_pred = dict()
            self._data_for_pred["scenario_id"] = self._scenario.token
            self._data_for_pred["scenario_name"] = self._scenario.token

            # Warm up CUDA context: force cuBLAS/cuDNN to load and pre-allocate memory pool
            if self._device.type == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    _dummy_a = torch.randn(2000, 2000, device=self._device)
                    _dummy_b = torch.randn(2000, 2000, device=self._device)
                    _dummy_c = torch.matmul(_dummy_a, _dummy_b)

                    # Block until GPU is fully initialized
                    torch.cuda.synchronize(self._device)
        else:
            print("Using saved predictions, skipping QCNet model loading", flush=True)
            self._predict_model = None
            self._feature_builder = None
            self._data_for_pred = None
            self._device = None

    def _get_valid_agent_tokens(self):
        """
        Return valid agent tokens for the saved-prediction path.

        Returns:
            List[str]: Tokens of agents that pass the QCNetFeatureBuilder validity filter
        """

        temp_feature_builder = QCNetFeatureBuilder()
        return temp_feature_builder._get_valid_agent_tokens(self._exo_agents)

    def make_prediction(self, iteration_index: int):
        """
        Dispatch prediction to the network or saved-data path depending on configuration.

        Args:
            iteration_index: Current simulation iteration index

        Returns:
            Tuple[List[str], int]: (agent token list, number of active agents)
        """

        if self._use_saved_predictions:
            return self._make_prediction_from_saved_data(iteration_index)
        else:
            return self._make_prediction_from_network(iteration_index)

    def _make_prediction_from_network(self, iteration_index: int):
        """
        Run QCNet inference and populate prediction buffers for all active agents.

        Args:
            iteration_index: Current simulation iteration index

        Returns:
            Tuple[List[str], int]: (agent token list, number of active agents)
        """

        # Pre-flight checks
        if self._use_saved_predictions:
            raise RuntimeError("_make_prediction_from_network should not be called when using saved predictions")

        if self._predict_model is None or self._data_for_pred is None:
            raise RuntimeError("QCNet model or data_for_pred is not initialized")

        # Cache frequently used attributes
        result_timesteps = self._result_timesteps
        offset_x = self._offset_x
        offset_y = self._offset_y
        use_pedestrian_simple = self._use_pedestrian_simple_prediction
        use_traj_opt = self._use_trajectory_optimization
        max_obs_exo_num = self._max_obs_exo_num
        num_historical_steps = self._feature_builder.num_historical_steps
        output_head = self._predict_model.output_head
        output_dim = self._predict_model.output_dim
        ego_agent = self._ego_agent
        data_for_pred_agent_id = self._data_for_pred["agent"]["id"]
        exo_agents = self._exo_agents

        pedestrian_type = int(TrackedObjectType.PEDESTRIAN)
        Parallel = RelativeHeading.PARALLEL
        Lateral = RelativePosition.LATERAL

        num_timesteps = 80
        time_deltas = 0.2 * np.arange(1, result_timesteps)

        # Model outputs at 0.1s intervals; downsample to 0.2s by taking odd frames 1,3,5,...
        # Frames beyond index 79 are clamped to the last frame.
        orig_indices = np.minimum(2 * np.arange(1, result_timesteps) - 1, num_timesteps - 1)

        # Heading/speed finite-difference needs a previous frame; frame at orig_index=0 has none
        valid_indices_mask = orig_indices > 0
        valid_indices = orig_indices[valid_indices_mask]
        valid_target_indices = np.arange(1, result_timesteps)[valid_indices_mask]

        # Run model inference
        # Prepare CPU-side data before GPU transfer to overlap CPU and GPU work
        exo_agents_token = list(exo_agents.keys())
        num_exo_agents = min(len(exo_agents_token), max_obs_exo_num)

        data_for_pred = HeteroData(self._data_for_pred).to(self._device, non_blocking=True)
        with torch.inference_mode():
            predictions = self._predict_model(data_for_pred)

        if output_head:
            refined_traj = torch.cat([
                predictions['loc_refine_pos'][..., :output_dim],
                predictions['loc_refine_head'],
                predictions['scale_refine_pos'][..., :output_dim],
                predictions['conc_refine_head']
            ], dim=-1)
        else:
            refined_traj = torch.cat([
                predictions['loc_refine_pos'][..., :output_dim],
                predictions['scale_refine_pos'][..., :output_dim]
            ], dim=-1)

        eval_mask = data_for_pred['agent']['valid_mask'][:, num_historical_steps - 1].reshape(-1)
        eval_mask_np = eval_mask.cpu().numpy()

        origin_eval = data_for_pred['agent']['position'][eval_mask, num_historical_steps - 1]
        theta_eval = data_for_pred['agent']['heading'][eval_mask, num_historical_steps - 1]
        cos_tensor, sin_tensor = theta_eval.cos(), theta_eval.sin()

        rot_mat = torch.stack([
            torch.stack([cos_tensor, sin_tensor], dim=-1),
            torch.stack([-sin_tensor, cos_tensor], dim=-1)
        ], dim=-2)

        traj_eval = torch.matmul(refined_traj[eval_mask, :, :, :2], rot_mat.unsqueeze(1)) + \
                    origin_eval[:, :2].reshape(-1, 1, 1, 2)

        pi_eval = F.softmax(predictions['pi'][eval_mask], dim=-1)

        if output_head:
            heading_diff = refined_traj[eval_mask, :, :, 2:3]
            absolute_heading = heading_diff + theta_eval.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            traj_eval = torch.cat([traj_eval, absolute_heading], dim=-1)

        traj_numpy = traj_eval.cpu().numpy()
        pi_numpy = pi_eval.cpu().numpy()

        if isinstance(data_for_pred_agent_id, np.ndarray):
            pred_agents_token = data_for_pred_agent_id[eval_mask_np].tolist()
        else:
            pred_agents_token = [data_for_pred_agent_id[i] for i in range(len(data_for_pred_agent_id)) if eval_mask_np[i]]

        token_to_pred_idx = {token: idx for idx, token in enumerate(pred_agents_token)}
        num_modes = traj_numpy.shape[1]
        has_heading_pred = output_head and traj_numpy.shape[-1] >= 3

        # Zero out active agent slices (not the full 192-agent buffer)
        self.pred_pi[:num_exo_agents].fill(0.0)
        self.pred_traj_x[:num_exo_agents].fill(0.0)
        self.pred_traj_y[:num_exo_agents].fill(0.0)
        self.pred_traj_v[:num_exo_agents].fill(0.0)
        self.pred_traj_theta[:num_exo_agents].fill(0.0)
        self.pred_traj_theta_cos[:num_exo_agents].fill(0.0)
        self.pred_traj_theta_sin[:num_exo_agents].fill(0.0)

        pred_pi = self.pred_pi[:num_exo_agents]
        pred_traj_x = self.pred_traj_x[:num_exo_agents]
        pred_traj_y = self.pred_traj_y[:num_exo_agents]
        pred_traj_v = self.pred_traj_v[:num_exo_agents]
        pred_traj_theta = self.pred_traj_theta[:num_exo_agents]
        pred_traj_theta_cos = self.pred_traj_theta_cos[:num_exo_agents]
        pred_traj_theta_sin = self.pred_traj_theta_sin[:num_exo_agents]

        # Batch-fill all agent predictions
        if num_exo_agents == 0:
            return (exo_agents_token[:num_exo_agents], num_exo_agents)

        active_tokens = exo_agents_token[:num_exo_agents]
        agents_list = [exo_agents[t] for t in active_tokens]

        # Extract all agent base attributes in one pass
        x0 = np.array([a._x for a in agents_list])
        y0 = np.array([a._y for a in agents_list])
        v0 = np.array([a._speed for a in agents_list])
        theta0 = np.array([a._heading for a in agents_list])
        agent_types = np.array([a.type for a in agents_list])
        cos_theta0 = np.cos(theta0)
        sin_theta0 = np.sin(theta0)

        # Fill t=0 from observed state
        pred_traj_x[:, :, 0] = x0[:, None]
        pred_traj_y[:, :, 0] = y0[:, None]
        pred_traj_v[:, :, 0] = v0[:, None]
        pred_traj_theta[:, :, 0] = theta0[:, None]
        pred_traj_theta_cos[:, :, 0] = cos_theta0[:, None]
        pred_traj_theta_sin[:, :, 0] = sin_theta0[:, None]

        # Classify agents into prediction groups
        is_pedestrian = (agent_types == pedestrian_type)
        is_simple_ped = is_pedestrian & use_pedestrian_simple
        is_pred = np.zeros(num_exo_agents, dtype=bool)
        pred_indices = np.zeros(num_exo_agents, dtype=int)
        is_low_speed = np.zeros(num_exo_agents, dtype=bool)
        v0_adj_pred = v0.copy()
        for i, token in enumerate(active_tokens):
            if token in token_to_pred_idx:
                is_pred[i] = True
                pred_indices[i] = token_to_pred_idx[token]

                if v0[i] < 1.5:
                    is_low_speed[i] = True
                    v_adj = v0[i]
                    agent = agents_list[i]

                    # Slightly boost speed for slow lateral-parallel agents to avoid stale-obstacle misclassification
                    if v0[i] > 0.5 and agent.heading_wrt_ego == Parallel and agent.position_wrt_ego == Lateral:
                        v_adj = min(2.0 * v0[i], 1.5) if v0[i] < 1.0 else 1.5
                    v0_adj_pred[i] = v_adj

        v0_adj_ped = v0.copy()
        if use_pedestrian_simple:
            for i, token in enumerate(active_tokens):
                if is_simple_ped[i]:
                    v0_adj_ped[i] = self._adjust_pedestrian_velocity(v0[i], x0[i], y0[i], ego_agent, agents_list[i], token == "b50e54ff51fd5559")

        # Four mutually exclusive groups: simple-ped / low-speed linear / model-predicted / static
        mask_ped = is_simple_ped
        mask_low_speed = is_pred & is_low_speed & ~mask_ped
        mask_model = is_pred & ~is_low_speed & ~mask_ped
        mask_static = ~(mask_ped | mask_low_speed | mask_model)

        # Vehicles/bicycles with zero model speed still use linear extrapolation (preserves coasting inertia);
        # pure static obstacles (type >=3) are fully frozen
        mask_linear = mask_ped | mask_low_speed | (mask_static & np.isin(agent_types, [0, 1, 2]))
        mask_pure_static = mask_static & ~np.isin(agent_types, [0, 1, 2])

        # Linear extrapolation group
        if np.any(mask_linear):
            v_extrapolate = np.where(mask_ped, v0_adj_ped, np.where(mask_low_speed, v0_adj_pred, v0))

            # Single deterministic mode; set mode-0 probability to 1
            pred_pi[mask_linear, 0] = 1.0
            dx = (v_extrapolate[mask_linear] * cos_theta0[mask_linear])[:, None] * time_deltas[None, :]
            dy = (v_extrapolate[mask_linear] * sin_theta0[mask_linear])[:, None] * time_deltas[None, :]

            pred_traj_x[mask_linear, :num_modes, 1:result_timesteps] = x0[mask_linear, None, None] + dx[:, None, :]
            pred_traj_y[mask_linear, :num_modes, 1:result_timesteps] = y0[mask_linear, None, None] + dy[:, None, :]
            pred_traj_v[mask_linear, :num_modes, 1:result_timesteps] = v_extrapolate[mask_linear, None, None]
            pred_traj_theta[mask_linear, :num_modes, 1:result_timesteps] = theta0[mask_linear, None, None]
            pred_traj_theta_cos[mask_linear, :num_modes, 1:result_timesteps] = cos_theta0[mask_linear, None, None]
            pred_traj_theta_sin[mask_linear, :num_modes, 1:result_timesteps] = sin_theta0[mask_linear, None, None]

        # Fully static group
        if np.any(mask_pure_static):
            pred_pi[mask_pure_static, 0] = 1.0
            pred_traj_x[mask_pure_static, :num_modes, 1:result_timesteps] = x0[mask_pure_static, None, None]
            pred_traj_y[mask_pure_static, :num_modes, 1:result_timesteps] = y0[mask_pure_static, None, None]
            pred_traj_v[mask_pure_static, :num_modes, 1:result_timesteps] = 0.0
            pred_traj_theta[mask_pure_static, :num_modes, 1:result_timesteps] = theta0[mask_pure_static, None, None]
            pred_traj_theta_cos[mask_pure_static, :num_modes, 1:result_timesteps] = cos_theta0[mask_pure_static, None, None]
            pred_traj_theta_sin[mask_pure_static, :num_modes, 1:result_timesteps] = sin_theta0[mask_pure_static, None, None]

        # Model-predicted group
        model_idx = np.where(mask_model)[0]
        if len(model_idx) > 0:
            p_idx = pred_indices[model_idx]

            pred_pi[model_idx, :] = pi_numpy[p_idx, :]
            traj_sliced = traj_numpy[p_idx, :num_modes]

            # Batch-extract trajectory positions for all active model agents
            traj_active = traj_sliced[:, :, orig_indices, :2]
            pred_traj_x[model_idx, :num_modes, 1:result_timesteps] = traj_active[..., 0] - offset_x
            pred_traj_y[model_idx, :num_modes, 1:result_timesteps] = traj_active[..., 1] - offset_y

            traj_val = traj_sliced[:, :, valid_indices, :2]
            traj_val_prev = traj_sliced[:, :, valid_indices - 1, :2]
            delta_active = traj_val - traj_val_prev

            if has_heading_pred:
                # output_head=True: network directly predicts heading; apply single-step jitter filter
                # to fix isolated spike frames without introducing turn-tracking lag from a lowpass
                headings_active = smooth_heading_batch(traj_sliced[:, :, orig_indices, 2], threshold=0.3)
            else:
                # output_head=False: back-compute heading from position differences, then smooth in
                # sin/cos domain to avoid angle wraparound artifacts near ±π
                headings_active = np.arctan2(delta_active[..., 1], delta_active[..., 0])
                sin_h = uniform_filter1d(np.sin(headings_active), size=3, axis=2, mode='nearest')
                cos_h = uniform_filter1d(np.cos(headings_active), size=3, axis=2, mode='nearest')
                headings_active = np.arctan2(sin_h, cos_h)

            # Smooth speed including v0 so t=0 observation is continuous with the predicted segment
            n_model = len(model_idx)
            speeds_buf = self._speeds_buf[:n_model, :num_modes, :result_timesteps]
            buf_theta = self._buf_theta[:n_model, :num_modes, :result_timesteps]
            buf_v = self._buf_v[:n_model, :num_modes, :result_timesteps]
            speeds_buf[...] = 0.0
            buf_theta[...] = 0.0
            buf_v[...] = 0.0
            speeds_buf[:, :, 0] = v0[model_idx, None]
            speeds_buf[:, :, 1:] = np.hypot(delta_active[..., 0], delta_active[..., 1]) * 10.0
            speeds_active = uniform_filter1d(speeds_buf, size=5, axis=2, mode='nearest')
            buf_theta[:, :, 0] = theta0[model_idx, None]
            buf_v[:, :, 0] = v0[model_idx, None]

            # Optional scipy arc-length re-parameterization for speed-position consistency
            if use_traj_opt:
                for m_idx, i in enumerate(model_idx):
                    for mode_idx in range(num_modes):
                        opt_x, opt_y = self._optimize_trajectory_with_smoothed_velocity(
                            pred_traj_x[i, mode_idx, :result_timesteps],
                            pred_traj_y[i, mode_idx, :result_timesteps],
                            speeds_active[m_idx, mode_idx, :],
                            headings_active[m_idx, mode_idx, :],
                            dt=0.2
                        )
                        pred_traj_x[i, mode_idx, :result_timesteps] = opt_x
                        pred_traj_y[i, mode_idx, :result_timesteps] = opt_y

            # Heading fallback has a sequential temporal dependency on buf_theta[t-1]; cannot vectorize
            for idx, target_idx in enumerate(valid_target_indices):
                curr_h = headings_active[:, :, idx]
                curr_s = speeds_active[:, :, target_idx]
                early_timestep = target_idx <= result_timesteps // 2
                prev_h = buf_theta[:, :, target_idx - 1]

                # curr_h = np.where(curr_s <= 1e-6, prev_h, curr_h)


                if has_heading_pred:
                    # Near-term predictions are trusted; freeze heading only at zero speed in the late horizon
                    curr_h = np.where((curr_s <= 1e-6) & ~early_timestep, prev_h, curr_h)
                    # print(f"curr_h: {curr_h}")
                else:
                    curr_h = np.where(curr_s <= 1e-6, prev_h, curr_h)
                    diff = (curr_h - prev_h + np.pi) % (2 * np.pi) - np.pi
                    large_diff_mask = np.abs(diff) >= 0.3
                    curr_h = np.where(large_diff_mask, prev_h + np.sign(diff) * 0.1, curr_h)

                buf_v[:, :, target_idx] = curr_s
                buf_theta[:, :, target_idx] = curr_h

            pred_traj_v[model_idx, :num_modes, 1:result_timesteps] = buf_v[:, :, 1:result_timesteps]
            pred_traj_theta[model_idx, :num_modes, 1:result_timesteps] = buf_theta[:, :, 1:result_timesteps]
            # pred_traj_theta_cos[model_idx, :num_modes, 1:result_timesteps] = np.cos(buf_theta[:, :, 1:result_timesteps])
            # pred_traj_theta_sin[model_idx, :num_modes, 1:result_timesteps] = np.sin(buf_theta[:, :, 1:result_timesteps])

        # Normalize heading angles for the active slice
        active_theta = self.pred_traj_theta[:num_exo_agents]
        active_theta[:] = normalize_angle(active_theta)
        self.pred_traj_theta_cos[:num_exo_agents] = np.cos(active_theta)
        self.pred_traj_theta_sin[:num_exo_agents] = np.sin(active_theta)

        del data_for_pred
        return (exo_agents_token[:num_exo_agents], num_exo_agents)

    def _make_prediction_from_saved_data(self, iteration_index: int):
        """
        Load trajectory predictions from saved disk files.

        Args:
            iteration_index: Current iteration index

        Returns:
            Tuple[List[str], int]: (agent token list, number of active agents)
        """

        exo_agents_token = list(self._exo_agents.keys())
        num_exo_agents = min(len(exo_agents_token), self._max_obs_exo_num)

        self.pred_pi.fill(0.0)
        self.pred_traj_x.fill(0.0)
        self.pred_traj_y.fill(0.0)
        self.pred_traj_v.fill(0.0)
        self.pred_traj_theta.fill(0.0)
        self.pred_traj_theta_cos.fill(0.0)
        self.pred_traj_theta_sin.fill(0.0)

        saved_data = self.load_prediction_data(self._scenario.token, iteration_index)

        if saved_data is None:
            print(f"Warning: no saved prediction found for iteration {iteration_index}, using constant-velocity fallback", flush=True)
            self._fill_default_predictions(exo_agents_token[:num_exo_agents])
        else:
            print(f"Loaded saved predictions for iteration {iteration_index}, {saved_data.get('num_agents', 0)} agents", flush=True)
            self._fill_predictions_from_saved_data(saved_data, exo_agents_token[:num_exo_agents])

        return (exo_agents_token[:num_exo_agents], num_exo_agents)

    # Trajectory optimization helpers
    def _optimize_trajectory_with_smoothed_velocity(self, xs: np.ndarray, ys: np.ndarray, smoothed_vs: np.ndarray, smoothed_headings: np.ndarray, dt: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-parameterize trajectory by arc length using smoothed speed to enforce speed-position consistency.

        Args:
            xs: Original x-coordinate sequence [num_timesteps]
            ys: Original y-coordinate sequence [num_timesteps]
            smoothed_vs: Smoothed speed sequence [num_timesteps]
            smoothed_headings: Smoothed heading sequence [num_timesteps]
            dt: Time step in seconds

        Returns:
            Re-sampled (xs_new, ys_new)
        """

        if len(xs) < 2 or len(ys) < 2 or len(smoothed_vs) < 2:
            return xs, ys

        try:
            # Compute cumulative arc length; avoid np.insert allocation overhead
            dx = np.diff(xs)
            dy = np.diff(ys)
            ds = np.sqrt(dx*dx + dy*dy)

            s_orig = np.empty(len(xs))
            s_orig[0] = 0.0
            np.cumsum(ds, out=s_orig[1:])  # In-place cumsum, zero-copy

            # Remove duplicate arc-length values to keep interpolation stable
            mask = np.empty(len(s_orig), dtype=bool)
            mask[0] = True
            mask[1:] = np.diff(s_orig) > 1e-10

            s_clean = s_orig[mask]
            xs_clean = xs[mask]
            ys_clean = ys[mask]

            # Degenerate case: vehicle is stationary
            if len(s_clean) < 2:
                return np.full_like(smoothed_vs, xs[0]), np.full_like(smoothed_vs, ys[0])

            # Integrate smoothed speed to get target arc-length positions
            s_target = np.empty(len(smoothed_vs))
            s_target[0] = 0.0
            np.cumsum(smoothed_vs[:-1] * dt, out=s_target[1:])

            # Interpolate within trajectory bounds using fast C-backed np.interp
            s_max = s_clean[-1]
            mask_in = s_target <= s_max

            xs_in = np.interp(s_target[mask_in], s_clean, xs_clean)
            ys_in = np.interp(s_target[mask_in], s_clean, ys_clean)

            # Linearly extrapolate beyond trajectory end using the final heading
            if not mask_in.all():
                s_extra = s_target[~mask_in] - s_max
                last_heading = smoothed_headings[-1]
                dx_ext = np.cos(last_heading) * s_extra
                dy_ext = np.sin(last_heading) * s_extra

                xs_extra = xs_clean[-1] + dx_ext
                ys_extra = ys_clean[-1] + dy_ext

                xs_new = np.concatenate([xs_in, xs_extra])
                ys_new = np.concatenate([ys_in, ys_extra])
            else:
                xs_new, ys_new = xs_in, ys_in

            return xs_new, ys_new

        except Exception:
            return xs, ys

    # Prediction buffer fillers (fallback / saved-data)
    def _fill_default_predictions(self, exo_agents_token):
        """
        Fill prediction arrays with constant-velocity linear extrapolation (fallback when no saved data).

        Args:
            exo_agents_token: List of agent tokens to fill
        """

        for i, token in enumerate(exo_agents_token):
            agent = self._exo_agents[token]

            x0 = agent._x
            y0 = agent._y
            v0 = agent._speed
            theta0 = agent._heading

            # t=0 reflects the current observed state
            self.pred_traj_x[i, :, 0].fill(x0)
            self.pred_traj_y[i, :, 0].fill(y0)
            self.pred_traj_v[i, :, 0].fill(v0)
            self.pred_traj_theta[i, :, 0].fill(theta0)
            self.pred_traj_theta_cos[i, :, 0].fill(np.cos(theta0))
            self.pred_traj_theta_sin[i, :, 0].fill(np.sin(theta0))

            if self._use_pedestrian_simple_prediction and agent.type == int(TrackedObjectType.PEDESTRIAN):
                self.pred_pi[i, 0] = 1.0
                time_deltas = 0.2 * np.arange(1, self._result_timesteps)  # shape (T-1,)
                v0 = self._adjust_pedestrian_velocity(v0, x0, y0, self._ego_agent, agent)
                delta_x = v0 * np.cos(theta0) * time_deltas  # shape (T-1,)
                delta_y = v0 * np.sin(theta0) * time_deltas  # shape (T-1,)
                self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps] = x0 + delta_x[None, :]  # shape (M, T-1)
                self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps] = y0 + delta_y[None, :]
                self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps] = v0
                self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps] = theta0
                self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps] = np.cos(theta0)
                self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps] = np.sin(theta0)
                continue

            self.pred_pi[i, 0] = 1.0

            if agent.type == 0 or agent.type == 1 or agent.type == 2:  # Dynamic agents: constant-velocity model
                time_deltas = 0.2 * np.arange(1, self._result_timesteps)  # shape (T-1,)

                delta_x = v0 * np.cos(theta0) * time_deltas  # shape (T-1,)
                delta_y = v0 * np.sin(theta0) * time_deltas  # shape (T-1,)

                # Broadcast to all modes (same prediction for each)
                self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps] = x0 + delta_x[None, :]  # shape (M, T-1)
                self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps] = y0 + delta_y[None, :]
                self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps] = v0
                self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps] = theta0
                self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps] = np.cos(theta0)
                self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps] = np.sin(theta0)
            else:  # Static objects: frozen in place
                self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps].fill(x0)
                self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps].fill(y0)
                self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps].fill(0.0)
                self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps].fill(theta0)
                self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps].fill(np.cos(theta0))
                self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps].fill(np.sin(theta0))

    def _fill_predictions_from_saved_data(self, saved_data, exo_agents_token):
        """
        Fill prediction arrays from saved data, preserving the agent ordering in exo_agents_token.

        Args:
            saved_data: Loaded prediction data dict
            exo_agents_token: Ordered list of agent tokens (defines slot assignment)
        """

        saved_agents_data = saved_data.get('agents_data', {})

        for i, token in enumerate(exo_agents_token):
            agent = self._exo_agents[token]

            # t=0 always reflects the current observed state, not the saved state
            x0 = agent._x
            y0 = agent._y
            v0 = agent._speed
            theta0 = agent._heading

            self.pred_traj_x[i, :, 0].fill(x0)
            self.pred_traj_y[i, :, 0].fill(y0)
            self.pred_traj_v[i, :, 0].fill(v0)
            self.pred_traj_theta[i, :, 0].fill(theta0)
            self.pred_traj_theta_cos[i, :, 0].fill(np.cos(theta0))
            self.pred_traj_theta_sin[i, :, 0].fill(np.sin(theta0))

            # Pedestrians always use the simple model, even when saved data is available
            if self._use_pedestrian_simple_prediction and agent.type == int(TrackedObjectType.PEDESTRIAN):
                self.pred_pi[i, 0] = 1.0
                time_deltas = 0.2 * np.arange(1, self._result_timesteps)  # shape (T-1,)
                v0 = self._adjust_pedestrian_velocity(v0, x0, y0, self._ego_agent, agent)
                delta_x = v0 * np.cos(theta0) * time_deltas  # shape (T-1,)
                delta_y = v0 * np.sin(theta0) * time_deltas  # shape (T-1,)
                self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps] = x0 + delta_x[None, :]  # shape (M, T-1)
                self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps] = y0 + delta_y[None, :]
                self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps] = v0
                self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps] = theta0
                self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps] = np.cos(theta0)
                self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps] = np.sin(theta0)
                continue

            if token in saved_agents_data:
                agent_data = saved_agents_data[token]

                self.pred_pi[i, :] = agent_data['pred_pi']
                self.pred_traj_x[i, :, :] = agent_data['pred_traj_x']
                self.pred_traj_y[i, :, :] = agent_data['pred_traj_y']
                self.pred_traj_v[i, :, :] = agent_data['pred_traj_v']
                self.pred_traj_theta[i, :, :] = agent_data['pred_traj_theta']
                self.pred_traj_theta_cos[i, :, :] = agent_data['pred_traj_theta_cos']
                self.pred_traj_theta_sin[i, :, :] = agent_data['pred_traj_theta_sin']

            else:
                print(f"Warning: token {token} not found in saved data, using constant-velocity fallback", flush=True)

                self.pred_pi[i, 0] = 1.0

                if agent.type == 0 or agent.type == 1 or agent.type == 2:  # Dynamic agents
                    time_deltas = 0.2 * np.arange(1, self._result_timesteps)  # shape (T-1,)

                    delta_x = v0 * np.cos(theta0) * time_deltas  # shape (T-1,)
                    delta_y = v0 * np.sin(theta0) * time_deltas  # shape (T-1,)

                    # Broadcast to all modes (same prediction for each)
                    self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps] = x0 + delta_x[None, :]  # shape (M, T-1)
                    self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps] = y0 + delta_y[None, :]
                    self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps] = v0
                    self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps] = theta0
                    self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps] = np.cos(theta0)
                    self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps] = np.sin(theta0)
                else:  # Static objects
                    self.pred_traj_x[i, :self._max_num_modes, 1:self._result_timesteps].fill(x0)
                    self.pred_traj_y[i, :self._max_num_modes, 1:self._result_timesteps].fill(y0)
                    self.pred_traj_v[i, :self._max_num_modes, 1:self._result_timesteps].fill(0.0)
                    self.pred_traj_theta[i, :self._max_num_modes, 1:self._result_timesteps].fill(theta0)
                    self.pred_traj_theta_cos[i, :self._max_num_modes, 1:self._result_timesteps].fill(np.cos(theta0))
                    self.pred_traj_theta_sin[i, :self._max_num_modes, 1:self._result_timesteps].fill(np.sin(theta0))

    def load_prediction_data(self, scenario_token: str, iteration_index: int) -> Dict:
        """
        Load saved prediction data for a given scenario and iteration.

        Args:
            scenario_token: Scenario identifier.
            iteration_index: Iteration index.

        Returns:
            Prediction data dict, or None if the file does not exist.
        """

        try:
            filepath = os.path.join("logs", "predicted_replay_logs", scenario_token, f"prediction_data_{iteration_index:03d}.pkl")
            if not os.path.exists(filepath):
                print(f"Prediction data file not found: {filepath}", flush=True)
                return None

            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            print(f"Loaded prediction data: {filepath}, {data.get('num_agents', 0)} agents", flush=True)
            return data

        except Exception as e:
            print(f"Error loading prediction data: {e}", flush=True)
            return None

