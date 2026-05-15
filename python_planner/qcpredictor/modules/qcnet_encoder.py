# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from modules.qcnet_agent_encoder import QCNetAgentEncoder
from modules.qcnet_map_encoder import QCNetMapEncoder


class QCNetEncoder(nn.Module):
    """Top-level scene encoder that composes the map encoder and agent encoder."""

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        """
        Initialize QCNetEncoder.

        Args:
            dataset: dataset name, one of {'argoverse_v2', 'nuplan'}.
            input_dim: spatial dimension of position inputs (2 or 3).
            hidden_dim: dimension of all hidden representations.
            num_historical_steps: number of past time steps to encode.
            pl2pl_radius: radius (meters) for polygon-to-polygon neighbor search in the map encoder.
            time_span: maximum temporal distance for time-graph edges in the agent encoder;
                defaults to num_historical_steps.
            pl2a_radius: radius (meters) for polygon-to-agent neighbor search.
            a2a_radius: radius (meters) for agent-to-agent neighbor search.
            num_freq_bands: number of frequency bands in Fourier embeddings.
            num_map_layers: number of stacked attention layers in the map encoder.
            num_agent_layers: number of stacked attention layers in the agent encoder.
            num_heads: number of attention heads per layer.
            head_dim: dimension per attention head.
            dropout: dropout probability applied in attention layers.
        """
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = QCNetAgentEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Encode the full scene: map polygons and agent histories.

        The map encoding is cached and reused across frames as long as the
        polygon count remains unchanged, avoiding redundant computation for
        static map elements.

        Args:
            data: heterogeneous graph containing agent states and map polygon features.

        Returns:
            Dict[str, torch.Tensor]: merged map and agent encoding with keys:
                - 'x_pl': polygon features of shape [num_historical_steps, num_polygons, hidden_dim].
                - 'x_a': agent features of shape [num_nodes, num_historical_steps, hidden_dim].
        """
        
        # lazily initialize cache attributes on first call
        if not hasattr(self, '_cached_map_enc'):
            self._cached_map_enc = None
            self._cached_map_nodes = -1

        current_map_nodes = data['map_polygon']['num_nodes']

        # recompute map features only on the first frame or when the polygon count changes
        # (e.g. switching simulation test cases / new scene)
        if self._cached_map_enc is None or current_map_nodes != self._cached_map_nodes:
            # print(">>> recomputing map encoder static features...")
            self._cached_map_enc = self.map_encoder(data)
            self._cached_map_nodes = current_map_nodes

        # compute agent features
        agent_enc = self.agent_encoder(data, self._cached_map_enc)

        return {**self._cached_map_enc, **agent_enc}
