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
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle


class QCNetAgentEncoder(nn.Module):
    """
    Agent encoder that builds temporal and spatial representations for all agents.
    """

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        """
        Initialize QCNetAgentEncoder.

        Args:
            dataset: dataset name, one of {'argoverse_v2', 'nuplan'}.
            input_dim: spatial dimension of position/velocity inputs (2 or 3).
            hidden_dim: dimension of all hidden representations.
            num_historical_steps: number of past time steps to encode.
            time_span: maximum temporal distance for time-graph edges; defaults to num_historical_steps.
            pl2a_radius: radius (meters) for polygon-to-agent neighbor search.
            a2a_radius: radius (meters) for agent-to-agent neighbor search.
            num_freq_bands: number of frequency bands in Fourier embeddings.
            num_layers: number of stacked attention layers.
            num_heads: number of attention heads per layer.
            head_dim: dimension per attention head.
            dropout: dropout probability applied in attention layers.
        """
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if dataset == 'argoverse_v2' or dataset == 'nuplan':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        # discrete feature embedding layer
        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)  # 10 agent types
        elif dataset == 'nuplan':
            self.type_a_emb = nn.Embedding(5, hidden_dim)  # 5 agent types (vehicle, pedestrian, cyclist)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

        # precompute static temporal connectivity graph for a single agent
        time_steps = torch.arange(self.num_historical_steps)

        # generate time grid
        grid_j, grid_i = torch.meshgrid(time_steps, time_steps, indexing='ij')
        base_edge_index = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=0)

        # apply static rules upfront: t2 > t1 and t2 - t1 <= time_span
        valid_time_mask = (base_edge_index[1] > base_edge_index[0]) & \
                          (base_edge_index[1] - base_edge_index[0] <= self.time_span)

        # register as buffer so it moves to GPU automatically with the model
        self.register_buffer('base_edge_index_t', base_edge_index[:, valid_time_mask], persistent=False)

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode all agents over historical time steps.

        Args:
            data: heterogeneous graph containing agent states (position, heading,
                  velocity, valid_mask, type) and map polygon features.
            map_enc: pre-computed map encoding with key 'x_pl'
                     of shape [num_historical_steps, num_polygons, hidden_dim].

        Returns:
            Dict[str, torch.Tensor]: {'x_a': agent features of shape
                [num_nodes, num_historical_steps, hidden_dim]}.
        """
        # mask [num_nodes, num_historical_steps]
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()

        # pos_a [num_nodes, num_historical_steps, input_dim]
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()

        if self.dataset == 'argoverse_v2' or self.dataset == 'nuplan':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()
            length = width = height = None
            
            # categorical embedding
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0),
            ]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2' or self.dataset == 'nuplan':
            # x_a: [motion_vec, motion_angle, vel_vec, vel_angle]
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        # fourier embedding
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        # pos_t [num_nodes * num_historical_steps, input_dim] (flattened)
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)

        # construct time-spatial graph
        # mask_t [num_nodes, num_historical_steps, num_historical_steps]
        # mask_t[node, t1, t2]=True when node is valid at both t1 and t2
        # mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        # convert dense 3D mask_t to sparse graph; flattened_index = agent_idx * num_historical_steps + time_step
        # edge_index_t = dense_to_sparse(mask_t)[0]
        # keep only edges where t2 > t1 to avoid double-counting
        # edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        # edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]

        # use precomputed graph + per-agent offset
        num_nodes = data['agent']['num_nodes']

        # broadcast single-agent graph to all agents: [2, num_base_edges] -> [num_nodes, 2, num_base_edges]
        edge_index_t = self.base_edge_index_t.unsqueeze(0).expand(num_nodes, -1, -1).clone()

        # add global flattened offset for each agent
        offsets = (torch.arange(num_nodes, device=pos_a.device) * self.num_historical_steps).view(-1, 1, 1)
        edge_index_t = edge_index_t + offsets

        # reshape to standard edge_index format [2, num_edges]
        edge_index_t = edge_index_t.transpose(0, 1).reshape(2, -1)

        # filter out edges involving invalid nodes using valid_mask
        mask_flat = mask.view(-1)
        valid_edge_mask = mask_flat[edge_index_t[0]] & mask_flat[edge_index_t[1]]
        edge_index_t = edge_index_t[:, valid_edge_mask]

        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])

        # relative position encoding for a single agent along the time dimension
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # flatten [num_nodes * num_historical_steps, input_dim]
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        
        # repeat polygon features for each time step
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)

        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(self.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])

        # search polygon neighbors within radius r for each historical position of each agent
        # handle empty polygon case: radius cannot process empty tensors
        num_polygons_original = data['map_polygon']['num_nodes']
        if num_polygons_original == 0:
            edge_index_pl2a = torch.empty(2, 0, dtype=torch.long, device=pos_s.device)
        else:
            edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                     batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps,
                               self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a, edge_index_pl2a)
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        return {'x_a': x_a}
