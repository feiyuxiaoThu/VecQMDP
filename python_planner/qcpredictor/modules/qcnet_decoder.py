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
import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class QCNetDecoder(nn.Module):
    """Multimodal trajectory decoder with propose-then-refine recurrent architecture."""

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        """
        Initialize QCNetDecoder.

        Args:
            dataset: dataset name, one of {'argoverse_v2', 'nuplan'}.
            input_dim: spatial dimension of position inputs (2 or 3).
            hidden_dim: dimension of all hidden representations.
            output_dim: spatial dimension of output trajectory (2 or 3).
            output_head: whether to predict heading angle in addition to position.
            num_historical_steps: number of past time steps available as context.
            num_future_steps: number of future time steps to predict.
            num_modes: number of trajectory modes (hypotheses) per agent.
            num_recurrent_steps: number of recurrent propose iterations.
            num_t2m_steps: number of most-recent historical steps used in T-to-M
                attention; defaults to num_historical_steps.
            pl2m_radius: radius (meters) for polygon-to-mode neighbor search.
            a2m_radius: radius (meters) for agent-to-mode neighbor search.
            num_freq_bands: number of frequency bands in Fourier embeddings.
            num_layers: number of stacked attention layers per stage.
            num_heads: number of attention heads per layer.
            head_dim: dimension per attention head.
            dropout: dropout probability applied in attention layers.
        """
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        # self.apply(weight_init)

        # decoder precomputed connectivity graphs
        # historical steps -> 1 mode (T to 1)
        t_idx = torch.arange(self.num_historical_steps)
        base_t2m_1 = torch.stack([t_idx, torch.zeros_like(t_idx)], dim=0)
        self.register_buffer('base_edge_t2m_1', base_t2m_1, persistent=False)

        # historical steps -> all modes (T to M)
        grid_t, grid_m = torch.meshgrid(
            torch.arange(self.num_historical_steps), torch.arange(self.num_modes), indexing='ij')
        base_t2m = torch.stack([grid_t.flatten(), grid_m.flatten()], dim=0)
        self.register_buffer('base_edge_t2m', base_t2m, persistent=False)

        grid_m1, grid_m2 = torch.meshgrid(
            torch.arange(self.num_modes), torch.arange(self.num_modes), indexing='ij')
        base_m2m = torch.stack([grid_m1.flatten(), grid_m2.flatten()], dim=0)
        self.register_buffer('base_edge_m2m', base_m2m, persistent=False)

        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run the propose-then-refine decode pass for all agents.

        Args:
            data: heterogeneous graph containing agent states (position, heading,
                  valid_mask, predict_mask) and map polygon features.
            scene_enc: scene encoding with keys:
                - 'x_a': agent features of shape [num_nodes, num_historical_steps, hidden_dim].
                - 'x_pl': polygon features of shape [num_historical_steps, num_polygons, hidden_dim].

        Returns:
            Dict[str, torch.Tensor] with keys:
                - 'loc_propose_pos': proposed position means,
                  shape [num_nodes, num_modes, num_future_steps, output_dim].
                - 'scale_propose_pos': proposed position scales (Laplace), same shape.
                - 'loc_propose_head': proposed heading means, shape [..., 1] (None if output_head=False).
                - 'conc_propose_head': proposed heading concentrations, same shape.
                - 'loc_refine_pos': refined position means, same shape as loc_propose_pos.
                - 'scale_refine_pos': refined position scales, same shape.
                - 'loc_refine_head': refined heading means (None if output_head=False).
                - 'conc_refine_head': refined heading concentrations.
                - 'pi': unnormalized mode log-probabilities, shape [num_nodes, num_modes].
        """

        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        # rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        # rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        # r_t2m = torch.stack(
        #     [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
        #      angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
        #      rel_head_t2m,
        #      (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        # r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        # edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        # r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        num_nodes = data['agent']['num_nodes']
        device = pos_m.device

        # build T to 1 graph
        e_t2m_1 = self.base_edge_t2m_1.unsqueeze(0).expand(num_nodes, -1, -1).clone()
        e_t2m_1[:, 0, :] += (torch.arange(num_nodes, device=device) * self.num_historical_steps).view(-1, 1)
        e_t2m_1[:, 1, :] += (torch.arange(num_nodes, device=device)).view(-1, 1)
        edge_index_t2m_1 = e_t2m_1.transpose(0, 1).reshape(2, -1)

        valid_mask_t2m_1 = mask_src.view(-1)[edge_index_t2m_1[0]] & mask_dst[:, 0][edge_index_t2m_1[1]]
        edge_index_t2m_1 = edge_index_t2m_1[:, valid_mask_t2m_1]

        rel_pos_t2m = pos_t[edge_index_t2m_1[0]] - pos_m[edge_index_t2m_1[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m_1[0]] - head_m[edge_index_t2m_1[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m_1[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m_1[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)

        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)

        # build T to M graph
        e_t2m = self.base_edge_t2m.unsqueeze(0).expand(num_nodes, -1, -1).clone()
        e_t2m[:, 0, :] += (torch.arange(num_nodes, device=device) * self.num_historical_steps).view(-1, 1)
        e_t2m[:, 1, :] += (torch.arange(num_nodes, device=device) * self.num_modes).view(-1, 1)
        edge_index_t2m = e_t2m.transpose(0, 1).reshape(2, -1)

        valid_mask_t2m = mask_src.view(-1)[edge_index_t2m[0]] & mask_dst.view(-1)[edge_index_t2m[1]]
        edge_index_t2m = edge_index_t2m[:, valid_mask_t2m]
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']

        # handle empty polygon case: radius cannot process empty tensors
        num_polygons = pos_pl.size(0)
        if num_polygons == 0:
            edge_index_pl2m = torch.empty(2, 0, dtype=torch.long, device=pos_m.device)
        else:
            edge_index_pl2m = radius(
                x=pos_m[:, :2],
                y=pos_pl[:, :2],
                r=self.pl2m_radius,
                batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
                batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
                max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])

        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])

        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        # edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        # build M to M graph
        e_m2m = self.base_edge_m2m.unsqueeze(0).expand(num_nodes, -1, -1).clone()
        offset_m2m = (torch.arange(num_nodes, device=device) * self.num_modes).view(-1, 1, 1)
        e_m2m = e_m2m + offset_m2m
        edge_index_m2m = e_m2m.transpose(0, 1).reshape(2, -1)

        valid_mask_m2m = mask_dst.view(-1)[edge_index_m2m[0]] & mask_dst.view(-1)[edge_index_m2m[1]]
        edge_index_m2m = edge_index_m2m[:, valid_mask_m2m]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps

        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)

        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) + 1.0,
            dim=-2) + 0.1

        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                     dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))

        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)

        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1

        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))
        pi = self.to_pi(m).squeeze(-1)

        return {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
