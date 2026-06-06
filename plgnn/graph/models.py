"""Graph neural network: encode-process-decode for mesh field reconstruction.

The model is constitutive-agnostic: it consumes ``graph.x`` (assembled node
features) and ``graph.edge_attr`` (edge weights) and returns a per-node field in
``local_stress``. Feature assembly and scaling are the caller's responsibility,
which keeps the GNN reusable across elastic, hyperelastic, and plastic cases.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch_geometric as PyG
from torch.nn import Linear, Sequential
from torch_geometric.nn import LayerNorm, MessagePassing

from plgnn.base import BaseModel
from plgnn.scaling import ModelStandardScaler


class Processor(MessagePassing):
    """A single message-passing step with residual node and edge updates."""

    def __init__(
        self,
        latent_size: int,
        input_nodes_features_size: int,
        input_edges_features_size: int,
    ):
        super().__init__(aggr="add")
        self.latent_size = latent_size

        self.edge_net = Sequential(
            Linear(input_edges_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.node_net = Sequential(
            Linear(input_nodes_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

    def forward(self, graph: PyG.data.Data) -> PyG.data.Data:
        edge_index = graph.edge_index
        x = graph.x
        edge_features = graph.edge_attr

        new_node_features = self.propagate(
            edge_index, x=x, edge_attr=edge_features
        )

        row, col = edge_index
        new_edge_features = self.edge_net(
            torch.cat([x[row], x[col], edge_features], dim=-1)
        )

        new_node_features = new_node_features + graph.x
        new_edge_features = new_edge_features + graph.edge_attr

        return PyG.data.Data(
            edge_index=edge_index,
            x=new_node_features,
            edge_attr=new_edge_features,
        )

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_net(features)

    def update(
        self, aggr_out: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        tmp = torch.cat([aggr_out, x], dim=-1)
        return self.node_net(tmp)


class EncodeProcessDecode(BaseModel):
    """Encoder -> repeated message passing -> decoder GNN."""

    def __init__(
        self,
        input_edges_features_size: int,
        message_passing_steps: int,
        latent_size: int,
        input_nodes_features_size: int,
        output_nodes_features_size: int,
        input_scaler: Optional[ModelStandardScaler] = None,
        output_scaler: Optional[ModelStandardScaler] = None,
        node_pos_scaler: Optional[ModelStandardScaler] = None,
        edge_scaler: Optional[ModelStandardScaler] = None,
    ):
        super().__init__(input_scaler, output_scaler)
        self.message_passing_steps = message_passing_steps
        self.input_edges_features_size = input_edges_features_size
        self.latent_size = latent_size
        self.input_nodes_features_size = input_nodes_features_size
        self.output_nodes_features_size = output_nodes_features_size
        self.node_pos_scaler = node_pos_scaler
        self.edge_scaler = edge_scaler
        self.node_encoder = Sequential(
            Linear(self.input_nodes_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.edge_encoder = Sequential(
            Linear(self.input_edges_features_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            LayerNorm(self.latent_size),
        )

        self.processor = Processor(
            self.latent_size,
            input_nodes_features_size=self.latent_size * 2,
            input_edges_features_size=self.latent_size * 3,
        )

        self.node_decoder = Sequential(
            Linear(self.latent_size, self.latent_size),
            torch.nn.ReLU(),
            Linear(self.latent_size, self.output_nodes_features_size),
        )

    def forward(self, mesh_graph: PyG.data.Data) -> PyG.data.Data:
        edge_index = mesh_graph.edge_index
        x = mesh_graph.x
        edge_weight = mesh_graph.edge_attr.unsqueeze(1)
        node_embedding = self.node_encoder(x)
        edge_embedding = self.edge_encoder(edge_weight)
        latent_graph = PyG.data.Data(
            edge_index=edge_index, x=node_embedding, edge_attr=edge_embedding
        )
        for _ in range(self.message_passing_steps):
            latent_graph = self.processor(latent_graph)

        decoded_nodes = self.node_decoder(latent_graph.x)
        return PyG.data.Data(
            local_stress=decoded_nodes,
            edge_index=mesh_graph.edge_index,
            pos=mesh_graph.pos,
        )
