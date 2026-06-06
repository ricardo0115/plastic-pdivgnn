"""Hybrid LSTM + GNN field model for path-dependent (e.g. plastic) problems.

`LstmGNN` composes two independently-trained families:

1. an :class:`~plgnn.lstm.AutoRegressiveStressRNN` constitutive law that turns a
   macro strain history into a macro stress history plus per-step hidden states;
2. an :class:`~plgnn.graph.EncodeProcessDecode` GNN that reconstructs the local
   field from node features.

The feature-assembly contract that connects them lives *only* here:
``node_features = concat(input_scaler(concat(hidden_state, macro_stress)), pos,
node_label)``. Neither family depends on this module, so they stay decoupled.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Batch, Data

from plgnn.base import get_device
from plgnn.graph import build_graph, normalize_graph
from plgnn.graph.convert import is_periodic
from plgnn.graph.models import EncodeProcessDecode
from plgnn.lstm.models import AutoRegressiveStressRNN


class LstmGNN(torch.nn.Module):
    """Wire an LSTM constitutive law to a GNN field reconstructor.

    Args:
        lstm_model: trained recurrent constitutive law (strain -> stress + states).
        gnn_model: trained GNN mapping assembled node features -> local field.
        mesh: pyvista mesh defining the geometry (built into a graph once).
        device: compute device; auto-detected when ``None``.
        periodic: build periodic wrap-around edges (default True). When True the
            mesh is asserted periodic.
    """

    def __init__(
        self,
        lstm_model: AutoRegressiveStressRNN,
        gnn_model: EncodeProcessDecode,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        device: Optional[str] = None,
        periodic: bool = True,
    ):
        super().__init__()
        self.device = device or get_device()
        self.periodic = periodic

        if periodic:
            assert is_periodic(mesh.points[:, :-1]), (
                "LstmGNN(periodic=True) expects a periodic input mesh. "
                "Pass periodic=False for a non-periodic mesh."
            )

        graph = build_graph(mesh, periodic=periodic)
        positions, node_labels, edge_attr, _ = normalize_graph(graph)

        # Store graph tensors on device once (they never change).
        self.positions = positions.to(self.device)  # [N, 2]
        self.node_labels = node_labels.to(self.device)  # [N]
        self.edge_index = graph.edge_index.to(self.device)  # [2, E]
        self.edge_attr = edge_attr.to(self.device)  # [E]

        self.lstm = lstm_model.to(self.device).eval()
        self.gnn = gnn_model.to(self.device).eval()

        # Move scalers to compute device (avoid per-call CPU<->GPU transfers).
        for scaler in (self.gnn.input_scaler, self.gnn.output_scaler):
            if isinstance(scaler.mean, torch.Tensor):
                scaler.mean = scaler.mean.to(self.device)
                scaler.std = scaler.std.to(self.device)

    @torch.no_grad()
    def run_lstm(
        self, strain_sequence: Sequence | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the LSTM: strain history -> (macro stress [T, C], hidden [T, H])."""
        strain = torch.as_tensor(strain_sequence, dtype=torch.float32)
        strain = self.lstm.input_scaler.transform(strain).float()
        strain = strain.to(self.device)
        preds, _, output_states = self.lstm(strain, return_hidden_states=True)
        macro_stress = self.lstm.output_scaler.inverse_transform(preds.cpu())
        hidden_states = output_states.detach().cpu()
        # Zero-input guarantee: a fully zero strain vector maps to zero stress.
        strain_in = torch.as_tensor(strain_sequence, dtype=torch.float32)
        if strain_in.ndim >= 2:
            zero_steps = torch.all(strain_in == 0.0, dim=-1)
            macro_stress[zero_steps] = 0.0
        return macro_stress, hidden_states

    @torch.no_grad()
    def run_gnn(
        self,
        mean_stresses: Sequence[torch.Tensor],  # [T, C]
        hidden_states: Sequence[torch.Tensor],  # [T, H]
        chunk_size: int = 16,
    ) -> np.ndarray:
        """Reconstruct the local field sequence from macro stress + hidden states.

        Returns an array of shape ``[T, N, output_comps]`` in physical units.
        The first timestep is zeroed (path-dependent initial state) when T > 1,
        matching the reference plastic_gnn pipeline.
        """
        assert len(mean_stresses) == len(hidden_states)
        n_timesteps = len(mean_stresses)
        n_nodes = self.positions.shape[0]

        all_predictions = []
        for start in range(0, n_timesteps, chunk_size):
            end = min(start + chunk_size, n_timesteps)
            data_list = []
            zero_steps = []
            for mean_stress, hidden_state in zip(
                mean_stresses[start:end], hidden_states[start:end]
            ):
                mean_stress = torch.as_tensor(mean_stress, dtype=torch.float32)
                hidden_state = torch.as_tensor(
                    hidden_state, dtype=torch.float32
                )
                zero_steps.append(bool(torch.all(mean_stress == 0)))
                if mean_stress.ndim == 1:
                    mean_stress = mean_stress.unsqueeze(0)
                if hidden_state.ndim == 1:
                    hidden_state = hidden_state.unsqueeze(0)

                x = torch.cat(
                    [
                        hidden_state.repeat(n_nodes, 1),
                        mean_stress.repeat(n_nodes, 1),
                    ],
                    dim=-1,
                )
                x = self.gnn.input_scaler.transform(x.to(self.device))
                x = torch.cat(
                    [x, self.positions, self.node_labels.unsqueeze(1)], dim=-1
                )
                data_list.append(
                    Data(
                        x=x,
                        pos=self.positions,
                        edge_index=self.edge_index,
                        edge_attr=self.edge_attr,
                    )
                )

            batch = Batch.from_data_list(data_list).to(self.device)
            out = self.gnn(batch).local_stress
            out = self.gnn.output_scaler.inverse_transform(out)
            out = out.view(end - start, n_nodes, -1)  # [chunk_T, N, C]
            # Zero-input guarantee: a fully zero macro stress maps to a zero field.
            for i, is_zero_step in enumerate(zero_steps):
                if is_zero_step:
                    out[i] = 0.0
            all_predictions.append(out.detach().cpu())

        predictions = torch.cat(all_predictions, dim=0).float().numpy()
        if predictions.shape[0] > 1:
            predictions[0] = 0
        return predictions

    @torch.no_grad()
    def forward(
        self,
        strain_sequence: Sequence | np.ndarray | torch.Tensor,
        chunk_size: int = 16,
    ) -> np.ndarray:
        """Full pipeline: strain history -> local field sequence ``[T, N, C]``."""
        mean_stresses, hidden_states = self.run_lstm(strain_sequence)
        return self.run_gnn(mean_stresses, hidden_states, chunk_size)
