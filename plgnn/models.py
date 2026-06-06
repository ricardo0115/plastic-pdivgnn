"""Inference wrappers used by the figure scripts, built on the plgnn models.

These two thin adapters keep the call sites of the figure-reproduction scripts
unchanged while delegating all model logic to the :mod:`plgnn` library:

- :class:`LstmConstitutiveLaw` wraps :class:`plgnn.lstm.AutoRegressiveStressRNN`
  (macro strain history -> macro stress + hidden states);
- :class:`PlasticGNN` wraps :class:`plgnn.graph.EncodeProcessDecode` and the
  periodic-graph build (macro stress + hidden states -> local stress field).

Equivalent to :class:`plgnn.hybrid.LstmGNN` run in two stages; kept separate
because the figures sometimes need the macro stress / hidden states on their own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Batch, Data

from plgnn.graph.build import build_graph, normalize_graph
from plgnn.graph.convert import is_periodic
from plgnn.graph.models import EncodeProcessDecode
from plgnn.lstm.models import AutoRegressiveStressRNN


class LstmConstitutiveLaw(torch.nn.Module):
    def __init__(self, weights_path: str, device: str, n_components: int = 3) -> None:
        super().__init__()
        self.device = device
        self.rnn = AutoRegressiveStressRNN(
            input_features_size=n_components,
            hidden_state_size=64,
            output_size=n_components,
            num_layers=2,
        )
        self.rnn.load_model_checkpoint(weights_path)
        self.rnn = self.rnn.to(device).eval()

    @torch.no_grad()
    def forward(
        self,
        strain_sequence: Sequence[float] | np.ndarray,
        return_hidden_states: bool = False,
    ):
        strain = torch.tensor(strain_sequence, dtype=torch.float32)
        strain = self.rnn.input_scaler.transform(strain).float().to(self.device)
        stress_sequence, _, hidden_states = self.rnn.forward(
            strain, return_hidden_states=True
        )
        stress_sequence = self.rnn.output_scaler.inverse_transform(
            stress_sequence.cpu()
        ).numpy()
        hidden_states = hidden_states.detach().cpu().numpy()
        # Zero-input guarantee: a fully zero strain vector maps to zero stress.
        strain_in = np.asarray(strain_sequence, dtype=np.float32)
        if strain_in.ndim >= 2:
            zero_steps = np.all(strain_in == 0.0, axis=-1)
            stress_sequence[zero_steps] = 0.0
        if return_hidden_states:
            return stress_sequence, hidden_states
        return stress_sequence


class PlasticGNN(torch.nn.Module):
    def __init__(
        self,
        weights_path: str,
        device: str,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        input_nodes_features_size: int = 70,
        output_nodes_features_size: int = 3,
        message_passing_steps: int = 10,
        latent_size: int = 128,
    ) -> None:
        super().__init__()
        self.device = device

        self.gnn = EncodeProcessDecode(
            input_edges_features_size=1,
            message_passing_steps=message_passing_steps,
            latent_size=latent_size,
            input_nodes_features_size=input_nodes_features_size,
            output_nodes_features_size=output_nodes_features_size,
        )

        assert is_periodic(mesh.points[:, :-1]), (
            "PlasticGNN expects a periodic input mesh."
        )
        graph = build_graph(mesh, periodic=True)
        positions, node_labels, edge_attr, _ = normalize_graph(graph)
        self.positions = positions.to(device)
        self.node_labels = node_labels.to(device)
        self.edge_index = graph.edge_index.to(device)
        self.edge_attr = edge_attr.to(device)

        self.gnn.load_model_checkpoint(weights_path)
        self.gnn = self.gnn.to(device).eval()
        for scaler in (self.gnn.input_scaler, self.gnn.output_scaler):
            if isinstance(scaler.mean, torch.Tensor):
                scaler.mean = scaler.mean.to(device)
                scaler.std = scaler.std.to(device)

        self._amp_dtype: torch.dtype | None = None

    def optimize_for_inference(
        self,
        compile: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        """Optional inference speedups (used by the benchmark figure)."""
        self._amp_dtype = amp_dtype
        if compile and self.device != "mps":
            self.gnn = torch.compile(
                self.gnn, mode="reduce-overhead", fullgraph=False
            )

    @torch.no_grad()
    def forward(
        self,
        mean_stresses: Sequence[torch.Tensor],
        hidden_states: Sequence[torch.Tensor],
        chunk_size: int = 16,
    ) -> np.ndarray:
        assert len(mean_stresses) == len(hidden_states)
        n_timesteps = len(mean_stresses)
        n_nodes = self.positions.shape[0]

        predictions: list[torch.Tensor] = []
        for start in range(0, n_timesteps, chunk_size):
            end = min(start + chunk_size, n_timesteps)
            data_list: list[Data] = []
            zero_steps: list[bool] = []
            for mean_stress, hidden_state in zip(
                mean_stresses[start:end], hidden_states[start:end]
            ):
                mean_stress = torch.as_tensor(mean_stress, dtype=torch.float32)
                hidden_state = torch.as_tensor(hidden_state, dtype=torch.float32)
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
            if self._amp_dtype is not None:
                with torch.autocast(device_type=self.device, dtype=self._amp_dtype):
                    out = self.gnn(batch).local_stress
            else:
                out = self.gnn(batch).local_stress
            out = self.gnn.output_scaler.inverse_transform(out)
            out = out.view(end - start, n_nodes, -1)
            # Zero-input guarantee: a fully zero macro stress maps to a zero field.
            for i, is_zero_step in enumerate(zero_steps):
                if is_zero_step:
                    out[i] = 0.0
            predictions.append(out.detach().cpu())

        result = torch.cat(predictions, dim=0).float().numpy()
        if result.shape[0] > 1:
            result[0] = 0
        return result
