"""Recurrent constitutive-law model (LSTM).

A standalone sequence model mapping an input history (e.g. macro strain) to an
output history (e.g. macro stress), optionally exposing the LSTM output states.
It is fully decoupled from the graph family; the hybrid pipeline in
:mod:`plgnn.hybrid` is what wires the two together.
"""

from __future__ import annotations

from typing import Optional

import torch

from plgnn.base import BaseModel
from plgnn.scaling import ModelStandardScaler


class AutoRegressiveStressRNN(BaseModel):
    """LSTM with a linear output head, mapping input to output sequences."""

    def __init__(
        self,
        input_features_size: int,
        hidden_state_size: int,
        output_size: int,
        num_layers: int,
        input_scaler: Optional[ModelStandardScaler] = None,
        output_scaler: Optional[ModelStandardScaler] = None,
    ):
        super().__init__(input_scaler, output_scaler)
        self.input_features_size = input_features_size
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = torch.nn.LSTM(
            input_features_size,
            hidden_state_size,
            num_layers,
            batch_first=True,
        )
        self.mlp_output_layer = torch.nn.Linear(hidden_state_size, output_size)

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ):
        if hidden_states is None or cell_state is None:
            output, (hidden_states, cell_state) = self.rnn(input_sequence)
        else:
            output, (hidden_states, cell_state) = self.rnn(
                input_sequence, (hidden_states, cell_state)
            )
        preds = self.mlp_output_layer(output)
        if return_hidden_states:
            return preds, (hidden_states, cell_state), output
        return preds, (hidden_states, cell_state)
