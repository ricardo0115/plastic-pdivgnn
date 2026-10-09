"""Recurrent constitutive-law model (LSTM).

A standalone sequence model mapping an input history (e.g. macro strain) to an
output history (e.g. macro stress), optionally exposing the LSTM output states.
It is fully decoupled from the graph family; :mod:`plgnn.models` is what wires
the two together.
"""

from __future__ import annotations

import torch

from plgnn.base import BaseModel
from plgnn.scaling import ModelStandardScaler

DEFAULT_INCREMENT_THRESHOLD = 3.25e-4


class AutoRegressiveStressRNN(BaseModel):
    """LSTM with a linear output head, mapping input to output sequences.

    The recurrence is guarded by ``increment_threshold``, the smallest move of
    the physical input (Euclidean norm over its components, in the units of
    ``input_scaler``) that advances the recurrent state. It is stored as a
    buffer, so it travels with the weights; the default is one tenth of the
    median strain increment of the training paths. Guarded, the sequence is
    processed step by step from a committed state: each step produces a
    provisional output from the committed state, the state and the committed
    input advance only where the increment reaches the threshold, and an input
    identical to the committed one returns the previously emitted output. The
    response therefore does not depend on how finely a loading path is
    discretized and stays constant while the input is held. With
    ``guarded=False``, or a threshold of zero, the whole sequence runs through
    the fused LSTM kernel, which is how the model is trained. Checkpoints
    written without the buffer load with the constructor value.
    """

    def __init__(
        self,
        input_features_size: int,
        hidden_state_size: int,
        output_size: int,
        num_layers: int,
        input_scaler: ModelStandardScaler | None = None,
        output_scaler: ModelStandardScaler | None = None,
        increment_threshold: float = DEFAULT_INCREMENT_THRESHOLD,
    ):
        super().__init__(input_scaler, output_scaler)
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
        self.register_buffer(
            "increment_threshold", torch.tensor(float(increment_threshold))
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(
            prefix + "increment_threshold", self.increment_threshold.clone()
        )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        cell_state: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        guarded: bool = True,
    ):
        unbatched = input_sequence.dim() == 2
        if unbatched:
            input_sequence = input_sequence.unsqueeze(0)
            if hidden_states is not None and cell_state is not None:
                hidden_states = hidden_states.unsqueeze(1)
                cell_state = cell_state.unsqueeze(1)
        state = (
            None
            if hidden_states is None or cell_state is None
            else (hidden_states, cell_state)
        )

        if not guarded or self.increment_threshold == 0:
            output, state = self.rnn(input_sequence, state)
        else:
            physical = input_sequence
            if self.input_scaler is not None:
                physical = self.input_scaler.inverse_transform(input_sequence)
            if state is None:
                hidden = input_sequence.new_zeros(
                    self.num_layers, len(input_sequence), self.hidden_state_size
                )
                state = (hidden, torch.zeros_like(hidden))
            committed = torch.full_like(physical[:, 0], torch.inf)
            outputs: list[torch.Tensor] = []
            for step in range(input_sequence.shape[1]):
                delta = torch.linalg.norm(physical[:, step] - committed, dim=-1)
                output, new_state = self.rnn(
                    input_sequence[:, step : step + 1], state
                )
                if outputs:
                    # A repeated input re-emits the previous output unchanged.
                    repeated = (delta == 0).view(-1, 1, 1)
                    output = torch.where(repeated, outputs[-1], output)
                outputs.append(output)
                commit = delta >= self.increment_threshold
                state = tuple(
                    torch.where(commit.view(1, -1, 1), new, old)
                    for new, old in zip(new_state, state, strict=True)
                )
                committed = torch.where(
                    commit.view(-1, 1), physical[:, step], committed
                )
            output = torch.cat(outputs, dim=1)

        preds = self.mlp_output_layer(output)
        if unbatched:
            preds, output = preds.squeeze(0), output.squeeze(0)
            state = tuple(s.squeeze(1) for s in state)
        if return_hidden_states:
            return preds, state, output
        return preds, state
