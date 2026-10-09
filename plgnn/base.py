"""Shared model infrastructure: the checkpoint format of every model family.

`BaseModel` is the common parent of every trainable model family
(:mod:`plgnn.graph`, :mod:`plgnn.lstm`). It carries optional input/output
scalers and a uniform checkpoint format so weights and normalization stats are
saved and restored together.
"""

from __future__ import annotations

from typing import Any

import torch

from plgnn.scaling import ModelStandardScaler


class BaseModel(torch.nn.Module):
    """Base model carrying optional scalers and a uniform checkpoint format."""

    def __init__(
        self,
        input_scaler: ModelStandardScaler | None = None,
        output_scaler: ModelStandardScaler | None = None,
    ):
        super().__init__()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def load_model_checkpoint(
        self,
        filename: str,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        checkpoint = torch.load(
            filename, weights_only=False, map_location="cpu"
        )
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        self.input_scaler = ModelStandardScaler(**checkpoint["input_scaler"])
        self.output_scaler = ModelStandardScaler(**checkpoint["output_scaler"])
        return epoch

    def save_model_checkpoint(
        self,
        filename: str,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = -1,
    ) -> None:
        if self.input_scaler is None or self.output_scaler is None:
            raise ValueError(
                "save_model_checkpoint requires input_scaler and output_scaler "
                "to be set on the model."
            )
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "input_scaler": {
                "mean": self.input_scaler.mean,
                "std": self.input_scaler.std,
            },
            "output_scaler": {
                "mean": self.output_scaler.mean,
                "std": self.output_scaler.std,
            },
        }
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, filename)
