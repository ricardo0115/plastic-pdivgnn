"""Normalization helpers and node-type labels shared across model families."""

from __future__ import annotations

from enum import IntEnum

import torch


class NodeType(IntEnum):
    """Mesh node classification used for boundary-aware operations."""

    INTERNAL_BOUNDARY = -1
    INTERNAL = 0
    EXTERNAL_BOUNDARY = 1


class ModelStandardScaler:
    """Tensor z-score scaler with fixed mean/std (stored in model checkpoints)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean: torch.Tensor | float = mean
        self.std: torch.Tensor | float = std

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def standardize(
    data: torch.Tensor,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
) -> torch.Tensor:
    return (data - mean) / std


def unstandardize(
    data: torch.Tensor,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
) -> torch.Tensor:
    return (data * std) + mean
