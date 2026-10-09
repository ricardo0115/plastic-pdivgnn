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
    """Tensor z-score scaler with fixed mean/std (stored in model checkpoints).

    The statistics are moved to the device and dtype of the tensor being
    scaled, so a scaler restored from a checkpoint on the CPU applies to
    tensors living on any device.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean: torch.Tensor | float = mean
        self.std: torch.Tensor | float = std

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor | float, ...]:
        return tuple(
            s.to(x.device, x.dtype) if isinstance(s, torch.Tensor) else s
            for s in (self.mean, self.std)
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(x)
        return (x - mean) / std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(x)
        return x * std + mean


def unstandardize(
    data: torch.Tensor,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
) -> torch.Tensor:
    return (data * std) + mean
