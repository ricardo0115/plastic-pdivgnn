"""Loss functions for field-prediction models."""

from __future__ import annotations

from typing import Union

import torch


def component_weighted_mse(
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    component_weights: torch.Tensor,
) -> torch.Tensor:
    """MSE normalized per component by fixed weights (1/variance)."""
    per_comp_mse = ((ground_truth.float() - predicted.float()) ** 2).mean(dim=0)
    return (per_comp_mse * component_weights).mean()


def normalized_mse_loss_single(
    ground_truth_local_stress: torch.Tensor,
    predicted_local_stress: torch.Tensor,
) -> Union[float, torch.Tensor]:
    """Per-component NMSE (MSE divided by the ground-truth variance), averaged.

    Forces float32: the NMSE division is numerically unstable in float16
    (autocast can reduce these to half, causing overflow/division-by-zero).
    """
    gt = ground_truth_local_stress.float()
    pred = predicted_local_stress.float()
    mean_gt = gt.mean(dim=0)
    mse = (gt - pred).square().sum(dim=0)
    normalization_term = (gt - mean_gt).square().sum(dim=0)
    loss = (mse / (normalization_term + 1e-8)).mean()
    return loss
