"""Temporal resampling of loading paths for training.

A recurrent model trained on paths sampled at one fixed step count can tie its
response to that step size. Resampling every mini-batch to a random number of
steps exposes the network to many increment sizes, so the learned response
depends on the path and not on its discretization.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def resample_in_time(sequences: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Linearly re-discretize ``(batch, steps, channels)`` sequences to ``n_steps``.

    The first and last samples are kept in place; intermediate samples are
    linearly interpolated between the original time points.
    """
    resampled = F.interpolate(
        sequences.permute(0, 2, 1),
        size=n_steps,
        mode="linear",
        align_corners=True,
    )
    return resampled.permute(0, 2, 1)


class RandomResampleCollate:
    """Collate that resamples each batch to a common random step count.

    The step count is drawn log-uniformly in ``[min_steps, max_steps]`` so that
    coarse and fine discretizations are equally represented. Both the input and
    the target sequences of the batch are resampled to the same step count.
    """

    def __init__(self, min_steps: int, max_steps: int):
        if not 1 < min_steps <= max_steps:
            raise ValueError("min_steps must satisfy 1 < min_steps <= max_steps")
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.stack([x for x, _ in batch])
        targets = torch.stack([y for _, y in batch])
        n_steps = round(
            math.exp(
                random.uniform(math.log(self.min_steps), math.log(self.max_steps))
            )
        )
        if n_steps == inputs.shape[1]:
            return inputs, targets
        return resample_in_time(inputs, n_steps), resample_in_time(targets, n_steps)
