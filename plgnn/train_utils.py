"""Paper-specific training data wiring (snapshot indices, train/val split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Timesteps kept per simulation for GNN training (one per macro strain step,
# matching the four-segment loading path used in the dataset).
SNAPSHOT_INDICES: tuple[int, ...] = (0, 24, 49, 74, 99)


def split_pairs(
    pairs: list[tuple[Path, Path]],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic train/validation split of (sim, hidden-state) path pairs."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_test = max(1, int(len(pairs) * test_fraction))
    val = [pairs[i] for i in perm[:n_test]]
    train = [pairs[i] for i in perm[n_test:]]
    return train, val
