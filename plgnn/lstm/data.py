"""Loading of the macroscopic loading paths written by the dataset generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def discover_simulations(data_dir: Path) -> list[Path]:
    """Return the sorted ``sim_*.npz`` files of a dataset directory."""
    paths = sorted(data_dir.glob("sim_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No simulation files sim_*.npz found in {data_dir}"
        )
    return paths


def load_macro_sequences(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(steps, 3)`` macro strain and stress paths of one simulation.

    The initial unloaded step is dropped, so the sequences start at the first
    load increment.
    """
    with np.load(npz_path) as npz:
        macro_strain = npz["macro_strain"].astype(np.float32)
        macro_stress = npz["macro_stress"].astype(np.float32)
    strain_seq = np.squeeze(macro_strain, axis=-1)
    stress_seq = np.squeeze(macro_stress, axis=-1)
    return strain_seq[1:], stress_seq[1:]
