"""Regenerate the FEM / GNN comparison movies.

Writes up to three ``.mp4`` files next to ``--output_path``:

* ``<name>.mp4`` - 3-row comparison: FEM stress row, P-DivGNN prediction row
  (sharing the FEM per-frame colorbar limits), and the macro strain-stress
  curve (FEM solid + LSTM dashed) with a moving dot;
* ``<name>_error.mp4`` - per-node normalized squared error field (3 components);
* ``<name>_divergence.mp4`` - FEM / GNN / P-DivGNN ``|div(sigma)|`` side by side
  (all sharing the FEM per-frame colorbar range).

The two extra movies are controlled by ``--include_error`` / ``--include_divergence``.

Example
-------
    python scripts/figures/fig_movie_fem_gnn.py \
        --lstm_checkpoint weights/lstm.pt \
        --gnn_checkpoint weights/pdivgnn.pt \
        --output_path outputs/fem_gnn_movie.mp4
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
import torch

from _common import fem_local_and_macro, random_strain_path
from plgnn.datagen import hole_plate_mesh
from plgnn.graph.build import compute_node_labels
from plgnn.models import LstmConstitutiveLaw, PlasticGNN
from plgnn.movie import (
    write_comparison_movie,
    write_divergence_field_movie,
    write_error_field_movie,
)
from plgnn.physics import compute_divergence_norm_field
from plgnn.physics_fem import compute_op_div_matrix


def _normalized_error_field(
    fem_field: np.ndarray, gnn_field: np.ndarray,
) -> np.ndarray:
    """Per-node, per-component normalized squared error (NSE), ``[T, N, 3]``.

    For each frame and component: ``(fem - gnn)^2`` normalized by the spatial
    variance of the FEM field, matching the legacy element-wise NSE definition.
    """
    errors: list[np.ndarray] = []
    for fem_t, gnn_t in zip(fem_field, gnn_field, strict=True):
        mean_gt: np.ndarray = fem_t.mean(axis=0)
        squared: np.ndarray = (fem_t - gnn_t) ** 2
        normalization: np.ndarray = ((fem_t - mean_gt) ** 2).sum(axis=0)
        errors.append(squared / (normalization + 1e-16))
    return np.stack(errors).astype(np.float32)


def _divergence_norm_sequence(
    field: np.ndarray,
    op_div_matrix: np.ndarray,
    node_labels: np.ndarray,
) -> np.ndarray:
    """Per-node ``|div(sigma)|`` for every frame, ``[T, N]``."""
    return np.stack(
        [
            compute_divergence_norm_field(field_t, op_div_matrix, node_labels)
            for field_t in field
        ],
    ).astype(np.float32)


def _run_gnn(
    checkpoint: str,
    device: str,
    mesh: pv.PolyData,
    lstm_stress_t: torch.Tensor,
    hidden_states_t: torch.Tensor,
    chunk_size: int,
) -> np.ndarray:
    """Reconstruct the local stress field from one GNN checkpoint, ``[T, N, 3]``."""
    gnn: PlasticGNN = PlasticGNN(checkpoint, device, mesh)
    gnn.eval()
    return np.asarray(
        gnn.forward(lstm_stress_t, hidden_states_t, chunk_size=chunk_size),
        dtype=np.float32,
    )


@torch.no_grad()
def main(
    lstm_checkpoint: str = "weights/lstm.pt",
    gnn_checkpoint: str = "weights/pdivgnn.pt",
    vanilla_gnn_checkpoint: str = "weights/gnn.pt",
    output_path: str = "outputs/fem_gnn_movie.mp4",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 23839,
    main_strain_steps: int = 4,
    increments_per_step: int = 25,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    global_mesh_refinement_size: float = 0.06,
    hole_refinement_factor: int = 7,
    gnn_chunk_size: int = 8,
    fps: int = 10,
    model_label: str = "P-DivGNN",
    include_error: bool = True,
    include_divergence: bool = True,
) -> None:
    required: list[tuple[str, str]] = [
        ("lstm", lstm_checkpoint), ("gnn", gnn_checkpoint),
    ]
    if include_divergence:
        required.append(("vanilla gnn", vanilla_gnn_checkpoint))
    for label, ckpt in required:
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {ckpt}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    with contextlib.suppress(Exception):
        pv.start_xvfb()

    mesh: pv.PolyData = hole_plate_mesh(
        width=1.0, height=1.0, radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
        mesh_type="quad",
    ).extract_surface()

    strain_interp, strain_states = random_strain_path(
        rng, strain_low, strain_high, main_strain_steps, increments_per_step,
    )

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()
    lstm_stress, hidden_states = lstm.forward(
        strain_interp, return_hidden_states=True,
    )
    lstm_stress_t: torch.Tensor = torch.from_numpy(
        np.asarray(lstm_stress, dtype=np.float32),
    ).to(device)
    hidden_states_t: torch.Tensor = torch.from_numpy(
        np.asarray(hidden_states, dtype=np.float32),
    ).to(device)

    gnn_field: np.ndarray = _run_gnn(
        gnn_checkpoint, device, mesh,
        lstm_stress_t, hidden_states_t, gnn_chunk_size,
    )

    fem_field, fem_macro, _ = fem_local_and_macro(mesh, strain_states, increments_per_step)
    lstm_macro: np.ndarray = np.asarray(lstm_stress, dtype=np.float32)

    out: Path = write_comparison_movie(
        mesh=mesh,
        fem_field=fem_field,
        gnn_field=gnn_field,
        macro_strain=strain_interp.astype(np.float32),
        fem_macro=fem_macro,
        lstm_macro=lstm_macro,
        out_path=output_path,
        fps=fps,
        model_label=model_label,
    )
    n_frames: int = fem_field.shape[0]
    print(f"Wrote {out} ({n_frames} frames @ {fps} fps)")

    base: Path = Path(output_path)
    if include_error:
        error_field: np.ndarray = _normalized_error_field(fem_field, gnn_field)
        error_out: Path = write_error_field_movie(
            mesh=mesh,
            error_field=error_field,
            out_path=base.with_name(f"{base.stem}_error{base.suffix}"),
            fps=fps,
        )
        print(f"Wrote {error_out} ({n_frames} frames @ {fps} fps)")

    if include_divergence:
        vanilla_field: np.ndarray = _run_gnn(
            vanilla_gnn_checkpoint, device, mesh,
            lstm_stress_t, hidden_states_t, gnn_chunk_size,
        )
        op_div = compute_op_div_matrix(mesh)
        node_labels: np.ndarray = compute_node_labels(mesh)
        # FEM reference, vanilla GNN, and the prediction model (default P-DivGNN).
        divergences: list[np.ndarray] = [
            _divergence_norm_sequence(fem_field, op_div, node_labels),
            _divergence_norm_sequence(vanilla_field, op_div, node_labels),
            _divergence_norm_sequence(gnn_field, op_div, node_labels),
        ]
        labels: list[str] = [
            "Divergence FEM", "Divergence GNN", f"Divergence {model_label}",
        ]
        div_out: Path = write_divergence_field_movie(
            mesh=mesh,
            divergences=divergences,
            labels=labels,
            out_path=base.with_name(f"{base.stem}_divergence{base.suffix}"),
            fps=fps,
        )
        print(f"Wrote {div_out} ({n_frames} frames @ {fps} fps)")


if __name__ == "__main__":
    fire.Fire(main)
