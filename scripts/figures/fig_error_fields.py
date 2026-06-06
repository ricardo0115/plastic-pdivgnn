"""Per-node error and divergence field maps on the quad mesh (last load step).

Two outputs:

* NMSE error maps for GNN and P-DivGNN against the FE reference - two rows
  (models) x three components, sharing a per-component scale across the rows. The
  per-(node, component) error is

      e(n, c) = (fem_{n,c} - pred_{n,c})^2 / sum_i (fem_{i,c} - mean_c)^2,

  whose sum over nodes equals the per-component NMSE.
* the per-node stress-divergence norm for FE / GNN / P-DivGNN on a shared scale,
  which is where P-DivGNN improves over the plain GNN.

    python scripts/figures/fig_error_fields.py \
        --lstm-checkpoint weights/lstm.pt --gnn-checkpoint weights/gnn.pt \
        --pdivgnn-checkpoint weights/pdivgnn.pt --output-dir outputs/figures
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export

from _common import fem_local_stress, per_node_nmse_field, random_strain_path
from plgnn.datagen import hole_plate_mesh_quad
from plgnn.figutils import render_field_row, vstack_rows
from plgnn.graph.build import compute_node_labels
from plgnn.models import LstmConstitutiveLaw, PlasticGNN
from plgnn.physics import compute_divergence_norm_field
from plgnn.physics_fem import compute_op_div_matrix

COMPS: tuple[str, str, str] = ("XX", "YY", "XY")


def _run_gnn(
    mesh: pv.PolyData,
    lstm_stress_t: torch.Tensor,
    hidden_states_t: torch.Tensor,
    gnn_checkpoint: str,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    gnn: PlasticGNN = PlasticGNN(gnn_checkpoint, device, mesh)
    gnn.eval()
    out: np.ndarray = np.asarray(
        gnn.forward(lstm_stress_t, hidden_states_t, chunk_size=chunk_size),
        dtype=np.float32,
    )[-1]
    del gnn
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _shared_component_clims(*fields: np.ndarray) -> list[tuple[float, float]]:
    clims: list[tuple[float, float]] = []
    for c in range(3):
        stacked: np.ndarray = np.concatenate([f[:, c] for f in fields])
        lo, hi = float(stacked.min()), float(stacked.max())
        if np.isclose(lo, hi):
            hi = lo + max(1e-12, abs(lo) * 1e-6)
        clims.append((lo, hi))
    return clims


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    gnn_checkpoint: str,
    pdivgnn_checkpoint: str,
    output_dir: str = "outputs/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 23839,
    main_strain_steps: int = 4,
    increments_per_step: int = 25,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    global_mesh_refinement_size: float = 0.06,
    hole_refinement_factor: int = 7,
    gnn_chunk_size: int = 8,
) -> None:
    for label, ckpt in (
        ("lstm", lstm_checkpoint),
        ("gnn", gnn_checkpoint),
        ("pdivgnn", pdivgnn_checkpoint),
    ):
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {ckpt}")

    outdir: Path = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    with contextlib.suppress(Exception):
        pv.start_xvfb()

    mesh: pv.PolyData = hole_plate_mesh_quad(
        width=1.0, height=1.0, radius=0.2, hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
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

    fem_last: np.ndarray = fem_local_stress(mesh, strain_states, increments_per_step)[-1]
    gnn_last: np.ndarray = _run_gnn(
        mesh, lstm_stress_t, hidden_states_t,
        gnn_checkpoint, device, gnn_chunk_size,
    )
    pdivgnn_last: np.ndarray = _run_gnn(
        mesh, lstm_stress_t, hidden_states_t,
        pdivgnn_checkpoint, device, gnn_chunk_size,
    )

    e_gnn: np.ndarray = per_node_nmse_field(fem_last, gnn_last)
    e_pdivgnn: np.ndarray = per_node_nmse_field(fem_last, pdivgnn_last)
    nmse_clims: list[tuple[float, float]] = _shared_component_clims(e_gnn, e_pdivgnn)

    nmse_rows: list[np.ndarray] = [
        render_field_row(
            mesh, e_gnn,
            ("GNN NMSE XX", "GNN NMSE YY", "GNN NMSE XY"), nmse_clims,
        ),
        render_field_row(
            mesh, e_pdivgnn,
            ("P-DivGNN NMSE XX", "P-DivGNN NMSE YY", "P-DivGNN NMSE XY"),
            nmse_clims,
        ),
    ]
    nmse_path: Path = outdir / "error_fields_nmse_quad.pdf"
    Image.fromarray(vstack_rows(nmse_rows)).save(nmse_path.as_posix())
    print(f"Wrote {nmse_path}")

    op_div_matrix = compute_op_div_matrix(mesh)
    node_labels: np.ndarray = compute_node_labels(mesh)
    div_fem: np.ndarray = compute_divergence_norm_field(
        fem_last, op_div_matrix, node_labels,
    )
    div_gnn: np.ndarray = compute_divergence_norm_field(
        gnn_last, op_div_matrix, node_labels,
    )
    div_pdivgnn: np.ndarray = compute_divergence_norm_field(
        pdivgnn_last, op_div_matrix, node_labels,
    )
    div_values: np.ndarray = np.column_stack(
        [div_fem, div_gnn, div_pdivgnn],
    ).astype(np.float32)
    div_hi: float = max(
        float(div_fem.max()), float(div_gnn.max()), float(div_pdivgnn.max()),
    )
    div_clims: list[tuple[float, float]] = [(0.0, max(div_hi, 1e-12))] * 3

    div_row: np.ndarray = render_field_row(
        mesh, div_values,
        ("FE Divergence", "GNN Divergence", "P-DivGNN Divergence"), div_clims,
    )
    div_path: Path = outdir / "divergence_fields_quad.pdf"
    Image.fromarray(div_row).save(div_path.as_posix())
    print(f"Wrote {div_path}")


if __name__ == "__main__":
    fire.Fire(main)
