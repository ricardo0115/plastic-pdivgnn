"""Per-node NMSE error maps on the fine tri6 mesh.

Two grids, both using the per-(node, component) NMSE field

    e_c(n) = (sigma^FE_{n,c} - sigma^pred_{n,c})^2 / sum_i (sigma^FE_{i,c} - mean_c)^2,

whose sum over nodes equals the per-component NMSE:

* fine-level (same mesh) - GNN(Tri6) and P-DivGNN(Tri6) vs FE(Tri6), exact, no
  interpolation;
* cross-resolution - FE / GNN / P-DivGNN evaluated on the coarse mesh and
  interpolated onto the fine mesh, compared against FE(fine). Each surrogate row
  folds the coarse->fine discretization gap on top of its own error; the FE(coarse)
  row is the pure discretization baseline.

    python scripts/figures/fig_refinement_error.py \
        --lstm-checkpoint weights/lstm.pt --gnn-checkpoint weights/gnn.pt \
        --pdivgnn-checkpoint weights/pdivgnn.pt --output-dir outputs/figures
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export
from scipy.spatial import cKDTree

import _tri6
from _common import (
    fem_local_stress,
    per_node_nmse_field,
    random_strain_path,
    run_gnn_on_mesh,
)
from _labels import add_row_labels_to_png
from plgnn.datagen import hole_plate_mesh_tri6
from plgnn.figutils import vstack_rows
from plgnn.models import LstmConstitutiveLaw

COMPS: tuple[str, str, str] = ("XX", "YY", "XY")
COARSE_SIZE: float = 0.20
FINE_SIZE: float = 0.06


def _tri6_connectivity(grid: pv.DataSet) -> np.ndarray:
    conn: np.ndarray = np.empty((grid.n_cells, 6), dtype=np.int64)
    for i in range(grid.n_cells):
        conn[i] = grid.get_cell(i).point_ids[:6]
    return conn


def _interp_to_fine(
    points_coarse: np.ndarray,
    conn_coarse: np.ndarray,
    field_coarse: np.ndarray,
    points_fine: np.ndarray,
) -> np.ndarray:
    """Interpolate a coarse tri6 nodal field onto the fine-mesh node positions.

    The coarse field is sampled on its genuine quadratic-triangle grid (curved
    hole boundary respected); fine nodes that fall outside every coarse cell fall
    back to the nearest coarse node.
    """
    def _to_3d(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p if p.shape[1] == 3 else np.column_stack([p, np.zeros(len(p))])

    cells: np.ndarray = np.hstack(
        [np.full((conn_coarse.shape[0], 1), 6, dtype=np.int64),
         conn_coarse.astype(np.int64)],
    ).ravel()
    cell_types: np.ndarray = np.full(
        conn_coarse.shape[0], pv.CellType.QUADRATIC_TRIANGLE, dtype=np.uint8,
    )
    grid = pv.UnstructuredGrid(cells, cell_types, _to_3d(points_coarse))
    grid.point_data["field"] = field_coarse.astype(np.float64)
    probe = pv.PolyData(_to_3d(points_fine)).sample(grid)
    out: np.ndarray = np.asarray(probe["field"], dtype=np.float64)
    valid: np.ndarray = np.asarray(probe["vtkValidPointMask"]).astype(bool)
    if not valid.all():
        _, nearest = cKDTree(points_coarse[:, :2]).query(points_fine[~valid, :2])
        out[~valid] = field_coarse[nearest]
    return out.astype(np.float32)


def _render_error_grid(
    fine_mesh: pv.UnstructuredGrid,
    errors: dict[str, np.ndarray],
    rows: tuple[tuple[str, str], ...],
    out_path: Path,
    row_label_font_size: int,
) -> None:
    clims: list[tuple[float, float]] = [
        (0.0, max(float(errors[key][:, c].max()) for key, _ in rows))
        for c in range(3)
    ]
    row_images: list[np.ndarray] = []
    for key, label in rows:
        titles: tuple[str, str, str] = tuple(  # type: ignore[assignment]
            f"{label} NMSE {c}" for c in COMPS
        )
        row_images.append(
            _tri6.render_field_row(fine_mesh, errors[key], titles, clims),
        )
    with tempfile.TemporaryDirectory(prefix="refinement_error_") as tmpdir:
        grid_png: Path = Path(tmpdir) / "grid.png"
        Image.fromarray(vstack_rows(row_images)).save(grid_png.as_posix())
        add_row_labels_to_png(
            grid_png, out_path,
            labels=[label for _, label in rows],
            font_size=row_label_font_size,
        )


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
    hole_refinement_factor: int = 7,
    gnn_chunk_size: int = 8,
    row_label_font_size: int = 44,
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

    def _tri6_mesh(size: float) -> pv.UnstructuredGrid:
        return hole_plate_mesh_tri6(
            width=1.0, height=1.0, radius=0.2, hole_center=(0.5, 0.5),
            hole_refinement_factor=hole_refinement_factor,
            global_mesh_refinement_size=size,
        )

    coarse_mesh: pv.UnstructuredGrid = _tri6_mesh(COARSE_SIZE)
    fine_mesh: pv.UnstructuredGrid = _tri6_mesh(FINE_SIZE)
    coarse_surface: pv.PolyData = coarse_mesh.extract_surface()
    fine_surface: pv.PolyData = fine_mesh.extract_surface()

    strain_interp, strain_states = random_strain_path(
        rng, strain_low, strain_high, main_strain_steps, increments_per_step,
    )

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()
    lstm_stress, hidden_states = lstm.forward(
        strain_interp, return_hidden_states=True,
    )
    lstm_stress_last: np.ndarray = np.asarray(lstm_stress)[-1]
    hidden_state_last: np.ndarray = np.asarray(hidden_states)[-1]

    fem_fine: np.ndarray = fem_local_stress(fine_mesh, strain_states, increments_per_step)[-1]
    gnn_fine: np.ndarray = run_gnn_on_mesh(
        fine_surface, lstm_stress_last, hidden_state_last,
        gnn_checkpoint, device, gnn_chunk_size,
    )
    pdivgnn_fine: np.ndarray = run_gnn_on_mesh(
        fine_surface, lstm_stress_last, hidden_state_last,
        pdivgnn_checkpoint, device, gnn_chunk_size,
    )

    fine_errors: dict[str, np.ndarray] = {
        "gnn": per_node_nmse_field(fem_fine, gnn_fine),
        "pdivgnn": per_node_nmse_field(fem_fine, pdivgnn_fine),
    }
    _render_error_grid(
        fine_mesh, fine_errors,
        (("gnn", "GNN(Tri6)"), ("pdivgnn", "P-DivGNN(Tri6)")),
        outdir / "refinement_error_fine.png", row_label_font_size,
    )
    print(f"Wrote {outdir / 'refinement_error_fine.png'}")

    fem_coarse: np.ndarray = fem_local_stress(
        coarse_mesh, strain_states, increments_per_step,
    )[-1]
    gnn_coarse: np.ndarray = run_gnn_on_mesh(
        coarse_surface, lstm_stress_last, hidden_state_last,
        gnn_checkpoint, device, gnn_chunk_size,
    )
    pdivgnn_coarse: np.ndarray = run_gnn_on_mesh(
        coarse_surface, lstm_stress_last, hidden_state_last,
        pdivgnn_checkpoint, device, gnn_chunk_size,
    )

    conn_coarse: np.ndarray = _tri6_connectivity(coarse_mesh)
    points_coarse: np.ndarray = np.asarray(coarse_mesh.points)[:, :2]
    points_fine: np.ndarray = np.asarray(fine_mesh.points)[:, :2]
    coarse_on_fine: dict[str, np.ndarray] = {
        key: _interp_to_fine(points_coarse, conn_coarse, field, points_fine)
        for key, field in (
            ("fem", fem_coarse), ("gnn", gnn_coarse), ("pdivgnn", pdivgnn_coarse)
        )
    }
    cross_errors: dict[str, np.ndarray] = {
        key: per_node_nmse_field(fem_fine, field)
        for key, field in coarse_on_fine.items()
    }
    _render_error_grid(
        fine_mesh, cross_errors,
        (("fem", "FE(coarse)"), ("gnn", "GNN(coarse)"),
         ("pdivgnn", "P-DivGNN(coarse)")),
        outdir / "refinement_error_cross.png", row_label_font_size,
    )
    print(f"Wrote {outdir / 'refinement_error_cross.png'}")


if __name__ == "__main__":
    fire.Fire(main)
