from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
import torch
from PIL import Image

import _tri6
from _common import fem_local_stress, random_strain_path, run_gnn_on_mesh
from _labels import add_row_labels_to_png
from plgnn.datagen import hole_plate_mesh, hole_plate_mesh_tri6
from plgnn.figutils import render_field_row, vstack_rows
from plgnn.models import LstmConstitutiveLaw

ROW_LABELS: tuple[str, str, str] = ("Coarse mesh", "Medium mesh", "Fine mesh")
_LEVELS: tuple[tuple[str, float], ...] = (
    ("coarse", 0.20), ("medium", 0.12), ("fine", 0.06),
)

_CAM_ZOOM: float = 0.95


def _build_tri6_refinement_meshes(
    width: float,
    height: float,
    radius: float,
    hole_refinement_factor: float,
) -> list[pv.UnstructuredGrid]:
    """Three genuine h-refinement quadratic-triangle (tri6) meshes."""
    return [
        hole_plate_mesh_tri6(
            width=width, height=height, radius=radius,
            hole_center=(width / 2.0, height / 2.0),
            hole_refinement_factor=hole_refinement_factor,
            global_mesh_refinement_size=size,
        )
        for _, size in _LEVELS
    ]


def _render_stress_grid(
    meshes: list[pv.UnstructuredGrid],
    final_stresses: list[np.ndarray],
    clims: list[tuple[float, float]],
    outpath: Path,
    model_label: str,
) -> None:
    titles: tuple[str, str, str] = (
        f"{model_label} Stress XX",
        f"{model_label} Stress YY",
        f"{model_label} Stress XY",
    )
    rows: list[np.ndarray] = [
        _tri6.render_field_row(mesh, stress_last, titles, clims)
        for mesh, stress_last in zip(meshes, final_stresses, strict=False)
    ]
    Image.fromarray(vstack_rows(rows)).save(outpath.as_posix())


def _save_quad_geometry(mesh: pv.PolyData, outpath: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.add_mesh(
        mesh, color="white", show_edges=True,
        edge_color="black", lighting=True,
    )
    plotter.camera_position = "xy"
    plotter.camera.zoom(_CAM_ZOOM)
    plotter.screenshot(outpath.as_posix())
    plotter.close()


def _save_quad_stress_last(
    mesh: pv.PolyData, stress_last: np.ndarray, outpath: Path,
) -> None:
    clims: list[tuple[float, float]] = [
        (float(stress_last[:, c].min()), float(stress_last[:, c].max()))
        for c in range(3)
    ]
    titles: tuple[str, str, str] = (
        "FEM (Quad) Stress XX",
        "FEM (Quad) Stress YY",
        "FEM (Quad) Stress XY",
    )
    row: np.ndarray = render_field_row(mesh, stress_last, titles, clims)
    Image.fromarray(row).save(outpath.as_posix())


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
    hole_refinement_factor: float = 7.0,
    ref_quad_hole_refinement_factor: float = 7.0,
    ref_quad_mesh_size: float = 0.06,
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

    tri_meshes: list[pv.UnstructuredGrid] = _build_tri6_refinement_meshes(
        width=1.0, height=1.0, radius=0.2,
        hole_refinement_factor=hole_refinement_factor,
    )
    tri_surfaces: list[pv.PolyData] = [m.extract_surface() for m in tri_meshes]

    quad_ref_mesh: pv.PolyData = hole_plate_mesh(
        width=1.0, height=1.0, radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=ref_quad_hole_refinement_factor,
        global_mesh_refinement_size=ref_quad_mesh_size,
        mesh_type="quad",
    ).extract_surface()
    _save_quad_geometry(
        quad_ref_mesh, outdir / "refinement_ref_quad_mesh_tri.png",
    )

    strain_interp, strain_states = random_strain_path(
        rng, strain_low, strain_high,
        main_strain_steps, increments_per_step,
    )
    quad_stress_seq: np.ndarray = fem_local_stress(
        quad_ref_mesh, strain_states, increments_per_step,
    )
    _save_quad_stress_last(
        quad_ref_mesh, quad_stress_seq[-1],
        outdir / "refinement_ref_quad_stress_tri.png",
    )
    quad_last: np.ndarray = quad_stress_seq[-1]
    clims: list[tuple[float, float]] = [
        (float(quad_last[:, c].min()), float(quad_last[:, c].max()))
        for c in range(3)
    ]

    fem_stresses_last: list[np.ndarray] = []
    for i, mesh in enumerate(tri_meshes):
        fem_seq: np.ndarray = fem_local_stress(mesh, strain_states, increments_per_step)
        fem_stresses_last.append(fem_seq[-1])
        print(f"FEM done mesh {i + 1}/{len(tri_meshes)}")

    with tempfile.TemporaryDirectory(prefix="fem_grid_") as tmpdir:
        fem_grid_tmp: Path = Path(tmpdir) / "fem.png"
        _render_stress_grid(
            tri_meshes, fem_stresses_last, clims, fem_grid_tmp, "FEM",
        )
        add_row_labels_to_png(
            fem_grid_tmp, outdir / "refinement_grid_fem_tri.png",
            labels=ROW_LABELS[: len(tri_meshes)],
            font_size=row_label_font_size,
        )

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()
    lstm_stress, hidden_states = lstm.forward(
        strain_interp, return_hidden_states=True,
    )
    lstm_stress_last: np.ndarray = np.asarray(lstm_stress)[-1]
    hidden_state_last: np.ndarray = np.asarray(hidden_states)[-1]

    gnn_stresses_last: list[np.ndarray] = [
        run_gnn_on_mesh(
            surface, lstm_stress_last, hidden_state_last,
            gnn_checkpoint, device, gnn_chunk_size,
        )
        for surface in tri_surfaces
    ]

    with tempfile.TemporaryDirectory(prefix="gnn_grid_") as tmpdir:
        gnn_grid_tmp: Path = Path(tmpdir) / "gnn.png"
        _render_stress_grid(
            tri_meshes, gnn_stresses_last, clims, gnn_grid_tmp, "GNN",
        )
        add_row_labels_to_png(
            gnn_grid_tmp, outdir / "refinement_grid_gnn_tri.png",
            labels=ROW_LABELS[: len(tri_meshes)],
            font_size=row_label_font_size,
        )

    pdivgnn_stresses_last: list[np.ndarray] = [
        run_gnn_on_mesh(
            surface, lstm_stress_last, hidden_state_last,
            pdivgnn_checkpoint, device, gnn_chunk_size,
        )
        for surface in tri_surfaces
    ]

    with tempfile.TemporaryDirectory(prefix="pdivgnn_grid_") as tmpdir:
        pdivgnn_grid_tmp: Path = Path(tmpdir) / "pdivgnn.png"
        _render_stress_grid(
            tri_meshes, pdivgnn_stresses_last, clims, pdivgnn_grid_tmp,
            "P-DivGNN",
        )
        add_row_labels_to_png(
            pdivgnn_grid_tmp, outdir / "refinement_grid_pdivgnn_tri.png",
            labels=ROW_LABELS[: len(tri_meshes)],
            font_size=row_label_font_size,
        )


if __name__ == "__main__":
    fire.Fire(main)
