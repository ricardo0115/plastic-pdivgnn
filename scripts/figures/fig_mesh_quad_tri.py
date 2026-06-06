from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export

from _common import component_clims, fem_local_stress, random_strain_path
from plgnn.datagen import hole_plate_mesh_quad
from plgnn.figutils import render_field_row

_WIN_W: int = 2000
_ROW_H: int = 760
_CAM_ZOOM: float = 0.96


def _render_row(
    mesh: pv.PolyData,
    values: np.ndarray,
    titles: tuple[str, str, str],
    clims: list[tuple[float, float]],
    outpath: Path,
    show_edges: bool = False,
) -> None:
    row: np.ndarray = render_field_row(mesh, values, titles, clims, show_edges)
    Image.fromarray(row).save(outpath.as_posix())


def _save_fem_compare_figure(
    mesh: pv.PolyData,
    stress_last: np.ndarray,
    label: str,
    outpath: Path,
) -> None:
    clims: list[tuple[float, float]] = component_clims(stress_last)
    titles: tuple[str, str, str] = (
        f"{label} Stress XX",
        f"{label} Stress YY",
        f"{label} Stress XY",
    )
    with tempfile.TemporaryDirectory(prefix="fem_compare_") as tmpdir:
        tmp_png: Path = Path(tmpdir) / "row.png"
        _render_row(mesh, stress_last, titles, clims, tmp_png)
        img: Image.Image = Image.open(tmp_png).convert("RGB")
        img.save(outpath)
        img.close()


def _save_quad_mesh_png(mesh: pv.PolyData, outpath: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(_WIN_W, _ROW_H))
    plotter.add_mesh(
        mesh, color="white", show_edges=True,
        edge_color="black", lighting=True,
    )
    plotter.view_xy()
    plotter.camera.zoom(_CAM_ZOOM)
    plotter.screenshot(outpath.as_posix())
    plotter.close()


def main(
    output_dir: str = "outputs/figures",
    seed: int = 23839,
    main_strain_steps: int = 4,
    increments_per_step: int = 25,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    global_mesh_refinement_size: float = 0.06,
    hole_refinement_factor: int = 7,
) -> None:
    outdir: Path = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    with contextlib.suppress(Exception):
        pv.start_xvfb()

    quad_mesh: pv.PolyData = hole_plate_mesh_quad(
        width=1.0,
        height=1.0,
        radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
    ).extract_surface()
    tri_mesh: pv.PolyData = quad_mesh.triangulate()

    if quad_mesh.n_points != tri_mesh.n_points:
        raise ValueError(
            f"Quad ({quad_mesh.n_points}) and tri ({tri_mesh.n_points}) "
            "meshes must share all nodes.",
        )

    _, strain_states = random_strain_path(
        rng, strain_low, strain_high,
        main_strain_steps, increments_per_step,
    )

    quad_stress: np.ndarray = fem_local_stress(quad_mesh, strain_states, increments_per_step)
    tri_stress: np.ndarray = fem_local_stress(tri_mesh, strain_states, increments_per_step)

    _save_fem_compare_figure(
        quad_mesh, quad_stress[-1], "FEM (Quad)",
        outdir / "fem_quad_compare.pdf",
    )
    _save_fem_compare_figure(
        tri_mesh, tri_stress[-1], "FEM (Tri)",
        outdir / "fem_tri_compare.pdf",
    )
    _save_quad_mesh_png(quad_mesh, outdir / "quad_mesh.png")


if __name__ == "__main__":
    fire.Fire(main)
