from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import fedoo as fd
import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export

from plgnn.scaling import NodeType
from plgnn.physics_fem import compute_op_div_matrix
from plgnn.datagen import Field, hole_plate_mesh_quad
from plgnn.graph.build import compute_node_labels
from plgnn.figutils import render_field_row
from plgnn.fem_sim import compute_mechanical_fields_non_linear
from plgnn.models import LstmConstitutiveLaw, PlasticGNN

COMP_LABELS: tuple[str, str, str] = (
    r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$",
)
LABELED_POINTS: tuple[tuple[float, float], ...] = (
    (0.50, 0.70),
    (0.20, 0.40),
    (0.70, 0.50),
    (0.70, 0.10),
)
POINT_NAMES: tuple[str, ...] = ("A", "B", "C", "D")
MATERIAL_PROPS: np.ndarray = np.array(
    [1e5, 0.3, 1e-5, 300.0, 1000.0, 0.3], dtype=float,
)

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "serif",
    }
)


def _random_strain_path(
    rng: np.random.Generator,
    low: float,
    high: float,
    n_macro_steps: int,
    n_increments_per_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    strain_states: np.ndarray = rng.uniform(
        low=low, high=high, size=(n_macro_steps, 3),
    )
    strain_states[:, 2] *= 2.0
    segments: list[np.ndarray] = [
        np.linspace(
            start=(0.0, 0.0, 0.0),
            stop=strain_states[0],
            num=n_increments_per_step,
            endpoint=False,
        )
    ]
    for i in range(n_macro_steps - 1):
        segments.append(
            np.linspace(
                start=strain_states[i],
                stop=strain_states[i + 1],
                num=n_increments_per_step,
                endpoint=False,
            )
        )
    segments.append(strain_states[-1])
    return np.vstack(segments), strain_states


def _run_fem(
    mesh: pv.PolyData,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> np.ndarray:
    mesh_fd: fd.Mesh = fd.Mesh.from_pyvista(mesh).as_2d()
    material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS.copy())
    local_fields, _ = compute_mechanical_fields_non_linear(
        strain_path=strain_states,
        mesh=mesh_fd,
        constitutive_law=material,
        n_increments_per_step=n_increments_per_step,
        modeling_space="2Dplane",
        verbose=False,
        nr_criterion_tol=1e-4,
    )
    return local_fields[Field.STRESS].transpose(0, 2, 1).astype(np.float32)


def _clim(arr: np.ndarray, c: int) -> tuple[float, float]:
    v: np.ndarray = arr[:, c]
    lo, hi = float(v.min()), float(v.max())
    if np.isclose(lo, hi):
        eps: float = max(1e-12, abs(lo) * 1e-4)
        lo -= eps
        hi += eps
    return lo, hi


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


def _vstack_images(paths: list[Path], outpath: Path, margin: int = 2) -> None:
    imgs: list[Image.Image] = [Image.open(p).convert("RGB") for p in paths]
    w: int = imgs[0].size[0]
    h_total: int = sum(i.size[1] for i in imgs) + margin * (len(imgs) - 1)
    canvas: Image.Image = Image.new("RGB", (w, h_total), (255, 255, 255))
    y: int = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.size[1] + margin
    canvas.save(outpath)
    for img in imgs:
        img.close()


def _save_stress_field_4row(
    quad_mesh: pv.PolyData,
    tri_mesh: pv.PolyData,
    fem_quad: np.ndarray,
    fem_tri: np.ndarray,
    gnn_quad: np.ndarray,
    gnn_tri: np.ndarray,
    outpath: Path,
) -> None:
    stress_clims: list[tuple[float, float]] = [_clim(fem_quad, c) for c in range(3)]
    rows_data: tuple = (
        (quad_mesh, fem_quad, "FEM(Quad)"),
        (tri_mesh, fem_tri, "FEM(Tri)"),
        (quad_mesh, gnn_quad, "GNN(Quad)"),
        (tri_mesh, gnn_tri, "GNN(Tri)"),
    )
    with tempfile.TemporaryDirectory(prefix="cross_mesh_4row_") as tmpdir:
        row_paths: list[Path] = []
        for i, (mesh, vals, label) in enumerate(rows_data):
            rpath: Path = Path(tmpdir) / f"row_{i:02d}.png"
            titles: tuple[str, str, str] = (
                f"{label} Stress XX",
                f"{label} Stress YY",
                f"{label} Stress XY",
            )
            _render_row(mesh, vals, titles, stress_clims, rpath)
            row_paths.append(rpath)
        _vstack_images(row_paths, outpath)


def _extract_point_series(
    stress: np.ndarray, point_ids: dict[str, int],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, pid in point_ids.items():
        out[name] = stress[:, pid, :]
    return out


def _plot_point_evolution_4sources(
    series: dict[str, dict[str, np.ndarray]],
    outpath: Path,
) -> None:
    points: list[str] = sorted(series.keys())
    n_points: int = len(points)
    sources: tuple[str, str, str, str] = (
        "FEM(Quad)", "FEM(Tri)", "GNN(Tri)", "P-DivGNN(Tri)",
    )
    # (linestyle, color, linewidth) — matches the published Figure 25
    source_styles: dict[str, tuple[str, str, float]] = {
        "FEM(Quad)": ("-", "#1f77b4", 1.6),
        "FEM(Tri)": ("-.", "#ff7f0e", 1.4),
        "GNN(Tri)": ("--", "#e377c2", 1.4),
        "P-DivGNN(Tri)": (":", "#17becf", 1.6),
    }

    fig, axes = plt.subplots(
        n_points, 3, figsize=(13, 3.0 * n_points),
        sharex=True, sharey="row", constrained_layout=True,
    )
    for row, pt in enumerate(points):
        for col in range(3):
            ax = axes[row, col]
            t: np.ndarray = np.arange(series[pt][sources[0]].shape[0])
            for src in sources:
                style, color, lw = source_styles[src]
                ax.plot(
                    t, series[pt][src][:, col], style,
                    color=color, linewidth=lw, label=src,
                )
            ax.grid(True, alpha=0.3)
            ax.tick_params(direction="in", which="both")
            if row == 0:
                ax.set_title(COMP_LABELS[col], fontsize=26)
            if col == 0:
                ax.set_ylabel(f"Point {pt}")
            if row == n_points - 1:
                ax.set_xlabel("Time step")

    axes[0, 2].legend(
        frameon=True, fancybox=True, framealpha=0.85,
        loc="upper right", fontsize=9,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_divergence_mean(
    stress: np.ndarray, op_div_matrix, node_labels: np.ndarray,
) -> float:
    stress_x_xy: np.ndarray = stress[:, [0, 2]].T.reshape(-1)
    stress_xy_y: np.ndarray = stress[:, [2, 1]].T.reshape(-1)
    stress_stack: np.ndarray = np.stack([stress_x_xy, stress_xy_y], axis=1)
    div_sigma: np.ndarray = op_div_matrix @ stress_stack
    ext_mask: np.ndarray = (node_labels == NodeType.EXTERNAL_BOUNDARY).squeeze()
    int_mask: np.ndarray = (node_labels == NodeType.INTERNAL_BOUNDARY).squeeze()
    div_sigma[ext_mask] = 0
    div_sigma[int_mask] = 0
    return float(np.sum(np.mean(np.abs(div_sigma), axis=0)))


def _plot_summary_divergence(
    divergences: dict[str, np.ndarray], outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    styles: dict[str, str] = {
        "FEM-quad": "s-", "FEM-tri": "o-",
        "GNN-quad": "s--", "GNN-tri": "o--",
    }
    for key, series in divergences.items():
        t: np.ndarray = np.arange(len(series))
        ax.plot(t, series, styles.get(key, "-"), markersize=3, label=key)
    ax.set_xlabel("Loading step")
    ax.set_ylabel("Mean $|\\nabla \\cdot \\sigma|$")
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in")
    ax.legend(frameon=True, fancybox=True, framealpha=0.85, ncol=2)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    gnn_checkpoint: str,
    pdivgnn_checkpoint: str,
    output_dir: str = "outputs/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 23839,
    strain_steps: int = 4,
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

    quad_mesh: pv.PolyData = hole_plate_mesh_quad(
        width=1.0, height=1.0, radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
    ).extract_surface()
    tri_mesh: pv.PolyData = quad_mesh.triangulate()
    if quad_mesh.n_points != tri_mesh.n_points:
        raise ValueError(
            f"Node count mismatch: quad={quad_mesh.n_points}, "
            f"tri={tri_mesh.n_points}",
        )

    strain_interp, strain_states = _random_strain_path(
        rng, strain_low, strain_high, strain_steps, increments_per_step,
    )

    fem_quad: np.ndarray = _run_fem(quad_mesh, strain_states, increments_per_step)
    fem_tri: np.ndarray = _run_fem(tri_mesh, strain_states, increments_per_step)

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()
    lstm_stress, hidden_states = lstm.forward(
        strain_interp, return_hidden_states=True,
    )
    stress_t: torch.Tensor = torch.from_numpy(
        np.asarray(lstm_stress, dtype=np.float32),
    ).to(device)
    hidden_t: torch.Tensor = torch.from_numpy(
        np.asarray(hidden_states, dtype=np.float32),
    ).to(device)

    gnn_quad_model: PlasticGNN = PlasticGNN(gnn_checkpoint, device, quad_mesh)
    gnn_quad_model.eval()
    gnn_quad: np.ndarray = np.asarray(
        gnn_quad_model.forward(stress_t, hidden_t, chunk_size=gnn_chunk_size),
        dtype=np.float32,
    )
    gnn_tri_model: PlasticGNN = PlasticGNN(gnn_checkpoint, device, tri_mesh)
    gnn_tri_model.eval()
    gnn_tri: np.ndarray = np.asarray(
        gnn_tri_model.forward(stress_t, hidden_t, chunk_size=gnn_chunk_size),
        dtype=np.float32,
    )

    pdivgnn_tri_model: PlasticGNN = PlasticGNN(pdivgnn_checkpoint, device, tri_mesh)
    pdivgnn_tri_model.eval()
    pdivgnn_tri: np.ndarray = np.asarray(
        pdivgnn_tri_model.forward(stress_t, hidden_t, chunk_size=gnn_chunk_size),
        dtype=np.float32,
    )

    _save_stress_field_4row(
        quad_mesh, tri_mesh,
        fem_quad[-1], fem_tri[-1], gnn_quad[-1], gnn_tri[-1],
        outdir / "cross_mesh_stress_field_4row.png",
    )

    point_ids: dict[str, int] = {
        name: int(quad_mesh.find_closest_point([x, y, 0.0]))
        for name, (x, y) in zip(POINT_NAMES, LABELED_POINTS, strict=False)
    }
    series: dict[str, dict[str, np.ndarray]] = {name: {} for name in POINT_NAMES}
    for src, stress in (
        ("FEM(Quad)", fem_quad), ("FEM(Tri)", fem_tri),
        ("GNN(Tri)", gnn_tri), ("P-DivGNN(Tri)", pdivgnn_tri),
    ):
        pt_series: dict[str, np.ndarray] = _extract_point_series(stress, point_ids)
        for name in POINT_NAMES:
            series[name][src] = pt_series[name]
    _plot_point_evolution_4sources(
        series, outdir / "cross_mesh_point_evolution.pdf",
    )

    op_quad = compute_op_div_matrix(quad_mesh)
    op_tri = compute_op_div_matrix(tri_mesh)
    labels_quad: np.ndarray = compute_node_labels(quad_mesh)
    labels_tri: np.ndarray = compute_node_labels(tri_mesh)
    divergences: dict[str, np.ndarray] = {
        "FEM-quad": np.array(
            [_compute_divergence_mean(s, op_quad, labels_quad) for s in fem_quad],
        ),
        "FEM-tri": np.array(
            [_compute_divergence_mean(s, op_tri, labels_tri) for s in fem_tri],
        ),
        "GNN-quad": np.array(
            [_compute_divergence_mean(s, op_quad, labels_quad) for s in gnn_quad],
        ),
        "GNN-tri": np.array(
            [_compute_divergence_mean(s, op_tri, labels_tri) for s in gnn_tri],
        ),
    }
    _plot_summary_divergence(
        divergences, outdir / "cross_mesh_summary_divergence.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
