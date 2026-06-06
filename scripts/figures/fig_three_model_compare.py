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

from plgnn.datagen import Field, hole_plate_mesh
from plgnn.fem_sim import compute_mechanical_fields_non_linear
from plgnn.figutils import render_field_row
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
COLORS: dict[str, str] = {
    "FEM": "#1f77b4", "GNN": "#d62728", "P-DivGNN": "#2ca02c",
}
STYLES: dict[str, str] = {"FEM": "-", "GNN": "--", "P-DivGNN": ":"}
LW: dict[str, float] = {"FEM": 1.6, "GNN": 1.4, "P-DivGNN": 1.4}

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.4,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (local stress [T, N, 3], macro stress [T, 3], macro strain [T, 3])."""
    mesh_fd: fd.Mesh = fd.Mesh.from_pyvista(mesh).as_2d()
    material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS.copy())
    local_fields, mean_fields = compute_mechanical_fields_non_linear(
        strain_path=strain_states,
        mesh=mesh_fd,
        constitutive_law=material,
        n_increments_per_step=n_increments_per_step,
        modeling_space="2Dplane",
        verbose=False,
        nr_criterion_tol=1e-4,
    )
    local_stress: np.ndarray = local_fields[Field.STRESS].transpose(0, 2, 1).astype(np.float32)
    macro_stress: np.ndarray = np.asarray(mean_fields[Field.STRESS]).reshape(-1, 3).astype(np.float32)
    macro_strain: np.ndarray = np.asarray(
        mean_fields[Field.TOTAL_STRAIN]
    ).reshape(-1, 3).astype(np.float32)
    return local_stress, macro_stress, macro_strain


def _component_clims(values: np.ndarray) -> list[tuple[float, float]]:
    clims: list[tuple[float, float]] = []
    for c in range(3):
        lo, hi = float(values[:, c].min()), float(values[:, c].max())
        if np.isclose(lo, hi):
            eps: float = max(1e-12, abs(lo) * 1e-6)
            lo -= eps
            hi += eps
        clims.append((lo, hi))
    return clims


def _trim_white_margins(
    path: Path, threshold: int = 245, pad: int = 0,
    *, trim_x: bool = False, trim_y: bool = True,
) -> None:
    img: Image.Image = Image.open(path).convert("RGB")
    arr: np.ndarray = np.asarray(img)
    mask: np.ndarray = np.any(arr < threshold, axis=2)
    if not mask.any():
        return
    ys, xs = np.where(mask)
    y0: int = max(0, int(ys.min()) - pad) if trim_y else 0
    y1: int = min(arr.shape[0], int(ys.max()) + 1 + pad) if trim_y else arr.shape[0]
    x0: int = max(0, int(xs.min()) - pad) if trim_x else 0
    x1: int = min(arr.shape[1], int(xs.max()) + 1 + pad) if trim_x else arr.shape[1]
    Image.fromarray(arr[y0:y1, x0:x1]).save(path)


def _concat_rows(row_paths: list[Path], outpath: Path, margin_px: int = 1) -> None:
    images: list[Image.Image] = []
    for p in row_paths:
        _trim_white_margins(p, trim_x=False, trim_y=True, pad=0)
        images.append(Image.open(p).convert("RGB"))
    widths: list[int] = [img.size[0] for img in images]
    if len(set(widths)) != 1:
        raise ValueError(f"Row widths differ: {widths}")
    total_h: int = sum(img.size[1] for img in images) + margin_px * (len(images) - 1)
    canvas: Image.Image = Image.new("RGB", (widths[0], total_h), (255, 255, 255))
    y: int = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.size[1] + margin_px
    canvas.save(outpath)
    for img in images:
        img.close()


def _render_row(
    mesh: pv.PolyData,
    values: np.ndarray,
    titles: tuple[str, str, str],
    clims: list[tuple[float, float]],
    outpath: Path,
) -> None:
    row: np.ndarray = render_field_row(mesh, values, titles, clims)
    Image.fromarray(row).save(outpath.as_posix())


def _plot_stress_field_grid(
    mesh: pv.PolyData,
    fem_last: np.ndarray,
    gnn_last: np.ndarray,
    pdivgnn_last: np.ndarray,
    outpath: Path,
) -> None:
    clims: list[tuple[float, float]] = _component_clims(fem_last)
    with tempfile.TemporaryDirectory(prefix="three_model_") as tmpdir:
        tmp: Path = Path(tmpdir)
        fem_path: Path = tmp / "fem.png"
        gnn_path: Path = tmp / "gnn.png"
        div_path: Path = tmp / "pdivgnn.png"
        _render_row(
            mesh, fem_last,
            ("FEM Stress XX", "FEM Stress YY", "FEM Stress XY"),
            clims, fem_path,
        )
        _render_row(
            mesh, gnn_last,
            ("GNN Stress XX", "GNN Stress YY", "GNN Stress XY"),
            clims, gnn_path,
        )
        _render_row(
            mesh, pdivgnn_last,
            ("P-DivGNN Stress XX", "P-DivGNN Stress YY", "P-DivGNN Stress XY"),
            clims, div_path,
        )
        _concat_rows([fem_path, gnn_path, div_path], outpath)


def _extract_point_series(
    stress: np.ndarray,
    point_ids: dict[str, int],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, pid in point_ids.items():
        out[name] = stress[:, pid, :]
    return out


def _plot_point_evolution(
    fem_series: dict[str, np.ndarray],
    gnn_series: dict[str, np.ndarray],
    pdivgnn_series: dict[str, np.ndarray],
    outpath: Path,
) -> None:
    points: list[str] = sorted(fem_series.keys())
    n_points: int = len(points)

    fig, axes = plt.subplots(
        n_points, 3, figsize=(13, 3.0 * n_points),
        sharex=True, sharey="row", constrained_layout=True,
    )
    for row, pt in enumerate(points):
        for col in range(3):
            ax = axes[row, col]
            t: np.ndarray = np.arange(fem_series[pt].shape[0])
            ax.plot(
                t, fem_series[pt][:, col], STYLES["FEM"],
                color=COLORS["FEM"], linewidth=LW["FEM"], label="FEM",
            )
            ax.plot(
                t, gnn_series[pt][:, col], STYLES["GNN"],
                color=COLORS["GNN"], linewidth=LW["GNN"], label="GNN",
            )
            ax.plot(
                t, pdivgnn_series[pt][:, col], STYLES["P-DivGNN"],
                color=COLORS["P-DivGNN"], linewidth=LW["P-DivGNN"],
                label="P-DivGNN",
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
        loc="upper right", fontsize=11,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_macro_stress_strain(
    fem_strain: np.ndarray,
    fem_stress: np.ndarray,
    lstm_strain: np.ndarray,
    lstm_stress: np.ndarray,
    outpath: Path,
) -> None:
    """Macroscopic stress-strain response, FEM (solid) vs LSTM (dashed)."""
    comps: tuple[str, str, str] = ("xx", "yy", "xy")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for c, ax in enumerate(axes):
        ax.plot(
            fem_strain[:, c], fem_stress[:, c], "-",
            color=COLORS["FEM"], linewidth=1.8, label="FEM",
        )
        ax.plot(
            lstm_strain[:, c], lstm_stress[:, c], "--",
            color="#d62728", linewidth=1.5, label="LSTM",
        )
        ax.set_xlabel(rf"Macro strain $\bar{{\varepsilon}}_{{{comps[c]}}}$")
        ax.set_ylabel(rf"Macro stress $\bar{{\sigma}}_{{{comps[c]}}}$ [MPa]")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both")
    axes[2].legend(
        frameon=True, fancybox=True, framealpha=0.85, loc="best", fontsize=11,
    )
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

    mesh: pv.PolyData = hole_plate_mesh(
        width=1.0, height=1.0, radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
        mesh_type="quad",
    ).extract_surface()

    strain_interp, strain_states = _random_strain_path(
        rng, strain_low, strain_high,
        main_strain_steps, increments_per_step,
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

    fem_stress, fem_macro_stress, fem_macro_strain = _run_fem(
        mesh, strain_states, increments_per_step,
    )

    gnn: PlasticGNN = PlasticGNN(gnn_checkpoint, device, mesh)
    gnn.eval()
    gnn_stress: np.ndarray = np.asarray(
        gnn.forward(lstm_stress_t, hidden_states_t, chunk_size=gnn_chunk_size),
        dtype=np.float32,
    )

    pdivgnn: PlasticGNN = PlasticGNN(pdivgnn_checkpoint, device, mesh)
    pdivgnn.eval()
    pdivgnn_stress: np.ndarray = np.asarray(
        pdivgnn.forward(lstm_stress_t, hidden_states_t, chunk_size=gnn_chunk_size),
        dtype=np.float32,
    )

    _plot_stress_field_grid(
        mesh,
        fem_stress[-1], gnn_stress[-1], pdivgnn_stress[-1],
        outdir / "stress_field_3model_compare_quad.png",
    )

    point_ids: dict[str, int] = {
        name: int(mesh.find_closest_point([x, y, 0.0]))
        for name, (x, y) in zip(POINT_NAMES, LABELED_POINTS, strict=False)
    }
    fem_series: dict[str, np.ndarray] = _extract_point_series(fem_stress, point_ids)
    gnn_series: dict[str, np.ndarray] = _extract_point_series(gnn_stress, point_ids)
    pdivgnn_series: dict[str, np.ndarray] = _extract_point_series(
        pdivgnn_stress, point_ids,
    )

    _plot_point_evolution(
        fem_series, gnn_series, pdivgnn_series,
        outdir / "point_evolution_3model_quad.pdf",
    )

    _plot_macro_stress_strain(
        fem_macro_strain, fem_macro_stress,
        np.asarray(strain_interp, dtype=np.float32),
        np.asarray(lstm_stress, dtype=np.float32),
        outdir / "stress_strain_fem_vs_lstm_quad.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
