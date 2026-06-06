from __future__ import annotations

import contextlib
from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.ticker import MaxNLocator

from plgnn.datagen import Field, hole_plate_mesh
from plgnn.models import LstmConstitutiveLaw

from _common import random_strain_path, solve_fem

COMPONENTS: tuple[str, ...] = ("xx", "yy", "xy")
LABELED_POINTS: tuple[tuple[float, float], ...] = (
    (0.50, 0.70),
    (0.20, 0.40),
    (0.70, 0.50),
    (0.70, 0.10),
)

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


def _alphabetical_labels(n: int) -> list[str]:
    labels: list[str] = []
    for i in range(n):
        s: str = ""
        k: int = i
        while True:
            s = chr(ord("A") + (k % 26)) + s
            k = k // 26 - 1
            if k < 0:
                break
        labels.append(s)
    return labels


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.margins(x=0.06, y=0.09)


def _pad_xlim(ax: plt.Axes, x: np.ndarray) -> None:
    if np.isfinite(x).all() and x.size > 1:
        xpad: float = 0.03 * (np.nanmax(x) - np.nanmin(x) + 1e-12)
        ax.set_xlim(np.nanmin(x) - xpad, np.nanmax(x) + xpad)


def _legend(ax: plt.Axes) -> None:
    ax.legend(
        loc="best", frameon=True, fancybox=True, framealpha=0.75,
        edgecolor="0.6", fontsize=11, borderpad=0.35,
        labelspacing=0.25, handlelength=1.6,
    )


def _plot_mesh_points_labeled(
    mesh: pv.PolyData,
    points: tuple[tuple[float, float], ...],
    outpath: Path,
) -> None:
    labels: list[str] = _alphabetical_labels(len(points))
    point_ids: list[int] = [
        int(mesh.find_closest_point([x, y, 0.0])) for (x, y) in points
    ]
    pts: np.ndarray = mesh.points[point_ids]

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    plotter.add_mesh(mesh, color="white", show_edges=True)
    pts_poly: pv.PolyData = pv.PolyData(pts)
    plotter.add_mesh(
        pts_poly, color="red", point_size=12.0,
        render_points_as_spheres=False,
    )
    plotter.add_point_labels(
        pts, labels, text_color="black",
        font_size=48, shape=None, point_color="red",
    )
    plotter.view_xy()
    plotter.save_graphic(outpath.as_posix())
    plotter.close()


def _plot_strain_path(strain_interp: np.ndarray, outpath: Path) -> None:
    t: np.ndarray = np.arange(strain_interp.shape[0])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for i, comp in enumerate(COMPONENTS):
        ax.plot(
            t, strain_interp[:, i], linestyle="--",
            solid_capstyle="round",
            label=rf"$\varepsilon_{{{comp}}}$",
        )
    ax.set_xlabel("Loading step")
    ax.set_ylabel("Total strain")
    _legend(ax)
    _style_axis(ax)
    _pad_xlim(ax, t)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_stress_strain_triptych(
    mean_fields: dict,
    lstm_stress: np.ndarray,
    outpath: Path,
) -> None:
    strain_fem: np.ndarray = mean_fields[Field.TOTAL_STRAIN][:, :, 0]
    stress_fem: np.ndarray = mean_fields[Field.STRESS][:, :, 0]

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), sharey=True, constrained_layout=True,
    )
    fig.set_constrained_layout_pads(wspace=0.1, hspace=0.08)

    all_stress: np.ndarray = np.concatenate([stress_fem, lstm_stress], axis=0)
    ymin: float = float(np.min(all_stress))
    ymax: float = float(np.max(all_stress))
    ypad: float = 0.06 * (ymax - ymin)

    for i, (comp, ax) in enumerate(zip(COMPONENTS, axes, strict=False)):
        x: np.ndarray = strain_fem[:, i]
        ax.plot(
            x, stress_fem[:, i], "-", color="blue", linewidth=1.4,
            label=rf"$\mathrm{{FE}}^2\ \bar{{\sigma}}_{{{comp}}}$",
        )
        ax.plot(
            x, lstm_stress[:, i], "--", color="red", linewidth=1.4,
            label=rf"LSTM $\bar{{\sigma}}_{{{comp}}}$",
        )
        ax.set_xlabel(rf"$\bar{{\varepsilon}}_{{{comp}}}$", fontsize=20)
        ax.set_ylabel(rf"$\bar{{\sigma}}_{{{comp}}}$ [MPa]", fontsize=20)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.legend(
            loc="lower right", frameon=True, fancybox=True, framealpha=0.75,
            edgecolor="0.6", fontsize=16, borderpad=0.35,
            labelspacing=0.25, handlelength=1.6,
        )
        _style_axis(ax)
        ax.tick_params(labelsize=15)
        _pad_xlim(ax, x)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    output_dir: str = "outputs/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 23839,
    main_strain_steps: int = 4,
    increments_per_step: int = 25,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    global_mesh_refinement_size: float = 0.06,
    hole_refinement_factor: int = 7,
) -> None:
    lstm_path: Path = Path(lstm_checkpoint)
    if not lstm_path.is_file():
        raise FileNotFoundError(f"LSTM checkpoint not found: {lstm_path}")

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

    _plot_mesh_points_labeled(
        mesh, LABELED_POINTS, outdir / "mesh_points_labeled.pdf",
    )

    strain_interp, strain_states = random_strain_path(
        rng, strain_low, strain_high,
        main_strain_steps, increments_per_step,
    )

    _plot_strain_path(strain_interp, outdir / "strain_path_microscopic.pdf")

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(str(lstm_path), device)
    lstm.eval()
    lstm_stress, _ = lstm.forward(strain_interp, return_hidden_states=True)

    _, mean_fields = solve_fem(mesh, strain_states, increments_per_step)
    _plot_stress_strain_triptych(
        mean_fields, np.asarray(lstm_stress, dtype=np.float32),
        outdir / "stress_strain_fem_vs_lstm_microscopic.pdf",
    )


if __name__ == "__main__":
    fire.Fire(main)
