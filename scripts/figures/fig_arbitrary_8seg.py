from __future__ import annotations

import contextlib
from pathlib import Path

import fedoo as fd
import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from plgnn.datagen import Field, hole_plate_mesh
from plgnn.fem_sim import compute_mechanical_fields_non_linear
from plgnn.models import LstmConstitutiveLaw

COMPONENTS: tuple[str, ...] = ("xx", "yy", "xy")
MATERIAL_PROPS: np.ndarray = np.array(
    [1e5, 0.3, 1e-5, 300.0, 1000.0, 0.3], dtype=float,
)

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.1,
    }
)


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
        edgecolor="0.6", fontsize=8, borderpad=0.35,
        labelspacing=0.25, handlelength=1.6,
    )


def _generate_strain_path(
    rng: np.random.Generator,
    n_steps: int,
    n_increments_per_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    strain_targets: np.ndarray = rng.uniform(
        -0.05, 0.05, size=(n_steps, 3),
    )
    strain_targets[:, 2] *= 2.0
    segments: list[np.ndarray] = [
        np.linspace(
            (0.0, 0.0, 0.0),
            strain_targets[0],
            num=n_increments_per_step + 1,
            endpoint=False,
        )
    ]
    for i in range(n_steps - 1):
        include_end: bool = i == n_steps - 2
        segments.append(
            np.linspace(
                strain_targets[i],
                strain_targets[i + 1],
                num=n_increments_per_step,
                endpoint=include_end,
            )
        )
    return strain_targets, np.vstack(segments)


def _run_fem(
    mesh: pv.PolyData,
    strain_targets: np.ndarray,
    n_increments_per_step: int,
) -> tuple[dict, dict]:
    mesh_fd: fd.Mesh = fd.Mesh.from_pyvista(mesh).as_2d()
    material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS.copy())
    local_fields, mean_fields = compute_mechanical_fields_non_linear(
        strain_path=strain_targets,
        mesh=mesh_fd,
        constitutive_law=material,
        n_increments_per_step=n_increments_per_step,
        modeling_space="2Dplane",
        verbose=False,
        nr_criterion_tol=1e-4,
    )
    return local_fields, mean_fields


def _plot_strain_path(full_path: np.ndarray, outpath: Path) -> None:
    t: np.ndarray = np.arange(full_path.shape[0])
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for i, comp in enumerate(COMPONENTS):
        ax.plot(
            t, full_path[:, i], linestyle="--",
            solid_capstyle="round",
            label=rf"$\varepsilon_{{{comp}}}$",
        )
    ax.set_xlabel(r"$\mathbf{Timestep}$")
    ax.set_ylabel(r"$\mathbf{Total\ Strain}\ \varepsilon$")
    _legend(ax)
    _style_axis(ax)
    _pad_xlim(ax, t)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_stress_strain_triptych(
    strain_fem: np.ndarray,
    stress_fem: np.ndarray,
    stress_lstm: np.ndarray,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.4), sharey=True, constrained_layout=True,
    )
    fig.set_constrained_layout_pads(wspace=0.1, hspace=0.08)

    all_stress: np.ndarray = np.concatenate([stress_fem, stress_lstm], axis=0)
    ymin: float = float(np.min(all_stress))
    ymax: float = float(np.max(all_stress))
    ypad: float = 0.06 * (ymax - ymin)

    for i, (comp, ax) in enumerate(zip(COMPONENTS, axes, strict=False)):
        x: np.ndarray = strain_fem[:, i]
        ax.plot(
            x, stress_fem[:, i], "-", color="blue", linewidth=1.1,
            solid_capstyle="round",
            label=rf"$\mathrm{{FE}}^2\ \bar{{\sigma}}_{{{comp}}}$",
        )
        ax.plot(
            x, stress_lstm[:, i], "--", color="red", linewidth=1.1,
            solid_capstyle="round",
            label=rf"LSTM $\bar{{\sigma}}_{{{comp}}}$",
        )
        ax.set_xlabel(
            rf"$\mathbf{{Macroscopic\ Strain}}\ \bar{{\varepsilon}}_{{{comp}}}$",
        )
        ylabel: str = (
            rf"$\mathbf{{Macroscopic\ Stress}}\ "
            rf"\bar{{\sigma}}_{{{comp}}}\ \mathrm{{[MPa]}}$"
        )
        ax.set_ylabel(ylabel)
        _legend(ax)
        _style_axis(ax)
        _pad_xlim(ax, x)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pca(hidden_states: np.ndarray, outpath: Path) -> None:
    t: np.ndarray = np.arange(hidden_states.shape[0])
    pca: PCA = PCA(n_components=2)
    pca_2d: np.ndarray = pca.fit_transform(hidden_states)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(
        pca_2d[:, 0], pca_2d[:, 1],
        c=t, cmap="viridis", s=20, edgecolors="none",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.colorbar(sc, ax=ax, label="Timestep")
    ax.tick_params(direction="in", which="both")
    ax.minorticks_on()
    ax.grid(which="major", linewidth=0.6, alpha=0.6)
    ax.grid(which="minor", linewidth=0.4, alpha=0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_tsne(hidden_states: np.ndarray, outpath: Path) -> None:
    t: np.ndarray = np.arange(hidden_states.shape[0])
    tsne: TSNE = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_2d: np.ndarray = tsne.fit_transform(hidden_states)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    sc = ax.scatter(
        tsne_2d[:, 0], tsne_2d[:, 1],
        c=t, cmap="viridis", s=20, edgecolors="none",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=ax, label="Timestep")
    ax.tick_params(direction="in", which="both")
    ax.minorticks_on()
    ax.grid(which="major", linewidth=0.6, alpha=0.6)
    ax.grid(which="minor", linewidth=0.4, alpha=0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(hidden_states: np.ndarray, outpath: Path) -> None:
    activation_matrix: np.ndarray = hidden_states.T
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(
        activation_matrix, aspect="auto",
        cmap="RdBu_r", interpolation="nearest",
    )
    ax.set_xlabel("Time step", fontsize=17)
    ax.set_ylabel("Hidden unit", fontsize=17)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Activation", fontsize=17)
    cbar.ax.tick_params(labelsize=13)
    ax.tick_params(direction="in", which="both", labelsize=14)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    output_dir: str = "outputs/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    n_steps: int = 8,
    n_increments_per_step: int = 25,
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

    strain_targets, full_strain_path = _generate_strain_path(
        rng, n_steps, n_increments_per_step,
    )

    mesh: pv.PolyData = hole_plate_mesh(
        width=1.0, height=1.0, radius=0.2,
        hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
        mesh_type="quad",
    ).extract_surface()

    _, mean_fields = _run_fem(mesh, strain_targets, n_increments_per_step)
    strain_fem: np.ndarray = mean_fields[Field.TOTAL_STRAIN][:, :, 0]
    stress_fem: np.ndarray = mean_fields[Field.STRESS][:, :, 0]

    law: LstmConstitutiveLaw = LstmConstitutiveLaw(str(lstm_path), device)
    law.eval()
    stress_lstm, hidden_states = law.forward(
        strain_fem, return_hidden_states=True,
    )
    stress_lstm = np.asarray(stress_lstm, dtype=np.float32)
    hidden_states = np.asarray(hidden_states, dtype=np.float32)

    _plot_strain_path(
        full_strain_path, outdir / "arbitrary_8seg_strain_path.pdf",
    )
    _plot_stress_strain_triptych(
        strain_fem, stress_fem, stress_lstm,
        outdir / "arbitrary_8seg_stress_strain.pdf",
    )
    _plot_pca(hidden_states, outdir / "arbitrary_8seg_pca.pdf")
    _plot_tsne(hidden_states, outdir / "arbitrary_8seg_tsne.pdf")
    _plot_heatmap(hidden_states, outdir / "arbitrary_8seg_heatmap.pdf")


if __name__ == "__main__":
    fire.Fire(main)
