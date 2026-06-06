"""Animated FEM-vs-GNN local-stress comparison movie.

Reimplements the 3-row comparison video from the legacy ``plastic_gnn`` repo
(``scripts/lstm_gnn_test_mod.py`` + ``plastic_gnn/plot.py``):

1. top row    - FEM local stress field (sigma_xx, sigma_yy, sigma_xy);
2. middle row - GNN-predicted field, rendered with the *same per-frame*
   colorbar limits as the FEM row so the comparison is fair;
3. bottom row - macro strain-stress curve (FEM solid + LSTM dashed) with a
   moving marker at the current increment.

Each frame is composed in-memory (pyvista screenshots for the fields,
matplotlib for the curves) and written with the ``imageio`` FFMPEG writer,
which relies on the bundled ``imageio_ffmpeg`` binary - no system ``ffmpeg``
is required.
"""

from __future__ import annotations

import os
from pathlib import Path

# Make the bundled ffmpeg binary discoverable before imageio is imported.
# imageio-ffmpeg ships a self-contained binary so no system ffmpeg is required;
# if it is absent we fall back to whatever ffmpeg imageio can locate.
try:
    import imageio_ffmpeg

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise ModuleNotFoundError(
        "imageio-ffmpeg is required to write the movie. Install it with "
        "`pip install imageio imageio-ffmpeg` (or `conda install -c conda-forge "
        "imageio imageio-ffmpeg`).",
    ) from exc

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PIL import Image  # noqa: E402

from plgnn.figutils import (  # noqa: E402
    hstack_panels,
    render_field_panel,
    render_field_row,
)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

COMPONENTS: tuple[str, str, str] = ("xx", "yy", "xy")

# Curve styling.
_FEM_COLOR: str = "#1f77b4"
_LSTM_COLOR: str = "#d62728"
_WHITE_THRESHOLD: int = 245


def _component_clims(values: np.ndarray) -> list[tuple[float, float]]:
    """Per-component (min, max) over a single ``[N, 3]`` field snapshot."""
    clims: list[tuple[float, float]] = []
    for c in range(3):
        lo, hi = float(values[:, c].min()), float(values[:, c].max())
        if np.isclose(lo, hi):
            eps: float = max(1e-12, abs(lo) * 1e-6)
            lo -= eps
            hi += eps
        clims.append((lo, hi))
    return clims


def _trim_white_vertical(
    arr: np.ndarray, threshold: int = _WHITE_THRESHOLD,
) -> np.ndarray:
    """Crop fully-white top/bottom bands, leaving full width untouched."""
    row_has_ink: np.ndarray = np.any(arr < threshold, axis=2).any(axis=1)
    if not row_has_ink.any():
        return arr
    ys: np.ndarray = np.where(row_has_ink)[0]
    return arr[int(ys.min()) : int(ys.max()) + 1]


def render_field_row_frame(
    mesh: pv.PolyData,
    values: np.ndarray,
    titles: tuple[str, str, str],
    clims: list[tuple[float, float]],
) -> np.ndarray:
    """Render one timestep's 3-component field row to an RGB array."""
    return render_field_row(mesh, values, titles, clims)


def render_curve_frame(
    macro_strain: np.ndarray,
    fem_macro: np.ndarray,
    lstm_macro: np.ndarray,
    frame: int,
) -> np.ndarray:
    """Render the macro strain-stress curve row (FEM + LSTM) with a moving dot."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.2), constrained_layout=True)
    for c, comp in enumerate(COMPONENTS):
        ax = axes[c]
        ax.plot(
            macro_strain[:, c], fem_macro[:, c],
            "-", color=_FEM_COLOR, linewidth=1.8,
            label=rf"FEM $\bar{{\sigma}}_{{{comp}}}$",
        )
        ax.plot(
            macro_strain[:, c], lstm_macro[:, c],
            "--", color=_LSTM_COLOR, linewidth=1.5,
            label=rf"LSTM $\bar{{\sigma}}_{{{comp}}}$",
        )
        ax.plot(
            [macro_strain[frame, c]], [fem_macro[frame, c]],
            "o", color=_FEM_COLOR, markersize=9,
        )
        ax.plot(
            [macro_strain[frame, c]], [lstm_macro[frame, c]],
            "o", color=_LSTM_COLOR, markersize=7,
        )
        ax.set_xlabel(rf"Macro strain $\bar{{\varepsilon}}_{{{comp}}}$")
        ax.set_ylabel(rf"Macro stress $\bar{{\sigma}}_{{{comp}}}$ [MPa]")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", which="both")
        ax.legend(loc="best", fontsize=11)
    fig.canvas.draw()
    arr: np.ndarray = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return _trim_white_vertical(arr)


def _resize_to_width(arr: np.ndarray, width: int) -> np.ndarray:
    img: Image.Image = Image.fromarray(arr)
    w, h = img.size
    new_h: int = max(1, round(h * width / w))
    return np.asarray(img.resize((width, new_h), Image.LANCZOS))


def stack_frames_vertically(
    rows: list[np.ndarray], width: int | None = None,
) -> np.ndarray:
    """Resize each row to a common width and vertically stack them.

    The final frame is padded to even height/width so libx264 accepts it.
    """
    if width is None:
        width = max(r.shape[1] for r in rows)
    resized: list[np.ndarray] = [_resize_to_width(r, width) for r in rows]
    canvas: np.ndarray = np.vstack(resized)
    pad_h: int = canvas.shape[0] % 2
    pad_w: int = canvas.shape[1] % 2
    if pad_h or pad_w:
        canvas = np.pad(
            canvas, ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant", constant_values=255,
        )
    return canvas


def write_comparison_movie(
    mesh: pv.PolyData,
    fem_field: np.ndarray,
    gnn_field: np.ndarray,
    macro_strain: np.ndarray,
    fem_macro: np.ndarray,
    lstm_macro: np.ndarray,
    out_path: str | Path,
    fps: int = 10,
    model_label: str = "P-DivGNN",
    target_width: int = 1600,
) -> Path:
    """Compose and encode the 3-row FEM / GNN / macro-curve comparison movie.

    Args:
        mesh: 2D pyvista surface the fields live on.
        fem_field: FEM local stress, shape ``[T, N, 3]``.
        gnn_field: GNN-predicted local stress, shape ``[T, N, 3]``.
        macro_strain: applied macro strain, shape ``[T, 3]`` (curve x-axis).
        fem_macro: FEM macro stress, shape ``[T, 3]``.
        lstm_macro: LSTM macro stress, shape ``[T, 3]``.
        out_path: destination ``.mp4`` path.
        fps: frames per second of the output.
        model_label: label used for the middle (prediction) row.
        target_width: common pixel width every row is resized to before stacking.
    """
    if not (fem_field.shape[0] == gnn_field.shape[0] == macro_strain.shape[0]):
        raise ValueError(
            "Sequence lengths differ: "
            f"fem={fem_field.shape[0]}, gnn={gnn_field.shape[0]}, "
            f"strain={macro_strain.shape[0]}",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames: int = fem_field.shape[0]
    fem_titles: tuple[str, str, str] = tuple(  # type: ignore[assignment]
        f"FEM Stress {c.upper()}" for c in COMPONENTS
    )
    gnn_titles: tuple[str, str, str] = tuple(  # type: ignore[assignment]
        f"{model_label} Stress {c.upper()}" for c in COMPONENTS
    )

    writer = imageio.get_writer(
        out_path.as_posix(), fps=fps, codec="libx264",
        macro_block_size=None, quality=8,
    )
    try:
        for t in range(n_frames):
            clims: list[tuple[float, float]] = _component_clims(fem_field[t])
            fem_row: np.ndarray = render_field_row_frame(
                mesh, fem_field[t], fem_titles, clims,
            )
            gnn_row: np.ndarray = render_field_row_frame(
                mesh, gnn_field[t], gnn_titles, clims,
            )
            curve_row: np.ndarray = render_curve_frame(
                macro_strain, fem_macro, lstm_macro, t,
            )
            frame: np.ndarray = stack_frames_vertically(
                [fem_row, gnn_row, curve_row], width=target_width,
            )
            writer.append_data(frame)
    finally:
        writer.close()

    return out_path


def _scalar_clim(values: np.ndarray) -> tuple[float, float]:
    """(min, max) of a scalar field, widened if degenerate."""
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        eps: float = max(1e-12, abs(lo) * 1e-6)
        lo -= eps
        hi += eps
    return lo, hi


def _open_writer(out_path: Path, fps: int):  # noqa: ANN202 - imageio writer type
    return imageio.get_writer(
        out_path.as_posix(), fps=fps, codec="libx264",
        macro_block_size=None, quality=8,
    )


def write_error_field_movie(
    mesh: pv.PolyData,
    error_field: np.ndarray,
    out_path: str | Path,
    fps: int = 10,
    target_width: int = 1600,
) -> Path:
    """Encode the per-component error field as a single 3-panel movie.

    Args:
        mesh: 2D pyvista surface the field lives on.
        error_field: per-node element-wise normalized squared error,
            shape ``[T, N, 3]`` (sigma_xx, sigma_yy, sigma_xy).
        out_path: destination ``.mp4`` path.
        fps: frames per second of the output.
        target_width: pixel width the row is resized to before encoding.

    The colorbar limits auto-scale per frame and per component (matching the
    legacy ``write_movie_stress_field`` behaviour for the NSE field).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    titles: tuple[str, str, str] = tuple(  # type: ignore[assignment]
        f"NSE {c.upper()}" for c in COMPONENTS
    )
    writer = _open_writer(out_path, fps)
    try:
        for t in range(error_field.shape[0]):
            clims: list[tuple[float, float]] = _component_clims(error_field[t])
            row: np.ndarray = render_field_row_frame(
                mesh, error_field[t], titles, clims,
            )
            writer.append_data(stack_frames_vertically([row], width=target_width))
    finally:
        writer.close()
    return out_path


def write_divergence_field_movie(
    mesh: pv.PolyData,
    divergences: list[np.ndarray],
    labels: list[str],
    out_path: str | Path,
    fps: int = 10,
    clim_reference: int = 0,
    target_width: int = 1600,
) -> Path:
    """Encode several models' divergence-norm fields side by side.

    Args:
        mesh: 2D pyvista surface the fields live on.
        divergences: per-model per-node ``|div(sigma)|``, each shape ``[T, N]``
            (e.g. FEM, GNN, P-DivGNN). Rendered left to right as panels.
        labels: panel titles, one per entry in ``divergences``.
        out_path: destination ``.mp4`` path.
        fps: frames per second of the output.
        clim_reference: index of the model whose per-frame range sets the shared
            colorbar limits (default 0, i.e. FEM), so every model's residual
            equilibrium error is read on the same scale.
        target_width: pixel width the row is resized to before encoding.
    """
    if not divergences:
        raise ValueError("divergences must contain at least one model")
    if len(divergences) != len(labels):
        raise ValueError(
            f"divergences ({len(divergences)}) and labels ({len(labels)}) "
            "must have the same length",
        )
    n_frames: int = divergences[0].shape[0]
    if any(d.shape[0] != n_frames for d in divergences):
        raise ValueError(
            f"Sequence lengths differ: {[d.shape[0] for d in divergences]}",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(out_path, fps)
    try:
        for t in range(n_frames):
            clim: tuple[float, float] = _scalar_clim(divergences[clim_reference][t])
            panels: list[np.ndarray] = [
                render_field_panel(mesh, div[t], clim, label)
                for div, label in zip(divergences, labels, strict=True)
            ]
            row: np.ndarray = hstack_panels(panels)
            writer.append_data(stack_frames_vertically([row], width=target_width))
    finally:
        writer.close()
    return out_path
