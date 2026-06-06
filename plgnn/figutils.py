"""Small helpers used only by the figure scripts."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import torch

PANEL_W: int = 2000 // 3
PANEL_H: int = 1240
CAMERA_ZOOM: float = 0.95
SB_TITLE_FS: int = 34
SB_LABEL_FS: int = 22
SB_POSITION_X: float = 0.25
SB_POSITION_Y: float = 0.24
SB_WIDTH: float = 0.50
SB_HEIGHT: float = 0.06
HGAP: int = 24
VGAP: int = 2

pv.global_theme.font.family = "times"


def von_mises_stress(
    mean_stress_x: torch.Tensor,
    mean_stress_y: torch.Tensor,
    mean_stress_xy: torch.Tensor,
) -> np.ndarray | float:
    return np.sqrt(
        0.5
        * (
            (mean_stress_x - mean_stress_y) ** 2
            + mean_stress_x**2
            + mean_stress_y**2
            + 6 * mean_stress_xy**2
        )
    )


def trim_white_vertical(arr: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Crop fully-white top/bottom bands, leaving the full width untouched."""
    mask: np.ndarray = np.any(arr[..., :3] < threshold, axis=2)
    if not mask.any():
        return arr
    rows: np.ndarray = np.where(mask.any(axis=1))[0]
    return arr[int(rows.min()) : int(rows.max()) + 1]


def render_field_panel(
    mesh: pv.PolyData,
    values_component: np.ndarray,
    clim: tuple[float, float],
    title: str,
    show_edges: bool = False,
) -> np.ndarray:
    """Render a single scalar component on a portrait panel (mesh on top, scalar
    bar in a clean band below) and return the white-trimmed RGB array."""
    plotter = pv.Plotter(
        off_screen=True, border=False, window_size=(PANEL_W, PANEL_H),
    )
    m: pv.PolyData = mesh.copy()
    m.point_data["field"] = np.asarray(values_component, dtype=np.float32)
    actor = plotter.add_mesh(
        m, scalars="field", cmap="jet", show_edges=show_edges,
        clim=clim, show_scalar_bar=False,
    )
    plotter.view_xy()
    plotter.camera.zoom(CAMERA_ZOOM)
    sb = plotter.add_scalar_bar(
        title=f"{title}__unique", mapper=actor.mapper, vertical=False,
        position_x=SB_POSITION_X, position_y=SB_POSITION_Y,
        width=SB_WIDTH, height=SB_HEIGHT,
        title_font_size=SB_TITLE_FS, label_font_size=SB_LABEL_FS,
        bold=True, n_labels=3, color="Black", fmt="%.2e",
    )
    sb.SetTitle(title)
    sb.GetTitleTextProperty().SetBold(True)
    sb.GetTitleTextProperty().SetFontSize(SB_TITLE_FS)
    sb.GetLabelTextProperty().SetBold(True)
    sb.GetLabelTextProperty().SetFontSize(SB_LABEL_FS)
    img: np.ndarray = np.asarray(plotter.screenshot(return_img=True))[..., :3]
    plotter.close()
    return trim_white_vertical(img)


def hstack_panels(panels: list[np.ndarray], gap: int = HGAP) -> np.ndarray:
    """Crop panels to a common height and stack them left to right."""
    height: int = min(p.shape[0] for p in panels)
    cropped: list[np.ndarray] = [p[:height] for p in panels]
    widths: list[int] = [p.shape[1] for p in cropped]
    total_w: int = sum(widths) + gap * (len(cropped) - 1)
    canvas: np.ndarray = np.full((height, total_w, 3), 255, dtype=np.uint8)
    x: int = 0
    for panel in cropped:
        canvas[:, x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return canvas


def vstack_rows(rows: list[np.ndarray], gap: int = VGAP) -> np.ndarray:
    """Crop rows to a common width and stack them top to bottom."""
    width: int = min(r.shape[1] for r in rows)
    cropped: list[np.ndarray] = [r[:, :width] for r in rows]
    total_h: int = sum(r.shape[0] for r in cropped) + gap * (len(cropped) - 1)
    canvas: np.ndarray = np.full((total_h, width, 3), 255, dtype=np.uint8)
    y: int = 0
    for row in cropped:
        canvas[y : y + row.shape[0]] = row
        y += row.shape[0] + gap
    return canvas


def render_field_row(
    mesh: pv.PolyData,
    values: np.ndarray,
    titles: tuple[str, str, str],
    clims: list[tuple[float, float]],
    show_edges: bool = False,
) -> np.ndarray:
    """Render the three stress components as a single stitched row."""
    panels: list[np.ndarray] = [
        render_field_panel(mesh, values[:, c], clims[c], titles[c], show_edges)
        for c in range(3)
    ]
    return hstack_panels(panels)
