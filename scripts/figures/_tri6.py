"""Genuine quadratic-triangle (tri6) geometry and field rendering.

A tri6 element carries mid-edge nodes, so its field is quadratic inside each
element. Rendering it with pyvista's ``show_edges`` tessellates every element into
four straight sub-triangles, which makes the field piecewise-linear instead of
quadratic. The helpers here instead evaluate the quadratic shape functions on a
fine reference tessellation, so the rendered field is the genuine tri6 one.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from plgnn.figutils import hstack_panels, render_field_panel

SUBDIVISIONS: int = 6


def _reference_tessellation(n: int) -> tuple[np.ndarray, np.ndarray]:
    index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            index[(i, j)] = len(points)
            points.append((i / n, j / n))
    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n - i):
            triangles.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if i + j < n - 1:
                triangles.append(
                    (index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)])
                )
    return np.asarray(points), np.asarray(triangles, dtype=np.int64)


def _tri6_shape_functions(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    l3: np.ndarray = 1.0 - l1 - l2
    return np.stack(
        [
            l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
            4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1,
        ],
        axis=-1,
    )


def _cell_connectivity(grid: pv.DataSet, n_nodes: int) -> np.ndarray:
    conn: np.ndarray = np.empty((grid.n_cells, n_nodes), dtype=np.int64)
    for i in range(grid.n_cells):
        conn[i] = grid.get_cell(i).point_ids[:n_nodes]
    return conn


def _build_refined(grid: pv.DataSet) -> tuple[pv.PolyData, np.ndarray, np.ndarray, int]:
    points: np.ndarray = np.asarray(grid.points)
    conn: np.ndarray = _cell_connectivity(grid, 6)
    reference, local_tris = _reference_tessellation(SUBDIVISIONS)
    shape: np.ndarray = _tri6_shape_functions(reference[:, 0], reference[:, 1])
    n_ref: int = shape.shape[0]
    coords: np.ndarray = np.empty((conn.shape[0] * n_ref, 3))
    faces: np.ndarray = np.empty(
        (conn.shape[0] * local_tris.shape[0], 4), dtype=np.int64
    )
    faces[:, 0] = 3
    for e, cell in enumerate(conn):
        offset: int = e * n_ref
        coords[offset : offset + n_ref, :2] = shape @ points[cell, :2]
        coords[offset : offset + n_ref, 2] = 0.0
        faces[e * local_tris.shape[0] : (e + 1) * local_tris.shape[0], 1:] = (
            local_tris + offset
        )
    return pv.PolyData(coords, faces.ravel()), shape, conn, n_ref


def _refine_field(
    node_field: np.ndarray, shape: np.ndarray, conn: np.ndarray, n_ref: int
) -> np.ndarray:
    refined: np.ndarray = np.empty(
        (conn.shape[0] * n_ref, node_field.shape[1]), dtype=np.float32
    )
    for e, cell in enumerate(conn):
        refined[e * n_ref : (e + 1) * n_ref] = shape @ node_field[cell]
    return refined


def render_field_row(
    grid: pv.DataSet,
    node_field: np.ndarray,
    titles: tuple[str, str, str],
    clims: list[tuple[float, float]],
) -> np.ndarray:
    """Render the three genuine tri6 stress components as a stitched row."""
    refined, shape, conn, n_ref = _build_refined(grid)
    refined_field: np.ndarray = _refine_field(node_field, shape, conn, n_ref)
    panels: list[np.ndarray] = [
        render_field_panel(refined, refined_field[:, c], clims[c], titles[c])
        for c in range(3)
    ]
    return hstack_panels(panels)
