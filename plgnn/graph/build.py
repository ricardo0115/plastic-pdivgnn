"""Graph construction and periodic boundary utilities.

Builds PyG graphs from pyvista meshes with periodic boundary edges, computes
node labels (boundary classification), and normalizes graph attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
import torch_geometric as PyG
from torch_geometric.data import Data

from plgnn.graph.convert import mesh_to_graph
from plgnn.scaling import NodeType


@dataclass
class GraphNormStats:
    """Normalization stats for graph positions, node labels, edge attributes."""

    mean_pos: torch.Tensor
    std_pos: torch.Tensor
    mean_node_labels: torch.Tensor
    std_node_labels: torch.Tensor
    mean_edge_attr: torch.Tensor
    std_edge_attr: torch.Tensor


def build_graph(
    mesh: pv.UnstructuredGrid | pv.PolyData,
    periodic: bool = True,
) -> Data:
    """Build a PyG graph with node labels and edge distances.

    When ``periodic`` is True (default), wrap-around edges connecting opposite
    boundaries are added (requires a periodic mesh). Set ``periodic=False`` to
    skip them and build a plain mesh graph. The resulting graph carries
    ``is_periodic`` so downstream code can tell which kind it received.
    """
    graph = mesh_to_graph(mesh)
    graph.edge_attr = _compute_node_distances_as_edge_weights(graph).float()
    if periodic:
        graph = compute_periodic_graph(graph)
    graph.is_periodic = periodic
    graph.node_labels = torch.Tensor(compute_node_labels(mesh))
    graph.pos = graph.pos[:, :-1].float()
    return graph


def normalize_graph(
    graph: Data,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GraphNormStats]:
    """Standardize positions, node labels, and edge attributes (zero mean, unit var).

    Returns ``(positions, node_labels, edge_attr, stats)`` where each tensor is
    standardized and ``stats`` carries the original means and standard deviations.
    """
    stats = GraphNormStats(
        mean_pos=graph.pos.mean(),
        std_pos=graph.pos.std(),
        mean_node_labels=graph.node_labels.mean(),
        std_node_labels=graph.node_labels.std(),
        mean_edge_attr=graph.edge_attr.mean(),
        std_edge_attr=graph.edge_attr.std(),
    )
    positions = (graph.pos - stats.mean_pos) / stats.std_pos
    node_labels = (
        graph.node_labels - stats.mean_node_labels
    ) / stats.std_node_labels
    edge_attr = (graph.edge_attr - stats.mean_edge_attr) / stats.std_edge_attr
    return positions, node_labels, edge_attr, stats


def _compute_node_distances_as_edge_weights(mesh_graph: Data) -> torch.Tensor:
    distances = (
        mesh_graph.pos[mesh_graph.edge_index[0]]
        - mesh_graph.pos[mesh_graph.edge_index[1]]
    )
    return torch.linalg.vector_norm(distances, dim=1)


def compute_periodic_graph(graph: PyG.data.Data) -> PyG.data.Data:
    """Add wrap-around edges connecting opposite boundaries (2D)."""
    points_2d = graph.pos[:, :-1].numpy()
    min_x, min_y = np.min(points_2d, axis=0)
    max_x, max_y = np.max(points_2d, axis=0)
    mesh_indices = np.arange(len(points_2d))
    left_side_points_mask = np.where(points_2d[:, 0] == min_x)[0]
    right_side_points_mask = np.where(points_2d[:, 0] == max_x)[0]
    upper_side_points_mask = np.where(points_2d[:, 1] == max_y)[0]
    lower_side_points_mask = np.where(points_2d[:, 1] == min_y)[0]

    (
        left_side_points_mask,
        right_side_points_mask,
        upper_side_points_mask,
        lower_side_points_mask,
    ) = [
        torch.from_numpy(point_mask[np.lexsort((points_2d[point_mask].T))])
        for point_mask in [
            left_side_points_mask,
            right_side_points_mask,
            upper_side_points_mask,
            lower_side_points_mask,
        ]
    ]
    left_lower_corner = mesh_indices[
        np.logical_and(*(points_2d == (min_x, min_y)).T)
    ]
    left_upper_corner = mesh_indices[
        np.logical_and(*(points_2d == (min_x, max_y)).T)
    ]
    right_lower_corner = mesh_indices[
        np.logical_and(*(points_2d == (max_x, min_y)).T)
    ]
    right_upper_corner = mesh_indices[
        np.logical_and(*(points_2d == (max_x, max_y)).T)
    ]
    corner_points = torch.from_numpy(
        np.array(
            [
                left_lower_corner,
                left_upper_corner,
                right_lower_corner,
                right_upper_corner,
            ]
        ).squeeze()
    )

    row, col = graph.edge_index
    n_row = torch.cat(
        (
            row,
            left_side_points_mask,
            right_side_points_mask,
            lower_side_points_mask,
            upper_side_points_mask,
            corner_points,
        )
    )
    n_col = torch.cat(
        (
            col,
            right_side_points_mask,
            left_side_points_mask,
            upper_side_points_mask,
            lower_side_points_mask,
            corner_points.flip(dims=[0]),
        )
    )
    n_edge_index = torch.vstack([n_row, n_col]).long()
    edge_attr = torch.zeros(n_edge_index.shape[1])
    edge_attr[: graph.num_edges] = graph.edge_attr
    return PyG.data.Data(
        edge_index=n_edge_index,
        pos=graph.pos,
        edge_attr=edge_attr,
        face=graph.face,
        org_edge_index=graph.edge_index,
    ).coalesce()


def _is_on_domain_edge(
    point: np.ndarray,
    bounds: tuple[float, float, float, float],
    tol: float = 1e-8,
) -> bool:
    x, y = point[0], point[1]
    xmin, xmax, ymin, ymax = bounds
    return (
        abs(x - xmin) < tol
        or abs(x - xmax) < tol
        or abs(y - ymin) < tol
        or abs(y - ymax) < tol
    )


def compute_node_labels(
    mesh: pv.PolyData | pv.UnstructuredGrid,
) -> npt.NDArray[np.int_]:
    """Classify each node as external/internal boundary or internal.

    Supports meshes with any number of holes (N holes -> N+1 boundary regions).
    The external boundary region is the one whose points lie on the domain edges;
    all other regions are internal (hole) boundaries.
    """
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_edges=False,
    )
    boundary_regions = edges.connectivity().cell_data_to_point_data()
    region_ids: npt.NDArray[np.int_] = boundary_regions.point_data["RegionId"]
    unique_regions = np.unique(region_ids)

    # Map boundary points back to original mesh indices by coordinate matching.
    # connectivity() reorders points and strips vtkOriginalPointIds in
    # pyvista>=0.44; n is small (~200 boundary pts) so O(n*N) is negligible.
    dists = np.linalg.norm(
        boundary_regions.points[:, None, :] - mesh.points[None, :, :],
        axis=2,
    )
    boundary_regions_mask = np.argmin(dists, axis=1)

    bounds_2d = mesh.bounds[:4]  # (xmin, xmax, ymin, ymax)
    external_region_id = None
    for rid in unique_regions:
        region_boundary_indices = boundary_regions_mask[region_ids == rid]
        region_points = mesh.points[region_boundary_indices]
        if any(_is_on_domain_edge(pt, bounds_2d) for pt in region_points):
            external_region_id = rid
            break

    if external_region_id is None:
        raise ValueError("Could not identify external boundary region")

    node_types = np.full(mesh.n_points, NodeType.INTERNAL, dtype=np.int_)
    external_mask = boundary_regions_mask[region_ids == external_region_id]
    node_types[external_mask] = NodeType.EXTERNAL_BOUNDARY

    for rid in unique_regions:
        if rid == external_region_id:
            continue
        internal_mask = boundary_regions_mask[region_ids == rid]
        node_types[internal_mask] = NodeType.INTERNAL_BOUNDARY

    return node_types
