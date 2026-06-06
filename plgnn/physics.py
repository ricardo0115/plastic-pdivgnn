"""Discrete stress-divergence operators for equilibrium-aware training/eval.

These functions apply a precomputed divergence operator (``op_div_matrix``) to a
stress field; constructing that operator from a mesh lives in
:mod:`plgnn.physics_fem` (which needs the optional ``fedoo`` dependency).
"""

from __future__ import annotations

import numpy as np
import scipy
import torch

from plgnn.scaling import NodeType


def compute_divergence_norm_field(
    local_stress_field: np.ndarray,
    op_div_matrix: scipy.sparse.csr_matrix,
    surface_nodes_ids: np.ndarray,
) -> np.ndarray:
    """Per-node norm of div(sigma) (2D), with external-boundary nodes zeroed."""
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = np.stack([stress_x_xy, stress_xy_y], axis=1)  # 2Nx2
    div_sigma = op_div_matrix @ stress_x_xy_xy_y
    external_boundary_nodes_mask = (
        surface_nodes_ids == NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    divergence_field = np.linalg.norm(div_sigma, axis=1)
    return divergence_field


def compute_divergence(
    local_stress_field: torch.Tensor,
    op_div_matrix: torch.Tensor,
    surface_nodes_ids: torch.Tensor,
    reduce_strategy: str = "square",
) -> torch.Tensor:
    """Scalar divergence penalty for a single 2D graph.

    Stress components are ordered [sigma_xx, sigma_yy, sigma_xy]; boundary nodes
    are zeroed before reduction.
    """
    if reduce_strategy not in ("abs", "square"):
        raise AttributeError("reduce_strategy must be 'abs' or 'square'")
    stress_x_xy = local_stress_field[:, [0, 2]].T.reshape(-1)
    stress_xy_y = local_stress_field[:, [2, 1]].T.reshape(-1)
    stress_x_xy_xy_y = torch.stack([stress_x_xy, stress_xy_y], dim=1)  # 2Nx2
    div_sigma = op_div_matrix @ stress_x_xy_xy_y
    external_boundary_nodes_mask = (
        surface_nodes_ids == NodeType.EXTERNAL_BOUNDARY
    ).squeeze()
    internal_boundary_nodes_mask = (
        surface_nodes_ids == NodeType.INTERNAL_BOUNDARY
    ).squeeze()
    div_sigma[external_boundary_nodes_mask] = 0
    div_sigma[internal_boundary_nodes_mask] = 0
    if reduce_strategy == "abs":
        div_sigma = torch.abs(div_sigma)
    elif reduce_strategy == "square":
        div_sigma = torch.square(div_sigma)
    div_sigma = torch.mean(div_sigma, dim=0)
    return torch.sum(div_sigma)


def compute_divergence_batch(
    predictions_phys: torch.Tensor,
    op_div_matrix: torch.Tensor,
    node_labels: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean divergence penalty over a batch of equally-sized graphs.

    All graphs share the same mesh, so ``predictions_phys`` is ``(B*N, comps)``
    in physical units.
    """
    n_total = predictions_phys.shape[0]
    n_graphs = n_total // num_nodes
    total_div = torch.tensor(0.0, device=predictions_phys.device)
    for i in range(n_graphs):
        start = i * num_nodes
        end = start + num_nodes
        graph_stress = predictions_phys[start:end]
        total_div = total_div + compute_divergence(
            graph_stress, op_div_matrix, node_labels
        )
    return total_div / n_graphs


def scipy_coo_to_torch_sparse(coo: scipy.sparse.coo_matrix) -> torch.Tensor:
    """Convert a SciPy COO matrix to a coalesced torch sparse tensor.

    Companion to the divergence operator assembled by
    :func:`plgnn.physics_fem.compute_op_div_matrix`; the resulting sparse tensor
    is consumed by :func:`compute_divergence_batch`.
    """
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(
        indices, values, torch.Size(coo.shape)
    ).coalesce()
