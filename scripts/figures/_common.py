"""Shared helpers for the figure-reproduction scripts.

The figure scripts all build the same finite-element reference, sample the same
kind of macro strain path, run the GNN on a mesh and reduce fields the same way.
Those building blocks live here so each ``fig_*.py`` only keeps its own plotting.
"""
from __future__ import annotations

import fedoo as fd
import numpy as np
import pyvista as pv
import torch

from plgnn.datagen import Field
from plgnn.fem_sim import compute_mechanical_fields_non_linear
from plgnn.models import PlasticGNN

# Simcoon EPICP parameters: [E, nu, eps_y0, sigma_y0, sigma_yinf, c_kinematic].
MATERIAL_PROPS: np.ndarray = np.array(
    [1e5, 0.3, 1e-5, 300.0, 1000.0, 0.3], dtype=float,
)
COMPONENTS: tuple[str, str, str] = ("xx", "yy", "xy")


def random_strain_path(
    rng: np.random.Generator,
    low: float,
    high: float,
    n_macro_steps: int,
    n_increments_per_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear macro strain path through random control states.

    Returns ``(interpolated_path, control_states)``; the interpolated path has
    ``n_increments_per_step * n_macro_steps + 1`` rows.
    """
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


def solve_fem(
    mesh: pv.DataSet,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> tuple[dict, dict]:
    """Solve the periodic elasto-plastic FE problem; return ``(local, mean)`` fields."""
    mesh_fd: fd.Mesh = fd.Mesh.from_pyvista(mesh).as_2d()
    material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS.copy())
    return compute_mechanical_fields_non_linear(
        strain_path=strain_states,
        mesh=mesh_fd,
        constitutive_law=material,
        n_increments_per_step=n_increments_per_step,
        modeling_space="2Dplane",
        verbose=False,
        nr_criterion_tol=1e-4,
    )


def fem_local_stress(
    mesh: pv.DataSet,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> np.ndarray:
    """FE local stress field, shape ``[T, N, 3]``."""
    local_fields, _ = solve_fem(mesh, strain_states, n_increments_per_step)
    return local_fields[Field.STRESS].transpose(0, 2, 1).astype(np.float32)


def fem_local_and_macro(
    mesh: pv.DataSet,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FE ``(local stress [T, N, 3], macro stress [T, 3], macro strain [T, 3])``."""
    local_fields, mean_fields = solve_fem(
        mesh, strain_states, n_increments_per_step,
    )
    local_stress: np.ndarray = local_fields[Field.STRESS].transpose(
        0, 2, 1
    ).astype(np.float32)
    macro_stress: np.ndarray = np.asarray(
        mean_fields[Field.STRESS]
    ).reshape(-1, 3).astype(np.float32)
    macro_strain: np.ndarray = np.asarray(
        mean_fields[Field.TOTAL_STRAIN]
    ).reshape(-1, 3).astype(np.float32)
    return local_stress, macro_stress, macro_strain


def run_gnn_on_mesh(
    mesh: pv.PolyData,
    lstm_stress_last: np.ndarray,
    hidden_state_last: np.ndarray,
    gnn_checkpoint: str,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    """Run a (P-Div)GNN for a single load step on ``mesh``; return ``[N, 3]``."""
    stress_t: torch.Tensor = torch.from_numpy(
        np.expand_dims(lstm_stress_last, axis=0).astype(np.float32),
    ).to(device)
    hidden_t: torch.Tensor = torch.from_numpy(
        np.expand_dims(hidden_state_last, axis=0).astype(np.float32),
    ).to(device)
    gnn: PlasticGNN = PlasticGNN(gnn_checkpoint, device, mesh)
    gnn.eval()
    out: np.ndarray = np.asarray(
        gnn.forward(stress_t, hidden_t, chunk_size=chunk_size), dtype=np.float32,
    )[-1]
    del gnn
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def per_node_nmse_field(
    fem_last: np.ndarray, pred_last: np.ndarray,
) -> np.ndarray:
    """Per-(node, component) NMSE field; summed over nodes it is the per-component NMSE."""
    fem: np.ndarray = fem_last.astype(np.float64)
    pred: np.ndarray = pred_last.astype(np.float64)
    mean_c: np.ndarray = fem.mean(axis=0, keepdims=True)
    squared_error: np.ndarray = (fem - pred) ** 2
    normalization: np.ndarray = ((fem - mean_c) ** 2).sum(axis=0, keepdims=True)
    return (squared_error / (normalization + 1e-8)).astype(np.float32)


def component_clims(values: np.ndarray) -> list[tuple[float, float]]:
    """Per-component (min, max) color limits over a single ``[N, 3]`` snapshot."""
    clims: list[tuple[float, float]] = []
    for c in range(3):
        lo, hi = float(values[:, c].min()), float(values[:, c].max())
        if np.isclose(lo, hi):
            eps: float = max(1e-12, abs(lo) * 1e-6)
            lo -= eps
            hi += eps
        clims.append((lo, hi))
    return clims
