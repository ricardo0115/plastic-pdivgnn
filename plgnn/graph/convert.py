"""Mesh <-> graph conversion and mesh geometry helpers.

Converts pyvista meshes to PyTorch Geometric graphs (triangular and quad
elements) and back, plus periodicity helpers (bounding box, face extraction,
periodicity check) used by :mod:`plgnn.graph.build`.

The mesh geometry helpers are inlined from microgen to avoid importing it
(microgen pulls cadquery -> nptyping -> np.bool8, broken on NumPy 2.x).
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
import torch_geometric as PyG

_DIM_2D = 2


# --------------------------------------------------------------------------- #
# Mesh <-> graph conversion (single element-type meshes)
# --------------------------------------------------------------------------- #
def _format_faces_from_pyvista(faces: np.ndarray) -> torch.Tensor:
    n_nodes_per_face = faces[0]
    # Copy because the original array is not writeable and could produce
    # unexpected tensor behaviors.
    mesh_faces = np.copy(faces.reshape(-1, n_nodes_per_face + 1)[:, 1:])
    return torch.from_numpy(mesh_faces).t().contiguous()


def _format_faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    n_nodes_per_face = faces.shape[1]
    formatted_faces = np.zeros(
        (faces.shape[0], n_nodes_per_face + 1), dtype=np.uint64
    )
    formatted_faces[:, 0] = n_nodes_per_face
    formatted_faces[:, 1:] = faces
    return formatted_faces


def _quad_face_to_edge(
    mesh_graph: PyG.data.Data, set_faces_to_none: bool = True
) -> None:
    face_indices = mesh_graph.face
    edge_index = torch.cat(
        [
            face_indices[:2],
            face_indices[1:3],
            face_indices[2:],
            face_indices[::3],
        ],
        dim=1,
    )
    edge_index = PyG.utils.to_undirected(
        edge_index, num_nodes=mesh_graph.num_nodes
    )
    mesh_graph.edge_index = edge_index
    if set_faces_to_none:
        mesh_graph.face = None


def mesh_to_graph(
    mesh: pv.UnstructuredGrid | pv.PolyData,
    remove_faces: bool = False,
) -> PyG.data.Data:
    """Build a graph from a surface mesh (triangular or quad elements)."""
    is_quad_mesh = mesh.get_cell(0).type == pv.CellType.QUAD
    faces: torch.Tensor = _format_faces_from_pyvista(mesh.faces)
    graph = PyG.data.Data(pos=torch.from_numpy(mesh.points), face=faces)
    if is_quad_mesh:
        _quad_face_to_edge(graph, remove_faces)
    else:
        face_to_edge = PyG.transforms.FaceToEdge(remove_faces=remove_faces)
        graph = face_to_edge(graph)
    return graph


def graph_to_mesh(graph: PyG.data.Data) -> pv.PolyData:
    """Reconstruct a pyvista surface mesh from a graph carrying ``face``."""
    if graph.pos.shape[1] == 2:
        # Add a zero Z column so _format_faces_to_pyvista works.
        pos = torch.zeros(size=(graph.pos.shape[0], 3))
        pos[:, :2] = graph.pos
        graph.pos = pos
    vertices = graph.pos.detach().cpu().numpy()
    faces = graph.face.detach().t().cpu().numpy()
    faces = _format_faces_to_pyvista(faces)
    return pv.PolyData(vertices, faces)


# --------------------------------------------------------------------------- #
# Mesh geometry / periodicity helpers (inlined from microgen)
# --------------------------------------------------------------------------- #
def _get_bounding_box(
    nodes_coords: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    min_point = np.min(nodes_coords, axis=0)
    max_point = np.max(nodes_coords, axis=0)
    return min_point, max_point


def _extract_face_nodes(
    nodes_coords: npt.NDArray[np.float64],
    min_point: npt.NDArray[np.float64],
    max_point: npt.NDArray[np.float64],
    tol: float,
    dim: int,
) -> dict[str, npt.NDArray[np.int64]]:
    axes = "xyz"[:dim]
    faces = {}
    for i, axis in enumerate(axes):
        for sign, point in zip("-+", [min_point, max_point]):
            face = f"{axis}{sign}"
            faces[face] = np.where(
                np.abs(nodes_coords[:, i] - point[i]) < tol
            )[0]
    return faces


def _sort_adjacent_faces_2d(
    faces: dict[str, npt.NDArray[np.int64]],
    nodes_coords: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.int64]]:
    complementary_indices = {"x": 1, "y": 0}
    for axis, sign in itertools.product("xy", "-+"):
        face = f"{axis}{sign}"
        idx = complementary_indices[axis]
        faces[face] = faces[face][np.argsort(nodes_coords[faces[face], idx])]
    return faces


def is_periodic(
    nodes_coords: npt.NDArray[np.float64],
    tol: float = 1e-8,
) -> bool:
    """Check whether a mesh is periodic, given its nodes' coordinates."""
    dim = nodes_coords.shape[1]
    axes = "xyz"[:dim]
    min_point, max_point = _get_bounding_box(nodes_coords)

    faces = _extract_face_nodes(nodes_coords, min_point, max_point, tol, dim)

    if dim == _DIM_2D:
        faces = _sort_adjacent_faces_2d(faces, nodes_coords)

    complementary_indices = {
        "x": slice(1, None),
        "y": slice(None, None, 2),
    }
    for axis in axes:
        face_m = faces[f"{axis}-"]
        face_p = faces[f"{axis}+"]

        if len(face_m) != len(face_p):
            return False

        slc = complementary_indices[axis]
        diff = nodes_coords[face_p, slc] - nodes_coords[face_m, slc]
        if (diff > tol).any():
            return False

    return True
