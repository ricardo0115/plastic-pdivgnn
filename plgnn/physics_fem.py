"""Construction of discrete divergence operators from meshes via ``fedoo``.

This module is isolated from the rest of the library because it requires the
optional ``fedoo`` finite-element dependency (install with the ``fem`` extra:
``pip install plgnn[fem]``). The resulting sparse operators are consumed by
:mod:`plgnn.physics`.
"""

from __future__ import annotations

import pyvista as pv
import scipy

try:
    import fedoo as fd
except ImportError as exc:  # pragma: no cover - exercised only without fedoo
    raise ImportError(
        "plgnn.physics_fem requires 'fedoo'. Install the fem extra: "
        "pip install plgnn[fem]"
    ) from exc


def compute_op_div_matrix(mesh: pv.PolyData) -> scipy.sparse.csr_matrix:
    """Assemble the 2D nodal divergence operator for a plane-stress mesh."""
    fd_mesh = fd.Mesh.from_pyvista(mesh)
    fd.Assembly.delete_memory()
    fd_mesh.remove_isolated_nodes()
    fd_mesh.reset_interpolation()
    fd.ModelingSpace("2Dplane")
    dummy_wf = fd.weakform.StressEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(0, 0)
    )
    assembly = fd.Assembly.create(dummy_wf, fd_mesh)
    op_div = (
        fd_mesh._get_gausspoint2node_mat()
        @ assembly._get_assembled_operator(assembly.space.op_div_u())
    )
    fd.Assembly.delete_memory()
    return op_div.tocoo()
