"""Finite-element meshes for the periodic plate-with-a-hole microstructure.

Quadrilateral, linear-triangle and quadratic-triangle (tri6) variants of the same
geometry are produced with gmsh. The quad mesh is the one used to train and
evaluate the surrogate; the triangular meshes are used for the cross-mesh and
mesh-refinement experiments.
"""

from enum import StrEnum, auto
from tempfile import NamedTemporaryFile
from typing import Literal

import gmsh
import meshio
import pyvista as pv

from plgnn.graph.convert import is_periodic


class Field(StrEnum):
    STRESS = auto()
    TOTAL_STRAIN = auto()
    PLASTIC_STRAIN = auto()
    EQ_PLASTIC_STRAIN = auto()


def hole_plate_mesh(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
    mesh_type: Literal["tri", "quad"] = "tri",
) -> pv.UnstructuredGrid:
    if mesh_type == "tri":
        return hole_plate_mesh_tri(
            width=width,
            height=height,
            hole_center=hole_center,
            radius=radius,
            hole_refinement_factor=hole_refinement_factor,
            global_mesh_refinement_size=global_mesh_refinement_size,
        )
    return hole_plate_mesh_quad(
        width=width,
        height=height,
        hole_center=hole_center,
        radius=radius,
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
    )


def hole_plate_mesh_quad(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> pv.UnstructuredGrid:
    hole_mesh_refinement_size = (
        global_mesh_refinement_size / hole_refinement_factor
    )
    gmsh.initialize()
    gmsh.model.add("PlateWithHole")

    rect_tag = gmsh.model.occ.addRectangle(0, 0, 0, width, height)

    cx, cy = hole_center
    hole_tag = gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)

    gmsh.model.occ.synchronize()

    hole_edges = gmsh.model.getBoundary([(2, hole_tag)], recursive=True)

    gmsh.model.occ.cut(
        [(2, rect_tag)], [(2, hole_tag)], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    for surface in gmsh.model.getEntities(2):
        gmsh.model.mesh.setRecombine(2, surface[1])

    gmsh.model.mesh.setSize(
        gmsh.model.getEntities(0), global_mesh_refinement_size
    )

    gmsh.model.mesh.setSize(hole_edges, hole_mesh_refinement_size)

    gmsh.model.mesh.generate(2)
    with NamedTemporaryFile(suffix=".msh", delete=True) as file:
        filename = file.name
        gmsh.write(filename)
        gmsh.finalize()
        m = meshio.read(filename)
        m.cell_sets = {}
        shape = pv.wrap(m)
        shape = shape.extract_cells_by_type(pv.CellType.QUAD)
    assert is_periodic(shape.points[:, :-1])
    return shape


def hole_plate_mesh_tri(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> pv.UnstructuredGrid:
    hole_mesh_refinement_size = (
        global_mesh_refinement_size / hole_refinement_factor
    )
    gmsh.initialize()

    square_points = [
        gmsh.model.geo.add_point(0, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, height, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(0, height, 0, global_mesh_refinement_size),
    ]

    square_lines = [
        gmsh.model.geo.add_line(square_points[0], square_points[1]),
        gmsh.model.geo.add_line(square_points[1], square_points[2]),
        gmsh.model.geo.add_line(square_points[2], square_points[3]),
        gmsh.model.geo.add_line(square_points[3], square_points[0]),
    ]

    center_x, center_y = hole_center
    center_point = gmsh.model.geo.add_point(
        center_x, center_y, 0, global_mesh_refinement_size
    )

    cp1 = gmsh.model.geo.add_point(
        center_x - radius, center_y, 0, hole_mesh_refinement_size
    )
    cp2 = gmsh.model.geo.add_point(
        center_x + radius, center_y, 0, hole_mesh_refinement_size
    )
    circle_arc0 = gmsh.model.geo.add_circle_arc(cp1, center_point, cp2)
    circle_arc1 = gmsh.model.geo.add_circle_arc(cp2, center_point, cp1)

    surface_plate = gmsh.model.geo.add_curve_loop(square_lines)
    surface_hole = gmsh.model.geo.add_curve_loop([circle_arc0, circle_arc1])

    mesh_surface = gmsh.model.geo.add_plane_surface(
        [surface_plate, surface_hole]
    )

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.set_algorithm(dim=2, tag=mesh_surface, val=5)
    mesh_dim = 2
    gmsh.model.mesh.generate(mesh_dim)

    with NamedTemporaryFile(suffix=".msh", delete=True) as file:
        filename = file.name
        gmsh.write(filename)
        gmsh.finalize()
        m = meshio.read(filename)
        m.cell_sets = {}
        shape = pv.wrap(m)
        shape = shape.extract_cells_by_type(pv.CellType.TRIANGLE)
    assert is_periodic(shape.points[:, :-1])
    return shape


def hole_plate_mesh_tri6(
    width: float,
    height: float,
    radius: float,
    hole_center: tuple[float, float],
    hole_refinement_factor: float = 10,
    global_mesh_refinement_size: float = 10,
) -> pv.UnstructuredGrid:
    """Same geometry as `hole_plate_mesh_tri` but with quadratic 6-node triangles."""
    hole_mesh_refinement_size = (
        global_mesh_refinement_size / hole_refinement_factor
    )
    gmsh.initialize()

    square_points = [
        gmsh.model.geo.add_point(0, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, 0, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(width, height, 0, global_mesh_refinement_size),
        gmsh.model.geo.add_point(0, height, 0, global_mesh_refinement_size),
    ]

    square_lines = [
        gmsh.model.geo.add_line(square_points[0], square_points[1]),
        gmsh.model.geo.add_line(square_points[1], square_points[2]),
        gmsh.model.geo.add_line(square_points[2], square_points[3]),
        gmsh.model.geo.add_line(square_points[3], square_points[0]),
    ]

    center_x, center_y = hole_center
    center_point = gmsh.model.geo.add_point(
        center_x, center_y, 0, global_mesh_refinement_size
    )

    cp1 = gmsh.model.geo.add_point(
        center_x - radius, center_y, 0, hole_mesh_refinement_size
    )
    cp2 = gmsh.model.geo.add_point(
        center_x + radius, center_y, 0, hole_mesh_refinement_size
    )
    circle_arc0 = gmsh.model.geo.add_circle_arc(cp1, center_point, cp2)
    circle_arc1 = gmsh.model.geo.add_circle_arc(cp2, center_point, cp1)

    surface_plate = gmsh.model.geo.add_curve_loop(square_lines)
    surface_hole = gmsh.model.geo.add_curve_loop([circle_arc0, circle_arc1])

    mesh_surface = gmsh.model.geo.add_plane_surface(
        [surface_plate, surface_hole]
    )

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.set_algorithm(dim=2, tag=mesh_surface, val=5)
    mesh_dim = 2
    gmsh.model.mesh.generate(mesh_dim)
    gmsh.model.mesh.setOrder(2)

    with NamedTemporaryFile(suffix=".msh", delete=True) as file:
        filename = file.name
        gmsh.write(filename)
        gmsh.finalize()
        m = meshio.read(filename)
        m.cell_sets = {}
        shape = pv.wrap(m)
        shape = shape.extract_cells_by_type(pv.CellType.QUADRATIC_TRIANGLE)
    return shape
