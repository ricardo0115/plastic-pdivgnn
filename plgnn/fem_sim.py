"""Finite-element solve of the periodic elasto-plastic boundary value problem.

`compute_mechanical_fields_non_linear` applies a piecewise-linear macro strain path
to a periodic unit cell and returns, at every increment, the local stress/strain
fields and their volume averages (the macroscopic response). The constitutive law
is a Simcoon EPICP model solved with a Newton-Raphson scheme through fedoo.
"""

from functools import partial
from typing import Literal

import fedoo as fd
import numpy as np

from plgnn.datagen import Field

LocalFields = dict[Field, np.ndarray]
MeanFields = dict[Field, tuple[float, float, float] | float]


class FiniteElementSimulationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"{self.message})"


def _extract_local_fields(problem: fd.core.Problem) -> LocalFields:
    assemb = problem.assembly
    res = problem.get_results(
        assemb, ["Disp", "Stress", "Strain", "Statev"], "Node"
    )
    xx_yy_xy_indices = [0, 1, 3]
    local_stress_field = res["Stress"][xx_yy_xy_indices]
    local_strain_field = res["Strain"][xx_yy_xy_indices]
    plastic_strain_slice = assemb.weakform.constitutivelaw.statev_label["EP"]
    equivalent_plastic_strain_slice = (
        assemb.weakform.constitutivelaw.statev_label["P"]
    )
    local_plastic_strain_field = res["Statev"][plastic_strain_slice][
        xx_yy_xy_indices
    ]

    local_equivalent_plastic_strain_field = res["Statev"][
        equivalent_plastic_strain_slice
    ]
    local_equivalent_plastic_strain_field = np.expand_dims(
        local_equivalent_plastic_strain_field, 0
    )
    local_fields_step: dict[Field, np.ndarray] = {
        Field.STRESS: local_stress_field,
        Field.TOTAL_STRAIN: local_strain_field,
        Field.PLASTIC_STRAIN: local_plastic_strain_field,
        Field.EQ_PLASTIC_STRAIN: local_equivalent_plastic_strain_field,
    }

    return local_fields_step


def compute_mechanical_fields_non_linear(
    strain_path: np.ndarray,
    mesh: fd.Mesh,
    constitutive_law: fd.ConstitutiveLaw,
    n_increments_per_step: int,
    modeling_space: Literal["2Dplane", "2Dstress"],
    verbose: bool,
    nr_criterion_tol: float = 2e-4,
) -> tuple[LocalFields, MeanFields]:
    fd.Assembly.delete_memory()
    mesh.reset_interpolation()
    fd.ModelingSpace(modeling_space)

    type_el = mesh.elm_type
    center = mesh.nearest_node(mesh.bounding_box.center)

    constitutive_law.use_elastic_lt = True
    wf = fd.weakform.StressEquilibrium(constitutive_law)
    if type_el == "quad4":
        wf.fbar = True
    assemb = fd.Assembly.create(wf, mesh, type_el)

    pb = fd.problem.NonLinear(assemb)

    bc_periodic = fd.constraint.PeriodicBC(periodicity_type="small_strain")
    pb.bc.add(bc_periodic)

    pb.bc.add("Dirichlet", center, "Disp", 0, name="center")
    sequence_length = (n_increments_per_step * len(strain_path)) + 1

    local_field_shape = (sequence_length, 3, mesh.n_nodes)
    mean_field_shape = (sequence_length, 3, 1)
    local_fields_sequence: LocalFields = {
        field_name: np.zeros(shape=local_field_shape)
        for field_name in Field
        if field_name is not Field.EQ_PLASTIC_STRAIN
    }
    local_fields_sequence[Field.EQ_PLASTIC_STRAIN] = np.zeros(
        shape=(
            sequence_length,
            1,
            mesh.n_nodes,
        )
    )

    mean_fields_sequence: MeanFields = {
        field_name: np.zeros(shape=mean_field_shape)
        for field_name in [Field.STRESS, Field.TOTAL_STRAIN]
    }

    _current_iteration = 1

    def callback(
        pb: fd.core.Problem,
        local_fields_sequence: dict[Field, np.ndarray],
        mean_fields_sequence: dict[Field, np.ndarray],
    ) -> None:
        nonlocal _current_iteration

        local_fields_step: dict[Field, np.ndarray] = _extract_local_fields(pb)
        mean_sigma = np.expand_dims(
            np.array(
                [
                    pb.mesh.integrate_field(field_component, type_field="Node")
                    for field_component in local_fields_step[Field.STRESS]
                ]
            ),
            axis=1,
        )
        for field_name in Field:
            local_fields_sequence[field_name][_current_iteration, :, :] = (
                local_fields_step[field_name][:, :mesh.n_nodes]
            )

        mean_strain = np.array([pb.get_dof_solution(component)[0] for component
                                in ["E_xx", "E_yy", "E_xy"]])[:, np.newaxis]

        mean_fields_sequence[Field.TOTAL_STRAIN][
            _current_iteration, :, :
        ] = mean_strain
        mean_fields_sequence[Field.STRESS][
            _current_iteration, :, :
        ] = mean_sigma

        _current_iteration += 1

    for step, (eps_xx, eps_yy, gamma_xy) in enumerate(strain_path):
        pb.bc.remove("_Strain")
        pb.bc.add("Dirichlet", "E_xx", eps_xx, name="_Strain")
        pb.bc.add("Dirichlet", "E_xy", gamma_xy, name="_Strain")
        pb.bc.add("Dirichlet", "E_yy", eps_yy, name="_Strain")
        mean_stress_callback = partial(
            callback,
            local_fields_sequence=local_fields_sequence,
            mean_fields_sequence=mean_fields_sequence,
        )
        dt = 1 / n_increments_per_step
        pb.set_nr_criterion(
            max_subiter=20, err0=None, tol=nr_criterion_tol, norm_type=2
        )
        try:
            pb.nlsolve(
                dt=dt,
                t0=step,
                tmax=step + 1,
                update_dt=True,
                print_info=1 if verbose else 0,
                interval_output=dt,
                callback=mean_stress_callback,
            )
        except Exception as e:
            raise FiniteElementSimulationError(str(e)) from e
    mean_fields_sequence["rawstrain_path"] = strain_path
    return local_fields_sequence, mean_fields_sequence
