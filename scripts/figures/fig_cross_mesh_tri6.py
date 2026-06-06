"""Cross-mesh stress-field comparison on quadratic triangles (tri6).

Four rows of the local stress field at the last load step, sharing the FEM(Quad)
colorbar range:

    FEM(Quad)  -  FEM(Tri6)  -  GNN(Tri6)  -  P-DivGNN(Tri6)

The quad-trained GNN and P-DivGNN are run on the tri6 surface; the three tri6 rows
are rendered with the genuine quadratic field (curved sides, mid-edge values).

    python scripts/figures/fig_cross_mesh_tri6.py \
        --lstm-checkpoint weights/lstm.pt --gnn-checkpoint weights/gnn.pt \
        --pdivgnn-checkpoint weights/pdivgnn.pt --output-dir outputs/figures
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import fedoo as fd
import fire
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from PIL import JpegImagePlugin  # noqa: F401 - register JPEG handler for PDF export

import _tri6
from plgnn.datagen import Field, hole_plate_mesh_quad, hole_plate_mesh_tri6
from plgnn.fem_sim import compute_mechanical_fields_non_linear
from plgnn.figutils import render_field_row, vstack_rows
from plgnn.models import LstmConstitutiveLaw, PlasticGNN

MATERIAL_PROPS: np.ndarray = np.array(
    [1e5, 0.3, 1e-5, 300.0, 1000.0, 0.3], dtype=float,
)
COMPS: tuple[str, str, str] = ("XX", "YY", "XY")


def _random_strain_path(
    rng: np.random.Generator,
    low: float,
    high: float,
    n_macro_steps: int,
    n_increments_per_step: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def _run_fem(
    mesh: pv.DataSet,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> np.ndarray:
    mesh_fd: fd.Mesh = fd.Mesh.from_pyvista(mesh).as_2d()
    material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS.copy())
    local_fields, _ = compute_mechanical_fields_non_linear(
        strain_path=strain_states,
        mesh=mesh_fd,
        constitutive_law=material,
        n_increments_per_step=n_increments_per_step,
        modeling_space="2Dplane",
        verbose=False,
        nr_criterion_tol=1e-4,
    )
    return local_fields[Field.STRESS].transpose(0, 2, 1).astype(np.float32)


def _run_gnn_on_mesh(
    mesh: pv.PolyData,
    lstm_stress_last: np.ndarray,
    hidden_state_last: np.ndarray,
    gnn_checkpoint: str,
    device: str,
    chunk_size: int,
) -> np.ndarray:
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


def _titles(label: str) -> tuple[str, str, str]:
    return tuple(f"{label} Stress {c}" for c in COMPS)  # type: ignore[return-value]


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    gnn_checkpoint: str,
    pdivgnn_checkpoint: str,
    output_dir: str = "outputs/figures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 23839,
    main_strain_steps: int = 4,
    increments_per_step: int = 25,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    global_mesh_refinement_size: float = 0.06,
    hole_refinement_factor: int = 7,
    gnn_chunk_size: int = 8,
) -> None:
    for label, ckpt in (
        ("lstm", lstm_checkpoint),
        ("gnn", gnn_checkpoint),
        ("pdivgnn", pdivgnn_checkpoint),
    ):
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {ckpt}")

    outdir: Path = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    with contextlib.suppress(Exception):
        pv.start_xvfb()

    mesh_kwargs = dict(
        width=1.0, height=1.0, radius=0.2, hole_center=(0.5, 0.5),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
    )
    quad_mesh: pv.PolyData = hole_plate_mesh_quad(**mesh_kwargs).extract_surface()
    tri6_mesh: pv.UnstructuredGrid = hole_plate_mesh_tri6(**mesh_kwargs)
    tri6_surface: pv.PolyData = tri6_mesh.extract_surface()

    strain_interp, strain_states = _random_strain_path(
        rng, strain_low, strain_high, main_strain_steps, increments_per_step,
    )

    fem_quad: np.ndarray = _run_fem(quad_mesh, strain_states, increments_per_step)[-1]
    fem_tri6: np.ndarray = _run_fem(tri6_mesh, strain_states, increments_per_step)[-1]

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()
    lstm_stress, hidden_states = lstm.forward(
        strain_interp, return_hidden_states=True,
    )
    lstm_stress_last: np.ndarray = np.asarray(lstm_stress)[-1]
    hidden_state_last: np.ndarray = np.asarray(hidden_states)[-1]

    gnn_tri6: np.ndarray = _run_gnn_on_mesh(
        tri6_surface, lstm_stress_last, hidden_state_last,
        gnn_checkpoint, device, gnn_chunk_size,
    )
    pdivgnn_tri6: np.ndarray = _run_gnn_on_mesh(
        tri6_surface, lstm_stress_last, hidden_state_last,
        pdivgnn_checkpoint, device, gnn_chunk_size,
    )

    clims: list[tuple[float, float]] = [
        (float(fem_quad[:, c].min()), float(fem_quad[:, c].max()))
        for c in range(3)
    ]

    rows: list[np.ndarray] = [
        render_field_row(quad_mesh, fem_quad, _titles("FEM(Quad)"), clims),
        _tri6.render_field_row(tri6_mesh, fem_tri6, _titles("FEM(Tri6)"), clims),
        _tri6.render_field_row(tri6_mesh, gnn_tri6, _titles("GNN(Tri6)"), clims),
        _tri6.render_field_row(
            tri6_mesh, pdivgnn_tri6, _titles("P-DivGNN(Tri6)"), clims,
        ),
    ]
    out_path: Path = outdir / "cross_mesh_tri6_field.pdf"
    Image.fromarray(vstack_rows(rows)).save(out_path.as_posix())
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
