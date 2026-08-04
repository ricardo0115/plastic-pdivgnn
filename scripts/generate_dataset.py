"""Generate the finite-element dataset of non-proportional loading paths.

Each sample applies a random four-segment macro strain path to the periodic
plate-with-a-hole unit cell and stores the macro strain/stress and the local
stress field. Output: one compressed ``sim_XXXXXX.npz`` per sample plus the shared
``mesh.vtk``, written under ``--data-dir`` (kept outside the git repository).

    python scripts/generate_dataset.py --data-dir <DATA_DIR> [--n-samples 10000]
"""

from __future__ import annotations

import os

_N_THREADS = "4"
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _N_THREADS)

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import fedoo as fd  # noqa: E402
import fire  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from plgnn.datagen import Field, hole_plate_mesh  # noqa: E402
from plgnn.fem_sim import (  # noqa: E402
    FiniteElementSimulationError,
    compute_mechanical_fields_non_linear,
)

# Simcoon EPICP parameters: [E, nu, alpha, sigma_y, k, m] (power-law hardening k*p**m).
MATERIAL_PROPS = np.array([1e5, 0.3, 1e-5, 300.0, 1000.0, 0.3])


def _sample_strain_paths(
    n_samples: int,
    strain_path_length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = rng.uniform(
        low=-0.05, high=0.05, size=(n_samples, strain_path_length, 3)
    )
    paths[:, :, 2] *= 2
    return paths


def _build_and_save_mesh(
    data_dir: Path,
    width: float,
    height: float,
    radius: float,
    hole_refinement_factor: float,
    global_mesh_refinement_size: float,
) -> Path:
    mesh_path = data_dir / "mesh.vtk"
    if mesh_path.exists():
        return mesh_path
    mesh: pv.UnstructuredGrid = hole_plate_mesh(
        width=width,
        height=height,
        radius=radius,
        hole_center=(width / 2.0, height / 2.0),
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
        mesh_type="quad",
    ).extract_surface()
    data_dir.mkdir(parents=True, exist_ok=True)
    mesh.save(mesh_path.as_posix())
    return mesh_path


def _run_single_simulation(
    sim_id: int,
    strain_path: np.ndarray,
    mesh_path: Path,
    data_dir: Path,
    n_increments_per_step: int,
    nr_criterion_tol: float,
) -> tuple[int, bool, str | None]:
    out_path = data_dir / f"sim_{sim_id:06d}.npz"
    if out_path.exists():
        return sim_id, True, None
    try:
        mesh = fd.Mesh.from_pyvista(pv.read(mesh_path.as_posix())).as_2d()
        material = fd.constitutivelaw.Simcoon("EPICP", MATERIAL_PROPS)
        local_fields, mean_fields = compute_mechanical_fields_non_linear(
            strain_path=strain_path,
            mesh=mesh,
            constitutive_law=material,
            n_increments_per_step=n_increments_per_step,
            modeling_space="2Dplane",
            verbose=False,
            nr_criterion_tol=nr_criterion_tol,
        )
    except FiniteElementSimulationError as err:
        return sim_id, False, str(err)

    local_stress = local_fields[Field.STRESS].astype(np.float32)
    macro_strain = np.asarray(mean_fields[Field.TOTAL_STRAIN]).astype(
        np.float32
    )
    macro_stress = np.asarray(mean_fields[Field.STRESS]).astype(np.float32)
    np.savez_compressed(
        out_path.as_posix(),
        macro_strain=macro_strain,
        macro_stress=macro_stress,
        local_stress=local_stress,
    )
    return sim_id, True, None


def main(
    data_dir: str,
    n_samples: int = 10000,
    strain_path_length: int = 4,
    n_increments_per_step: int = 25,
    seed: int = 69,
    workers: int | None = None,
    width: float = 1.0,
    height: float = 1.0,
    radius: float = 0.2,
    hole_refinement_factor: float = 7.0,
    global_mesh_refinement_size: float = 0.06,
    nr_criterion_tol: float = 1e-4,
) -> None:
    data_path = Path(data_dir).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    mesh_path = _build_and_save_mesh(
        data_dir=data_path,
        width=width,
        height=height,
        radius=radius,
        hole_refinement_factor=hole_refinement_factor,
        global_mesh_refinement_size=global_mesh_refinement_size,
    )

    strain_paths = _sample_strain_paths(
        n_samples=n_samples,
        strain_path_length=strain_path_length,
        seed=seed,
    )

    n_workers = workers if workers is not None else (os.cpu_count() or 1)
    start = time.perf_counter()
    failures: list[tuple[int, str]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(
                _run_single_simulation,
                sim_id,
                strain_paths[sim_id],
                mesh_path,
                data_path,
                n_increments_per_step,
                nr_criterion_tol,
            )
            for sim_id in range(n_samples)
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="FEM sims",
        ):
            sim_id, ok, err = fut.result()
            if not ok:
                failures.append((sim_id, err or "unknown"))

    elapsed = time.perf_counter() - start
    n_ok = n_samples - len(failures)
    print(
        f"Generated {n_ok}/{n_samples} simulations in {elapsed:.1f}s "
        f"(mesh: {mesh_path})"
    )
    if failures:
        print(f"Failed sim ids: {[sid for sid, _ in failures][:20]}")


if __name__ == "__main__":
    fire.Fire(main)
