from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import torch

from plgnn.datagen import hole_plate_mesh
from plgnn.models import LstmConstitutiveLaw, PlasticGNN

from _common import solve_fem


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
    for i in range(strain_states.shape[0] - 1):
        segments.append(
            np.linspace(
                start=strain_states[i],
                stop=strain_states[i + 1],
                num=n_increments_per_step,
                endpoint=False,
            )
        )
    segments.append(strain_states[-1])
    return (
        np.vstack(segments).astype(np.float32),
        strain_states.astype(np.float32),
    )


def _fem_worker(
    mesh: pv.PolyData,
    strain_states: np.ndarray,
    n_increments_per_step: int,
) -> tuple[int, float]:
    t0: float = time.perf_counter()
    _ = solve_fem(mesh, strain_states, n_increments_per_step)
    return os.getpid(), time.perf_counter() - t0


def _benchmark_fem_parallel(
    mesh: pv.PolyData,
    strain_batch: list[np.ndarray],
    n_increments_per_step: int,
    workers: int,
    mp_start: str,
    parallel: bool,
) -> tuple[float, list[float]]:
    if not parallel or workers <= 1:
        sample_times: list[float] = []
        t0: float = time.perf_counter()
        for strain_states in strain_batch:
            _, t = _fem_worker(mesh, strain_states, n_increments_per_step)
            sample_times.append(t)
        wall: float = time.perf_counter() - t0
        return wall, sample_times

    ctx = mp.get_context(mp_start)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        results = list(
            ex.map(
                _fem_worker,
                [mesh] * len(strain_batch),
                strain_batch,
                [n_increments_per_step] * len(strain_batch),
            )
        )
    wall = time.perf_counter() - t0
    sample_times = [t for _, t in results]
    return wall, sample_times


@torch.no_grad()
def _benchmark_lstm_gnn_last_step(
    lstm: LstmConstitutiveLaw,
    gnn: PlasticGNN,
    strain_sequence: np.ndarray,
    device: str,
    chunk_size: int = 1,
    amp: bool = False,
) -> tuple[float, float, float]:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0: float = time.perf_counter()
    lstm_stress, hidden_states = lstm.forward(
        strain_sequence, return_hidden_states=True,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_lstm: float = time.perf_counter() - t0

    lstm_last: np.ndarray = np.asarray(lstm_stress)[-1]
    hidden_last: np.ndarray = np.asarray(hidden_states)[-1]
    stress_in: np.ndarray = np.expand_dims(lstm_last, axis=0).astype(np.float32)
    hidden_in: np.ndarray = np.expand_dims(hidden_last, axis=0).astype(np.float32)

    stress_t: torch.Tensor = torch.from_numpy(stress_in).to(device)
    hidden_t: torch.Tensor = torch.from_numpy(hidden_in).to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1: float = time.perf_counter()
    use_amp: bool = amp and device.startswith("cuda")
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=use_amp,
    ):
        _ = gnn.forward(stress_t, hidden_t, chunk_size=chunk_size)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_gnn: float = time.perf_counter() - t1
    return t_lstm, t_gnn, t_lstm + t_gnn


def _plot_full(data: dict[str, list[float]], outpath: Path) -> None:
    n_nodes: list[float] = data["n_nodes"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_nodes, data["fem"], label="FEM (plastic)", marker="o")
    ax.plot(n_nodes, data["lstm"], label="LSTM", marker="x")
    ax.plot(n_nodes, data["gnn_last"], label="GNN (last step)", marker="s")
    ax.plot(n_nodes, data["lstm+gnn"], label="LSTM-GNN (total)", marker="^")
    ax.set_xlabel("Number of mesh nodes")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Benchmark: FEM vs LSTM-GNN (last-step field)")
    ax.legend()
    ax.grid(True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_zoom(data: dict[str, list[float]], outpath: Path) -> None:
    n_nodes: list[float] = data["n_nodes"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_nodes, data["lstm"], label="LSTM", marker="x")
    ax.plot(n_nodes, data["gnn_last"], label="GNN (last step)", marker="s")
    ax.plot(n_nodes, data["lstm+gnn"], label="LSTM-GNN (total)", marker="^")
    ax.set_xlabel("Number of mesh nodes")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Benchmark: LSTM-GNN timing (zoom)")
    ax.legend()
    ax.grid(True)
    all_vals: np.ndarray = np.asarray(
        [data["lstm"], data["gnn_last"], data["lstm+gnn"]], dtype=float,
    )
    y_min: float = float(np.nanmin(all_vals))
    y_max: float = float(np.nanmax(all_vals))
    pad: float = (y_max - y_min) * 0.08 if y_max > y_min else 0.01
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_and_plot(csv_path: Path, outdir: Path) -> None:
    df: pd.DataFrame = pd.read_csv(csv_path)
    data: dict[str, list[float]] = {k: df[k].tolist() for k in df.columns}
    _plot_full(data, outdir / "benchmark_fem_vs_lstm_gnn.pdf")
    _plot_zoom(data, outdir / "benchmark_fem_vs_lstm_gnn_zoom.pdf")


def _parse_extra_fem_table(
    raw: str | dict | None,
) -> dict[int, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): float(v) for k, v in raw.items()}
    return {int(k): float(v) for k, v in json.loads(raw).items()}


def _parse_explicit_sizes(
    raw: list[int] | tuple[int, ...] | str | None,
) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return sorted({int(x) for x in raw})
    return sorted({int(x) for x in json.loads(raw)})


def _closest_match(
    target: int,
    candidates: list[int],
    max_rel_diff: float,
) -> int | None:
    if not candidates:
        return None
    best: int = min(candidates, key=lambda c: abs(c - target))
    if abs(best - target) / max(target, 1) <= max_rel_diff:
        return best
    return None


@torch.no_grad()
def main(
    lstm_checkpoint: str,
    gnn_checkpoint: str,
    output_dir: str = "outputs/figures",
    csv: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    steps: int = 10,
    min_nodes: int = 100,
    max_nodes: int = 20000,
    mesh_sizes_explicit: list[int] | str | None = None,
    hole_refinement_factor: float = 4.0,
    mesh_search_samples: int = 100,
    seed: int = 69,
    strain_low: float = -0.05,
    strain_high: float = 0.05,
    n_macro_steps: int = 4,
    n_increments_per_step: int = 25,
    n_mean_steps: int = 10,
    gnn_chunk_size: int = 1,
    amp: bool = True,
    compile_gnn: bool = False,
    skip_fem: bool = False,
    fem_lstm_csv: str | None = None,
    extra_fem_table: str | dict | None = None,
    fem_parallel: bool = False,
    fem_workers: int = 0,
    fem_mp_start: str = "spawn",
) -> None:
    outdir: Path = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if csv is not None:
        csv_path: Path = Path(csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        _load_and_plot(csv_path, outdir)
        return

    for label, ckpt in (("lstm", lstm_checkpoint), ("gnn", gnn_checkpoint)):
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {ckpt}")

    explicit_targets: list[int] | None = _parse_explicit_sizes(mesh_sizes_explicit)
    extra_fem: dict[int, float] = _parse_extra_fem_table(extra_fem_table)

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng: np.random.Generator = np.random.default_rng(seed=seed)

    if explicit_targets is not None:
        min_nodes = explicit_targets[0]
        max_nodes = explicit_targets[-1]
        targets: np.ndarray = np.asarray(explicit_targets, dtype=float)
        geo_lo: float = min(0.50, 2.4 / max(np.sqrt(min_nodes), 1.0))
        geo_hi: float = max(0.005, 2.4 / (np.sqrt(max_nodes) * 1.5))
        n_search_samples: int = max(mesh_search_samples, 12 * len(explicit_targets))
    else:
        targets = np.linspace(min_nodes, max_nodes, steps)
        geo_lo, geo_hi = 0.30, 0.01
        n_search_samples = mesh_search_samples

    mesh_sizes: np.ndarray = np.geomspace(geo_lo, geo_hi, n_search_samples)
    upper_bound: float = max_nodes * 1.5 if explicit_targets is not None else max_nodes
    candidates: list[tuple[float, pv.PolyData, int, int]] = []
    for size in mesh_sizes:
        mesh: pv.PolyData = hole_plate_mesh(
            width=1.0, height=1.0, radius=0.2,
            hole_center=(0.5, 0.5),
            hole_refinement_factor=float(hole_refinement_factor),
            global_mesh_refinement_size=float(size),
            mesh_type="tri",
        ).extract_surface()
        if mesh.n_points > upper_bound:
            break
        candidates.append((float(size), mesh, mesh.n_points, mesh.n_cells))
    candidates.sort(key=lambda x: x[2])

    in_range: list[tuple[float, pv.PolyData, int, int]] = [
        c for c in candidates if min_nodes * 0.5 <= c[2] <= upper_bound
    ]
    if not in_range:
        raise ValueError(
            f"No mesh candidate in [{min_nodes}, {upper_bound}] nodes.",
        )

    node_counts: np.ndarray = np.asarray([c[2] for c in in_range], dtype=float)

    chosen: list[int] = []
    if explicit_targets is not None:
        for target in targets:
            local: int = int(np.argmin(np.abs(node_counts - target)))
            chosen.append(local)
    else:
        prev: int = -1
        for target in targets:
            cand: np.ndarray = np.arange(prev + 1, len(in_range))
            if cand.size == 0:
                raise ValueError("Not enough candidates for linear node spacing.")
            local = int(np.argmin(np.abs(node_counts[cand] - target)))
            chosen.append(int(cand[local]))
            prev = chosen[-1]
    selected: list[tuple[float, pv.PolyData, int, int]] = [in_range[i] for i in chosen]

    lstm: LstmConstitutiveLaw = LstmConstitutiveLaw(lstm_checkpoint, device)
    lstm.eval()

    times_lstm: list[float] = []
    times_gnn: list[float] = []
    times_total: list[float] = []
    times_fem: list[float] = []
    n_nodes_out: list[int] = []
    n_cells_out: list[int] = []

    for idx, (_, mesh, nn, nc) in enumerate(selected):
        print(f"Iteration {idx + 1}/{len(selected)}: n_nodes={nn}, n_cells={nc}")
        n_nodes_out.append(nn)
        n_cells_out.append(nc)

        gnn: PlasticGNN = PlasticGNN(gnn_checkpoint, device, mesh)
        gnn.eval()
        gnn.optimize_for_inference(
            compile=compile_gnn,
            amp_dtype=torch.float16 if amp else None,
        )

        for _ in range(2):
            strain_sequence_warm, strain_states_warm = _random_strain_path(
                rng, strain_low, strain_high,
                n_macro_steps, n_increments_per_step,
            )
            _ = _benchmark_lstm_gnn_last_step(
                lstm, gnn, strain_sequence_warm,
                device, gnn_chunk_size, amp,
            )

        if not skip_fem:
            _ = solve_fem(mesh, strain_states_warm, n_increments_per_step)

        total_lstm: float = 0.0
        total_gnn: float = 0.0
        total_both: float = 0.0
        strain_samples: list[np.ndarray] = []
        for _ in range(n_mean_steps):
            strain_sequence, strain_states = _random_strain_path(
                rng, strain_low, strain_high,
                n_macro_steps, n_increments_per_step,
            )
            strain_samples.append(strain_states)
            tl, tg, tt = _benchmark_lstm_gnn_last_step(
                lstm, gnn, strain_sequence, device, gnn_chunk_size, amp,
            )
            total_lstm += tl
            total_gnn += tg
            total_both += tt

        if skip_fem:
            t_fem: float = float("nan")
        else:
            if fem_workers <= 0:
                workers: int = min(n_mean_steps, os.cpu_count() or 1)
            else:
                workers = min(fem_workers, n_mean_steps)
            workers = max(1, workers)
            _, sample_times = _benchmark_fem_parallel(
                mesh, strain_samples, n_increments_per_step,
                workers=workers, mp_start=fem_mp_start, parallel=fem_parallel,
            )
            t_fem = float(np.mean(sample_times))

        times_lstm.append(total_lstm / n_mean_steps)
        times_gnn.append(total_gnn / n_mean_steps)
        times_total.append(total_both / n_mean_steps)
        times_fem.append(t_fem)

        del gnn
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if skip_fem:
        ext_csv_lstm: dict[int, float] = {}
        ext_csv_fem: dict[int, float] = {}
        ext_lstm_mean: float | None = None
        if fem_lstm_csv is not None:
            ext_csv_path: Path = Path(os.path.expanduser(fem_lstm_csv))
            if not ext_csv_path.is_file():
                raise FileNotFoundError(
                    f"fem_lstm_csv not found: {ext_csv_path}",
                )
            ext_df: pd.DataFrame = pd.read_csv(ext_csv_path)
            for _, row in ext_df.iterrows():
                ext_csv_lstm[int(row["n_nodes"])] = float(row["lstm"])
                ext_csv_fem[int(row["n_nodes"])] = float(row["fem"])
            ext_lstm_mean = float(np.mean(ext_df["lstm"].values))

        csv_node_keys: list[int] = sorted(ext_csv_lstm.keys())
        extra_node_keys: list[int] = sorted(extra_fem.keys())

        merged_lstm: list[float] = []
        merged_fem: list[float] = []
        for nn in n_nodes_out:
            csv_match: int | None = _closest_match(nn, csv_node_keys, 0.10)
            if csv_match is not None:
                merged_lstm.append(ext_csv_lstm[csv_match])
                merged_fem.append(ext_csv_fem[csv_match])
                continue
            extra_match: int | None = _closest_match(nn, extra_node_keys, 0.10)
            lstm_fallback: float = (
                ext_lstm_mean if ext_lstm_mean is not None else float("nan")
            )
            merged_lstm.append(lstm_fallback)
            if extra_match is not None:
                merged_fem.append(extra_fem[extra_match])
            else:
                merged_fem.append(float("nan"))

        times_lstm = merged_lstm
        times_fem = merged_fem
        times_total = [a + b for a, b in zip(times_lstm, times_gnn)]

    data: dict[str, list[float]] = {
        "n_nodes": [float(x) for x in n_nodes_out],
        "n_cells": [float(x) for x in n_cells_out],
        "lstm": times_lstm,
        "gnn_last": times_gnn,
        "lstm+gnn": times_total,
        "fem": times_fem,
    }
    df: pd.DataFrame = pd.DataFrame(data)
    csv_name: str = (
        "benchmark_data_optimized.csv" if skip_fem else "benchmark_data.csv"
    )
    df.to_csv(outdir / csv_name, index=False)
    _plot_full(data, outdir / "benchmark_fem_vs_lstm_gnn.pdf")
    _plot_zoom(data, outdir / "benchmark_fem_vs_lstm_gnn_zoom.pdf")


if __name__ == "__main__":
    fire.Fire(main)
