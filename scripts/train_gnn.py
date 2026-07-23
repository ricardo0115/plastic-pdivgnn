"""Train the GNN field reconstructor (and the divergence-regularized P-DivGNN).

Node features are the LSTM hidden state (precomputed) + the macro stress, broadcast
over the periodic mesh graph. With ``--div-target-ratio > 0`` a discrete divergence
penalty is added with a linear warm-up (the P-DivGNN variant). Models and the
divergence operators come from the plgnn library.

    # vanilla GNN
    python scripts/train_gnn.py --data-dir <D> --hidden-dir <D>/hidden \
        --mesh-path <D>/mesh.vtk --output-dir <OUT>
    # P-DivGNN
    python scripts/train_gnn.py ... --div-target-ratio 0.1
"""

from __future__ import annotations

import random
from pathlib import Path

import fire
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from plgnn.graph.build import build_graph, normalize_graph
from plgnn.graph.models import EncodeProcessDecode
from plgnn.losses import normalized_mse_loss_single
from plgnn.physics import compute_divergence_batch, scipy_coo_to_torch_sparse
from plgnn.physics_fem import compute_op_div_matrix
from plgnn.scaling import ModelStandardScaler, unstandardize
from plgnn.train_utils import SNAPSHOT_INDICES, split_pairs


def _discover_pairs(
    data_dir: Path, hidden_dir: Path
) -> list[tuple[Path, Path]]:
    sims = sorted(data_dir.glob("sim_*.npz"))
    if not sims:
        raise FileNotFoundError(f"No sim_*.npz files in {data_dir}")
    pairs: list[tuple[Path, Path]] = []
    for sim_path in sims:
        hidden_path = hidden_dir / f"{sim_path.stem}.npy"
        if not hidden_path.is_file():
            raise FileNotFoundError(
                "Missing hidden-state sidecar for "
                f"{sim_path.name}: expected {hidden_path}"
            )
        pairs.append((sim_path, hidden_path))
    return pairs


class SnapshotGraphDataset(Dataset):
    """One sample = one sim, returning every snapshot as a list of `Data`.

    Each `__getitem__` yields `len(snapshot_indices)` graphs of the same sim;
    pair this with `_collate_sims` so a DataLoader batch of N sims becomes a
    single PyG `Batch` of N x n_snapshots graphs. This preserves the
    "contiguous loading subsequence per batch" semantics. Per-sim tensors are
    cached lazily; `compute_xy_stats()` then `set_normalization()` must run
    before iteration.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        positions: torch.Tensor,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        snapshot_indices: tuple[int, ...],
        use_lstm_hidden_states: bool = True,
        macro_all: np.ndarray | None = None,
    ) -> None:
        self.pairs = pairs
        self.positions = positions
        self.node_labels = node_labels
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.snapshot_indices = snapshot_indices
        self.use_lstm_hidden_states = use_lstm_hidden_states
        self.macro_all = macro_all
        self.n_nodes = positions.shape[0]
        self._cache: list[dict[str, torch.Tensor]] = [{} for _ in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_sim(self, sim_idx: int) -> dict[str, torch.Tensor]:
        entry = self._cache[sim_idx]
        if entry:
            return entry
        sim_path, hidden_path = self.pairs[sim_idx]
        with np.load(sim_path) as npz:
            local_stress = npz["local_stress"].astype(np.float32)[1:]

        stress = torch.from_numpy(local_stress[list(self.snapshot_indices)])
        stress = stress.permute(0, 2, 1).contiguous()
        if self.macro_all is None:
            mean_stress = stress.mean(dim=1)
        else:
            # Stored Hill-Mandel (volume-integrated) macro stress, indexed by sim id.
            sim_id = int(sim_path.stem.split("_")[1])
            mean_stress = torch.from_numpy(
                self.macro_all[sim_id][1:][list(self.snapshot_indices)]
            )
        entry["local_stress"] = stress
        entry["mean_stress"] = mean_stress
        if self.use_lstm_hidden_states:
            hidden = np.load(hidden_path).astype(np.float32)
            entry["hidden_states"] = torch.from_numpy(
                hidden[list(self.snapshot_indices)]
            )
        self._cache[sim_idx] = entry
        return entry

    def compute_xy_stats(
        self, use_cache: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_x: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        for sim_idx in tqdm(range(len(self.pairs)), desc="Normalization stats"):
            data = self._load_sim(sim_idx)
            if self.use_lstm_hidden_states:
                x_components = [data["hidden_states"], data["mean_stress"]]
            else:
                x_components = [data["mean_stress"]]
            all_x.append(torch.cat(x_components, dim=-1))
            all_y.append(data["local_stress"])
        x_cat = torch.cat(all_x, dim=0)
        y_cat = torch.cat(all_y, dim=0)
        if not use_cache:
            self._cache = [{} for _ in self.pairs]
        return x_cat.mean(), x_cat.std(), y_cat.mean(), y_cat.std()

    def set_normalization(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __getitem__(self, sim_idx: int) -> list[Data]:
        data = self._load_sim(sim_idx)
        graphs: list[Data] = []
        for t in range(len(self.snapshot_indices)):
            mean_stress = data["mean_stress"][t].unsqueeze(0)
            local_stress = data["local_stress"][t]

            if self.use_lstm_hidden_states:
                hidden_state = data["hidden_states"][t]
                x = torch.cat(
                    [
                        hidden_state.repeat(self.n_nodes, 1),
                        mean_stress.repeat(self.n_nodes, 1),
                    ],
                    dim=-1,
                )
            else:
                x = mean_stress.repeat(self.n_nodes, 1)
            x = (x - self.x_mean) / self.x_std
            y = (local_stress - self.y_mean) / self.y_std
            x = torch.cat(
                [x, self.positions, self.node_labels.unsqueeze(1)],
                dim=-1,
            )
            graphs.append(
                Data(
                    x=x,
                    pos=self.positions,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                    y=y,
                )
            )
        return graphs


def _collate_sims(samples: list[list[Data]]) -> Batch:
    return Batch.from_data_list([g for sample in samples for g in sample])


def _run_epoch(
    model: EncodeProcessDecode,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    current_ratio: float,
    op_div_matrix: torch.Tensor | None,
    raw_node_labels: torch.Tensor | None,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    num_nodes: int,
    desc: str,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_nmse = 0.0
    total_div = 0.0
    total_combined = 0.0
    total_eff_weight = 0.0
    count = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)
            pred = model.forward(batch).local_stress
            nmse = normalized_mse_loss_single(batch.y, pred)

            if current_ratio > 0.0:
                pred_phys = unstandardize(pred, y_mean, y_std)
                div = compute_divergence_batch(
                    pred_phys, op_div_matrix, raw_node_labels, num_nodes
                )
                if div > 0:
                    eff_weight = current_ratio * (
                        nmse.detach() / (div.detach() + 1e-8)
                    )
                    combined = nmse + eff_weight * div
                else:
                    eff_weight = torch.tensor(0.0, device=device)
                    combined = nmse
            else:
                div = torch.tensor(0.0, device=device)
                eff_weight = torch.tensor(0.0, device=device)
                combined = nmse

            if is_train:
                optimizer.zero_grad()
                combined.backward()
                optimizer.step()

            n = batch.num_graphs
            total_nmse += float(nmse.detach()) * n
            total_div += float(div.detach()) * n
            total_combined += float(combined.detach()) * n
            total_eff_weight += float(eff_weight.detach()) * n
            count += n
    denom = max(count, 1)
    return (
        total_nmse / denom,
        total_div / denom,
        total_combined / denom,
        total_eff_weight / denom,
    )


def main(
    data_dir: str,
    hidden_dir: str,
    mesh_path: str,
    output_dir: str,
    div_target_ratio: float = 0.1,
    warmup_epochs: int = 20,
    epochs: int = 100,
    batch_size: int = 7,
    learning_rate: float = 1e-3,
    message_passing_steps: int = 10,
    latent_size: int = 128,
    hidden_state_size: int = 64,
    test_fraction: float = 0.3,
    seed: int = 69,
    integrated_macro: bool = False,
    macro_sidecar: str | None = None,
    limit_sims: int | None = None,
    resume_from: str | None = None,
    snapshot_indices: tuple[int, ...] = SNAPSHOT_INDICES,
    use_lstm_hidden_states: bool = True,
) -> None:
    data_path = Path(data_dir).expanduser().resolve()
    hidden_path = Path(hidden_dir).expanduser().resolve()
    mesh_file = Path(mesh_path).expanduser().resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(f"Mesh file missing: {mesh_file}")
    if not hidden_path.is_dir():
        raise FileNotFoundError(f"Hidden-states dir missing: {hidden_path}")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = pv.read(mesh_file.as_posix()).extract_surface()
    graph = build_graph(mesh, periodic=True)
    positions, node_labels, edge_attr, _ = normalize_graph(graph)
    edge_index = graph.edge_index
    n_nodes = positions.shape[0]

    pairs = _discover_pairs(data_path, hidden_path)
    train_pairs, val_pairs = split_pairs(pairs, test_fraction, seed)
    if limit_sims is not None:
        train_pairs = train_pairs[:limit_sims]
        val_pairs = val_pairs[: max(1, limit_sims // 4)]
        print(
            f"[limit_sims] subset -> {len(train_pairs)} train / "
            f"{len(val_pairs)} val sims"
        )

    macro_all = None
    if integrated_macro:
        if macro_sidecar is None:
            raise ValueError(
                "integrated_macro=True requires --macro-sidecar pointing to the "
                "stored Hill-Mandel macro stresses."
            )
        macro_all = np.load(Path(macro_sidecar).expanduser().resolve())
        print(
            f"[macro input] stored Hill-Mandel from {macro_sidecar} "
            f"{macro_all.shape}"
        )

    train_dataset = SnapshotGraphDataset(
        train_pairs,
        positions,
        node_labels,
        edge_index,
        edge_attr,
        snapshot_indices,
        use_lstm_hidden_states=use_lstm_hidden_states,
        macro_all=macro_all,
    )
    val_dataset = SnapshotGraphDataset(
        val_pairs,
        positions,
        node_labels,
        edge_index,
        edge_attr,
        snapshot_indices,
        use_lstm_hidden_states=use_lstm_hidden_states,
        macro_all=macro_all,
    )
    x_mean, x_std, y_mean, y_std = train_dataset.compute_xy_stats()
    train_dataset.set_normalization(x_mean, x_std, y_mean, y_std)
    val_dataset.set_normalization(x_mean, x_std, y_mean, y_std)

    input_nodes_features_size = (
        (hidden_state_size if use_lstm_hidden_states else 0) + 3 + 2 + 1
    )
    print(
        f"use_lstm_hidden_states={use_lstm_hidden_states}, "
        f"input_nodes_features_size={input_nodes_features_size}"
    )
    model = EncodeProcessDecode(
        input_edges_features_size=1,
        message_passing_steps=message_passing_steps,
        latent_size=latent_size,
        input_nodes_features_size=input_nodes_features_size,
        output_nodes_features_size=3,
        input_scaler=ModelStandardScaler(mean=x_mean, std=x_std),
        output_scaler=ModelStandardScaler(mean=y_mean, std=y_std),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if resume_from is not None:
        ckpt_path = Path(resume_from).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint missing: {ckpt_path}")
        start_epoch = model.load_model_checkpoint(
            ckpt_path.as_posix(), optimizer=optimizer
        )
        for scaler in (model.input_scaler, model.output_scaler):
            if isinstance(scaler.mean, torch.Tensor):
                scaler.mean = scaler.mean.to(device)
                scaler.std = scaler.std.to(device)
        print(
            f"[resume] loaded {ckpt_path} @ epoch {start_epoch}; "
            f"continuing to {epochs}"
        )

    op_div_matrix: torch.Tensor | None = None
    raw_node_labels: torch.Tensor | None = None
    if div_target_ratio > 0.0:
        op_div_scipy = compute_op_div_matrix(mesh)
        op_div_matrix = scipy_coo_to_torch_sparse(op_div_scipy).to(device)
        raw_node_labels = node_labels.long().to(device)

    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_sims,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_sims,
    )

    writer = SummaryWriter(log_dir=(output_path / "tb").as_posix())
    best_path = output_path / "best.pt"
    last_path = output_path / "last.pt"
    best_val = float("inf")

    for epoch in range(start_epoch + 1, epochs + 1):
        warmup_factor = (
            min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
        )
        current_ratio = div_target_ratio * warmup_factor

        train_nmse, train_div, train_combined, train_eff_weight = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            current_ratio,
            op_div_matrix,
            raw_node_labels,
            y_mean_dev,
            y_std_dev,
            n_nodes,
            desc=f"Train {epoch:03d}",
        )
        val_nmse, val_div, val_combined, val_eff_weight = _run_epoch(
            model,
            val_loader,
            None,
            device,
            current_ratio,
            op_div_matrix,
            raw_node_labels,
            y_mean_dev,
            y_std_dev,
            n_nodes,
            desc=f"Val   {epoch:03d}",
        )
        writer.add_scalar("Loss/NMSE_train", train_nmse, epoch)
        writer.add_scalar("Loss/NMSE_val", val_nmse, epoch)
        writer.add_scalar("Loss/Combined_train", train_combined, epoch)
        writer.add_scalar("Loss/Combined_val", val_combined, epoch)
        writer.add_scalar("Divergence/train", train_div, epoch)
        writer.add_scalar("Divergence/val", val_div, epoch)
        writer.add_scalar("Penalty/warmup_factor", warmup_factor, epoch)
        writer.add_scalar("Penalty/current_ratio", current_ratio, epoch)
        writer.add_scalar(
            "Penalty/effective_weight_train", train_eff_weight, epoch
        )
        writer.add_scalar("Penalty/effective_weight_val", val_eff_weight, epoch)
        print(
            f"Epoch {epoch:03d} | warmup={warmup_factor:.2f} "
            f"eff_w={train_eff_weight:.2e} | "
            f"train NMSE={train_nmse:.6f} div={train_div:.3e} | "
            f"val NMSE={val_nmse:.6f} div={val_div:.3e}"
        )
        if val_combined < best_val:
            best_val = val_combined
            model.save_model_checkpoint(
                best_path.as_posix(), optimizer=optimizer, epoch=epoch
            )
        model.save_model_checkpoint(
            last_path.as_posix(), optimizer=optimizer, epoch=epoch
        )

    writer.close()
    print(f"Best val loss: {best_val:.6f}. Checkpoint: {best_path}")


if __name__ == "__main__":
    fire.Fire(main)
