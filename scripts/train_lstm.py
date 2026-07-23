"""Train the LSTM constitutive model (macro strain history -> macro stress).

Reads the per-simulation ``sim_*.npz`` macro sequences produced by
``generate_dataset.py`` and trains an :class:`plgnn.lstm.AutoRegressiveStressRNN`
with a normalized-MSE loss. The best checkpoint (``best.pt``) stores the weights and
the input/output scalers together.

    python scripts/train_lstm.py --data-dir <DATA_DIR> --output-dir <OUT_DIR>
"""
from __future__ import annotations

import random
from pathlib import Path

import fire
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plgnn.losses import normalized_mse_loss_single
from plgnn.lstm.models import AutoRegressiveStressRNN
from plgnn.scaling import ModelStandardScaler


def _load_sim(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as npz:
        macro_strain = npz["macro_strain"].astype(np.float32)
        macro_stress = npz["macro_stress"].astype(np.float32)
    strain_seq = np.squeeze(macro_strain, axis=-1)
    stress_seq = np.squeeze(macro_stress, axis=-1)
    return strain_seq[1:], stress_seq[1:]


class MacroPathDataset(Dataset):
    def __init__(self, npz_paths: list[Path]) -> None:
        if not npz_paths:
            raise FileNotFoundError("MacroPathDataset received an empty list.")
        self.npz_paths = npz_paths
        strains: list[np.ndarray] = []
        stresses: list[np.ndarray] = []
        for path in tqdm(npz_paths, desc="Loading macro sequences"):
            eps, sig = _load_sim(path)
            strains.append(eps)
            stresses.append(sig)
        self.strains = np.stack(strains, axis=0)
        self.stresses = np.stack(stresses, axis=0)

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.strains[idx]),
            torch.from_numpy(self.stresses[idx]),
        )


def _discover_sims(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("sim_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No simulation files sim_*.npz found in {data_dir}"
        )
    return paths


def _split_indices(
    n: int, test_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_fraction))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def _fit_standard_scaler(data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    flat = data.reshape(-1, data.shape[-1])
    mean = torch.from_numpy(flat.mean(axis=0)).float()
    std = torch.from_numpy(flat.std(axis=0)).float()
    std[std < 1e-8] = 1.0
    return mean, std


def _apply_scaler(
    tensor: torch.Tensor, scaler: ModelStandardScaler
) -> torch.Tensor:
    mean = scaler.mean.to(tensor.device, dtype=tensor.dtype)
    std = scaler.std.to(tensor.device, dtype=tensor.dtype)
    return (tensor - mean) / std


def _run_epoch(
    model: AutoRegressiveStressRNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for strain_batch, stress_batch in loader:
            strain_batch = strain_batch.to(device)
            stress_batch = stress_batch.to(device)
            x = _apply_scaler(strain_batch, model.input_scaler)
            y = _apply_scaler(stress_batch, model.output_scaler)
            pred, _ = model.forward(x)
            loss = normalized_mse_loss_single(
                y.reshape(-1, y.shape[-1]),
                pred.reshape(-1, pred.shape[-1]),
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * strain_batch.shape[0]
            count += strain_batch.shape[0]
    return total / max(count, 1)


def main(
    data_dir: str,
    output_dir: str,
    hidden_state_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 64,
    epochs: int = 2000,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.3,
    seed: int = 69,
    num_workers: int = 0,
) -> None:
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sim_paths = _discover_sims(data_path)
    train_idx, val_idx = _split_indices(len(sim_paths), test_fraction, seed)

    train_dataset = MacroPathDataset([sim_paths[i] for i in train_idx])
    val_dataset = MacroPathDataset([sim_paths[i] for i in val_idx])

    input_mean, input_std = _fit_standard_scaler(train_dataset.strains)
    output_mean, output_std = _fit_standard_scaler(train_dataset.stresses)

    input_scaler = ModelStandardScaler(mean=input_mean, std=input_std)
    output_scaler = ModelStandardScaler(mean=output_mean, std=output_std)

    model = AutoRegressiveStressRNN(
        input_features_size=3,
        hidden_state_size=hidden_state_size,
        output_size=3,
        num_layers=num_layers,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    writer = SummaryWriter(log_dir=(output_path / "tb").as_posix())
    best_path = output_path / "best.pt"
    last_path = output_path / "last.pt"
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        val_loss = _run_epoch(model, val_loader, None, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            model.save_model_checkpoint(
                best_path.as_posix(), optimizer=optimizer, epoch=epoch
            )

    model.save_model_checkpoint(
        last_path.as_posix(), optimizer=optimizer, epoch=epochs
    )
    writer.close()
    print(f"Best val NMSE: {best_val:.6f}. Checkpoint: {best_path}")


if __name__ == "__main__":
    fire.Fire(main)
