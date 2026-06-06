"""Precompute the LSTM hidden-state sequences used as GNN node features.

Runs the trained LSTM over every simulation's macro strain history and stores the
per-step hidden output as ``sim_XXXXXX.npy`` under ``--hidden-dir`` (default:
``<data-dir>/hidden``). These feed ``train_gnn.py``.

    python scripts/precompute_hidden_states.py --data-dir <DATA_DIR> \
        --lstm-checkpoint <LSTM_CKPT>
"""
from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import torch
from tqdm import tqdm

from plgnn.lstm.models import AutoRegressiveStressRNN


def _discover_sims(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("sim_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No simulation files sim_*.npz found in {data_dir}"
        )
    return paths


@torch.no_grad()
def main(
    data_dir: str,
    lstm_checkpoint: str,
    hidden_dir: str | None = None,
    hidden_state_size: int = 64,
    num_layers: int = 2,
) -> None:
    data_path = Path(data_dir).expanduser().resolve()
    ckpt_path = Path(lstm_checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LSTM checkpoint missing: {ckpt_path}")

    hidden_path = (
        Path(hidden_dir).expanduser().resolve()
        if hidden_dir is not None
        else data_path / "hidden"
    )
    hidden_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRegressiveStressRNN(
        input_features_size=3,
        hidden_state_size=hidden_state_size,
        output_size=3,
        num_layers=num_layers,
    )
    model.load_model_checkpoint(ckpt_path.as_posix())
    model = model.to(device).eval()
    for scaler in (model.input_scaler, model.output_scaler):
        if isinstance(scaler.mean, torch.Tensor):
            scaler.mean = scaler.mean.to(device)
            scaler.std = scaler.std.to(device)

    sim_paths = _discover_sims(data_path)
    for sim_path in tqdm(sim_paths, desc="LSTM hidden states"):
        sim_id = sim_path.stem.split("_")[-1]
        out_path = hidden_path / f"sim_{sim_id}.npy"
        with np.load(sim_path) as npz:
            strain = npz["macro_strain"].astype(np.float32)
        strain_seq = np.squeeze(strain, axis=-1)[1:]
        x = torch.from_numpy(strain_seq).to(device)
        x = model.input_scaler.transform(x).float()
        sequence_output, _ = model.rnn.forward(x)
        np.save(out_path.as_posix(), sequence_output.detach().cpu().numpy())

    print(f"Wrote {len(sim_paths)} hidden-state files to {hidden_path}")


if __name__ == "__main__":
    fire.Fire(main)
