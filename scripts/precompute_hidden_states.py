"""Precompute the LSTM hidden-state sequences used as GNN node features.

Runs the trained LSTM in evaluation mode, so with its stored increment guard, over
every simulation's macro strain history and stores the per-step hidden output as
``sim_XXXXXX.npy`` under ``--hidden-dir`` (default: ``<data-dir>/hidden``). These
feed ``train_gnn.py``. Simulations are processed in batches of ``--batch-size``
paths, which requires every path to have the same number of steps.

    python scripts/precompute_hidden_states.py --data-dir <DATA_DIR> \
        --lstm-checkpoint <LSTM_CKPT>
"""
from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import torch
from tqdm import tqdm

from plgnn.lstm import AutoRegressiveStressRNN
from plgnn.lstm.data import discover_simulations, load_macro_sequences


@torch.no_grad()
def main(
    data_dir: str,
    lstm_checkpoint: str,
    hidden_dir: str | None = None,
    hidden_state_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 256,
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

    sim_paths = discover_simulations(data_path)
    for start in tqdm(
        range(0, len(sim_paths), batch_size), desc="LSTM hidden states"
    ):
        chunk = sim_paths[start : start + batch_size]
        strains = np.stack([load_macro_sequences(path)[0] for path in chunk])
        x = model.input_scaler.transform(torch.from_numpy(strains).to(device))
        _, _, sequence_output = model(x, return_hidden_states=True)
        for sim_path, hidden in zip(chunk, sequence_output.cpu().numpy(), strict=True):
            np.save(hidden_path / sim_path.with_suffix(".npy").name, hidden)

    print(f"Wrote {len(sim_paths)} hidden-state files to {hidden_path}")


if __name__ == "__main__":
    fire.Fire(main)
