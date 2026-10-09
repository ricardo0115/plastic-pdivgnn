"""Increment guard of the constitutive LSTM."""

from __future__ import annotations

import pytest
import torch

from plgnn.lstm import AutoRegressiveStressRNN, resample_in_time
from plgnn.scaling import ModelStandardScaler

HIDDEN = 8
LAYERS = 2
STEPS = 12
TOLERANCE = 1e-6


def _model(threshold: float) -> AutoRegressiveStressRNN:
    torch.manual_seed(0)
    model = AutoRegressiveStressRNN(
        input_features_size=3,
        hidden_state_size=HIDDEN,
        output_size=3,
        num_layers=LAYERS,
        input_scaler=ModelStandardScaler(
            mean=torch.tensor([0.01, -0.02, 0.0]), std=torch.tensor([0.05, 0.04, 0.03])
        ),
        output_scaler=ModelStandardScaler(mean=torch.zeros(3), std=torch.ones(3)),
        increment_threshold=threshold,
    )
    return model.eval()


def _scaled(model: AutoRegressiveStressRNN, physical: torch.Tensor) -> torch.Tensor:
    return model.input_scaler.transform(physical)


def _random_path(seed: int, n_cases: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    increments = torch.randn(n_cases, STEPS, 3, generator=generator) * 1e-3
    return increments.cumsum(dim=1)


def test_near_zero_threshold_matches_the_fused_recurrence():
    model = _model(1e-12)
    x = _scaled(model, _random_path(1))
    with torch.no_grad():
        guarded = model(x, return_hidden_states=True)
        model.increment_threshold.zero_()
        fused = model(x, return_hidden_states=True)
    assert torch.allclose(guarded[0], fused[0], atol=TOLERANCE)
    assert torch.allclose(guarded[1][0], fused[1][0], atol=TOLERANCE)
    assert torch.allclose(guarded[1][1], fused[1][1], atol=TOLERANCE)
    assert torch.allclose(guarded[2], fused[2], atol=TOLERANCE)


def test_repeated_input_returns_the_held_output():
    model = _model(5e-4)
    path = _random_path(2)
    held_from = STEPS - 5
    path[:, held_from:] = path[:, held_from - 1 : held_from]
    with torch.no_grad():
        preds, _, output = model(_scaled(model, path), return_hidden_states=True)
    reference_pred = preds[:, held_from - 1 : held_from]
    reference_output = output[:, held_from - 1 : held_from]
    assert torch.equal(preds[:, held_from:], reference_pred.expand(-1, 5, -1))
    assert torch.equal(output[:, held_from:], reference_output.expand(-1, 5, -1))


def test_refined_path_reaches_the_same_state():
    increment = 2e-3
    generator = torch.Generator().manual_seed(3)
    directions = torch.randn(2, STEPS, 3, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    coarse = (directions * increment).cumsum(dim=1)
    fine = resample_in_time(coarse, 10 * (STEPS - 1) + 1)
    model = _model(0.99 * increment)
    with torch.no_grad():
        _, (h_coarse, c_coarse) = model(_scaled(model, coarse))
        _, (h_fine, c_fine) = model(_scaled(model, fine))
    assert torch.allclose(h_coarse, h_fine, atol=1e-5)
    assert torch.allclose(c_coarse, c_fine, atol=1e-5)


def test_unbatched_input_keeps_unbatched_shapes():
    model = _model(5e-4)
    x = _scaled(model, _random_path(4, n_cases=1))[0]
    with torch.no_grad():
        preds, (h, c), output = model(x, return_hidden_states=True)
    assert preds.shape == (STEPS, 3)
    assert output.shape == (STEPS, HIDDEN)
    assert h.shape == (LAYERS, HIDDEN) and c.shape == (LAYERS, HIDDEN)


def test_checkpoint_round_trip_restores_the_threshold(tmp_path):
    model = _model(3.25e-4)
    path = tmp_path / "model.pt"
    model.save_model_checkpoint(path.as_posix(), epoch=1)
    loaded = _model(0.0)
    loaded.load_model_checkpoint(path.as_posix())
    assert float(loaded.increment_threshold) == pytest.approx(3.25e-4)

    without_key = {
        k: v for k, v in model.state_dict().items() if k != "increment_threshold"
    }
    fallback = _model(1e-3)
    fallback.load_state_dict(without_key, strict=True)
    assert float(fallback.increment_threshold) == pytest.approx(1e-3)
