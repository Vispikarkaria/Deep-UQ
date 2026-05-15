"""Tests for post-hoc calibration methods."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.calibration import (
    IsotonicCalibration,
    TemperatureScaling,
    VectorScaling,
)
from deepuq.types import UQResult


def _make_miscalibrated_model(num_classes: int = 5) -> nn.Module:
    """Create a simple model that outputs overconfident logits."""
    model = nn.Linear(10, num_classes)
    # Scale weights to produce overconfident predictions
    with torch.no_grad():
        model.weight.mul_(5.0)
    return model


def _make_val_loader(
    model: nn.Module, n: int = 200, num_classes: int = 5
) -> DataLoader:
    """Create a validation loader with random data."""
    x = torch.randn(n, 10)
    # Random labels
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


class TestTemperatureScaling:
    def test_identity_when_t_equals_one(self) -> None:
        """When T=1, calibrated probs equal raw softmax probs."""
        model = nn.Linear(10, 5)
        ts = TemperatureScaling(model)
        ts.temperature = nn.Parameter(torch.tensor(1.0))

        x = torch.randn(8, 10)
        calibrated = ts.predict_calibrated(x)
        with torch.no_grad():
            expected = torch.softmax(model(x), dim=-1)

        assert torch.allclose(calibrated, expected, atol=1e-6)

    def test_fit_changes_temperature(self) -> None:
        """fit() should move T away from initial value on miscalibrated model."""
        model = _make_miscalibrated_model()
        ts = TemperatureScaling(model)
        initial_t = ts.temperature.item()

        val_loader = _make_val_loader(model)
        ts.fit(val_loader, max_iter=50, lr=0.01)

        assert ts.temperature.item() != initial_t

    def test_predict_uq_returns_uqresult(self) -> None:
        """predict_uq should return a valid UQResult."""
        model = nn.Linear(10, 5)
        ts = TemperatureScaling(model)
        x = torch.randn(4, 10)

        result = ts.predict_uq(x)

        assert isinstance(result, UQResult)
        assert result.mean.shape == (4, 5)
        assert result.epistemic_var.shape == (4,)
        assert result.probs is not None


class TestVectorScaling:
    def test_fit_and_predict(self) -> None:
        """VectorScaling should fit and produce valid probabilities."""
        model = _make_miscalibrated_model()
        vs = VectorScaling(model)
        val_loader = _make_val_loader(model)
        vs.fit(val_loader, max_iter=50)

        x = torch.randn(8, 10)
        probs = vs.predict_calibrated(x)

        assert probs.shape == (8, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
        assert (probs >= 0).all()


class TestIsotonicCalibration:
    def test_produces_valid_probabilities(self) -> None:
        """Isotonic calibration should produce probs in [0,1] summing to 1."""
        model = _make_miscalibrated_model()
        ic = IsotonicCalibration(model)
        val_loader = _make_val_loader(model, n=300)
        ic.fit(val_loader)

        x = torch.randn(16, 10)
        probs = ic.predict_calibrated(x)

        assert probs.shape == (16, 5)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)

    def test_predict_uq_returns_uqresult(self) -> None:
        """predict_uq should return a valid UQResult."""
        model = _make_miscalibrated_model()
        ic = IsotonicCalibration(model)
        val_loader = _make_val_loader(model, n=200)
        ic.fit(val_loader)

        x = torch.randn(4, 10)
        result = ic.predict_uq(x)

        assert isinstance(result, UQResult)
        assert result.mean.shape == (4, 5)
        assert result.epistemic_var.shape == (4,)
