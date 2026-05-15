"""Tests for selective prediction module."""

import torch
import torch.nn as nn

from deepuq.methods.mc_dropout import MCDropoutWrapper
from deepuq.methods.selective import (
    SelectiveMetrics,
    SelectivePredictor,
    SelectiveResult,
)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=4, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _make_model():
    mlp = SimpleMLP()
    wrapper = MCDropoutWrapper(mlp, n_mc=10, apply_softmax=False)
    return wrapper


def test_predict_with_rejection_coverage():
    """coverage=0.5 accepts exactly half the samples."""
    model = _make_model()
    x = torch.randn(20, 4)
    sp = SelectivePredictor(model, criterion="epistemic_var")
    result = sp.predict_with_rejection(x, coverage=0.5)

    assert isinstance(result, SelectiveResult)
    assert result.accepted_mask.sum().item() == 10
    assert abs(result.coverage - 0.5) < 1e-6


def test_find_threshold_returns_float():
    """find_threshold returns a reasonable float."""
    model = _make_model()
    x = torch.randn(50, 4)
    y = torch.randn(50, 1)
    sp = SelectivePredictor(model, criterion="epistemic_var")
    threshold = sp.find_threshold(x, y, target_coverage=0.8)

    assert isinstance(threshold, float)
    assert threshold >= 0.0


def test_higher_uncertainty_rejected_first():
    """Higher uncertainty samples are rejected first."""
    model = _make_model()
    x = torch.randn(30, 4)
    sp = SelectivePredictor(model, criterion="epistemic_var")

    # Use the result's own uncertainties to verify ordering
    result = sp.predict_with_rejection(x, coverage=0.5)
    accepted_unc = result.uncertainties[result.accepted_mask]
    rejected_unc = result.uncertainties[~result.accepted_mask]

    assert accepted_unc.max() <= rejected_unc.min() + 1e-6


def test_selective_metrics_fields():
    """SelectiveMetrics has all expected fields."""
    model = _make_model()
    x = torch.randn(40, 4)
    y = torch.randn(40, 1)
    sp = SelectivePredictor(model, criterion="epistemic_var")
    metrics = sp.evaluate(x, y)

    assert isinstance(metrics, SelectiveMetrics)
    assert hasattr(metrics, "coverage")
    assert hasattr(metrics, "selective_mse")
    assert hasattr(metrics, "rejection_rate")
    assert hasattr(metrics, "oracle_accuracy")
    assert hasattr(metrics, "aurc")


def test_evaluate_returns_valid_metrics():
    """evaluate returns valid metrics with sensible values."""
    model = _make_model()
    x = torch.randn(50, 4)
    y = torch.randn(50, 1)
    sp = SelectivePredictor(model, criterion="epistemic_var")
    metrics = sp.evaluate(x, y)

    assert 0.0 <= metrics.coverage <= 1.0
    assert metrics.selective_mse >= 0.0
    assert 0.0 <= metrics.rejection_rate <= 1.0
    assert metrics.aurc >= 0.0
