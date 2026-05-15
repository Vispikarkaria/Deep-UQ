"""Tests for Multi-Fidelity Gaussian Process."""

import torch

from deepuq.models.gp.multifidelity import MultiFidelityGP
from deepuq.types import UQResult


def _make_toy_data():
    torch.manual_seed(42)
    X_lo = torch.linspace(0, 5, 20).unsqueeze(-1)
    y_lo = torch.sin(X_lo).squeeze() + 0.3 * torch.randn(20)
    X_hi = torch.linspace(0, 5, 8).unsqueeze(-1)
    y_hi = torch.sin(X_hi).squeeze() + 0.05 * torch.randn(8)
    return X_lo, y_lo, X_hi, y_hi


def test_fit_runs_without_error():
    X_lo, y_lo, X_hi, y_hi = _make_toy_data()
    model = MultiFidelityGP()
    model.fit(X_lo, y_lo, X_hi, y_hi)


def test_predict_uq_returns_valid_uqresult():
    X_lo, y_lo, X_hi, y_hi = _make_toy_data()
    model = MultiFidelityGP()
    model.fit(X_lo, y_lo, X_hi, y_hi)
    X_new = torch.linspace(0, 5, 10).unsqueeze(-1)
    result = model.predict_uq(X_new)
    assert isinstance(result, UQResult)
    assert result.mean.shape == (10,)
    assert result.epistemic_var.shape == (10,)
    assert (result.epistemic_var >= 0).all()


def test_high_fidelity_lower_variance_than_low():
    X_lo, y_lo, X_hi, y_hi = _make_toy_data()
    model = MultiFidelityGP()
    model.fit(X_lo, y_lo, X_hi, y_hi)
    # Predict at high-fidelity training locations
    result_hi = model.predict_uq(X_hi, fidelity="high")
    result_lo = model.predict_uq(X_hi, fidelity="low")
    # High-fidelity should generally have lower total variance
    # (at least on average at training points)
    assert result_hi.total_var.mean() < result_lo.total_var.mean()


def test_predict_uq_low_fidelity():
    X_lo, y_lo, X_hi, y_hi = _make_toy_data()
    model = MultiFidelityGP()
    model.fit(X_lo, y_lo, X_hi, y_hi)
    X_new = torch.linspace(0, 5, 5).unsqueeze(-1)
    result = model.predict_uq(X_new, fidelity="low")
    assert isinstance(result, UQResult)
    assert result.metadata["fidelity"] == "low"


def test_rho_is_learnable():
    X_lo, y_lo, X_hi, y_hi = _make_toy_data()
    model = MultiFidelityGP()
    rho_before = model.rho.item()
    model.optimize(X_lo, y_lo, X_hi, y_hi, n_iter=50, lr=0.01)
    rho_after = model.rho.item()
    assert rho_before != rho_after
