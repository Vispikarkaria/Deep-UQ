"""Tests for linearized (GLM) Laplace predictive."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.laplace import LaplaceWrapper


def _make_regression_setup():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 1))
    x_train = torch.randn(50, 3)
    y_train = torch.randn(50, 1)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)

    # Train briefly
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    return model, loader


class TestLinearizedLaplace:
    def test_glm_returns_valid_uqresult(self):
        model, loader = _make_regression_setup()
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader, prior_precision=1.0)

        x_test = torch.randn(5, 3)
        result = la.predict_uq(x_test, method="glm")

        assert result.mean is not None
        assert result.mean.shape == (5, 1)
        assert result.total_var is not None
        assert result.total_var.shape == (5, 1)

    def test_glm_variance_positive(self):
        model, loader = _make_regression_setup()
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader, prior_precision=1.0)

        x_test = torch.randn(5, 3)
        result = la.predict_uq(x_test, method="glm")

        assert (result.total_var > 0).all()

    def test_glm_differs_from_sampling(self):
        model, loader = _make_regression_setup()
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader, prior_precision=1.0)

        x_test = torch.randn(5, 3)
        result_glm = la.predict_uq(x_test, method="glm")
        result_sampling = la.predict_uq(x_test, method="sampling")

        # Both should produce valid variance; metadata should differ
        assert result_glm.metadata["method"] == "laplace_glm"
        assert result_sampling.metadata["method"] == "laplace"
        # Both produce positive variance
        assert (result_glm.total_var > 0).all()
        assert (result_sampling.total_var > 0).all()

    def test_glm_metadata(self):
        model, loader = _make_regression_setup()
        la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
        la.fit(loader, prior_precision=1.0)

        x_test = torch.randn(3, 3)
        result = la.predict_uq(x_test, method="glm")
        assert result.metadata["method"] == "laplace_glm"
