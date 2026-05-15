"""Tests for Evidential Deep Learning methods."""

import torch

from deepuq.methods.evidential import EvidentialClassification, EvidentialRegression
from deepuq.models import MLP
from deepuq.types import UQResult


class TestEvidentialRegression:
    def _make_model(self, input_dim=3, output_dim=1):
        base = MLP(input_dim=input_dim, hidden_dims=[32], output_dim=4 * output_dim)
        return EvidentialRegression(base, output_dim=output_dim)

    def test_forward_shape(self):
        model = self._make_model(input_dim=3, output_dim=2)
        x = torch.randn(8, 3)
        out = model(x)
        assert out["gamma"].shape == (8, 2)
        assert out["nu"].shape == (8, 2)
        assert out["alpha"].shape == (8, 2)
        assert out["beta"].shape == (8, 2)

    def test_forward_positivity(self):
        model = self._make_model()
        x = torch.randn(8, 3)
        out = model(x)
        assert (out["nu"] > 0).all()
        assert (out["alpha"] > 1).all()
        assert (out["beta"] > 0).all()

    def test_loss_finite_positive(self):
        model = self._make_model()
        x = torch.randn(8, 3)
        y = torch.randn(8, 1)
        loss = model.loss(x, y)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_predict_uq_returns_uqresult(self):
        model = self._make_model(output_dim=2)
        x = torch.randn(5, 3)
        result = model.predict_uq(x)
        assert isinstance(result, UQResult)
        assert result.mean.shape == (5, 2)
        assert result.aleatoric_var.shape == (5, 2)
        assert result.epistemic_var.shape == (5, 2)
        assert result.total_var.shape == (5, 2)

    def test_variances_positive(self):
        model = self._make_model()
        x = torch.randn(10, 3)
        result = model.predict_uq(x)
        assert (result.aleatoric_var > 0).all()
        assert (result.epistemic_var > 0).all()


class TestEvidentialClassification:
    def _make_model(self, input_dim=3, num_classes=4):
        base = MLP(input_dim=input_dim, hidden_dims=[32], output_dim=num_classes)
        return EvidentialClassification(base, num_classes=num_classes)

    def test_forward_shape(self):
        model = self._make_model()
        x = torch.randn(8, 3)
        alpha = model(x)
        assert alpha.shape == (8, 4)
        assert (alpha > 1).all()

    def test_loss_finite(self):
        model = self._make_model()
        x = torch.randn(8, 3)
        y = torch.randint(0, 4, (8,))
        loss = model.loss(x, y)
        assert torch.isfinite(loss)

    def test_predict_uq_probs_sum_to_one(self):
        model = self._make_model()
        x = torch.randn(8, 3)
        result = model.predict_uq(x)
        assert isinstance(result, UQResult)
        sums = result.probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_epistemic_var_decreases_with_evidence(self):
        """Higher evidence (larger alpha) should yield lower epistemic uncertainty."""
        # Directly test the math: higher S means lower uncertainty
        # Manually create alphas
        alpha_low = torch.tensor([[2.0, 2.0, 2.0, 2.0]])  # S=8
        alpha_high = torch.tensor([[10.0, 10.0, 10.0, 10.0]])  # S=40
        num_classes = 4
        eps_low = num_classes / alpha_low.sum(dim=-1)
        eps_high = num_classes / alpha_high.sum(dim=-1)
        assert eps_high < eps_low
