"""Tests for Batch Ensemble and Packed Ensemble methods."""

import torch
from torch import nn

from deepuq.methods.batch_ensemble import (
    BatchEnsembleLinear,
    BatchEnsembleWrapper,
    PackedEnsembleWrapper,
)
from deepuq.types import UQResult


def _make_simple_model(in_features=10, out_features=1):
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, out_features),
    )


class TestBatchEnsembleLinear:
    def test_output_shape(self):
        ensemble_size = 4
        layer = BatchEnsembleLinear(10, 5, ensemble_size)
        x = torch.randn(8 * ensemble_size, 10)
        out = layer(x)
        assert out.shape == (8 * ensemble_size, 5)

    def test_different_outputs_per_member(self):
        ensemble_size = 4
        layer = BatchEnsembleLinear(10, 5, ensemble_size)
        batch_size = 8
        # Same input replicated
        x_single = torch.randn(1, 10).expand(batch_size, -1)
        x_rep = x_single.unsqueeze(0).expand(ensemble_size, -1, -1)
        x_rep = x_rep.reshape(ensemble_size * batch_size, 10)

        out = layer(x_rep)
        out = out.view(ensemble_size, batch_size, 5)

        # Different members should produce different outputs
        # (since r and s differ across members)
        assert not torch.allclose(out[0], out[1], atol=1e-6)


class TestBatchEnsembleWrapper:
    def test_predict_uq_returns_uqresult(self):
        model = _make_simple_model()
        wrapper = BatchEnsembleWrapper(model, ensemble_size=4)
        x = torch.randn(16, 10)
        result = wrapper.predict_uq(x)
        assert isinstance(result, UQResult)
        assert result.mean.shape == (16, 1)
        assert result.epistemic_var.shape == (16, 1)
        assert result.metadata["method"] == "batch_ensemble"

    def test_memory_efficiency(self):
        """BatchEnsemble should use much less memory than full ensemble."""
        model = _make_simple_model(in_features=64, out_features=8)
        ensemble_size = 4

        # Full ensemble parameter count
        full_params = sum(p.numel() for p in model.parameters()) * ensemble_size

        # Batch ensemble parameter count
        wrapper = BatchEnsembleWrapper(model, ensemble_size=ensemble_size)
        be_params = sum(p.numel() for p in wrapper.parameters())

        # Batch ensemble should be significantly smaller
        assert be_params < full_params

    def test_forward_shape(self):
        model = _make_simple_model()
        wrapper = BatchEnsembleWrapper(model, ensemble_size=4)
        x = torch.randn(8, 10)
        out = wrapper(x)
        assert out.shape == (8 * 4, 1)


class TestPackedEnsembleWrapper:
    def test_predict_uq_returns_uqresult(self):
        model = _make_simple_model()
        wrapper = PackedEnsembleWrapper(model, num_packs=4, alpha=2)
        x = torch.randn(16, 10)
        result = wrapper.predict_uq(x)
        assert isinstance(result, UQResult)
        assert result.mean.shape[0] == 16
        assert result.epistemic_var is not None
        assert result.metadata["method"] == "packed_ensemble"

    def test_output_has_variance(self):
        model = _make_simple_model()
        wrapper = PackedEnsembleWrapper(model, num_packs=4, alpha=2)
        x = torch.randn(8, 10)
        result = wrapper.predict_uq(x)
        # Variance should be non-negative
        assert (result.total_var >= 0).all()
