"""Tests for SVGD implementation."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.svgd import RBFKernel, SVGDOptimizer, SVGDWrapper
from deepuq.models import MLP
from deepuq.types import UQResult


def _make_mlp():
    return MLP(input_dim=2, hidden_dims=[16], output_dim=1)


class TestRBFKernel:
    def test_symmetric_positive_diagonal(self):
        kernel = RBFKernel()
        X = torch.randn(5, 10)
        K, grad_K = kernel(X)

        assert K.shape == (5, 5)
        assert grad_K.shape == (5, 10)
        # Symmetric
        assert torch.allclose(K, K.T, atol=1e-6)
        # Positive diagonal
        assert (K.diag() > 0).all()
        # Diagonal should be 1 (K(x,x) = exp(0) = 1)
        assert torch.allclose(K.diag(), torch.ones(5), atol=1e-6)


class TestSVGDOptimizer:
    def test_step_changes_parameters(self):
        particles = [_make_mlp() for _ in range(3)]
        opt = SVGDOptimizer(particles, lr=0.1)

        # Record initial params
        params_before = [
            torch.cat([p.flatten() for p in m.parameters()]).clone() for m in particles
        ]

        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        opt.step(torch.nn.functional.mse_loss, x, y)

        params_after = [
            torch.cat([p.flatten() for p in m.parameters()]) for m in particles
        ]

        # At least one particle should have changed
        changed = any(
            not torch.allclose(b, a, atol=1e-8)
            for b, a in zip(params_before, params_after)
        )
        assert changed

    def test_particles_diverge(self):
        """Particles should not collapse to same parameters."""
        torch.manual_seed(42)
        particles = [_make_mlp() for _ in range(5)]
        opt = SVGDOptimizer(particles, lr=0.01)

        x = torch.randn(16, 2)
        y = torch.randn(16, 1)

        for _ in range(10):
            opt.step(torch.nn.functional.mse_loss, x, y)

        # Check particles are not all identical
        params = [torch.cat([p.flatten() for p in m.parameters()]) for m in particles]
        diffs = []
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                diffs.append((params[i] - params[j]).norm().item())

        assert max(diffs) > 1e-6, "Particles collapsed to same point"


class TestSVGDWrapper:
    def test_predict_uq_returns_valid_result(self):
        torch.manual_seed(0)
        wrapper = SVGDWrapper(model_fn=_make_mlp, n_particles=5, lr=0.01)

        # Create simple dataset
        x = torch.randn(32, 2)
        y = torch.randn(32, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=16)

        wrapper.fit(loader, n_epochs=3)

        x_test = torch.randn(4, 2)
        result = wrapper.predict_uq(x_test)

        assert isinstance(result, UQResult)
        assert result.mean.shape == (4, 1)
        assert result.epistemic_var.shape == (4, 1)

    def test_epistemic_var_positive(self):
        torch.manual_seed(1)
        wrapper = SVGDWrapper(model_fn=_make_mlp, n_particles=10, lr=0.01)

        x = torch.randn(32, 2)
        y = torch.randn(32, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=16)

        wrapper.fit(loader, n_epochs=5)

        x_test = torch.randn(8, 2)
        result = wrapper.predict_uq(x_test)

        assert (result.epistemic_var > 0).all()
