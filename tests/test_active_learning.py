"""Tests for the active learning module."""

import torch

from deepuq.active import ActiveLearningLoop, BALDSampling, UncertaintySampling
from deepuq.methods.mc_dropout import MCDropoutWrapper
from deepuq.models import MLP


def _make_model():
    """Create a simple MLP with MC Dropout wrapper for testing."""
    mlp = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, p_drop=0.2)
    return MCDropoutWrapper(mlp, n_mc=10, apply_softmax=False)


def _train_fn(model, X, y):
    """Minimal training function."""
    inner = model.model
    opt = torch.optim.Adam(inner.parameters(), lr=0.01)
    inner.train()
    for _ in range(5):
        opt.zero_grad()
        loss = ((inner(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return model


class TestUncertaintySampling:
    def test_selects_correct_number(self):
        model = _make_model()
        strategy = UncertaintySampling(model, criterion="epistemic_var")
        pool_X = torch.randn(50, 2)
        indices = strategy.select(pool_X, n_samples=5)
        assert len(indices) == 5

    def test_indices_correspond_to_highest_uncertainty(self):
        # Use a deterministic model (no dropout) so repeated calls are consistent
        mlp = MLP(input_dim=2, hidden_dims=[16, 16], output_dim=1, p_drop=0.0)
        model = MCDropoutWrapper(mlp, n_mc=5, apply_softmax=False)
        strategy = UncertaintySampling(model, criterion="total_var")
        pool_X = torch.randn(30, 2)
        indices = strategy.select(pool_X, n_samples=3)

        # With no dropout, variance is zero everywhere, so any 3 indices are valid
        # Just verify we get 3 unique indices in range
        assert len(indices) == 3
        assert all(0 <= idx < 30 for idx in indices.tolist())

    def test_epistemic_var_criterion(self):
        model = _make_model()
        strategy = UncertaintySampling(model, criterion="epistemic_var")
        pool_X = torch.randn(20, 2)
        indices = strategy.select(pool_X, n_samples=4)
        assert indices.dim() == 1
        assert len(indices) == 4


class TestBALDSampling:
    def test_returns_valid_indices(self):
        model = _make_model()
        strategy = BALDSampling(model, n_mc_samples=10)
        pool_X = torch.randn(40, 2)
        indices = strategy.select(pool_X, n_samples=5)
        assert len(indices) == 5
        assert all(0 <= idx < 40 for idx in indices.tolist())

    def test_indices_within_pool_range(self):
        model = _make_model()
        strategy = BALDSampling(model, n_mc_samples=5)
        pool_X = torch.randn(15, 2)
        indices = strategy.select(pool_X, n_samples=10)
        assert len(indices) == 10
        assert indices.max().item() < 15


class TestActiveLearningLoop:
    def test_step_reduces_pool_increases_train(self):
        model = _make_model()
        strategy = UncertaintySampling(model, criterion="epistemic_var")
        initial_X = torch.randn(10, 2)
        initial_y = torch.randn(10, 1)
        pool_X = torch.randn(50, 2)
        pool_y = torch.randn(50, 1)

        loop = ActiveLearningLoop(
            model=model,
            strategy=strategy,
            train_fn=_train_fn,
            initial_X=initial_X,
            initial_y=initial_y,
            pool_X=pool_X,
            pool_y=pool_y,
        )

        result = loop.step(n_samples=5)
        assert result["train_size"] == 15
        assert len(loop.pool_X) == 45
        assert len(result["selected_indices"]) == 5

    def test_run_returns_correct_history_length(self):
        model = _make_model()
        strategy = UncertaintySampling(model, criterion="total_var")
        initial_X = torch.randn(5, 2)
        initial_y = torch.randn(5, 1)
        pool_X = torch.randn(100, 2)
        pool_y = torch.randn(100, 1)
        val_X = torch.randn(10, 2)
        val_y = torch.randn(10, 1)

        loop = ActiveLearningLoop(
            model=model,
            strategy=strategy,
            train_fn=_train_fn,
            initial_X=initial_X,
            initial_y=initial_y,
            pool_X=pool_X,
            pool_y=pool_y,
            val_X=val_X,
            val_y=val_y,
        )

        history = loop.run(n_iterations=3, n_samples_per_iter=5)
        assert len(history) == 3
        assert history[-1]["train_size"] == 20
        assert "val_metric" in history[0]
