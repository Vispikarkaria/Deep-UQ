"""Tests for spatiotemporal uncertainty propagation."""

import torch
import torch.nn as nn

from deepuq.methods.mc_dropout import MCDropoutWrapper
from deepuq.propagation import MomentMatchingPropagator, SamplingPropagator, UncertaintyRollout


class SimpleMLP(nn.Module):
    """Simple autoregressive MLP: D -> D with expansive dynamics."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 16),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(16, dim),
        )
        # Initialize with moderately large weights to ensure expansive dynamics
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_model(dim: int = 4):
    mlp = SimpleMLP(dim)
    return MCDropoutWrapper(mlp, n_mc=10, apply_softmax=False)


class TestSamplingPropagator:
    def test_step_shape(self):
        model = _make_model(4)
        prop = SamplingPropagator(model, n_samples=50)
        mean = torch.randn(1, 4)
        var = torch.ones(1, 4) * 0.01
        new_mean, new_var = prop.step(mean, var)
        assert new_mean.shape == (1, 4)
        assert new_var.shape == (1, 4)

    def test_step_1d_input(self):
        model = _make_model(4)
        prop = SamplingPropagator(model, n_samples=50)
        mean = torch.randn(4)
        var = torch.ones(4) * 0.01
        new_mean, new_var = prop.step(mean, var)
        assert new_mean.shape == (4,)
        assert new_var.shape == (4,)


class TestMomentMatchingPropagator:
    def test_step_shape(self):
        model = _make_model(4)
        prop = MomentMatchingPropagator(model)
        mean = torch.randn(1, 4)
        var = torch.ones(1, 4) * 0.01
        new_mean, new_var = prop.step(mean, var)
        assert new_mean.shape == (1, 4)
        assert new_var.shape == (1, 4)

    def test_step_1d_input(self):
        model = _make_model(4)
        prop = MomentMatchingPropagator(model)
        mean = torch.randn(4)
        var = torch.ones(4) * 0.01
        new_mean, new_var = prop.step(mean, var)
        assert new_mean.shape == (4,)
        assert new_var.shape == (4,)


class TestUncertaintyRollout:
    def test_sampling_trajectory_length(self):
        model = _make_model(4)
        rollout = UncertaintyRollout(model, propagation="sampling", n_samples=50)
        x0 = torch.randn(4)
        traj = rollout.predict_trajectory(x0, n_steps=5)
        assert len(traj) == 5

    def test_variance_grows(self):
        model = _make_model(4)
        rollout = UncertaintyRollout(model, propagation="sampling", n_samples=200)
        x0 = torch.zeros(4)
        traj = rollout.predict_trajectory(x0, n_steps=3)
        # Variance should grow: last step > first step
        # (first step has only model variance, subsequent steps accumulate)
        assert traj[-1].epistemic_var.mean().item() > traj[0].epistemic_var.mean().item()

    def test_uncertainty_growth_rate_positive(self):
        model = _make_model(4)
        rollout = UncertaintyRollout(model, propagation="sampling", n_samples=200)
        x0 = torch.zeros(4)
        traj = rollout.predict_trajectory(x0, n_steps=3)
        rate = rollout.uncertainty_growth_rate(traj)
        assert isinstance(rate, float)
        assert rate > 1.0

    def test_moment_matching_trajectory(self):
        model = _make_model(4)
        rollout = UncertaintyRollout(model, propagation="moment_matching")
        x0 = torch.randn(4)
        traj = rollout.predict_trajectory(x0, n_steps=5)
        assert len(traj) == 5
        for uq in traj:
            assert uq.mean.shape == (1, 4)
            assert uq.epistemic_var.shape == (1, 4)
