"""Tests for Weighted and Adaptive Conformal Prediction."""

import torch
import torch.nn as nn

from deepuq.methods.conformal import (
    AdaptiveConformalPredictor,
    WeightedConformalPredictor,
)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        return self.linear(x)


def test_weighted_predict_set_shape():
    model = LinearModel()
    wcp = WeightedConformalPredictor(model)
    cal_X = torch.randn(50, 5)
    cal_y = model(cal_X).detach().squeeze() + torch.randn(50) * 0.1
    weights = torch.ones(50)
    wcp.calibrate(cal_X, cal_y, weights)

    x_test = torch.randn(10, 5)
    lower, upper = wcp.predict_set(x_test, alpha=0.1)
    assert lower.shape == (10,)
    assert upper.shape == (10,)
    assert (upper >= lower).all()


def test_uniform_weights_matches_standard():
    """Uniform weights should behave like standard conformal."""
    model = LinearModel()
    cal_X = torch.randn(100, 5)
    cal_y = model(cal_X).detach().squeeze() + torch.randn(100) * 0.1

    # Uniform weights
    wcp = WeightedConformalPredictor(model)
    wcp.calibrate(cal_X, cal_y, torch.ones(100))
    x_test = torch.randn(5, 5)
    lower1, upper1 = wcp.predict_set(x_test, alpha=0.1)

    # Different uniform scale (should give same result since weights are normalized)
    wcp2 = WeightedConformalPredictor(model)
    wcp2.calibrate(cal_X, cal_y, torch.ones(100) * 5.0)
    lower2, upper2 = wcp2.predict_set(x_test, alpha=0.1)

    assert torch.allclose(lower1, lower2)
    assert torch.allclose(upper1, upper2)


def test_adaptive_update_changes_threshold():
    model = LinearModel()
    acp = AdaptiveConformalPredictor(model, target_coverage=0.9, gamma=0.05)
    cal_X = torch.randn(50, 5)
    cal_y = model(cal_X).detach().squeeze() + torch.randn(50) * 0.1
    acp.calibrate(cal_X, cal_y)

    initial_threshold = acp.threshold

    # Feed points that are far from predictions (misses) to increase threshold
    x_new = torch.randn(1, 5)
    y_far = model(x_new).detach().squeeze() + 100.0  # way outside
    acp.update(x_new, y_far)

    assert acp.threshold != initial_threshold


def test_adaptive_coverage():
    """AdaptiveConformalPredictor should achieve approximately target coverage."""
    torch.manual_seed(42)
    model = LinearModel()

    # Generate calibration data
    cal_X = torch.randn(200, 5)
    cal_y = model(cal_X).detach().squeeze() + torch.randn(200) * 0.5

    acp = AdaptiveConformalPredictor(model, target_coverage=0.9, gamma=0.01)
    acp.calibrate(cal_X, cal_y)

    # Test coverage on new data
    test_X = torch.randn(500, 5)
    test_y = model(test_X).detach().squeeze() + torch.randn(500) * 0.5

    lower, upper = acp.predict_set(test_X)
    covered = ((test_y >= lower) & (test_y <= upper)).float().mean().item()

    # Coverage should be approximately 0.9 (within reasonable tolerance)
    assert covered >= 0.75, f"Coverage too low: {covered}"
