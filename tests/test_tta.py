"""Tests for Test-Time Augmentation UQ wrapper."""

import torch
import torch.nn as nn

from deepuq.methods.tta import TTAWrapper
from deepuq.types import UQResult


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        return self.linear(x)


def test_predict_uq_returns_uqresult():
    model = SimpleModel()
    wrapper = TTAWrapper(model, n_augmentations=10)
    x = torch.randn(8, 5)
    result = wrapper.predict_uq(x)
    assert isinstance(result, UQResult)
    assert result.mean.shape == (8, 1)
    assert result.epistemic_var.shape == (8, 1)
    assert result.total_var.shape == (8, 1)


def test_variance_is_positive():
    model = SimpleModel()
    wrapper = TTAWrapper(model, n_augmentations=20)
    x = torch.randn(16, 5)
    result = wrapper.predict_uq(x)
    # With noise augmentations (i>0), variance should be > 0
    assert (result.epistemic_var > 0).any()


def test_works_with_plain_module():
    """TTAWrapper should work with any nn.Module without predict_uq."""
    model = nn.Sequential(nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 2))
    wrapper = TTAWrapper(model, n_augmentations=15)
    x = torch.randn(4, 3)
    result = wrapper.predict_uq(x)
    assert result.mean.shape == (4, 2)


def test_custom_augmentations():
    model = SimpleModel()
    custom_augs = [
        lambda x: x * 1.1,
        lambda x: x * 0.9,
        lambda x: x + 0.5,
    ]
    wrapper = TTAWrapper(model, augmentations=custom_augs)
    assert len(wrapper.augmentations) == 3
    x = torch.randn(4, 5)
    result = wrapper.predict_uq(x)
    assert isinstance(result, UQResult)
    assert result.metadata["n_augmentations"] == 3
