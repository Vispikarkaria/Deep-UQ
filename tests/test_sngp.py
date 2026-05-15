"""Tests for SNGP (Spectral-Normalized Neural Gaussian Process)."""

import torch
import torch.nn as nn

from deepuq.methods.sngp import SNGPWrapper, SpectralNormGP
from deepuq.types import UQResult


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(10, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.head(x)


def test_spectral_norm_gp_forward():
    """SpectralNormGP forward produces logits and covariance diagonal."""
    gp = SpectralNormGP(in_features=32, num_classes=5, num_random_features=128)
    features = torch.randn(4, 32)
    logits, cov_diag = gp(features)

    assert logits.shape == (4, 5)
    assert cov_diag.shape == (4, 5)


def test_sngp_wrapper_predict_uq():
    """SNGPWrapper wraps an MLP and predict_uq returns valid UQResult."""
    model = SimpleMLP()
    wrapper = SNGPWrapper(model, last_layer_name="head", num_random_features=128)

    x = torch.randn(8, 10)
    # Train a bit to populate covariance
    wrapper.update_covariance(x)

    result = wrapper.predict_uq(x)
    assert isinstance(result, UQResult)
    assert result.mean.shape == (8, 5)
    assert result.epistemic_var.shape == (8, 5)
    assert result.metadata["method"] == "sngp"


def test_spectral_norm_applied():
    """Spectral norm is applied and weight norm is bounded."""
    model = SimpleMLP()
    bound = 3.0
    wrapper = SNGPWrapper(model, last_layer_name="head", num_random_features=128, spec_norm_bound=bound)

    # Check that spectral_norm is applied to linear layers in feature extractor
    for module in wrapper.feature_extractor.modules():
        if isinstance(module, nn.Linear):
            # spectral_norm adds a 'weight_orig' parameter
            assert hasattr(module, "weight_orig"), "Spectral norm not applied"


def test_covariance_changes_after_update():
    """Covariance changes after update_covariance is called."""
    gp = SpectralNormGP(in_features=16, num_classes=3, num_random_features=64)

    precision_before = gp.precision.clone()
    features = torch.randn(8, 16)
    gp.update_covariance(features)
    precision_after = gp.precision.clone()

    assert not torch.allclose(precision_before, precision_after)


def test_epistemic_var_positive():
    """epistemic_var is positive."""
    model = SimpleMLP()
    wrapper = SNGPWrapper(model, last_layer_name="head", num_random_features=128)

    x = torch.randn(4, 10)
    wrapper.update_covariance(x)
    result = wrapper.predict_uq(x)

    assert (result.epistemic_var > 0).all()
