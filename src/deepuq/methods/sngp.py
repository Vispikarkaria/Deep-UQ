"""Spectral-Normalized Neural Gaussian Process (SNGP).

This module implements the SNGP approach which combines spectral normalization
of hidden layers with a Gaussian Process output layer based on random Fourier
features, yielding distance-aware uncertainty estimates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepuq.types import UQResult


class SpectralNormGP(nn.Module):
    """Random-feature GP last layer.

    Replaces a standard classification head with a Gaussian Process
    approximated via Random Fourier Features (RFF). Maintains a running
    precision matrix that can be inverted at inference time to produce
    predictive covariance.

    Parameters
    ----------
    in_features:
        Dimensionality of the input feature vector.
    num_classes:
        Number of output classes.
    num_random_features:
        Number of random Fourier features (D).
    momentum:
        Momentum for the running precision matrix update.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_random_features: int = 1024,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_random_features = num_random_features
        self.momentum = momentum

        # Random Fourier Feature parameters (frozen)
        self.register_buffer(
            "rff_weights", torch.randn(in_features, num_random_features)
        )
        self.register_buffer("rff_bias", torch.rand(num_random_features) * 2 * math.pi)

        # Output linear layer on top of random features
        self.beta = nn.Linear(num_random_features, num_classes)

        # Precision matrix (running average)
        self.register_buffer("precision", torch.eye(num_random_features))
        self.register_buffer("is_fitted", torch.tensor(False))

    def _compute_random_features(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Φ(x) = sqrt(2/D) * cos(Wx + b)."""
        projected = features @ self.rff_weights + self.rff_bias
        scale = math.sqrt(2.0 / self.num_random_features)
        return scale * torch.cos(projected)

    def reset_covariance(self) -> None:
        """Reset precision matrix accumulator. Call at the start of each epoch."""
        self.precision.copy_(
            torch.eye(self.num_random_features, device=self.precision.device)
        )
        self.is_fitted.fill_(False)

    def update_covariance(self, features: torch.Tensor) -> None:
        """Accumulate precision: Σ^-1 += Φ(x)^T Φ(x) with momentum.

        Parameters
        ----------
        features:
            Raw features from the backbone, shape (B, in_features).
        """
        phi = self._compute_random_features(features)  # (B, D)
        batch_precision = phi.t() @ phi  # (D, D)

        if self.is_fitted:
            self.precision.copy_(
                self.momentum * self.precision + (1 - self.momentum) * batch_precision
            )
        else:
            self.precision.copy_(batch_precision)
            self.is_fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and covariance diagonal.

        Parameters
        ----------
        features:
            Input features of shape (B, in_features).

        Returns
        -------
        (logits, covariance_diagonal):
            logits has shape (B, num_classes), covariance_diagonal has shape (B, num_classes).
        """
        phi = self._compute_random_features(features)  # (B, D)
        logits = self.beta(phi)  # (B, C)

        # Compute per-sample per-class variance: var_{b,c} = w_c^T Σ_b w_c
        # where Σ_b approximated as feature_var_b * Σ_global
        # Proper computation: project phi through precision inverse then through beta
        # cov_features = Σ^{-1} solved: Σ phi^T => phi Σ
        phi_cov = torch.linalg.solve(self.precision, phi.t()).t()  # (B, D)
        # Project through beta weights: (B, D) @ (D, C) -> (B, C)
        # var per sample per class = sum over d of (phi_cov * beta_weight) squared
        # Actually: var_{b,c} = phi_b^T Σ phi_b projected through w_c
        # = (phi_cov_b @ w_c)^2 ... no, correct is w_c^T Σ w_c (sample independent)
        # But we want sample-dependent variance: phi_b^T (precision^-1) phi_b gives scalar
        # Then scale by w_c^T w_c? No.
        # Correct: the predictive variance for logit c at input x is:
        # var_c(x) = phi(x)^T Σ phi(x) * ||w_c||^2  (approximation)
        # Better: use phi_cov directly with beta
        # logit_c = w_c^T phi, so var(logit_c) = w_c^T Cov(phi) w_c
        # But Cov(phi) here is the posterior covariance = precision^{-1}
        # So var_c = w_c^T precision^{-1} w_c (independent of input x for GP)
        # For input-dependent variance, use: var_c(x) = phi(x)^T precision^{-1} phi(x)
        # which is same for all classes. Scale by 1 for each class.
        # Actually the standard SNGP uses: var(x) = phi(x)^T precision^{-1} phi(x)
        feature_var = (phi * phi_cov).sum(dim=-1, keepdim=True)  # (B, 1)
        covariance_diagonal = feature_var.expand(-1, self.num_classes)  # (B, C)

        return logits, covariance_diagonal


class SNGPWrapper(nn.Module):
    """Wrap an existing model with SNGP: spectral norm + GP last layer.

    Parameters
    ----------
    base_model:
        The neural network to wrap.
    last_layer_name:
        Attribute name of the final classification layer to replace.
    num_random_features:
        Number of random Fourier features for the GP layer.
    spec_norm_bound:
        Upper bound on the spectral norm of weight matrices.
    """

    def __init__(
        self,
        base_model: nn.Module,
        last_layer_name: str,
        num_random_features: int = 1024,
        spec_norm_bound: float = 6.0,
    ):
        super().__init__()
        self.spec_norm_bound = spec_norm_bound
        self.last_layer_name = last_layer_name

        # Extract the last layer to determine dimensions
        last_layer = getattr(base_model, last_layer_name)
        in_features = last_layer.in_features
        num_classes = last_layer.out_features

        # Remove last layer and store feature extractor
        setattr(base_model, last_layer_name, nn.Identity())
        self.feature_extractor = base_model

        # Apply spectral norm to all Linear/Conv2d layers in the feature extractor
        self.apply_spectral_norm()

        # Create GP last layer
        self.gp_layer = SpectralNormGP(
            in_features=in_features,
            num_classes=num_classes,
            num_random_features=num_random_features,
        )

    def apply_spectral_norm(self) -> None:
        """Apply spectral normalization to all Linear and Conv2d layers."""
        for module in self.feature_extractor.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.utils.spectral_norm(module)

    def reset_covariance(self) -> None:
        """Reset the GP layer's precision matrix."""
        self.gp_layer.reset_covariance()

    def update_covariance(self, x: torch.Tensor) -> None:
        """Forward x through the feature extractor and update GP covariance."""
        with torch.no_grad():
            features = self.feature_extractor(x)
        self.gp_layer.update_covariance(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning logits (for training with CE loss)."""
        features = self.feature_extractor(x)
        logits, _ = self.gp_layer(features)
        return logits

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return predictive uncertainty as a :class:`deepuq.types.UQResult`.

        The GP covariance diagonal provides the epistemic uncertainty estimate.
        """
        self.eval()
        features = self.feature_extractor(x)
        logits, covariance_diagonal = self.gp_layer(features)

        probs = torch.softmax(logits, dim=-1)
        epistemic_var = covariance_diagonal

        return UQResult(
            mean=probs,
            epistemic_var=epistemic_var,
            aleatoric_var=None,
            total_var=epistemic_var,
            probs=probs,
            probs_var=epistemic_var,
            metadata={
                "method": "sngp",
                "num_random_features": self.gp_layer.num_random_features,
                "spec_norm_bound": self.spec_norm_bound,
            },
        )
