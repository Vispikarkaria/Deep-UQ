"""Sampling-based uncertainty propagation."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepuq.types import UQResult


class SamplingPropagator:
    """Propagate uncertainty by sampling from the input distribution.

    Parameters
    ----------
    model:
        Model with a ``predict_uq(x) -> UQResult`` method.
    n_samples:
        Number of samples drawn from N(mean, diag(var)).
    """

    def __init__(self, model: nn.Module, n_samples: int = 50):
        self.model = model
        self.n_samples = n_samples

    def step(
        self, mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate one step: (mean, var) -> (new_mean, new_var).

        Draws samples from N(mean, diag(var)), passes them through the model,
        and computes output statistics.
        """
        squeeze = mean.dim() == 1
        if squeeze:
            mean = mean.unsqueeze(0)
            var = var.unsqueeze(0)

        # Draw samples: (n_samples, D)
        std = torch.sqrt(var + 1e-12)  # avoid sqrt(0)
        eps = torch.randn(
            self.n_samples, mean.shape[-1], device=mean.device, dtype=mean.dtype
        )
        samples = mean + eps * std  # (n_samples, D)

        # Batch forward pass - get point predictions
        with torch.no_grad():
            outputs = (
                self.model.model(samples)
                if hasattr(self.model, "model")
                else self.model(samples)
            )

        # Compute statistics of outputs
        new_mean = outputs.mean(dim=0, keepdim=True)  # (1, D)
        sample_var = outputs.var(dim=0, unbiased=True, keepdim=True)  # (1, D)

        # Add mean epistemic variance from model
        uq: UQResult = self.model.predict_uq(mean)
        model_var = uq.epistemic_var  # (1, D)
        new_var = sample_var + model_var

        if squeeze:
            new_mean = new_mean.squeeze(0)
            new_var = new_var.squeeze(0)

        return new_mean, new_var
