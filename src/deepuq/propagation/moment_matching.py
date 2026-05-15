"""Moment-matching uncertainty propagation via linearization."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepuq.types import UQResult


class MomentMatchingPropagator:
    """Propagate uncertainty through a model using first-order Taylor expansion.

    Parameters
    ----------
    model:
        Model with a ``predict_uq(x) -> UQResult`` method.
    eps:
        Finite-difference step size for Jacobian approximation.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-4):
        self.model = model
        self.eps = eps

    def step(
        self, mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate one step: (mean, var) -> (new_mean, new_var).

        Uses linearization: new_var = J @ diag(var) @ J^T + model_var
        where J is approximated via finite differences.
        """
        # mean shape: (D,) or (1, D)
        squeeze = mean.dim() == 1
        if squeeze:
            mean = mean.unsqueeze(0)
            var = var.unsqueeze(0)

        # Get model prediction and epistemic variance at mean
        uq: UQResult = self.model.predict_uq(mean)
        new_mean = uq.mean  # (1, D)
        model_var = uq.epistemic_var  # (1, D)

        # Approximate Jacobian via finite differences
        D = mean.shape[-1]
        J = torch.zeros(D, D, device=mean.device, dtype=mean.dtype)
        for i in range(D):
            perturb = torch.zeros_like(mean)
            perturb[0, i] = self.eps
            uq_plus = self.model.predict_uq(mean + perturb)
            J[:, i] = (uq_plus.mean[0] - new_mean[0]) / self.eps

        # new_var = J @ diag(var) @ J^T + model_var
        # For a single sample: diag(var[0]) is (D,D)
        var_diag = torch.diag(var[0])  # (D, D)
        propagated = J @ var_diag @ J.T  # (D, D)
        new_var = torch.diag(propagated).unsqueeze(0) + model_var  # (1, D)

        if squeeze:
            new_mean = new_mean.squeeze(0)
            new_var = new_var.squeeze(0)

        return new_mean, new_var
