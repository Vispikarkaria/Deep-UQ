"""Autoregressive uncertainty rollout."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepuq.propagation.moment_matching import MomentMatchingPropagator
from deepuq.propagation.sampling import SamplingPropagator
from deepuq.types import UQResult


class UncertaintyRollout:
    """Autoregressive rollout with uncertainty propagation.

    Parameters
    ----------
    model:
        Model with a ``predict_uq(x) -> UQResult`` method.
    propagation:
        Propagation strategy: ``"moment_matching"`` or ``"sampling"``.
    n_samples:
        Number of samples for sampling propagation.
    """

    def __init__(
        self,
        model: nn.Module,
        propagation: str = "moment_matching",
        n_samples: int = 50,
    ):
        self.model = model
        if propagation == "moment_matching":
            self.propagator = MomentMatchingPropagator(model)
        elif propagation == "sampling":
            self.propagator = SamplingPropagator(model, n_samples=n_samples)
        else:
            raise ValueError(f"Unknown propagation method: {propagation}")

    def predict_trajectory(self, x0: torch.Tensor, n_steps: int) -> list[UQResult]:
        """Roll out model autoregressively for n_steps.

        Parameters
        ----------
        x0:
            Initial state tensor of shape ``(D,)`` or ``(1, D)``.
        n_steps:
            Number of autoregressive steps.

        Returns
        -------
        list[UQResult]:
            UQResult at each timestep (length n_steps).
        """
        squeeze = x0.dim() == 1
        mean = x0.unsqueeze(0) if squeeze else x0
        var = torch.zeros_like(mean)

        trajectory: list[UQResult] = []
        for _ in range(n_steps):
            new_mean, new_var = self.propagator.step(mean.detach(), var.detach())
            # Ensure 2D
            if new_mean.dim() == 1:
                new_mean = new_mean.unsqueeze(0)
                new_var = new_var.unsqueeze(0)
            trajectory.append(
                UQResult(
                    mean=new_mean.detach().clone(),
                    epistemic_var=new_var.detach().clone(),
                    total_var=new_var.detach().clone(),
                )
            )
            mean = new_mean
            var = new_var

        return trajectory

    @staticmethod
    def uncertainty_growth_rate(trajectory: list[UQResult]) -> float:
        """Compute average rate of epistemic variance growth per step.

        Returns the geometric-mean ratio mean(var[t+1]) / mean(var[t]).
        """
        if len(trajectory) < 2:
            return 1.0

        ratios: list[float] = []
        for t in range(len(trajectory) - 1):
            v_curr = trajectory[t].epistemic_var.mean().item()
            v_next = trajectory[t + 1].epistemic_var.mean().item()
            if v_curr > 1e-12:
                ratios.append(v_next / v_curr)

        if not ratios:
            return 1.0
        return sum(ratios) / len(ratios)
