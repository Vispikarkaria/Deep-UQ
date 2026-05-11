"""Conformal Classification (APS and RAPS)."""

from __future__ import annotations

from typing import Any

import torch

from deepuq.types import UQResult

from ._base import BaseConformalPredictor, _extract_data, _model_forward
from ._utils import conformal_quantile


class ConformalClassifier(BaseConformalPredictor):
    """Prediction sets with marginal coverage guarantee.

    Supports Adaptive Prediction Sets (APS) and Regularized APS (RAPS).
    """

    def __init__(
        self,
        model: Any,
        alpha: float = 0.1,
        method: str = "aps",
        k_reg: int = 1,
        lambda_reg: float = 0.01,
    ):
        super().__init__(model, alpha)
        self.method = method
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self._q_hat: float | None = None

    def _compute_aps_scores(
        self, probs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute APS/RAPS nonconformity scores."""
        sorted_probs, sorted_idx = probs.sort(dim=1, descending=True)
        # Find rank of true label
        n = probs.shape[0]
        # Cumulative sum of sorted probabilities
        cumsum = sorted_probs.cumsum(dim=1)

        # For each sample, find where the true label sits in the sorted order
        # and return the cumulative probability up to (and including) that point
        ranks = (sorted_idx == labels.unsqueeze(1)).nonzero(as_tuple=True)[1]
        scores = cumsum[torch.arange(n), ranks]

        if self.method == "raps":
            # Add regularization penalty for large sets
            penalty = self.lambda_reg * torch.clamp(
                ranks.float() - self.k_reg + 1, min=0
            )
            scores = scores + penalty

        return scores

    def calibrate(self, cal_data) -> ConformalClassifier:
        x_cal, y_cal = _extract_data(cal_data)
        logits = _model_forward(self.model, x_cal)
        probs = torch.softmax(logits, dim=1)
        y_cal = y_cal.long().squeeze()

        scores = self._compute_aps_scores(probs, y_cal)
        self._scores = scores
        self._q_hat = conformal_quantile(scores, self.alpha)
        self._is_calibrated = True
        return self

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_uq().")

        logits = _model_forward(self.model, x)
        probs = torch.softmax(logits, dim=1)

        # Build prediction sets
        sorted_probs, sorted_idx = probs.sort(dim=1, descending=True)
        cumsum = sorted_probs.cumsum(dim=1)

        if self.method == "raps":
            ranks = torch.arange(probs.shape[1], device=probs.device).unsqueeze(0)
            penalty = self.lambda_reg * torch.clamp(
                ranks.float() - self.k_reg + 1, min=0
            )
            cumsum = cumsum + penalty

        # Include classes until cumsum exceeds q_hat
        include_sorted = cumsum <= self._q_hat
        # Always include at least the top class
        include_sorted[:, 0] = True

        # Map back to original class indices
        prediction_sets = torch.zeros_like(probs, dtype=torch.bool)
        prediction_sets.scatter_(1, sorted_idx, include_sorted)

        set_sizes = prediction_sets.sum(dim=1)

        return UQResult(
            mean=probs,
            probs=probs,
            metadata={
                "prediction_sets": prediction_sets,
                "set_sizes": set_sizes,
                "coverage_alpha": self.alpha,
                "q_hat": self._q_hat,
            },
        )
