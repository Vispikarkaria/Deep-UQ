"""Selective prediction module for Deep-UQ.

Implements uncertainty-based rejection to improve predictive quality by
abstaining on high-uncertainty inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from deepuq.types import UQResult


@dataclass
class SelectiveResult:
    """Output of selective prediction with rejection."""

    predictions: torch.Tensor
    accepted_mask: torch.Tensor
    uncertainties: torch.Tensor
    coverage: float
    selective_risk: float


@dataclass
class SelectiveMetrics:
    """Metrics for evaluating selective prediction quality."""

    coverage: float
    selective_mse: float
    rejection_rate: float
    oracle_accuracy: float
    aurc: float


class SelectivePredictor:
    """Uncertainty-based selective predictor that rejects uncertain inputs.

    Parameters
    ----------
    model:
        A model with a ``predict_uq(x)`` method returning a UQResult.
    criterion:
        One of "epistemic_var", "total_var", "entropy", or a callable
        that maps UQResult -> tensor of per-sample uncertainty scores.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: str | Callable[[UQResult], torch.Tensor] = "epistemic_var",
    ):
        self.model = model
        self.criterion = criterion

    def _get_uncertainties(self, uq_result: UQResult) -> torch.Tensor:
        """Extract per-sample uncertainty scores from a UQResult."""
        if callable(self.criterion) and not isinstance(self.criterion, str):
            return self.criterion(uq_result)

        if self.criterion == "epistemic_var":
            var = uq_result.epistemic_var
            if var is None:
                raise ValueError("UQResult has no epistemic_var")
            # Sum over output dims if multi-dimensional
            if var.dim() > 1:
                return var.sum(dim=-1)
            return var

        elif self.criterion == "total_var":
            var = uq_result.total_var
            if var is None:
                raise ValueError("UQResult has no total_var")
            if var.dim() > 1:
                return var.sum(dim=-1)
            return var

        elif self.criterion == "entropy":
            probs = uq_result.mean
            if probs is None:
                raise ValueError("UQResult has no mean for entropy")
            # Clamp for numerical stability
            p = probs.clamp(min=1e-8)
            entropy = -(p * p.log()).sum(dim=-1)
            return entropy

        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def predict_with_rejection(
        self,
        x: torch.Tensor,
        threshold: float | None = None,
        coverage: float | None = None,
        y: torch.Tensor | None = None,
    ) -> SelectiveResult:
        """Make predictions, rejecting uncertain samples.

        Exactly one of threshold or coverage must be provided.
        """
        if (threshold is None) == (coverage is None):
            raise ValueError("Exactly one of threshold or coverage must be provided")

        uq_result = self.model.predict_uq(x)
        uncertainties = self._get_uncertainties(uq_result)

        if threshold is not None:
            accepted_mask = uncertainties <= threshold
        else:
            # Accept top-coverage% most certain (lowest uncertainty)
            n = len(uncertainties)
            k = max(1, int(round(coverage * n)))  # type: ignore[operator]
            # Find the k-th smallest uncertainty as threshold
            sorted_unc, _ = uncertainties.sort()
            thresh = sorted_unc[min(k - 1, n - 1)]
            accepted_mask = uncertainties <= thresh
            # If ties cause more than k to be accepted, keep exactly k
            if accepted_mask.sum() > k:
                indices = uncertainties.argsort()
                accepted_mask = torch.zeros_like(accepted_mask, dtype=torch.bool)
                accepted_mask[indices[:k]] = True

        predictions = uq_result.mean[accepted_mask]
        actual_coverage = float(accepted_mask.sum()) / len(accepted_mask)

        # Compute selective risk if y is provided
        selective_risk = 0.0
        if y is not None and accepted_mask.sum() > 0:
            accepted_preds = uq_result.mean[accepted_mask]
            accepted_y = y[accepted_mask]
            if accepted_preds.dim() > 1:
                accepted_preds = accepted_preds.squeeze(-1)
            if accepted_y.dim() > 1:
                accepted_y = accepted_y.squeeze(-1)
            selective_risk = float(((accepted_preds - accepted_y) ** 2).mean())

        return SelectiveResult(
            predictions=predictions,
            accepted_mask=accepted_mask,
            uncertainties=uncertainties,
            coverage=actual_coverage,
            selective_risk=selective_risk,
        )

    def find_threshold(
        self,
        val_X: torch.Tensor,
        val_y: torch.Tensor,
        target_coverage: float = 0.8,
    ) -> float:
        """Find uncertainty threshold achieving target_coverage on validation data."""
        uq_result = self.model.predict_uq(val_X)
        uncertainties = self._get_uncertainties(uq_result)
        n = len(uncertainties)
        k = max(1, int(round(target_coverage * n)))
        sorted_unc, _ = uncertainties.sort()
        threshold = float(sorted_unc[min(k - 1, n - 1)])
        return threshold

    def evaluate(
        self,
        test_X: torch.Tensor,
        test_y: torch.Tensor,
    ) -> SelectiveMetrics:
        """Compute selective prediction metrics on test data."""
        uq_result = self.model.predict_uq(test_X)
        uncertainties = self._get_uncertainties(uq_result)
        predictions = uq_result.mean

        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        if test_y.dim() > 1:
            test_y = test_y.squeeze(-1)

        # Sort by uncertainty (ascending = most certain first)
        sorted_indices = uncertainties.argsort()
        sorted_errors = (predictions[sorted_indices] - test_y[sorted_indices]) ** 2

        n = len(test_y)

        # Use all samples (coverage=1.0) for basic metrics
        coverage = 1.0
        selective_mse = float(sorted_errors.mean())
        rejection_rate = 0.0

        # Oracle: best possible MSE at same coverage (reject highest-error samples)
        oracle_errors_sorted, _ = ((predictions - test_y) ** 2).sort()
        oracle_accuracy = float(oracle_errors_sorted.mean())

        # AURC: area under risk-coverage curve
        cumulative_errors = sorted_errors.cumsum(dim=0)
        coverages = torch.arange(1, n + 1, dtype=torch.float32)
        risks = cumulative_errors / coverages  # risk at each coverage level
        # Integrate using trapezoidal rule (coverage increments are uniform 1/n)
        aurc = float(risks.sum() / n)

        return SelectiveMetrics(
            coverage=coverage,
            selective_mse=selective_mse,
            rejection_rate=rejection_rate,
            oracle_accuracy=oracle_accuracy,
            aurc=aurc,
        )
