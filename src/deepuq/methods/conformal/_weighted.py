"""Weighted and Adaptive Conformal Prediction methods."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _weighted_quantile(
    scores: torch.Tensor, weights: torch.Tensor, quantile: float
) -> torch.Tensor:
    """Compute weighted quantile of scores."""
    sorted_indices = torch.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    cumulative_weights = cumulative_weights / cumulative_weights[-1]
    idx = torch.searchsorted(cumulative_weights, quantile)
    idx = idx.clamp(max=len(sorted_scores) - 1)
    return sorted_scores[idx]


class WeightedConformalPredictor:
    """Conformal predictor with importance-weighted calibration.

    Parameters
    ----------
    model:
        Trained model (callable on input tensors).
    score_fn:
        Callable ``(model, x, y) -> scores``. Defaults to absolute residual.
    """

    def __init__(
        self,
        model: nn.Module,
        score_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.score_fn = score_fn or self._default_score_fn
        self.threshold: torch.Tensor | None = None

    @staticmethod
    def _default_score_fn(
        model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            preds = model(x)
        return torch.abs(preds.squeeze() - y.squeeze())

    def calibrate(
        self, cal_X: torch.Tensor, cal_y: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Calibrate using weighted nonconformity scores.

        Parameters
        ----------
        cal_X: Calibration inputs.
        cal_y: Calibration targets.
        weights: Importance weights per calibration sample.
        """
        self.model.eval()
        scores = self.score_fn(self.model, cal_X, cal_y)
        # Store for potential reuse
        self._cal_scores = scores
        self._cal_weights = weights
        # Default alpha=0.1, store quantile at 1-alpha=0.9
        # We'll recompute at predict time with the actual alpha
        self._scores = scores
        self._weights = weights

    def predict_set(
        self, x: torch.Tensor, alpha: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prediction intervals.

        Parameters
        ----------
        x: Input tensor.
        alpha: Miscoverage level (default 0.1 for 90% coverage).

        Returns
        -------
        Tuple of (lower, upper) bounds.
        """
        assert self._scores is not None, "Must call calibrate() first."
        quantile_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(self._scores)))
        threshold = _weighted_quantile(self._scores, self._weights, quantile_level)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).squeeze()

        lower = preds - threshold
        upper = preds + threshold
        return lower, upper


class AdaptiveConformalPredictor:
    """Online adaptive conformal predictor with coverage guarantees.

    Parameters
    ----------
    model:
        Trained model.
    target_coverage:
        Target coverage level (default 0.9).
    gamma:
        Step size for online threshold updates.
    """

    def __init__(
        self,
        model: nn.Module,
        target_coverage: float = 0.9,
        gamma: float = 0.01,
    ) -> None:
        self.model = model
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.gamma = gamma
        self.threshold: float = 0.0

    def calibrate(self, cal_X: torch.Tensor, cal_y: torch.Tensor) -> None:
        """Initial calibration on held-out data."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(cal_X).squeeze()
        scores = torch.abs(preds - cal_y.squeeze())
        n = len(scores)
        quantile_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        sorted_scores, _ = torch.sort(scores)
        idx = int(quantile_level * n)
        idx = min(idx, n - 1)
        self.threshold = sorted_scores[idx].item()

    def predict_set(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict intervals using current threshold."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).squeeze()
        lower = preds - self.threshold
        upper = preds + self.threshold
        return lower, upper

    def update(self, x_new: torch.Tensor, y_new: torch.Tensor) -> None:
        """Online update of threshold based on observed coverage.

        Parameters
        ----------
        x_new: New observation input.
        y_new: New observation target.
        """
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_new).squeeze()
        residual = torch.abs(pred - y_new.squeeze())
        # miss_indicator = 1 if y was NOT covered
        miss = (residual > self.threshold).float().mean().item()
        # threshold += gamma * (alpha - miss_indicator)
        # If miss > alpha, threshold increases; if miss < alpha, decreases
        self.threshold += self.gamma * (miss - self.alpha)
