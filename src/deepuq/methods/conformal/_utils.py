"""Utility functions for conformal prediction."""

from __future__ import annotations

import math

import torch


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """Compute the conformal quantile of nonconformity scores.

    Returns the ceil((n+1)*(1-alpha))/n quantile of the scores.
    """
    n = scores.shape[0]
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return torch.quantile(scores.float(), level).item()


def check_coverage(
    y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> float:
    """Compute empirical coverage fraction."""
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    return covered.mean().item()
