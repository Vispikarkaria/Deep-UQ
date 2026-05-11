"""Nonconformity score functions for conformal prediction."""

from __future__ import annotations

import torch


def absolute_residual_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Absolute residual: |y - y_hat|."""
    return (y_true - y_pred).abs()


def signed_residual_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Signed residual: y - y_hat."""
    return y_true - y_pred


def normalized_residual_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Normalized absolute residual: |y - y_hat| / sigma."""
    return (y_true - y_pred).abs() / (sigma + 1e-8)


def quantile_score(
    y_true: torch.Tensor, q_lo: torch.Tensor, q_hi: torch.Tensor
) -> torch.Tensor:
    """CQR score: max(q_lo - y, y - q_hi)."""
    return torch.max(q_lo - y_true, y_true - q_hi)
