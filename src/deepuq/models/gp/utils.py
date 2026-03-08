"""Numerical helpers for Gaussian process models."""

from __future__ import annotations

import torch


def pairwise_squared_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distances between rows of ``x1`` and ``x2``."""
    x1_sq = (x1**2).sum(dim=-1, keepdim=True)
    x2_sq = (x2**2).sum(dim=-1).unsqueeze(0)
    return (x1_sq + x2_sq - 2.0 * x1 @ x2.t()).clamp_min(0.0)


def prepare_lengthscale(lengthscale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Validate and broadcast scalar or ARD lengthscales."""
    if lengthscale.ndim == 0:
        return lengthscale
    if lengthscale.ndim == 1 and lengthscale.shape[0] == x.shape[-1]:
        return lengthscale.view(1, -1)
    raise ValueError(
        "lengthscale must be a scalar or a 1D tensor matching input dimension."
    )


def add_diagonal(matrix: torch.Tensor, value: float | torch.Tensor) -> torch.Tensor:
    """Add ``value`` to diagonal of ``matrix`` and return the result."""
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return matrix + value * eye


def stable_cholesky(
    matrix: torch.Tensor,
    jitter_base: float = 1e-6,
    jitter_max: float = 1e-2,
) -> tuple[torch.Tensor, float]:
    """Compute a numerically stable Cholesky factor with jitter escalation."""
    jitter = 0.0
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    while True:
        try:
            chol = torch.linalg.cholesky(matrix + jitter * eye)
            return chol, jitter
        except RuntimeError as exc:
            if jitter == 0.0:
                jitter = jitter_base
            else:
                jitter *= 10.0
            if jitter > jitter_max:
                raise RuntimeError(
                    f"Cholesky failed even after jitter escalation up to {jitter_max}."
                ) from exc


def solve_cholesky(chol: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve linear system ``A x = rhs`` where ``chol`` is Cholesky of ``A``."""
    return torch.cholesky_solve(rhs, chol)
