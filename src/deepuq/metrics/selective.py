"""Selective prediction (risk-coverage) metrics."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def risk_coverage_curve(
    uncertainties: "np.ndarray | torch.Tensor",
    errors: "np.ndarray | torch.Tensor",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve.

    Samples are sorted by uncertainty (ascending). At each coverage level,
    the risk is the mean error of the selected (least uncertain) samples.

    Args:
        uncertainties: Uncertainty scores per sample, shape (N,).
        errors: Error values per sample (e.g., squared error), shape (N,).

    Returns:
        (coverages, risks) — arrays of shape (N,).
    """
    unc = _to_numpy(uncertainties).ravel()
    err = _to_numpy(errors).ravel()
    n = len(unc)
    if n == 0:
        return np.array([]), np.array([])

    order = np.argsort(unc)
    sorted_errors = err[order]

    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    return coverages, risks


def aurc(
    uncertainties: "np.ndarray | torch.Tensor",
    errors: "np.ndarray | torch.Tensor",
) -> float:
    """Compute Area Under the Risk-Coverage curve (AURC).

    Args:
        uncertainties: Uncertainty scores per sample, shape (N,).
        errors: Error values per sample, shape (N,).

    Returns:
        AURC as a float.
    """
    coverages, risks = risk_coverage_curve(uncertainties, errors)
    if len(coverages) == 0:
        return 0.0
    try:
        return float(np.trapezoid(risks, coverages))
    except AttributeError:
        return float(np.trapz(risks, coverages))
