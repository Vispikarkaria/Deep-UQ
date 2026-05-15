"""Calibration metrics for classification and regression UQ."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def expected_calibration_error(
    probs: "np.ndarray | torch.Tensor",
    labels: "np.ndarray | torch.Tensor",
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for classification.

    Args:
        probs: Predicted confidence (max probability) per sample, shape (N,).
        labels: Binary correctness labels (1 if prediction correct), shape (N,).
        n_bins: Number of equal-width bins.

    Returns:
        ECE as a float.
    """
    probs_np = _to_numpy(probs).ravel()
    labels_np = _to_numpy(labels).ravel()
    n = len(probs_np)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_np > bin_edges[i]) & (probs_np <= bin_edges[i + 1])
        # Include left edge for first bin
        if i == 0:
            mask = (probs_np >= bin_edges[i]) & (probs_np <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = probs_np[mask].mean()
        avg_accuracy = labels_np[mask].mean()
        ece += (count / n) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def maximum_calibration_error(
    probs: "np.ndarray | torch.Tensor",
    labels: "np.ndarray | torch.Tensor",
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    Args:
        probs: Predicted confidence per sample, shape (N,).
        labels: Binary correctness labels, shape (N,).
        n_bins: Number of equal-width bins.

    Returns:
        MCE as a float.
    """
    probs_np = _to_numpy(probs).ravel()
    labels_np = _to_numpy(labels).ravel()
    if len(probs_np) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (probs_np > bin_edges[i]) & (probs_np <= bin_edges[i + 1])
        if i == 0:
            mask = (probs_np >= bin_edges[i]) & (probs_np <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = probs_np[mask].mean()
        avg_accuracy = labels_np[mask].mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))
    return float(mce)


def prediction_interval_coverage(
    lower: "np.ndarray | torch.Tensor",
    upper: "np.ndarray | torch.Tensor",
    y_true: "np.ndarray | torch.Tensor",
) -> float:
    """Compute Prediction Interval Coverage Probability (PICP).

    Args:
        lower: Lower bounds of prediction intervals, shape (N,).
        upper: Upper bounds of prediction intervals, shape (N,).
        y_true: True target values, shape (N,).

    Returns:
        Fraction of y_true within [lower, upper].
    """
    lower_np = _to_numpy(lower).ravel()
    upper_np = _to_numpy(upper).ravel()
    y_np = _to_numpy(y_true).ravel()
    if len(y_np) == 0:
        return 0.0
    covered = ((y_np >= lower_np) & (y_np <= upper_np)).mean()
    return float(covered)


def calibration_curve_regression(
    predicted_std: "np.ndarray | torch.Tensor",
    residuals: "np.ndarray | torch.Tensor",
    quantiles: "np.ndarray | torch.Tensor | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve for regression (expected vs observed coverage).

    For each quantile level q, we check what fraction of |residuals| fall within
    predicted_std * Phi^{-1}((1+q)/2).

    Args:
        predicted_std: Predicted standard deviations, shape (N,).
        residuals: (y_true - y_pred), shape (N,).
        quantiles: Quantile levels to evaluate (default: linspace 0.05..0.95).

    Returns:
        (expected_coverage, observed_coverage) arrays.
    """
    from scipy.stats import norm

    std_np = _to_numpy(predicted_std).ravel()
    res_np = _to_numpy(residuals).ravel()

    if quantiles is None:
        expected = np.linspace(0.05, 0.95, 19)
    else:
        expected = _to_numpy(quantiles).ravel()

    observed = np.zeros_like(expected)
    for i, q in enumerate(expected):
        z = norm.ppf((1.0 + q) / 2.0)
        threshold = std_np * z
        observed[i] = (np.abs(res_np) <= threshold).mean()

    return expected, observed
