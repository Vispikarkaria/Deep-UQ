"""Proper scoring rules for probabilistic predictions."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


_EPS = 1e-8


def negative_log_likelihood(
    mean: "np.ndarray | torch.Tensor",
    var: "np.ndarray | torch.Tensor",
    y_true: "np.ndarray | torch.Tensor",
) -> float:
    """Compute mean Gaussian negative log-likelihood.

    Args:
        mean: Predicted means, shape (N,).
        var: Predicted variances, shape (N,). Clipped to eps for stability.
        y_true: True targets, shape (N,).

    Returns:
        Mean NLL as a float.
    """
    mu = _to_numpy(mean).ravel()
    v = np.maximum(_to_numpy(var).ravel(), _EPS)
    y = _to_numpy(y_true).ravel()

    nll = 0.5 * np.log(2 * np.pi * v) + 0.5 * ((y - mu) ** 2) / v
    return float(nll.mean())


def continuous_ranked_probability_score(
    mean: "np.ndarray | torch.Tensor",
    std: "np.ndarray | torch.Tensor",
    y_true: "np.ndarray | torch.Tensor",
) -> float:
    """Compute mean CRPS for Gaussian predictive distributions.

    Formula: sigma * [z*Phi(z) + phi(z) - 1/sqrt(pi)] where z = (y - mu) / sigma.

    Args:
        mean: Predicted means, shape (N,).
        std: Predicted standard deviations, shape (N,). Clipped to eps.
        y_true: True targets, shape (N,).

    Returns:
        Mean CRPS as a float (non-negative).
    """
    from scipy.stats import norm

    mu = _to_numpy(mean).ravel()
    sigma = np.maximum(_to_numpy(std).ravel(), _EPS)
    y = _to_numpy(y_true).ravel()

    z = (y - mu) / sigma
    crps = sigma * (z * norm.cdf(z) + norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(crps.mean())


def brier_score(
    probs: "np.ndarray | torch.Tensor",
    labels: "np.ndarray | torch.Tensor",
) -> float:
    """Compute mean Brier score for binary classification.

    Args:
        probs: Predicted probabilities for positive class, shape (N,).
        labels: True binary labels (0 or 1), shape (N,).

    Returns:
        Mean Brier score as a float.
    """
    p = _to_numpy(probs).ravel()
    y = _to_numpy(labels).ravel()
    return float(((p - y) ** 2).mean())


def interval_score(
    lower: "np.ndarray | torch.Tensor",
    upper: "np.ndarray | torch.Tensor",
    y_true: "np.ndarray | torch.Tensor",
    alpha: float = 0.1,
) -> float:
    """Compute mean interval score.

    Score = (upper - lower) + (2/alpha)*(lower - y)*(y < lower)
            + (2/alpha)*(y - upper)*(y > upper)

    Args:
        lower: Lower prediction bounds, shape (N,).
        upper: Upper prediction bounds, shape (N,).
        y_true: True targets, shape (N,).
        alpha: Significance level (e.g., 0.1 for 90% interval).

    Returns:
        Mean interval score as a float.
    """
    lo = _to_numpy(lower).ravel()
    up = _to_numpy(upper).ravel()
    y = _to_numpy(y_true).ravel()

    width = up - lo
    penalty_lo = (2.0 / alpha) * (lo - y) * (y < lo)
    penalty_hi = (2.0 / alpha) * (y - up) * (y > up)
    score = width + penalty_lo + penalty_hi
    return float(score.mean())
