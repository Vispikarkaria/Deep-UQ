"""Out-of-distribution detection metrics."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def auroc_ood(
    in_scores: "np.ndarray | torch.Tensor",
    out_scores: "np.ndarray | torch.Tensor",
) -> float:
    """Compute AUROC for OOD detection.

    Higher scores should indicate OOD. The metric measures separability
    between in-distribution and out-of-distribution scores.

    Args:
        in_scores: Uncertainty scores for in-distribution data, shape (N,).
        out_scores: Uncertainty scores for OOD data, shape (M,).

    Returns:
        AUROC as a float in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    in_np = _to_numpy(in_scores).ravel()
    out_np = _to_numpy(out_scores).ravel()

    labels = np.concatenate([np.zeros(len(in_np)), np.ones(len(out_np))])
    scores = np.concatenate([in_np, out_np])
    return float(roc_auc_score(labels, scores))


def fpr_at_tpr(
    in_scores: "np.ndarray | torch.Tensor",
    out_scores: "np.ndarray | torch.Tensor",
    tpr: float = 0.95,
) -> float:
    """Compute FPR at a given TPR threshold for OOD detection.

    Args:
        in_scores: Uncertainty scores for in-distribution data, shape (N,).
        out_scores: Uncertainty scores for OOD data, shape (M,).
        tpr: Target true positive rate (default 0.95).

    Returns:
        False positive rate at the given TPR.
    """
    from sklearn.metrics import roc_curve

    in_np = _to_numpy(in_scores).ravel()
    out_np = _to_numpy(out_scores).ravel()

    labels = np.concatenate([np.zeros(len(in_np)), np.ones(len(out_np))])
    scores = np.concatenate([in_np, out_np])

    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find FPR at the desired TPR
    idx = np.searchsorted(tpr_arr, tpr)
    idx = min(idx, len(fpr_arr) - 1)
    return float(fpr_arr[idx])
