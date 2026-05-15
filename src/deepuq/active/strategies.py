"""Acquisition strategies for active learning."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepuq.types import UQResult


class UncertaintySampling:
    """Select pool points with highest uncertainty according to a criterion.

    Parameters
    ----------
    model:
        Model exposing a ``predict_uq(x) -> UQResult`` method.
    criterion:
        Which UQResult field to rank by. One of "epistemic_var", "total_var",
        or "entropy".
    """

    def __init__(self, model: nn.Module, criterion: str = "epistemic_var"):
        if criterion not in ("epistemic_var", "total_var", "entropy"):
            raise ValueError(
                f"criterion must be 'epistemic_var', 'total_var', or 'entropy', got '{criterion}'"
            )
        self.model = model
        self.criterion = criterion

    def select(self, pool_X: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Return indices of the top-n most uncertain points in pool_X."""
        uq: UQResult = self.model.predict_uq(pool_X)

        if self.criterion == "entropy":
            # For classification: entropy of predictive probs
            probs = uq.probs
            if probs is None:
                raise ValueError("entropy criterion requires probs in UQResult")
            scores = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        else:
            var_tensor = getattr(uq, self.criterion)
            if var_tensor is None:
                raise ValueError(f"{self.criterion} is None in UQResult")
            # Collapse to per-sample scalar if multi-dimensional
            if var_tensor.dim() > 1:
                scores = var_tensor.sum(dim=tuple(range(1, var_tensor.dim())))
            else:
                scores = var_tensor

        _, indices = scores.topk(min(n_samples, len(scores)))
        return indices


class BALDSampling:
    """Bayesian Active Learning by Disagreement.

    Computes mutual information I[y; theta | x, D] by measuring disagreement
    across multiple stochastic forward passes.

    Parameters
    ----------
    model:
        Model that supports stochastic forward passes (e.g. MC Dropout enabled
        via ``model.model.train(True)``). Must expose ``predict_uq`` or a
        callable forward.
    n_mc_samples:
        Number of MC forward passes to estimate BALD score.
    """

    def __init__(self, model: nn.Module, n_mc_samples: int = 20):
        self.model = model
        self.n_mc_samples = n_mc_samples

    @torch.inference_mode()
    def select(self, pool_X: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Return indices with highest BALD (mutual information) score."""
        # Collect MC samples
        if hasattr(self.model, "model"):
            self.model.model.train(True)  # enable dropout

        samples = []
        for _ in range(self.n_mc_samples):
            out = self.model.model(pool_X) if hasattr(self.model, "model") else self.model(pool_X)
            samples.append(out.unsqueeze(0))

        preds = torch.cat(samples, dim=0)  # [K, B, D]

        # For regression: BALD ~ variance of means (epistemic)
        # Total variance = Var[E[y|x,theta]] + E[Var[y|x,theta]]
        # Since we only have point predictions, BALD = Var across samples
        # which is the disagreement measure
        var_across_samples = preds.var(dim=0)  # [B, D]

        # Collapse to per-sample score
        if var_across_samples.dim() > 1:
            scores = var_across_samples.sum(dim=tuple(range(1, var_across_samples.dim())))
        else:
            scores = var_across_samples

        if hasattr(self.model, "model"):
            self.model.model.eval()

        _, indices = scores.topk(min(n_samples, len(scores)))
        return indices
