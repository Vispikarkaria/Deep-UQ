"""Active learning loop orchestration."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ActiveLearningLoop:
    """Iteratively select informative samples, retrain, and track progress.

    Parameters
    ----------
    model:
        The UQ-aware model to train.
    strategy:
        Acquisition strategy with a ``select(pool_X, n_samples)`` method.
    train_fn:
        User-provided training function with signature
        ``train_fn(model, X_train, y_train) -> model``.
    initial_X, initial_y:
        Starting labeled dataset.
    pool_X, pool_y:
        Unlabeled pool to draw from.
    val_X, val_y:
        Optional validation set for tracking MSE.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy,
        train_fn: Callable,
        initial_X: torch.Tensor,
        initial_y: torch.Tensor,
        pool_X: torch.Tensor,
        pool_y: torch.Tensor,
        val_X: torch.Tensor | None = None,
        val_y: torch.Tensor | None = None,
    ):
        self.model = model
        self.strategy = strategy
        self.train_fn = train_fn
        self.train_X = initial_X.clone()
        self.train_y = initial_y.clone()
        self.pool_X = pool_X.clone()
        self.pool_y = pool_y.clone()
        self.val_X = val_X
        self.val_y = val_y

    def step(self, n_samples: int = 10) -> dict:
        """Run one active learning iteration.

        Returns dict with selected_indices, train_size, and val_metric (if val set provided).
        """
        indices = self.strategy.select(self.pool_X, n_samples)

        # Move selected points from pool to training set
        self.train_X = torch.cat([self.train_X, self.pool_X[indices]], dim=0)
        self.train_y = torch.cat([self.train_y, self.pool_y[indices]], dim=0)

        # Remove selected from pool
        mask = torch.ones(len(self.pool_X), dtype=torch.bool)
        mask[indices] = False
        self.pool_X = self.pool_X[mask]
        self.pool_y = self.pool_y[mask]

        # Retrain
        self.model = self.train_fn(self.model, self.train_X, self.train_y)

        result = {
            "selected_indices": indices,
            "train_size": len(self.train_X),
        }

        # Compute validation metric if available
        if self.val_X is not None and self.val_y is not None:
            with torch.no_grad():
                if hasattr(self.model, "predict_uq"):
                    preds = self.model.predict_uq(self.val_X).mean
                elif hasattr(self.model, "model"):
                    self.model.model.eval()
                    preds = self.model.model(self.val_X)
                else:
                    self.model.eval()
                    preds = self.model(self.val_X)
                mse = ((preds - self.val_y) ** 2).mean().item()
            result["val_metric"] = mse

        return result

    def run(self, n_iterations: int = 10, n_samples_per_iter: int = 10) -> list[dict]:
        """Run multiple active learning steps.

        Returns list of step result dicts.
        """
        history = []
        for _ in range(n_iterations):
            result = self.step(n_samples=n_samples_per_iter)
            history.append(result)
        return history
