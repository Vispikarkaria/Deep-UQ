"""Isotonic regression calibration for classification models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from deepuq.types import UQResult


class IsotonicCalibration:
    """Per-class isotonic regression calibration.

    Fits an isotonic regression model per class on validation set predictions,
    mapping raw softmax probabilities to calibrated probabilities.

    Parameters
    ----------
    model:
        A PyTorch model that outputs raw logits.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._regressors: list[IsotonicRegression] | None = None

    def fit(self, val_loader: torch.utils.data.DataLoader) -> IsotonicCalibration:
        """Fit isotonic regression per class on validation data.

        Parameters
        ----------
        val_loader:
            DataLoader yielding (inputs, targets) batches.

        Returns
        -------
        self
        """
        self.model.eval()
        probs_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits = self.model(inputs)
                probs_list.append(F.softmax(logits, dim=-1))
                labels_list.append(targets)

        probs = torch.cat(probs_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()

        num_classes = probs.shape[-1]
        self._regressors = []
        for c in range(num_classes):
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            binary_labels = (labels == c).astype(np.float64)
            ir.fit(probs[:, c], binary_labels)
            self._regressors.append(ir)

        return self

    def predict_calibrated(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for input x.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Calibrated class probabilities (normalized to sum to 1).
        """
        if self._regressors is None:
            raise RuntimeError("Must call fit() before predict_calibrated().")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).numpy()

        num_classes = probs.shape[-1]
        calibrated = np.zeros_like(probs)
        for c in range(num_classes):
            calibrated[:, c] = self._regressors[c].predict(probs[:, c])

        # Normalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        calibrated = calibrated / row_sums

        return torch.from_numpy(calibrated).float()

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return a UQResult with calibrated probabilities and entropy-based uncertainty."""
        probs = self.predict_calibrated(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return UQResult(
            mean=probs,
            epistemic_var=entropy,
            probs=probs,
            metadata={"method": "isotonic_calibration"},
        )
