"""Base class for conformal predictors."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from deepuq.types import UQResult


def _extract_data(cal_data) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (x, y) tensors from DataLoader or tuple."""
    if isinstance(cal_data, DataLoader):
        xs, ys = [], []
        for batch in cal_data:
            xs.append(batch[0])
            ys.append(batch[1])
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    else:
        return cal_data[0], cal_data[1]


def _model_forward(model: Any, x: torch.Tensor) -> torch.Tensor:
    """Call model on input, handling nn.Module and callables."""
    with torch.no_grad():
        return model(x)


class BaseConformalPredictor:
    """Base class for conformal prediction methods."""

    def __init__(self, model: Any, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self._is_calibrated = False
        self._scores: torch.Tensor | None = None

    def calibrate(self, cal_data) -> BaseConformalPredictor:
        """Calibrate on calibration data. Subclasses must override."""
        raise NotImplementedError

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Produce prediction with uncertainty. Subclasses must override."""
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_uq().")
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
