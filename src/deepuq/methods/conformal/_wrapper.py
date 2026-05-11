"""Wrapper to conformalize any UQ method."""

from __future__ import annotations

from typing import Any

import torch

from deepuq.types import UQResult

from ._base import BaseConformalPredictor, _extract_data
from ._scores import normalized_residual_score
from ._utils import conformal_quantile


class ConformalUQWrapper(BaseConformalPredictor):
    """Calibrate intervals from any predict_uq()-compatible method.

    Wraps Laplace, Ensembles, MC Dropout, etc. to guarantee coverage.
    """

    def __init__(self, uq_method: Any, alpha: float = 0.1, symmetric: bool = True):
        super().__init__(uq_method, alpha)
        self.uq_method = uq_method
        self.symmetric = symmetric
        self._q_hat: float | None = None

    def _get_predictions(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and std from the wrapped UQ method."""
        if hasattr(self.uq_method, "predict_uq"):
            result = self.uq_method.predict_uq(x)
            mean = result.mean.squeeze(-1)
            var = (
                result.total_var
                if result.total_var is not None
                else result.epistemic_var
            )
            if var is None:
                var = torch.ones_like(mean)
            std = var.squeeze(-1).clamp_min(1e-12).sqrt()
        else:
            with torch.no_grad():
                out = self.uq_method(x)
            if isinstance(out, tuple):
                mean, var = out[0].squeeze(-1), out[1].squeeze(-1)
                std = var.clamp_min(1e-12).sqrt()
            else:
                mean = out.squeeze(-1)
                std = torch.ones_like(mean)
        return mean, std

    def calibrate(self, cal_data) -> ConformalUQWrapper:
        x_cal, y_cal = _extract_data(cal_data)
        y_cal = y_cal.squeeze(-1)
        mean, std = self._get_predictions(x_cal)

        scores = normalized_residual_score(mean, y_cal, std)
        self._scores = scores
        self._q_hat = conformal_quantile(scores, self.alpha)
        self._is_calibrated = True
        return self

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_uq().")

        mean, std = self._get_predictions(x)
        q = self._q_hat
        lower = mean - q * std
        upper = mean + q * std

        return UQResult(
            mean=mean,
            total_var=(q * std) ** 2,
            metadata={
                "conformal_lower": lower,
                "conformal_upper": upper,
                "coverage_alpha": self.alpha,
                "q_hat": q,
            },
        )
