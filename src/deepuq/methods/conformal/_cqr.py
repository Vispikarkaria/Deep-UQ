"""Conformalized Quantile Regression."""

from __future__ import annotations

from typing import Any

import torch

from deepuq.types import UQResult

from ._base import BaseConformalPredictor, _extract_data, _model_forward
from ._scores import quantile_score
from ._utils import conformal_quantile


class CQRPredictor(BaseConformalPredictor):
    """Adaptive conformal intervals using quantile regression.

    Requires a model that outputs (lower_quantile, upper_quantile) as a
    tensor of shape (N, 2) or a tuple.
    """

    def __init__(self, model: Any, alpha: float = 0.1):
        super().__init__(model, alpha)
        self._q_hat: float | None = None

    def _get_quantiles(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = _model_forward(self.model, x)
        if isinstance(out, tuple):
            return out[0].squeeze(-1), out[1].squeeze(-1)
        # Assume shape (N, 2)
        return out[:, 0], out[:, 1]

    def calibrate(self, cal_data) -> CQRPredictor:
        x_cal, y_cal = _extract_data(cal_data)
        y_cal = y_cal.squeeze(-1)
        q_lo, q_hi = self._get_quantiles(x_cal)

        scores = quantile_score(y_cal, q_lo, q_hi)
        self._scores = scores
        self._q_hat = conformal_quantile(scores, self.alpha)
        self._is_calibrated = True
        return self

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_uq().")

        q_lo, q_hi = self._get_quantiles(x)
        q = self._q_hat
        lower = q_lo - q
        upper = q_hi + q
        mean = (lower + upper) / 2
        width = upper - lower

        return UQResult(
            mean=mean,
            total_var=(width / 2) ** 2,
            metadata={
                "conformal_lower": lower,
                "conformal_upper": upper,
                "coverage_alpha": self.alpha,
                "q_hat": q,
                "raw_q_lo": q_lo,
                "raw_q_hi": q_hi,
            },
        )
