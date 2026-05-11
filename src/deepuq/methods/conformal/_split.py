"""Split Conformal Regression."""

from __future__ import annotations

from typing import Any

import torch

from deepuq.types import UQResult

from ._base import BaseConformalPredictor, _extract_data, _model_forward
from ._scores import (
    absolute_residual_score,
    signed_residual_score,
)
from ._utils import conformal_quantile

_SCORE_FNS = {
    "absolute_residual": absolute_residual_score,
    "signed_residual": signed_residual_score,
}


class SplitConformalRegressor(BaseConformalPredictor):
    """Distribution-free prediction intervals for any regression model.

    Guarantees P(Y in [lower, upper]) >= 1 - alpha for exchangeable data.
    """

    def __init__(
        self, model: Any, alpha: float = 0.1, score_fn: str = "absolute_residual"
    ):
        super().__init__(model, alpha)
        self.score_fn_name = score_fn
        self._q_hat: float | None = None

    def calibrate(self, cal_data) -> SplitConformalRegressor:
        x_cal, y_cal = _extract_data(cal_data)
        y_pred = _model_forward(self.model, x_cal).squeeze(-1)
        y_cal = y_cal.squeeze(-1)

        score_fn = _SCORE_FNS[self.score_fn_name]
        scores = score_fn(y_pred, y_cal)
        self._scores = scores
        self._q_hat = conformal_quantile(scores, self.alpha)
        self._is_calibrated = True
        return self

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before predict_uq().")

        y_pred = _model_forward(self.model, x).squeeze(-1)
        q = self._q_hat
        lower = y_pred - q
        upper = y_pred + q

        return UQResult(
            mean=y_pred,
            total_var=torch.full_like(y_pred, q**2),
            metadata={
                "conformal_lower": lower,
                "conformal_upper": upper,
                "coverage_alpha": self.alpha,
                "q_hat": q,
            },
        )
