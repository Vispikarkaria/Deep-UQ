"""Conformal Prediction methods for distribution-free uncertainty quantification."""

from ._classification import ConformalClassifier
from ._cqr import CQRPredictor
from ._scores import (
    absolute_residual_score,
    normalized_residual_score,
    quantile_score,
    signed_residual_score,
)
from ._split import SplitConformalRegressor
from ._utils import check_coverage, conformal_quantile
from ._wrapper import ConformalUQWrapper

__all__ = [
    "SplitConformalRegressor",
    "ConformalClassifier",
    "CQRPredictor",
    "ConformalUQWrapper",
    "absolute_residual_score",
    "signed_residual_score",
    "normalized_residual_score",
    "quantile_score",
    "conformal_quantile",
    "check_coverage",
]
