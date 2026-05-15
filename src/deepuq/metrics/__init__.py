"""UQ metrics for calibration, scoring, OOD detection, and selective prediction."""

from deepuq.metrics.calibration import (
    calibration_curve_regression,
    expected_calibration_error,
    maximum_calibration_error,
    prediction_interval_coverage,
)
from deepuq.metrics.ood import auroc_ood, fpr_at_tpr
from deepuq.metrics.scoring import (
    brier_score,
    continuous_ranked_probability_score,
    interval_score,
    negative_log_likelihood,
)
from deepuq.metrics.selective import aurc, risk_coverage_curve

__all__ = [
    "expected_calibration_error",
    "maximum_calibration_error",
    "prediction_interval_coverage",
    "calibration_curve_regression",
    "negative_log_likelihood",
    "continuous_ranked_probability_score",
    "brier_score",
    "interval_score",
    "auroc_ood",
    "fpr_at_tpr",
    "risk_coverage_curve",
    "aurc",
]
