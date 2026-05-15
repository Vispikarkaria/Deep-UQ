"""Tests for deepuq.metrics module."""

import numpy as np
import pytest
import torch

from deepuq.metrics import (
    auroc_ood,
    continuous_ranked_probability_score,
    expected_calibration_error,
    fpr_at_tpr,
    prediction_interval_coverage,
    risk_coverage_curve,
    aurc,
)


class TestECE:
    def test_perfectly_calibrated(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # Confidence matches accuracy in each bin
        np.random.seed(42)
        n = 10000
        probs = np.random.uniform(0, 1, n)
        # For each sample, label=1 with probability equal to confidence
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)
        ece = expected_calibration_error(probs, labels, n_bins=10)
        assert ece < 0.05  # Should be near zero with enough samples

    def test_overconfident(self):
        """ECE should be > 0 for overconfident predictions."""
        # Always predict 0.9 confidence but only 50% accurate
        probs = np.full(100, 0.9)
        labels = np.concatenate([np.ones(50), np.zeros(50)])
        ece = expected_calibration_error(probs, labels)
        assert ece > 0.3

    def test_torch_input(self):
        """Should accept torch tensors."""
        probs = torch.tensor([0.9, 0.9, 0.9, 0.9])
        labels = torch.tensor([1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(probs, labels)
        assert isinstance(ece, float)


class TestPICP:
    def test_all_covered(self):
        """PICP = 1.0 when all points fall within intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        lower = y_true - 1.0
        upper = y_true + 1.0
        assert prediction_interval_coverage(lower, upper, y_true) == 1.0

    def test_none_covered(self):
        """PICP = 0.0 when no points fall within intervals."""
        y_true = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert prediction_interval_coverage(lower, upper, y_true) == 0.0

    def test_torch_input(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        lo = torch.tensor([0.5, 1.5, 2.5])
        hi = torch.tensor([1.5, 2.5, 3.5])
        assert prediction_interval_coverage(lo, hi, y) == 1.0


class TestCRPS:
    def test_non_negative(self):
        """CRPS should always be >= 0."""
        np.random.seed(0)
        mean = np.random.randn(100)
        std = np.abs(np.random.randn(100)) + 0.1
        y = np.random.randn(100)
        crps = continuous_ranked_probability_score(mean, std, y)
        assert crps >= 0.0

    def test_perfect_prediction_small_std(self):
        """CRPS should be small when predictions are accurate with small std."""
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.01, 0.01, 0.01])
        y = np.array([1.0, 2.0, 3.0])
        crps = continuous_ranked_probability_score(mean, std, y)
        assert crps < 0.01


class TestAUROC:
    def test_perfectly_separated(self):
        """AUROC = 1.0 for perfectly separated distributions."""
        in_scores = np.zeros(50)
        out_scores = np.ones(50)
        assert auroc_ood(in_scores, out_scores) == 1.0

    def test_identical_distributions(self):
        """AUROC ~ 0.5 for identical distributions."""
        np.random.seed(42)
        in_scores = np.random.randn(1000)
        out_scores = np.random.randn(1000)
        auc = auroc_ood(in_scores, out_scores)
        assert 0.45 < auc < 0.55

    def test_torch_input(self):
        in_s = torch.zeros(50)
        out_s = torch.ones(50)
        assert auroc_ood(in_s, out_s) == 1.0


class TestAURC:
    def test_returns_float(self):
        unc = np.array([0.1, 0.5, 0.9, 0.2])
        err = np.array([0.0, 1.0, 1.0, 0.0])
        result = aurc(unc, err)
        assert isinstance(result, float)
        assert result >= 0.0
