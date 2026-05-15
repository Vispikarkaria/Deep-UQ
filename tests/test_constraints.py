"""Tests for physics constraint utilities."""

import torch

from deepuq.constraints import (
    BoundConstraint,
    ConservationConstraint,
    MonotonicityConstraint,
    PositivityConstraint,
    apply_constraints,
)
from deepuq.types import UQResult


def _make_result(mean, var=None):
    m = torch.tensor(mean, dtype=torch.float32)
    v = torch.tensor(var, dtype=torch.float32) if var is not None else None
    return UQResult(mean=m, total_var=v)


class TestPositivityConstraint:
    def test_clips_negative_means(self):
        result = _make_result([-1.0, 2.0, -3.0], [1.0, 1.0, 1.0])
        out = PositivityConstraint().apply(result)
        assert (out.mean >= 0).all()
        assert out.mean[1] == 2.0

    def test_variance_still_positive(self):
        result = _make_result([-1.0, 2.0, -0.5], [1.0, 1.0, 1.0])
        out = PositivityConstraint().apply(result)
        assert (out.total_var >= 0).all()


class TestBoundConstraint:
    def test_keeps_in_range(self):
        result = _make_result([-5.0, 0.5, 10.0], [1.0, 1.0, 1.0])
        out = BoundConstraint(lower=0.0, upper=1.0).apply(result)
        assert (out.mean >= 0.0).all()
        assert (out.mean <= 1.0).all()

    def test_lower_only(self):
        result = _make_result([-2.0, 3.0])
        out = BoundConstraint(lower=0.0).apply(result)
        assert out.mean[0] == 0.0
        assert out.mean[1] == 3.0

    def test_variance_still_positive(self):
        result = _make_result([5.0, -5.0], [2.0, 2.0])
        out = BoundConstraint(lower=0.0, upper=1.0).apply(result)
        assert (out.total_var >= 0).all()


class TestConservationConstraint:
    def test_integral_equals_conserved_quantity(self):
        weights = torch.ones(5)
        result = _make_result([1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 5)
        out = ConservationConstraint(weights, conserved_quantity=10.0).apply(result)
        integral = (out.mean * weights).sum()
        assert torch.isclose(integral, torch.tensor(10.0), atol=1e-5)

    def test_variance_still_positive(self):
        weights = torch.ones(3)
        result = _make_result([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        out = ConservationConstraint(weights, conserved_quantity=0.0).apply(result)
        assert (out.total_var >= 0).all()


class TestMonotonicityConstraint:
    def test_produces_nondecreasing(self):
        result = _make_result([1.0, 3.0, 2.0, 5.0, 4.0], [1.0] * 5)
        out = MonotonicityConstraint(direction="increasing").apply(result)
        for i in range(len(out.mean) - 1):
            assert out.mean[i] <= out.mean[i + 1]

    def test_decreasing(self):
        result = _make_result([5.0, 2.0, 4.0, 1.0], [1.0] * 4)
        out = MonotonicityConstraint(direction="decreasing").apply(result)
        for i in range(len(out.mean) - 1):
            assert out.mean[i] >= out.mean[i + 1]

    def test_variance_still_positive(self):
        result = _make_result([3.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        out = MonotonicityConstraint().apply(result)
        assert (out.total_var >= 0).all()


class TestApplyConstraints:
    def test_chains_multiple(self):
        result = _make_result([-1.0, 3.0, 2.0, 5.0], [1.0] * 4)
        out = apply_constraints(
            result, [PositivityConstraint(), MonotonicityConstraint(direction="increasing")]
        )
        assert (out.mean >= 0).all()
        for i in range(len(out.mean) - 1):
            assert out.mean[i] <= out.mean[i + 1]
