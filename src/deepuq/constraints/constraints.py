"""Physics constraint implementations for UQResult."""

from __future__ import annotations

from dataclasses import replace

import torch

from deepuq.types import UQResult


def _adjust_variance(
    original_var: torch.Tensor | None, mean: torch.Tensor, new_mean: torch.Tensor
) -> torch.Tensor | None:
    """Scale variance proportionally where mean was clipped."""
    if original_var is None:
        return None
    # Where mean changed, scale variance by (new_mean / mean)^2 clamped
    with torch.no_grad():
        ratio = torch.where(
            mean.abs() > 1e-12,
            (new_mean / mean).clamp(0.0, 1.0),
            torch.where(new_mean == mean, torch.ones_like(mean), torch.zeros_like(mean)),
        )
    return (original_var * ratio.square()).clamp(min=0.0)


class PositivityConstraint:
    """Clips mean to >= 0 and adjusts variance proportionally."""

    def apply(self, result: UQResult) -> UQResult:
        new_mean = result.mean.clamp(min=0.0)
        return replace(
            result,
            mean=new_mean,
            epistemic_var=_adjust_variance(result.epistemic_var, result.mean, new_mean),
            aleatoric_var=_adjust_variance(result.aleatoric_var, result.mean, new_mean),
            total_var=_adjust_variance(result.total_var, result.mean, new_mean),
        )


class BoundConstraint:
    """Clips mean to [lower, upper] and adjusts variance."""

    def __init__(self, lower: float | None = None, upper: float | None = None):
        self.lower = lower
        self.upper = upper

    def apply(self, result: UQResult) -> UQResult:
        new_mean = result.mean.clone()
        if self.lower is not None:
            new_mean = new_mean.clamp(min=self.lower)
        if self.upper is not None:
            new_mean = new_mean.clamp(max=self.upper)
        return replace(
            result,
            mean=new_mean,
            epistemic_var=_adjust_variance(result.epistemic_var, result.mean, new_mean),
            aleatoric_var=_adjust_variance(result.aleatoric_var, result.mean, new_mean),
            total_var=_adjust_variance(result.total_var, result.mean, new_mean),
        )


class ConservationConstraint:
    """Adjusts mean so that weighted integral equals conserved_quantity."""

    def __init__(self, integration_weights: torch.Tensor, conserved_quantity: float = 1.0):
        self.weights = integration_weights
        self.conserved_quantity = conserved_quantity

    def apply(self, result: UQResult) -> UQResult:
        current_integral = (result.mean * self.weights).sum()
        correction = (self.conserved_quantity - current_integral) / self.weights.sum()
        new_mean = result.mean + correction

        # Project out constraint direction from variance
        # The constraint removes one degree of freedom uniformly
        n = result.mean.numel()
        reduction_factor = (n - 1.0) / n if n > 1 else 1.0

        def reduce_var(v: torch.Tensor | None) -> torch.Tensor | None:
            if v is None:
                return None
            return (v * reduction_factor).clamp(min=0.0)

        return replace(
            result,
            mean=new_mean,
            epistemic_var=reduce_var(result.epistemic_var),
            aleatoric_var=reduce_var(result.aleatoric_var),
            total_var=reduce_var(result.total_var),
        )


class MonotonicityConstraint:
    """Enforces monotonicity via Pool Adjacent Violators algorithm."""

    def __init__(self, direction: str = "increasing", dim: int = 0):
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        self.direction = direction
        self.dim = dim

    def _isotonic_regression(self, y: torch.Tensor) -> torch.Tensor:
        """Pool Adjacent Violators for non-decreasing sequence."""
        n = y.shape[0]
        result = y.clone()
        # PAV algorithm
        blocks_start = list(range(n))
        blocks_size = [1] * n
        block_val = result.tolist()

        i = 0
        while i < n - 1:
            if block_val[i] > block_val[i + 1]:
                # Merge blocks
                total = block_val[i] * blocks_size[i] + block_val[i + 1] * blocks_size[i + 1]
                new_size = blocks_size[i] + blocks_size[i + 1]
                block_val[i] = total / new_size
                blocks_size[i] = new_size
                del block_val[i + 1]
                del blocks_size[i + 1]
                del blocks_start[i + 1]
                n -= 1
                # Check backward
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Reconstruct
        out = torch.empty_like(y)
        idx = 0
        for val, size in zip(block_val, blocks_size):
            out[idx : idx + size] = val
            idx += size
        return out

    def apply(self, result: UQResult) -> UQResult:
        mean = result.mean
        if self.direction == "decreasing":
            mean_work = -mean
        else:
            mean_work = mean

        # Apply along dim 0 for each slice
        if mean_work.dim() <= 1:
            new_mean_work = self._isotonic_regression(mean_work.flatten())
            new_mean_work = new_mean_work.reshape(mean.shape)
        else:
            # Move target dim to front, flatten rest
            mean_perm = mean_work.movedim(self.dim, 0)
            shape = mean_perm.shape
            flat = mean_perm.reshape(shape[0], -1)
            out_flat = torch.empty_like(flat)
            for col in range(flat.shape[1]):
                out_flat[:, col] = self._isotonic_regression(flat[:, col])
            new_mean_work = out_flat.reshape(shape).movedim(0, self.dim)

        if self.direction == "decreasing":
            new_mean = -new_mean_work
        else:
            new_mean = new_mean_work

        # Adjust variance where mean changed
        return replace(
            result,
            mean=new_mean,
            epistemic_var=_adjust_variance(result.epistemic_var, result.mean, new_mean),
            aleatoric_var=_adjust_variance(result.aleatoric_var, result.mean, new_mean),
            total_var=_adjust_variance(result.total_var, result.mean, new_mean),
        )


def apply_constraints(result: UQResult, constraints: list) -> UQResult:
    """Apply a list of constraints sequentially."""
    for constraint in constraints:
        result = constraint.apply(result)
    return result
