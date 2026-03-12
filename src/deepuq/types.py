"""Shared public types for Deep-UQ outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class UQResult:
    """Standardized uncertainty output container.

    Fields are method-agnostic and can be partially populated depending on the
    inference algorithm and task type.

    Parameters
    ----------
    mean:
        Predictive mean tensor for regression-style outputs. For classification
        methods this usually stores the same value as ``probs`` for
        convenience.
    epistemic_var:
        Variance attributed to model or posterior uncertainty. Present for
        methods that expose between-sample or between-model spread.
    aleatoric_var:
        Variance attributed to data noise or likelihood noise. Present only for
        methods that model observation noise explicitly.
    total_var:
        Total predictive variance. When both epistemic and aleatoric terms are
        available this should be their sum.
    probs:
        Predictive class probabilities for classification methods.
    probs_var:
        Probability-space variance or disagreement summary for classification
        methods.
    metadata:
        Free-form method metadata such as backend, sample count, or likelihood
        settings.
    """

    mean: torch.Tensor
    epistemic_var: torch.Tensor | None = None
    aleatoric_var: torch.Tensor | None = None
    total_var: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    probs_var: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
