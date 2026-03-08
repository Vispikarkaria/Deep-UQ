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
    """

    mean: torch.Tensor
    epistemic_var: torch.Tensor | None = None
    aleatoric_var: torch.Tensor | None = None
    total_var: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    probs_var: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
