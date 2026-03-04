"""Shared public types for Deep-UQ outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class UQResult:
    """Standardized uncertainty output container.

    Fields are method-agnostic and can be partially populated depending on the
    inference algorithm and task type.
    """

    mean: torch.Tensor
    epistemic_var: Optional[torch.Tensor] = None
    aleatoric_var: Optional[torch.Tensor] = None
    total_var: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    probs_var: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
