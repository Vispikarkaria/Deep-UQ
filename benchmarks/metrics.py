from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def regression_metrics(
    y_true: torch.Tensor,
    mean: torch.Tensor,
    var: Optional[torch.Tensor],
) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    mean = mean.reshape(-1)
    rmse = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    mae = torch.mean(torch.abs(mean - y_true)).item()

    if var is None:
        return {
            "rmse": rmse,
            "mae": mae,
            "nll": float("nan"),
            "coverage95": float("nan"),
            "interval_width95": float("nan"),
        }

    var = var.reshape(-1).clamp_min(1e-8)
    nll = 0.5 * (torch.log(2.0 * math.pi * var) + ((y_true - mean) ** 2) / var)
    std = torch.sqrt(var)
    lo = mean - 1.96 * std
    hi = mean + 1.96 * std
    coverage = ((y_true >= lo) & (y_true <= hi)).float().mean().item()
    width = (hi - lo).mean().item()
    return {
        "rmse": rmse,
        "mae": mae,
        "nll": float(nll.mean().item()),
        "coverage95": coverage,
        "interval_width95": width,
    }
