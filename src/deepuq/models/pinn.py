from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _PINNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for idx in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[idx], dims[idx + 1]), nn.Tanh()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)


class PINN1D(_PINNBase):
    """Coordinate-input MLP for 1D physics-informed learning."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        output_dim: int = 1,
    ) -> None:
        super().__init__(input_dim=1, hidden_dims=hidden_dims, output_dim=output_dim)


class PINN2D(_PINNBase):
    """Coordinate-input MLP for 2D physics-informed learning."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128, 128, 128),
        output_dim: int = 1,
    ) -> None:
        super().__init__(input_dim=2, hidden_dims=hidden_dims, output_dim=output_dim)
