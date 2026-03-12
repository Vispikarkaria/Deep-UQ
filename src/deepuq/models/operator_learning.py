"""DeepONet-style operator-learning models for fixed query grids.

The implementations here keep ``forward`` compatible with the package's current
``LaplaceWrapper`` convention: the default call accepts only the branch-input
tensor and evaluates the learned operator on an internally stored query grid.
"""

from __future__ import annotations

import torch
from torch import nn


def _build_tanh_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
) -> nn.Sequential:
    if depth < 2:
        raise ValueError(
            "depth must be at least 2 (one hidden layer and one output layer)."
        )

    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class _DeepONetFixedGrid(nn.Module):
    """Shared DeepONet implementation for fixed-grid operator learning.

    The default ``forward`` accepts only the branch input tensor so the model can
    be used with existing ``LaplaceWrapper`` dataloaders that expect
    ``(inputs, targets)`` pairs of tensors.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        query_grid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.branch_input_dim = int(branch_input_dim)
        self.trunk_input_dim = int(trunk_input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)

        self.branch_net = _build_tanh_mlp(
            input_dim=self.branch_input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            depth=self.depth,
        )
        self.trunk_net = _build_tanh_mlp(
            input_dim=self.trunk_input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            depth=self.depth,
        )
        self.output_head = nn.Linear(self.latent_dim, 1)
        self.register_buffer("query_grid", torch.empty(0, self.trunk_input_dim))

        if query_grid is not None:
            self.set_query_grid(query_grid)

    def set_query_grid(self, query_grid: torch.Tensor) -> None:
        """Store the coordinates used by the default ``forward`` call."""
        if query_grid.dim() != 2 or query_grid.size(-1) != self.trunk_input_dim:
            raise ValueError("query_grid must have shape [n_query, trunk_input_dim].")
        self.query_grid = query_grid.detach().clone().to(self.output_head.weight.device)

    def predict_on_coords(
        self,
        branch_inputs: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the operator on arbitrary coordinates.

        Parameters
        ----------
        branch_inputs:
            Tensor with shape ``[batch, branch_input_dim]`` containing the
            discretized input function for each sample.
        coords:
            Tensor with shape ``[n_query, trunk_input_dim]`` containing query
            coordinates.
        """
        if branch_inputs.dim() != 2:
            raise ValueError("branch_inputs must have shape [batch, branch_input_dim].")
        if coords.dim() != 2 or coords.size(-1) != self.trunk_input_dim:
            raise ValueError("coords must have shape [n_query, trunk_input_dim].")

        branch_latent = self.branch_net(branch_inputs)
        trunk_latent = self.trunk_net(coords)
        fused = branch_latent.unsqueeze(1) * trunk_latent.unsqueeze(0)
        outputs = self.output_head(fused.reshape(-1, self.latent_dim))
        return outputs.reshape(branch_inputs.size(0), coords.size(0))

    def forward(self, branch_inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the operator on the stored fixed query grid."""
        if self.query_grid.numel() == 0:
            raise RuntimeError(
                "query_grid is not set. Provide query_grid at construction time "
                "or call set_query_grid(...) before forward()."
            )
        return self.predict_on_coords(branch_inputs, self.query_grid)


class DeepONet1D(_DeepONetFixedGrid):
    """A compact DeepONet for 1D operator learning on a fixed query grid."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 1,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        query_grid: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            query_grid=query_grid,
        )


class DeepONet2D(_DeepONetFixedGrid):
    """A compact DeepONet for 2D operator learning on a fixed query grid."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 4,
        query_grid: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            query_grid=query_grid,
        )
