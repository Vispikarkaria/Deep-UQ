"""Graph-based neural operators for regular-grid scientific fields.

The models in this module treat a Cartesian lattice as a graph and perform
message passing directly on node embeddings. They are intended as pragmatic
operator-learning baselines for settings where a graph inductive bias is useful
or where a future migration to unstructured meshes is anticipated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class _GridGraphCacheEntry:
    src: torch.Tensor
    dst: torch.Tensor
    edge_attr: torch.Tensor
    coords: torch.Tensor


class _MessagePassingBlock(nn.Module):
    """One message-passing block for the graph operator.

    Parameters
    ----------
    hidden_dim:
        Node embedding width.
    message_dim:
        Hidden width used for edge messages.
    use_edge_mlp:
        If ``True``, combine source and destination node states with edge
        features in a learned message MLP. Otherwise use a simpler linear
        transform of the source state plus a small edge projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int,
        use_edge_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.message_dim = int(message_dim)
        self.use_edge_mlp = bool(use_edge_mlp)
        if self.use_edge_mlp:
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_dim + 3, self.message_dim),
                nn.GELU(),
                nn.Linear(self.message_dim, self.message_dim),
            )
        else:
            self.message_linear = nn.Linear(self.hidden_dim, self.message_dim)
            self.edge_linear = nn.Linear(3, self.message_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.message_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Update node embeddings.

        Parameters
        ----------
        node_state:
            Tensor of shape ``[batch, n_nodes, hidden_dim]``.
        src, dst:
            Edge source and destination indices with shape ``[n_edges]``.
        edge_attr:
            Edge features with shape ``[n_edges, 3]`` storing ``dx``, ``dy``,
            and Euclidean distance.
        """
        src_state = node_state[:, src, :]
        dst_state = node_state[:, dst, :]
        edge_features = edge_attr.unsqueeze(0).expand(node_state.size(0), -1, -1)
        if self.use_edge_mlp:
            message_input = torch.cat((src_state, dst_state, edge_features), dim=-1)
            messages = self.message_mlp(message_input)
        else:
            messages = self.message_linear(src_state) + self.edge_linear(edge_features)

        aggregated = torch.zeros(
            node_state.size(0),
            node_state.size(1),
            self.message_dim,
            device=node_state.device,
            dtype=node_state.dtype,
        )
        aggregated.index_add_(1, dst, messages)
        update = self.update_mlp(torch.cat((node_state, aggregated), dim=-1))
        return self.norm(node_state + update)


class GraphNeuralOperator2D(nn.Module):
    """Message-passing neural operator on a 2D grid treated as a graph.

    Parameters
    ----------
    in_channels:
        Number of input field channels. In the Gray-Scott notebook this is ``2``
        for the species fields ``A`` and ``B``.
    hidden_dim:
        Node embedding width.
    message_dim:
        Width of the edge-message hidden state.
    n_message_passing_steps:
        Number of residual message-passing blocks.
    out_channels:
        Number of predicted output channels.
    radius:
        Grid-neighborhood radius. ``1`` corresponds to an 8-neighbor stencil.
    use_edge_mlp:
        If ``True``, use a learned edge MLP that mixes source, destination, and
        edge features. Otherwise use a lighter linear message path.

    Notes
    -----
    The public ``forward`` accepts channels-last grid tensors with shape
    ``[batch, H, W, C]`` and returns channels-last outputs with shape
    ``[batch, H, W, out_channels]``. The final decoder ends in ``nn.Linear`` so
    last-layer Laplace remains compatible.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        message_dim: int = 64,
        n_message_passing_steps: int = 4,
        out_channels: int = 2,
        radius: int = 1,
        use_edge_mlp: bool = True,
    ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be at least 1.")
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.message_dim = int(message_dim)
        self.n_message_passing_steps = int(n_message_passing_steps)
        self.out_channels = int(out_channels)
        self.radius = int(radius)
        self.use_edge_mlp = bool(use_edge_mlp)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.in_channels + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _MessagePassingBlock(
                    self.hidden_dim,
                    self.message_dim,
                    use_edge_mlp=self.use_edge_mlp,
                )
                for _ in range(self.n_message_passing_steps)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_channels),
        )
        self._graph_cache: dict[tuple[int, int, str], _GridGraphCacheEntry] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.2 if module is self.decoder[-1] else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if isinstance(self.decoder[-1], nn.Linear):
            nn.init.normal_(self.decoder[-1].weight, mean=0.0, std=1e-3)
            if self.decoder[-1].bias is not None:
                nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = torch.arange(height, device=device, dtype=dtype) / max(height - 1, 1)
        y = torch.arange(width, device=device, dtype=dtype) / max(width - 1, 1)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)

    def _build_graph(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GridGraphCacheEntry:
        key = (height, width, str(device))
        cached = self._graph_cache.get(key)
        if cached is not None and cached.edge_attr.dtype == dtype:
            return cached

        src_list: list[int] = []
        dst_list: list[int] = []
        edge_features: list[tuple[float, float, float]] = []
        inv_h = 1.0 / max(height - 1, 1)
        inv_w = 1.0 / max(width - 1, 1)
        for i in range(height):
            for j in range(width):
                dst_idx = i * width + j
                for di in range(-self.radius, self.radius + 1):
                    for dj in range(-self.radius, self.radius + 1):
                        if di == 0 and dj == 0:
                            continue
                        ii = i + di
                        jj = j + dj
                        if ii < 0 or ii >= height or jj < 0 or jj >= width:
                            continue
                        src_idx = ii * width + jj
                        dx = float(di) * inv_h
                        dy = float(dj) * inv_w
                        dist = (dx * dx + dy * dy) ** 0.5
                        src_list.append(src_idx)
                        dst_list.append(dst_idx)
                        edge_features.append((dx, dy, dist))

        src = torch.tensor(src_list, device=device, dtype=torch.long)
        dst = torch.tensor(dst_list, device=device, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, device=device, dtype=dtype)
        coords = self._coordinate_grid(height, width, device, dtype)
        entry = _GridGraphCacheEntry(src=src, dst=dst, edge_attr=edge_attr, coords=coords)
        self._graph_cache[key] = entry
        return entry

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a channels-last 2D input field to a channels-last output field."""
        if x.dim() != 4:
            raise ValueError("GraphNeuralOperator2D expects inputs with shape [B, H, W, C].")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.size(-1)}."
            )

        batch, height, width, _ = x.shape
        graph = self._build_graph(height, width, x.device, x.dtype)
        node_features = x.reshape(batch, height * width, self.in_channels)
        coords = graph.coords.unsqueeze(0).expand(batch, -1, -1)
        node_state = self.node_encoder(torch.cat((node_features, coords), dim=-1))
        for block in self.blocks:
            node_state = block(node_state, graph.src, graph.dst, graph.edge_attr)
        output = self.decoder(node_state)
        return output.reshape(batch, height, width, self.out_channels)
