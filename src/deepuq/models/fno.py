from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _as_mode_tuple(modes: Sequence[int]) -> tuple[int, int, int]:
    if len(modes) != 3:
        raise ValueError("modes must contain exactly three integers for 3D FNO.")
    mode_tuple = tuple(int(max(m, 1)) for m in modes)
    return mode_tuple  # type: ignore[return-value]


class SpectralConv3D(nn.Module):
    """3D spectral convolution with truncated Fourier modes.

    The layer follows the standard FNO pattern: transform to Fourier space,
    multiply a small set of learnable low-frequency modes, and transform back.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = _as_mode_tuple(modes)
        scale = 1.0 / max(self.in_channels * self.out_channels, 1)
        weight_shape = (
            self.in_channels,
            self.out_channels,
            self.modes[0],
            self.modes[1],
            self.modes[2],
            2,
        )
        self.weights_x0_y0 = nn.Parameter(scale * torch.randn(weight_shape))
        self.weights_x1_y0 = nn.Parameter(scale * torch.randn(weight_shape))
        self.weights_x0_y1 = nn.Parameter(scale * torch.randn(weight_shape))
        self.weights_x1_y1 = nn.Parameter(scale * torch.randn(weight_shape))

    @staticmethod
    def _compl_mul3d(
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("SpectralConv3D expects [batch, channels, nx, ny, nz].")

        batch, _, nx, ny, nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

        mx = min(self.modes[0], nx)
        my = min(self.modes[1], ny)
        mz = min(self.modes[2], x_ft.size(-1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :mx, :my, :mz] = self._compl_mul3d(
            x_ft[:, :, :mx, :my, :mz],
            torch.view_as_complex(self.weights_x0_y0[:, :, :mx, :my, :mz].contiguous()),
        )
        out_ft[:, :, -mx:, :my, :mz] = self._compl_mul3d(
            x_ft[:, :, -mx:, :my, :mz],
            torch.view_as_complex(self.weights_x1_y0[:, :, :mx, :my, :mz].contiguous()),
        )
        out_ft[:, :, :mx, -my:, :mz] = self._compl_mul3d(
            x_ft[:, :, :mx, -my:, :mz],
            torch.view_as_complex(self.weights_x0_y1[:, :, :mx, :my, :mz].contiguous()),
        )
        out_ft[:, :, -mx:, -my:, :mz] = self._compl_mul3d(
            x_ft[:, :, -mx:, -my:, :mz],
            torch.view_as_complex(self.weights_x1_y1[:, :, :mx, :my, :mz].contiguous()),
        )
        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1), norm="ortho")


class FNOBlock3D(nn.Module):
    """One 3D FNO block with spectral mixing and a local 1x1 skip path."""

    def __init__(
        self,
        width: int,
        modes: Sequence[int],
        use_nonlinearity: bool = True,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv3D(width, width, modes=modes)
        self.local = nn.Conv3d(width, width, kernel_size=1)
        self.use_nonlinearity = bool(use_nonlinearity)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x) + self.local(x)
        if self.use_nonlinearity:
            x = self.activation(x)
        return x


class FNO3D(nn.Module):
    """A compact 3D Fourier Neural Operator for scalar field-to-field maps."""

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 24,
        modes: Sequence[int] = (6, 6, 6),
        n_blocks: int = 4,
        head_hidden_dim: int = 64,
        use_coordinate_features: bool = True,
        use_nonlinearity: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.modes = _as_mode_tuple(modes)
        self.n_blocks = int(n_blocks)
        self.head_hidden_dim = int(head_hidden_dim)
        self.use_coordinate_features = bool(use_coordinate_features)
        self.use_nonlinearity = bool(use_nonlinearity)

        lifted_channels = self.in_channels + (3 if self.use_coordinate_features else 0)
        self.input_projection = nn.Linear(lifted_channels, self.width)
        self.blocks = nn.ModuleList(
            [
                FNOBlock3D(
                    self.width,
                    self.modes,
                    use_nonlinearity=self.use_nonlinearity,
                )
                for _ in range(self.n_blocks)
            ]
        )
        if self.head_hidden_dim > 0:
            head_layers = [nn.Linear(self.width, self.head_hidden_dim)]
            if self.use_nonlinearity:
                head_layers.append(nn.GELU())
            head_layers.append(nn.Linear(self.head_hidden_dim, 1))
            self.output_head = nn.Sequential(*head_layers)
        else:
            self.output_head = nn.Linear(self.width, 1)

    @staticmethod
    def _coordinate_grid(
        shape: tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        nx, ny, nz = shape
        x = torch.arange(nx, device=device, dtype=dtype) / max(nx, 1)
        y = torch.arange(ny, device=device, dtype=dtype) / max(ny, 1)
        z = torch.arange(nz, device=device, dtype=dtype) / max(nz, 1)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("FNO3D expects inputs with shape [B, Nx, Ny, Nz, C].")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got {x.size(-1)}."
            )

        if self.use_coordinate_features:
            coords = self._coordinate_grid(
                (x.size(1), x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
            coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1, -1)
            x = torch.cat((x, coords), dim=-1)
        x = self.input_projection(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.output_head(x).squeeze(-1)
        return x
