"""Diffusion-model building blocks for scientific field reconstruction.

The classes here support conditional denoising notebooks where uncertainty is
estimated from sample spread rather than Bayesian posterior moments.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by diffusion denoisers.

    Parameters
    ----------
    embedding_dim:
        Width of the returned timestep embedding.

    Shape contract
    --------------
    - input: ``timesteps`` with shape ``[batch]``
    - output: embedding tensor with shape ``[batch, embedding_dim]``
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            raise ValueError("timesteps must have shape [batch].")
        half_dim = self.embedding_dim // 2
        scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _TimeResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + residual


class _Downsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(x)


class ConditionalUNet2D(nn.Module):
    """A compact conditional U-Net denoiser for 2D diffusion notebooks.

    Parameters
    ----------
    x_channels:
        Number of noisy input channels to denoise.
    cond_channels:
        Number of conditioning channels supplied alongside the noisy input.
    base_channels:
        Base feature width used by the U-Net encoder/decoder.
    time_dim:
        Width of the timestep embedding processed by the residual blocks.
    dropout_p:
        Spatial dropout probability inside the residual blocks.
    use_coordinate_features:
        Whether to append normalized ``(x, y)`` coordinate channels.

    Shape contract
    --------------
    - ``x_t``: ``[batch, x_channels, height, width]``
    - ``timesteps``: ``[batch]``
    - ``condition``: ``[batch, cond_channels, height, width]``
    - output: denoised tensor with shape ``[batch, x_channels, height, width]``

    Example
    -------
    ```python
    model = ConditionalUNet2D(x_channels=1, cond_channels=2, base_channels=32)
    eps_hat = model(x_t, timesteps, condition)
    ```
    """

    def __init__(
        self,
        x_channels: int = 1,
        cond_channels: int = 2,
        base_channels: int = 32,
        time_dim: int = 128,
        dropout_p: float = 0.0,
        use_coordinate_features: bool = True,
    ) -> None:
        super().__init__()
        self.x_channels = int(x_channels)
        self.cond_channels = int(cond_channels)
        self.base_channels = int(base_channels)
        self.time_dim = int(time_dim)
        self.use_coordinate_features = bool(use_coordinate_features)

        in_channels = self.x_channels + self.cond_channels
        if self.use_coordinate_features:
            in_channels += 2

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4

        self.input_projection = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = _TimeResidualBlock2D(c1, c1, self.time_dim, dropout_p=dropout_p)
        self.pool1 = _Downsample2D(c1)
        self.down2 = _TimeResidualBlock2D(c1, c2, self.time_dim, dropout_p=dropout_p)
        self.pool2 = _Downsample2D(c2)
        self.mid = _TimeResidualBlock2D(c2, c3, self.time_dim, dropout_p=dropout_p)
        self.up2 = _Upsample2D(c3)
        self.up_block2 = _TimeResidualBlock2D(
            c3 + c2, c2, self.time_dim, dropout_p=dropout_p
        )
        self.up1 = _Upsample2D(c2)
        self.up_block1 = _TimeResidualBlock2D(
            c2 + c1, c1, self.time_dim, dropout_p=dropout_p
        )
        self.output_projection = nn.Sequential(
            _group_norm(c1),
            nn.GELU(),
            nn.Conv2d(c1, self.x_channels, kernel_size=3, padding=1),
        )
        self.reset_parameters()

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a normalized coordinate grid with shape ``[2, H, W]``."""
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last_conv = self.output_projection[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise or residual field for a conditioned diffusion step."""
        if x_t.dim() != 4:
            raise ValueError("x_t must have shape [B, C, H, W].")
        if condition.dim() != 4:
            raise ValueError("condition must have shape [B, C_cond, H, W].")
        if x_t.shape[0] != condition.shape[0] or x_t.shape[-2:] != condition.shape[-2:]:
            raise ValueError("x_t and condition must share batch and spatial shape.")
        if x_t.shape[1] != self.x_channels:
            raise ValueError(
                f"Expected {self.x_channels} noisy input channel(s), got {x_t.shape[1]}."
            )
        if condition.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected {self.cond_channels} conditioning channel(s), got {condition.shape[1]}."
            )

        if self.use_coordinate_features:
            coords = self._coordinate_grid(
                height=x_t.shape[-2],
                width=x_t.shape[-1],
                device=x_t.device,
                dtype=x_t.dtype,
            ).unsqueeze(0)
            coords = coords.expand(x_t.shape[0], -1, -1, -1)
            h = torch.cat((x_t, condition, coords), dim=1)
        else:
            h = torch.cat((x_t, condition), dim=1)

        time_embedding = self.time_embedding(timesteps)
        h0 = self.input_projection(h)
        h1 = self.down1(h0, time_embedding)
        h2 = self.down2(self.pool1(h1), time_embedding)
        hm = self.mid(self.pool2(h2), time_embedding)

        hu = self.up2(hm)
        if hu.shape[-2:] != h2.shape[-2:]:
            hu = torch.nn.functional.interpolate(
                hu, size=h2.shape[-2:], mode="bilinear", align_corners=False
            )
        hu = self.up_block2(torch.cat((hu, h2), dim=1), time_embedding)
        hu = self.up1(hu)
        if hu.shape[-2:] != h1.shape[-2:]:
            hu = torch.nn.functional.interpolate(
                hu, size=h1.shape[-2:], mode="bilinear", align_corners=False
            )
        hu = self.up_block1(torch.cat((hu, h1), dim=1), time_embedding)
        return self.output_projection(hu)
