from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout3d(dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNRegressor2D(nn.Module):
    """Compact image-to-image CNN baseline for 2D scientific fields."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: Sequence[int] = (32, 48, 64, 64),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [in_channels, *hidden_channels]
        blocks = []
        for idx in range(len(channels) - 1):
            blocks.append(_ConvBlock2D(channels[idx], channels[idx + 1], dropout_p))
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class _ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        layers.extend(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResNetRegressor2D(nn.Module):
    """Residual 2D field-to-field regressor with optional spatial dropout."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_blocks: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock2D(width, dropout_p=dropout_p) for _ in range(n_blocks)]
        )
        self.head = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        return self.head(features)


class _DownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = _ConvBlock2D(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class _UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.block = _ConvBlock2D(
            in_channels + skip_channels,
            out_channels,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)
        if up.shape[-2:] != skip.shape[-2:]:
            up = torch.nn.functional.interpolate(
                up, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.block(torch.cat([up, skip], dim=1))


class UNet2D(nn.Module):
    """2D U-Net backbone for field-to-field scientific regression."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.enc1 = _ConvBlock2D(in_channels, c1, dropout_p=0.0)
        self.enc2 = _DownBlock2D(c1, c2, dropout_p=dropout_p)
        self.enc3 = _DownBlock2D(c2, c3, dropout_p=dropout_p)
        self.bottleneck = _DownBlock2D(c3, c4, dropout_p=dropout_p)
        self.up3 = _UpBlock2D(c4, c3, c3, dropout_p=dropout_p)
        self.up2 = _UpBlock2D(c3, c2, c2, dropout_p=dropout_p)
        self.up1 = _UpBlock2D(c2, c1, c1, dropout_p=dropout_p)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


class _DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.block = _ConvBlock3D(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class _UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )
        self.block = _ConvBlock3D(
            in_channels + skip_channels,
            out_channels,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)
        if up.shape[-3:] != skip.shape[-3:]:
            up = torch.nn.functional.interpolate(
                up, size=skip.shape[-3:], mode="trilinear", align_corners=False
            )
        return self.block(torch.cat([up, skip], dim=1))


class UNet3D(nn.Module):
    """3D U-Net backbone for volumetric scientific regression."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.enc1 = _ConvBlock3D(in_channels, c1, dropout_p=0.0)
        self.enc2 = _DownBlock3D(c1, c2, dropout_p=dropout_p)
        self.enc3 = _DownBlock3D(c2, c3, dropout_p=dropout_p)
        self.bottleneck = _DownBlock3D(c3, c4, dropout_p=dropout_p)
        self.up3 = _UpBlock3D(c4, c3, c3, dropout_p=dropout_p)
        self.up2 = _UpBlock3D(c3, c2, c2, dropout_p=dropout_p)
        self.up1 = _UpBlock3D(c2, c1, c1, dropout_p=dropout_p)
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)
