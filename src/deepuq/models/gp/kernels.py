"""Kernel implementations for Gaussian process models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch

from .utils import pairwise_squared_distance, prepare_lengthscale

LengthscaleLike = Union[float, torch.Tensor]


class Kernel:
    """Base kernel interface with composition support."""

    jitter: float = 0.0

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __add__(self, other: Kernel) -> Kernel:
        if not isinstance(other, Kernel):
            return NotImplemented
        return SumKernel(self, other)

    def __mul__(self, other: Kernel) -> Kernel:
        if not isinstance(other, Kernel):
            return NotImplemented
        return ProductKernel(self, other)


@dataclass
class SumKernel(Kernel):
    """Additive kernel composition."""

    left: Kernel
    right: Kernel

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.left(x1, x2) + self.right(x1, x2)


@dataclass
class ProductKernel(Kernel):
    """Multiplicative kernel composition."""

    left: Kernel
    right: Kernel

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.left(x1, x2) * self.right(x1, x2)


@dataclass
class RBFKernel(Kernel):
    """Squared exponential (RBF) kernel with scalar or ARD lengthscales."""

    lengthscale: LengthscaleLike = 1.0
    outputscale: float = 1.0
    jitter: float = 1e-6

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls = torch.as_tensor(self.lengthscale, device=x1.device, dtype=x1.dtype)
        ls = prepare_lengthscale(ls, x1)
        x1_scaled = x1 / ls
        x2_scaled = x2 / ls
        squared_dist = pairwise_squared_distance(x1_scaled, x2_scaled)
        cov = self.outputscale * torch.exp(-0.5 * squared_dist)
        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov


@dataclass
class MaternKernel(Kernel):
    """Matérn kernel with ν in {1.5, 2.5}."""

    lengthscale: LengthscaleLike = 1.0
    outputscale: float = 1.0
    nu: float = 1.5
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        if self.nu not in {1.5, 2.5}:
            raise ValueError("MaternKernel only supports nu=1.5 or nu=2.5.")

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls = torch.as_tensor(self.lengthscale, device=x1.device, dtype=x1.dtype)
        ls = prepare_lengthscale(ls, x1)
        x1_scaled = x1 / ls
        x2_scaled = x2 / ls
        r = torch.sqrt(pairwise_squared_distance(x1_scaled, x2_scaled) + 1e-12)

        if self.nu == 1.5:
            c = math.sqrt(3.0)
            cov = self.outputscale * (1.0 + c * r) * torch.exp(-c * r)
        else:
            c = math.sqrt(5.0)
            cov = (
                self.outputscale
                * (1.0 + c * r + (5.0 / 3.0) * r**2)
                * torch.exp(-c * r)
            )

        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov


@dataclass
class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel."""

    lengthscale: LengthscaleLike = 1.0
    outputscale: float = 1.0
    alpha: float = 1.0
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0 for RationalQuadraticKernel.")

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls = torch.as_tensor(self.lengthscale, device=x1.device, dtype=x1.dtype)
        ls = prepare_lengthscale(ls, x1)
        x1_scaled = x1 / ls
        x2_scaled = x2 / ls
        sq = pairwise_squared_distance(x1_scaled, x2_scaled)
        cov = self.outputscale * torch.pow(1.0 + sq / (2.0 * self.alpha), -self.alpha)
        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov


@dataclass
class PeriodicKernel(Kernel):
    """Periodic kernel with scalar or ARD lengthscales."""

    lengthscale: LengthscaleLike = 1.0
    outputscale: float = 1.0
    period: float = 1.0
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be > 0 for PeriodicKernel.")

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls = torch.as_tensor(self.lengthscale, device=x1.device, dtype=x1.dtype)
        ls = prepare_lengthscale(ls, x1)

        tau = x1[:, None, :] - x2[None, :, :]
        sin_sq = torch.sin(math.pi * tau / self.period).pow(2)
        if ls.ndim == 0:
            scaled = sin_sq / (ls**2)
        else:
            scaled = sin_sq / (ls.view(1, 1, -1) ** 2)

        cov = self.outputscale * torch.exp(-2.0 * scaled.sum(dim=-1))
        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov


@dataclass
class LinearKernel(Kernel):
    """Linear kernel."""

    variance: float = 1.0
    bias: float = 0.0
    jitter: float = 1e-6

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cov = self.variance * (x1 @ x2.t()) + self.bias
        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov


@dataclass
class SpectralMixtureKernel(Kernel):
    """Spectral mixture kernel with fixed mixture parameters."""

    weights: torch.Tensor
    means: torch.Tensor
    scales: torch.Tensor
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must have shape [Q].")
        if self.means.ndim != 2 or self.scales.ndim != 2:
            raise ValueError("means/scales must have shape [Q, D].")
        if self.means.shape != self.scales.shape:
            raise ValueError("means and scales must share shape [Q, D].")
        if self.means.shape[0] != self.weights.shape[0]:
            raise ValueError("weights and means/scales must share Q mixtures.")

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(device=x1.device, dtype=x1.dtype)
        means = self.means.to(device=x1.device, dtype=x1.dtype)
        scales = self.scales.to(device=x1.device, dtype=x1.dtype)

        tau = x1[:, None, :] - x2[None, :, :]
        q, d = means.shape
        if x1.shape[-1] != d:
            raise ValueError(
                "Input feature dimension must match spectral mixture parameter dimension."
            )

        tau_q = tau.unsqueeze(0)  # [Q, N, M, D]
        means_q = means.view(q, 1, 1, d)
        scales_q = scales.view(q, 1, 1, d)

        exp_term = torch.exp(
            -2.0 * (math.pi**2) * (tau_q**2) * torch.clamp(scales_q, min=1e-8)
        ).prod(dim=-1)
        cos_term = torch.cos(2.0 * math.pi * tau_q * means_q).prod(dim=-1)
        cov = (weights.view(q, 1, 1) * exp_term * cos_term).sum(dim=0)

        if x1.shape == x2.shape and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(
                x1.shape[0], device=x1.device, dtype=x1.dtype
            )
        return cov
