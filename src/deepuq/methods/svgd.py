"""Stein Variational Gradient Descent (SVGD) for Bayesian deep learning."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepuq.types import UQResult


class RBFKernel:
    """RBF kernel with median heuristic bandwidth."""

    def __init__(self, bandwidth: str | float = "median"):
        self.bandwidth = bandwidth

    def __call__(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute kernel matrix and gradient.

        Parameters
        ----------
        X : (n_particles, n_params) parameter matrix

        Returns
        -------
        K : (n_particles, n_particles) kernel matrix
        grad_K : (n_particles, n_params) sum of kernel gradients for each particle
        """
        n = X.shape[0]
        # Pairwise squared distances
        diff = X.unsqueeze(0) - X.unsqueeze(1)  # (n, n, d)
        dist_sq = (diff**2).sum(-1)  # (n, n)

        if self.bandwidth == "median":
            med = torch.median(dist_sq[dist_sq > 0])
            h = med / max(torch.log(torch.tensor(float(n))).item(), 1.0)
        else:
            h = float(self.bandwidth)

        K = torch.exp(-dist_sq / (h + 1e-8))  # (n, n)

        # grad_K[i] = sum_j ∇_{x_j} K(x_j, x_i) = sum_j K(x_j, x_i) * 2*(x_i - x_j)/h
        # diff[j, i] = X[i] - X[j], so (x_i - x_j) = diff[j, i]
        grad_K = (K.unsqueeze(-1) * diff).sum(0) * (2.0 / (h + 1e-8))  # (n, d)

        return K, grad_K


class SVGDOptimizer:
    """SVGD optimizer that updates a set of particle networks."""

    def __init__(
        self,
        particles: list[nn.Module],
        kernel: RBFKernel | None = None,
        lr: float = 0.01,
    ):
        self.particles = particles
        self.kernel = kernel or RBFKernel()
        self.lr = lr

    def _flatten_params(self) -> torch.Tensor:
        """Flatten all particle parameters into (K, D) matrix."""
        return torch.stack(
            [torch.cat([p.flatten() for p in m.parameters()]) for m in self.particles]
        )

    def _get_grads(self) -> torch.Tensor:
        """Get flattened gradients for all particles as (K, D) matrix."""
        grads = []
        for m in self.particles:
            g = torch.cat(
                [
                    p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                    for p in m.parameters()
                ]
            )
            grads.append(g)
        return torch.stack(grads)

    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> None:
        """Perform one SVGD update step.

        Parameters
        ----------
        loss_fn : callable(pred, target) -> scalar loss (negative log likelihood)
        x : input batch
        y : target batch
        """
        n = len(self.particles)

        # Compute gradients for each particle
        for m in self.particles:
            m.zero_grad()
            pred = m(x)
            loss = loss_fn(pred, y)
            loss.backward()

        # ∇log p(θ) = -∇loss (we want to maximize log posterior)
        neg_grads = -self._get_grads()  # (K, D)

        # Get parameter matrix
        theta = self._flatten_params().detach()  # (K, D)

        # Kernel
        K, grad_K = self.kernel(theta)  # (K, K), (K, D)

        # SVGD update: φ(θ_i) = (1/K) * Σ_j [K(θ_j, θ_i) * ∇log p(θ_j) + ∇_{θ_j} K(θ_j, θ_i)]
        # K @ neg_grads: (K, K) @ (K, D) -> (K, D) gives Σ_j K(j,i)*grad_j for each i
        phi = (K @ neg_grads + grad_K) / n  # (K, D)

        # Apply updates to particles
        for i, m in enumerate(self.particles):
            offset = 0
            for p in m.parameters():
                numel = p.numel()
                p.data.add_(self.lr * phi[i, offset : offset + numel].reshape(p.shape))
                offset += numel


class SVGDWrapper:
    """High-level SVGD wrapper for uncertainty quantification."""

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_particles: int = 10,
        lr: float = 0.01,
        kernel: RBFKernel | None = None,
    ):
        self.model_fn = model_fn
        self.n_particles = n_particles
        self.lr = lr
        self.kernel = kernel
        self.particles: list[nn.Module] = []
        self.optimizer: SVGDOptimizer | None = None

    def fit(
        self,
        train_loader,
        n_epochs: int,
        loss_fn: Callable = F.mse_loss,
    ) -> None:
        """Train particles using SVGD."""
        self.particles = [self.model_fn() for _ in range(self.n_particles)]
        self.optimizer = SVGDOptimizer(self.particles, kernel=self.kernel, lr=self.lr)

        for _ in range(n_epochs):
            for x, y in train_loader:
                self.optimizer.step(loss_fn, x, y)

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Get predictions with uncertainty from particle ensemble."""
        preds = torch.stack([m(x) for m in self.particles])  # (K, batch, out)
        mean = preds.mean(dim=0)
        epistemic_var = preds.var(dim=0)
        return UQResult(
            mean=mean,
            epistemic_var=epistemic_var,
            total_var=epistemic_var,
            metadata={"method": "svgd", "n_particles": self.n_particles},
        )
