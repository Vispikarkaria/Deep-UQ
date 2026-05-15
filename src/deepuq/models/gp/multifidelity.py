"""Multi-Fidelity Gaussian Process regression model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from deepuq.types import UQResult


class _RBFKernel(nn.Module):
    """Simple RBF kernel with learnable scale and lengthscale."""

    def __init__(self, scale: float = 1.0, lengthscale: float = 1.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.log_lengthscale.exp()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist_sq = torch.cdist(x1, x2, p=2).pow(2)
        return self.scale * torch.exp(-0.5 * dist_sq / self.lengthscale.pow(2))


def _gp_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_new: torch.Tensor,
    kernel: _RBFKernel,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GP posterior mean and variance."""
    K = kernel.forward(x_train, x_train)
    n = K.shape[0]
    K = K + noise * torch.eye(n, device=K.device, dtype=K.dtype)
    K_s = kernel.forward(x_train, x_new)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y_train.reshape(-1, 1), L)
    mean = K_s.T @ alpha
    v = torch.cholesky_solve(K_s, L)
    var = kernel.forward(x_new, x_new).diag() - (K_s * v).sum(dim=0)
    return mean.squeeze(-1), var.clamp_min(0.0)


def _log_marginal_likelihood(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: _RBFKernel,
    noise: float,
) -> torch.Tensor:
    """Compute log marginal likelihood."""
    K = kernel.forward(x, x)
    n = K.shape[0]
    K = K + noise * torch.eye(n, device=K.device, dtype=K.dtype)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.reshape(-1, 1), L)
    data_fit = -0.5 * (y.reshape(-1, 1).T @ alpha).squeeze()
    log_det = -torch.log(torch.diagonal(L)).sum()
    constant = -0.5 * n * math.log(2.0 * math.pi)
    return data_fit + log_det + constant


class MultiFidelityGP(nn.Module):
    """Multi-fidelity GP using autoregressive Kennedy-O'Hagan model.

    f_hi(x) = rho * f_lo(x) + f_delta(x)
    """

    def __init__(
        self,
        kernel_lo: _RBFKernel | None = None,
        kernel_hi: _RBFKernel | None = None,
        noise_lo: float = 0.1,
        noise_hi: float = 0.01,
    ) -> None:
        super().__init__()
        self.kernel_lo = kernel_lo or _RBFKernel()
        self.kernel_hi = kernel_hi or _RBFKernel()
        self.noise_lo = noise_lo
        self.noise_hi = noise_hi
        self.rho = nn.Parameter(torch.tensor(1.0))

        self._x_lo: torch.Tensor | None = None
        self._y_lo: torch.Tensor | None = None
        self._x_hi: torch.Tensor | None = None
        self._y_delta: torch.Tensor | None = None

    def fit(
        self,
        X_lo: torch.Tensor,
        y_lo: torch.Tensor,
        X_hi: torch.Tensor,
        y_hi: torch.Tensor,
    ) -> MultiFidelityGP:
        """Fit the multi-fidelity model."""
        X_lo = X_lo.float()
        y_lo = y_lo.float()
        X_hi = X_hi.float()
        y_hi = y_hi.float()

        if X_lo.ndim == 1:
            X_lo = X_lo.unsqueeze(-1)
        if X_hi.ndim == 1:
            X_hi = X_hi.unsqueeze(-1)

        self._x_lo = X_lo
        self._y_lo = y_lo

        # Compute low-fidelity predictions at high-fidelity locations
        with torch.no_grad():
            mean_lo_at_hi, _ = _gp_predict(
                X_lo, y_lo, X_hi, self.kernel_lo, self.noise_lo
            )

        # Residuals
        rho_val = self.rho.detach()
        self._y_delta = y_hi - rho_val * mean_lo_at_hi
        self._x_hi = X_hi

        return self

    def predict_uq(self, X_new: torch.Tensor, fidelity: str = "high") -> UQResult:
        """Predict with uncertainty quantification."""
        X_new = X_new.float()
        if X_new.ndim == 1:
            X_new = X_new.unsqueeze(-1)

        assert self._x_lo is not None and self._y_lo is not None

        mean_lo, var_lo = _gp_predict(
            self._x_lo, self._y_lo, X_new, self.kernel_lo, self.noise_lo
        )

        if fidelity == "low":
            return UQResult(
                mean=mean_lo,
                epistemic_var=var_lo,
                aleatoric_var=torch.full_like(var_lo, self.noise_lo),
                total_var=(var_lo + self.noise_lo).clamp_min(0.0),
                probs=None,
                probs_var=None,
                metadata={"method": "multifidelity_gp", "fidelity": "low"},
            )

        assert self._x_hi is not None and self._y_delta is not None

        mean_delta, var_delta = _gp_predict(
            self._x_hi, self._y_delta, X_new, self.kernel_hi, self.noise_hi
        )

        rho_val = self.rho.detach()
        mean_hi = rho_val * mean_lo + mean_delta
        var_hi = rho_val**2 * var_lo + var_delta

        return UQResult(
            mean=mean_hi,
            epistemic_var=var_hi,
            aleatoric_var=torch.full_like(var_hi, self.noise_hi),
            total_var=(var_hi + self.noise_hi).clamp_min(0.0),
            probs=None,
            probs_var=None,
            metadata={"method": "multifidelity_gp", "fidelity": "high"},
        )

    def optimize(
        self,
        X_lo: torch.Tensor,
        y_lo: torch.Tensor,
        X_hi: torch.Tensor,
        y_hi: torch.Tensor,
        n_iter: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Optimize hyperparameters via marginal likelihood."""
        X_lo = X_lo.float()
        y_lo = y_lo.float()
        X_hi = X_hi.float()
        y_hi = y_hi.float()

        if X_lo.ndim == 1:
            X_lo = X_lo.unsqueeze(-1)
        if X_hi.ndim == 1:
            X_hi = X_hi.unsqueeze(-1)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses: list[float] = []

        for _ in range(n_iter):
            optimizer.zero_grad()

            # Low-fidelity marginal likelihood
            lml_lo = _log_marginal_likelihood(X_lo, y_lo, self.kernel_lo, self.noise_lo)

            # Compute residuals with current rho
            mean_lo_at_hi, _ = _gp_predict(
                X_lo, y_lo, X_hi, self.kernel_lo, self.noise_lo
            )
            residuals = y_hi - self.rho * mean_lo_at_hi

            # High-fidelity (delta) marginal likelihood
            lml_hi = _log_marginal_likelihood(
                X_hi, residuals, self.kernel_hi, self.noise_hi
            )

            loss = -(lml_lo + lml_hi)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Refit with optimized parameters
        self.fit(X_lo, y_lo, X_hi, y_hi)
        return losses
