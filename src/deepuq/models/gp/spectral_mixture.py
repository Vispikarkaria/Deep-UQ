"""Spectral mixture Gaussian process regression."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from deepuq.types import UQResult

from .utils import stable_cholesky


class SpectralMixtureGaussianProcessRegressor:
    """Exact GP regression with a trainable spectral mixture kernel."""

    def __init__(
        self,
        num_mixtures: int = 4,
        opt_steps: int = 300,
        lr: float = 3e-2,
        noise: float = 1e-3,
        jitter: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        verbose: bool = False,
    ) -> None:
        self.num_mixtures = num_mixtures
        self.opt_steps = opt_steps
        self.lr = lr
        self.noise = noise
        self.jitter = jitter
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self._x_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._chol: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._weights: Optional[torch.Tensor] = None
        self._means: Optional[torch.Tensor] = None
        self._scales: Optional[torch.Tensor] = None
        self._noise: Optional[torch.Tensor] = None

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def _kernel(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        tau = x1[:, None, :] - x2[None, :, :]
        q, d = means.shape
        tau_q = tau.unsqueeze(0)  # [Q, N, M, D]
        means_q = means.view(q, 1, 1, d)
        scales_q = scales.view(q, 1, 1, d)

        exp_term = torch.exp(-2.0 * (math.pi**2) * (tau_q**2) * scales_q).prod(dim=-1)
        cos_term = torch.cos(2.0 * math.pi * tau_q * means_q).prod(dim=-1)
        return (weights.view(q, 1, 1) * exp_term * cos_term).sum(dim=0)

    def _init_parameters(
        self, x: torch.Tensor
    ) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
        d = x.shape[1]
        span = (x.max(dim=0).values - x.min(dim=0).values).clamp_min(1e-3)
        inv_span = 1.0 / span

        log_weights = nn.Parameter(
            torch.log(
                torch.full(
                    (self.num_mixtures,),
                    1.0 / self.num_mixtures,
                    device=x.device,
                    dtype=x.dtype,
                )
            )
        )
        raw_means = nn.Parameter(
            torch.rand(self.num_mixtures, d, device=x.device, dtype=x.dtype)
            * inv_span.view(1, -1)
        )
        log_scales = nn.Parameter(
            torch.log(0.5 * inv_span.view(1, -1).repeat(self.num_mixtures, 1))
        )
        log_noise = nn.Parameter(
            torch.log(torch.tensor(self.noise, device=x.device, dtype=x.dtype))
        )
        return log_weights, raw_means, log_scales, log_noise

    def fit(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> "SpectralMixtureGaussianProcessRegressor":
        """Fit spectral mixture GP by maximizing log marginal likelihood."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        log_weights, raw_means, log_scales, log_noise = self._init_parameters(x)
        params = [log_weights, raw_means, log_scales, log_noise]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        n = x.shape[0]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)

        for step in range(self.opt_steps):
            optimizer.zero_grad(set_to_none=True)
            weights = torch.softmax(log_weights, dim=0)
            means = torch.abs(raw_means)
            scales = torch.exp(log_scales).clamp_min(1e-8)
            noise = torch.exp(log_noise).clamp_min(1e-8)

            K = self._kernel(x, x, weights, means, scales)
            K = K + (noise + self.jitter) * eye
            chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)
            alpha = torch.cholesky_solve(y, chol)

            nll = (
                0.5 * (y.t() @ alpha)
                + torch.log(torch.diagonal(chol)).sum()
                + 0.5 * n * math.log(2.0 * math.pi)
            )
            nll.squeeze().backward()
            optimizer.step()

            if self.verbose and (step + 1) % max(1, self.opt_steps // 10) == 0:
                print(
                    f"[SMGP] step {step + 1:04d}/{self.opt_steps} nll={nll.item():.4f}"
                )

        with torch.no_grad():
            weights = torch.softmax(log_weights, dim=0)
            means = torch.abs(raw_means)
            scales = torch.exp(log_scales).clamp_min(1e-8)
            noise = torch.exp(log_noise).clamp_min(1e-8)
            K = self._kernel(x, x, weights, means, scales)
            K = K + (noise + self.jitter) * eye
            chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)
            alpha = torch.cholesky_solve(y, chol)

        self._x_train = x
        self._y_train = y
        self._weights = weights.detach()
        self._means = means.detach()
        self._scales = scales.detach()
        self._noise = noise.detach()
        self._chol = chol.detach()
        self._alpha = alpha.detach()
        return self

    def _check_fit(self) -> None:
        if (
            self._x_train is None
            or self._weights is None
            or self._means is None
            or self._scales is None
            or self._noise is None
            or self._chol is None
            or self._alpha is None
        ):
            raise RuntimeError("Model must be fit before prediction.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict posterior mean and variance/covariance for test points."""
        self._check_fit()
        x_star = self._prepare(x_star)
        if x_star.ndim != 2:
            raise ValueError("x_star must have shape [N*, D].")

        assert (
            self._x_train is not None
            and self._weights is not None
            and self._means is not None
            and self._scales is not None
            and self._noise is not None
            and self._chol is not None
            and self._alpha is not None
        )

        K_xs = self._kernel(
            x_star, self._x_train, self._weights, self._means, self._scales
        )
        mean = K_xs @ self._alpha

        K_train_star = K_xs.t()
        solved = torch.cholesky_solve(K_train_star, self._chol)

        if return_cov:
            K_ss = self._kernel(
                x_star, x_star, self._weights, self._means, self._scales
            )
            cov = K_ss - K_xs @ solved
            if include_noise:
                cov = cov + self._noise * torch.eye(
                    x_star.shape[0],
                    device=x_star.device,
                    dtype=x_star.dtype,
                )
            cov = 0.5 * (cov + cov.t())
            return mean.squeeze(-1), cov

        K_ss_diag = self._kernel(
            x_star,
            x_star,
            self._weights,
            self._means,
            self._scales,
        ).diag()
        var = K_ss_diag - (K_xs * solved.t()).sum(dim=1)
        if include_noise:
            var = var + self._noise
        var = var.clamp_min(1e-10)
        return mean.squeeze(-1), var

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ output for spectral mixture GP regression."""
        mean, total = self.predict(x_star, return_cov=False, include_noise=True)
        _, epistemic = self.predict(x_star, return_cov=False, include_noise=False)
        aleatoric = (total - epistemic).clamp_min(0.0)
        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=total,
            probs=None,
            probs_var=None,
            metadata={
                "method": "spectral_mixture_gp",
                "num_mixtures": self.num_mixtures,
            },
        )
