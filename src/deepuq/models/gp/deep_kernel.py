"""Deep kernel learning GP regression."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from deepuq.types import UQResult

from .utils import stable_cholesky


class _FeatureExtractor(nn.Module):
    """Small MLP feature extractor used by deep kernel GP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        feature_dim: int,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, feature_dim]
        layers = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepKernelGaussianProcessRegressor:
    """Regression-only deep kernel GP with end-to-end marginal-likelihood training."""

    def __init__(
        self,
        feature_dim: int = 16,
        hidden_dims: tuple[int, int] = (64, 64),
        epochs: int = 300,
        lr: float = 1e-3,
        noise: float = 1e-3,
        jitter: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        verbose: bool = False,
    ) -> None:
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.noise = noise
        self.jitter = jitter
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self.feature_extractor: _FeatureExtractor | None = None
        self._x_train: torch.Tensor | None = None
        self._z_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._chol: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._lengthscale: torch.Tensor | None = None
        self._outputscale: torch.Tensor | None = None
        self._noise: torch.Tensor | None = None

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def _rbf(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        lengthscale: torch.Tensor,
        outputscale: torch.Tensor,
    ) -> torch.Tensor:
        z1_scaled = z1 / lengthscale
        z2_scaled = z2 / lengthscale
        z1_sq = (z1_scaled**2).sum(dim=-1, keepdim=True)
        z2_sq = (z2_scaled**2).sum(dim=-1).unsqueeze(0)
        sq = z1_sq + z2_sq - 2.0 * z1_scaled @ z2_scaled.t()
        return outputscale * torch.exp(-0.5 * sq)

    def fit(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> DeepKernelGaussianProcessRegressor:
        """Fit DKL-GP on regression targets."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("DeepKernelGaussianProcessRegressor is regression-only.")

        self.feature_extractor = _FeatureExtractor(
            input_dim=x.shape[1],
            hidden_dims=self.hidden_dims,
            feature_dim=self.feature_dim,
        ).to(device=x.device, dtype=x.dtype)

        log_lengthscale = nn.Parameter(
            torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype))
        )
        log_outputscale = nn.Parameter(
            torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype))
        )
        log_noise = nn.Parameter(
            torch.log(torch.tensor(self.noise, device=x.device, dtype=x.dtype))
        )

        params = [
            *self.feature_extractor.parameters(),
            log_lengthscale,
            log_outputscale,
            log_noise,
        ]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        n = x.shape[0]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)

        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            z = self.feature_extractor(x)
            lengthscale = torch.exp(log_lengthscale).clamp_min(1e-6)
            outputscale = torch.exp(log_outputscale).clamp_min(1e-6)
            noise = torch.exp(log_noise).clamp_min(1e-8)

            K = self._rbf(z, z, lengthscale, outputscale)
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

            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(
                    f"[DKL-GP] epoch {epoch + 1:04d}/{self.epochs} nll={nll.item():.4f}"
                )

        with torch.no_grad():
            z = self.feature_extractor(x)
            lengthscale = torch.exp(log_lengthscale).clamp_min(1e-6)
            outputscale = torch.exp(log_outputscale).clamp_min(1e-6)
            noise = torch.exp(log_noise).clamp_min(1e-8)

            K = self._rbf(z, z, lengthscale, outputscale)
            K = K + (noise + self.jitter) * eye
            chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)
            alpha = torch.cholesky_solve(y, chol)

        self._x_train = x
        self._z_train = z.detach()
        self._y_train = y
        self._chol = chol.detach()
        self._alpha = alpha.detach()
        self._lengthscale = lengthscale.detach()
        self._outputscale = outputscale.detach()
        self._noise = noise.detach()
        return self

    def _check_fit(self) -> None:
        if (
            self.feature_extractor is None
            or self._x_train is None
            or self._z_train is None
            or self._chol is None
            or self._alpha is None
            or self._lengthscale is None
            or self._outputscale is None
            or self._noise is None
        ):
            raise RuntimeError("Model must be fit before prediction.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict posterior mean and variance/covariance."""
        self._check_fit()
        x_star = self._prepare(x_star)
        if x_star.ndim != 2:
            raise ValueError("x_star must have shape [N*, D].")

        assert (
            self.feature_extractor is not None
            and self._z_train is not None
            and self._chol is not None
            and self._alpha is not None
            and self._lengthscale is not None
            and self._outputscale is not None
            and self._noise is not None
        )

        with torch.no_grad():
            z_star = self.feature_extractor(x_star)

        K_xs = self._rbf(z_star, self._z_train, self._lengthscale, self._outputscale)
        mean = K_xs @ self._alpha
        K_train_star = K_xs.t()
        solved = torch.cholesky_solve(K_train_star, self._chol)

        if return_cov:
            K_ss = self._rbf(z_star, z_star, self._lengthscale, self._outputscale)
            cov = K_ss - K_xs @ solved
            if include_noise:
                cov = cov + self._noise * torch.eye(
                    x_star.shape[0],
                    device=x_star.device,
                    dtype=x_star.dtype,
                )
            cov = 0.5 * (cov + cov.t())
            return mean.squeeze(-1), cov

        K_ss_diag = self._rbf(
            z_star,
            z_star,
            self._lengthscale,
            self._outputscale,
        ).diag()
        var = K_ss_diag - (K_xs * solved.t()).sum(dim=1)
        if include_noise:
            var = var + self._noise
        var = var.clamp_min(1e-10)
        return mean.squeeze(-1), var

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ output for deep kernel GP regression."""
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
                "method": "deep_kernel_gp",
                "feature_dim": self.feature_dim,
            },
        )
