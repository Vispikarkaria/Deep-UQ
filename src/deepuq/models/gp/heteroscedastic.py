"""Heteroscedastic Gaussian process regression."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from deepuq.types import UQResult

from .exact_regression import GaussianProcessRegressor
from .kernels import Kernel, RBFKernel


class HeteroscedasticGaussianProcessRegressor:
    """Two-stage alternating GP that models input-dependent observation noise."""

    def __init__(
        self,
        mean_kernel: Optional[Kernel] = None,
        noise_kernel: Optional[Kernel] = None,
        num_alternations: int = 6,
        mean_noise: float = 1e-3,
        noise_floor: float = 1e-5,
        residual_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.mean_kernel = mean_kernel or RBFKernel(lengthscale=1.0, outputscale=1.0)
        self.noise_kernel = noise_kernel or RBFKernel(lengthscale=1.0, outputscale=0.5)
        self.num_alternations = num_alternations
        self.mean_noise = mean_noise
        self.noise_floor = noise_floor
        self.residual_eps = residual_eps
        self.device = device
        self.dtype = dtype

        self._mean_gp: Optional[GaussianProcessRegressor] = None
        self._noise_gp: Optional[GaussianProcessRegressor] = None

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> "HeteroscedasticGaussianProcessRegressor":
        """Fit mean and log-noise GPs via alternating residual updates."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        noise_level = max(float(y.var().item()) * 0.05, self.mean_noise)
        mean_gp = GaussianProcessRegressor(
            kernel=self.mean_kernel,
            noise=noise_level,
            device=self.device,
            dtype=self.dtype,
        )
        noise_gp = GaussianProcessRegressor(
            kernel=self.noise_kernel,
            noise=1e-3,
            device=self.device,
            dtype=self.dtype,
        )

        for _ in range(self.num_alternations):
            mean_gp.noise = noise_level
            mean_gp.fit(x, y)
            with torch.no_grad():
                mean_train, _ = mean_gp.predict(x, return_cov=False, return_var=True)
                resid2 = (y.squeeze(-1) - mean_train).pow(2)
                log_resid = torch.log(resid2 + self.residual_eps)
                noise_gp.fit(x, log_resid.unsqueeze(-1))
                pred_log_noise, _ = noise_gp.predict(
                    x,
                    return_cov=False,
                    return_var=True,
                )
                noise_level = max(
                    float(torch.exp(pred_log_noise).mean().item()),
                    self.noise_floor,
                )

        self._mean_gp = mean_gp
        self._noise_gp = noise_gp
        return self

    def _check_fit(self) -> None:
        if self._mean_gp is None or self._noise_gp is None:
            raise RuntimeError("Model must be fit before prediction.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance/covariance at test inputs."""
        self._check_fit()
        assert self._mean_gp is not None and self._noise_gp is not None
        x_star = self._prepare(x_star)

        mean, epistemic = self._mean_gp.predict(
            x_star,
            return_cov=return_cov,
            return_var=not return_cov,
        )
        if return_cov:
            if not include_noise:
                return mean, epistemic
            log_noise_mean, _ = self._noise_gp.predict(x_star)
            aleatoric_diag = torch.exp(log_noise_mean).clamp_min(self.noise_floor)
            cov = epistemic + torch.diag(aleatoric_diag)
            cov = 0.5 * (cov + cov.t())
            return mean, cov

        if not include_noise:
            return mean, epistemic

        log_noise_mean, _ = self._noise_gp.predict(
            x_star,
            return_cov=False,
            return_var=True,
        )
        aleatoric = torch.exp(log_noise_mean).clamp_min(self.noise_floor)
        total = (epistemic + aleatoric).clamp_min(0.0)
        return mean, total

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ fields for heteroscedastic GP regression."""
        self._check_fit()
        assert self._mean_gp is not None and self._noise_gp is not None

        mean, epistemic = self._mean_gp.predict(
            self._prepare(x_star),
            return_cov=False,
            return_var=True,
        )
        log_noise_mean, _ = self._noise_gp.predict(
            self._prepare(x_star),
            return_cov=False,
            return_var=True,
        )
        aleatoric = torch.exp(log_noise_mean).clamp_min(self.noise_floor)
        total = (epistemic + aleatoric).clamp_min(0.0)

        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=total,
            probs=None,
            probs_var=None,
            metadata={"method": "heteroscedastic_gp"},
        )
