"""Exact Gaussian process regression model."""

from __future__ import annotations

import math

import torch

from deepuq.types import UQResult

from .kernels import Kernel, RBFKernel
from .utils import stable_cholesky


class GaussianProcessRegressor:
    """Exact GP regression using torch tensors."""

    def __init__(
        self,
        kernel: Kernel | None = None,
        noise: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        jitter_base: float = 1e-6,
        jitter_max: float = 1e-2,
    ) -> None:
        self.kernel = kernel or RBFKernel()
        self.noise = float(noise)
        self.device = device
        self.dtype = dtype
        self.jitter_base = jitter_base
        self.jitter_max = jitter_max

        self._x_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._chol: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._effective_jitter: float = 0.0

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> GaussianProcessRegressor:
        """Fit model on training features ``x`` and targets ``y``."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor of shape [N, D].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        k_xx = self.kernel(x, x)
        eye = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        k_xx = k_xx + self.noise * eye
        chol, jitter = stable_cholesky(
            k_xx,
            jitter_base=self.jitter_base,
            jitter_max=self.jitter_max,
        )
        alpha = torch.cholesky_solve(y, chol)

        self._x_train = x
        self._y_train = y
        self._chol = chol
        self._alpha = alpha
        self._effective_jitter = jitter
        return self

    def _check_is_fit(self) -> None:
        if self._x_train is None or self._chol is None or self._alpha is None:
            raise RuntimeError("The model must be fit before making predictions.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        return_var: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior predictive mean and variance/covariance."""
        self._check_is_fit()
        x_star = self._prepare(x_star)
        if x_star.ndim != 2:
            raise ValueError("x_star must be a 2D tensor of shape [N, D].")

        assert (
            self._x_train is not None
            and self._chol is not None
            and self._alpha is not None
        )

        k_xs = self.kernel(self._x_train, x_star)
        pred_mean = k_xs.transpose(0, 1) @ self._alpha
        v = torch.cholesky_solve(k_xs, self._chol)

        if return_cov:
            k_ss = self.kernel(x_star, x_star)
            pred_cov = k_ss - k_xs.transpose(0, 1) @ v
            pred_cov = 0.5 * (pred_cov + pred_cov.t())
            return pred_mean.squeeze(-1), pred_cov

        k_ss_diag = self.kernel(x_star, x_star).diag()
        pred_var = k_ss_diag - (k_xs * v).sum(dim=0)
        if return_var:
            pred_var = pred_var.clamp_min(0.0)
        return pred_mean.squeeze(-1), pred_var

    def posterior_samples(self, x_star: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Draw samples from posterior predictive distribution."""
        mean, cov = self.predict(x_star, return_cov=True)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return dist.rsample((n_samples,))

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ output for exact GP regression."""
        mean, epistemic = self.predict(x_star, return_cov=False, return_var=True)
        aleatoric = torch.full_like(epistemic, self.noise)
        total = (epistemic + aleatoric).clamp_min(0.0)
        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=total,
            probs=None,
            probs_var=None,
            metadata={
                "method": "exact_gp",
                "noise": self.noise,
                "effective_jitter": self._effective_jitter,
            },
        )

    def log_marginal_likelihood(self) -> float:
        """Return the exact GP log marginal likelihood for fitted data."""
        self._check_is_fit()
        assert (
            self._y_train is not None
            and self._alpha is not None
            and self._chol is not None
        )
        data_fit = -0.5 * torch.matmul(self._y_train.T, self._alpha)
        log_det = -torch.log(torch.diagonal(self._chol)).sum()
        constant = -0.5 * self._y_train.shape[0] * math.log(2.0 * math.pi)
        return float((data_fit + log_det).item() + constant)
