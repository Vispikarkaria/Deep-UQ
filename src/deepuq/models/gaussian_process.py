"""
Simple Gaussian Process regression utilities built on PyTorch tensors.

The implementation is intentionally lightweight so the tutorial notebooks can
run without additional dependencies.  The provided ``GaussianProcessRegressor``
supports closed-form posterior inference under a zero-mean GP prior with an RBF
kernel.  The API mirrors scikit-learn's GaussianProcessRegressor where it
makes sense, but keeps tensors on the device chosen by the user.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class RBFKernel:
    """Squared exponential (RBF) kernel.

    Parameters
    ----------
    lengthscale:
        Characteristic lengthscale for the kernel. Larger values lead to
        smoother functions.
    outputscale:
        Overall variance scale (sometimes called signal variance).
    jitter:
        Numerical stabiliser added to the diagonal whenever the kernel is used
        to form a covariance matrix.
    """

    lengthscale: float = 1.0
    outputscale: float = 1.0
    jitter: float = 1e-6

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1 x2^T
        x1_sq = (x1**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2**2).sum(dim=-1).unsqueeze(0)
        squared_dist = x1_sq + x2_sq - 2.0 * x1 @ x2.t()
        cov = self.outputscale * torch.exp(-0.5 * squared_dist)
        if x1.shape[0] == x2.shape[0] and torch.equal(x1, x2):
            cov = cov + self.jitter * torch.eye(x1.shape[0], device=x1.device, dtype=x1.dtype)
        return cov


class GaussianProcessRegressor:
    """Exact GP regression using torch tensors.

    Parameters
    ----------
    kernel:
        Kernel function mapping two matrices of shape ``[N, D]`` and ``[M, D]``
        to a covariance matrix ``[N, M]``. Defaults to :class:`RBFKernel`.
    noise:
        Observation noise variance. Acts as a lower bound on predictive
        variance and is added to the training covariance diagonal.
    device:
        Optional device to move inputs and cached factors to.
    dtype:
        Optional dtype for computations. Defaults to ``torch.float32``.
    """

    def __init__(
        self,
        kernel: Optional[RBFKernel] = None,
        noise: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.kernel = kernel or RBFKernel()
        self.noise = noise
        self.device = device
        self.dtype = dtype

        self._x_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._chol: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device=self.device, dtype=self.dtype, copy=False)
        return tensor

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianProcessRegressor":
        """Fit the GP to observed inputs ``x`` and targets ``y``."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor of shape [N, D].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        k_xx = self.kernel(x, x)
        noise_mat = (self.noise + self.kernel.jitter) * torch.eye(
            x.shape[0], device=x.device, dtype=x.dtype
        )
        k_xx = k_xx + noise_mat
        chol = torch.linalg.cholesky(k_xx)
        alpha = torch.cholesky_solve(y, chol)

        self._x_train = x
        self._y_train = y
        self._chol = chol
        self._alpha = alpha
        return self

    def _check_is_fit(self) -> None:
        if self._x_train is None or self._chol is None or self._alpha is None:
            raise RuntimeError("The model must be fit before making predictions.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        return_var: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior predictive mean and variance/covariance."""
        self._check_is_fit()
        x_star = self._prepare(x_star)
        if x_star.ndim != 2:
            raise ValueError("x_star must be a 2D tensor of shape [N, D].")

        assert self._x_train is not None and self._chol is not None and self._alpha is not None
        k_xs = self.kernel(self._x_train, x_star)
        pred_mean = k_xs.transpose(0, 1) @ self._alpha  # [N*, 1]
        v = torch.cholesky_solve(k_xs, self._chol)

        if return_cov:
            k_ss = self.kernel(x_star, x_star)
            pred_cov = k_ss - k_xs.transpose(0, 1) @ v
        else:
            k_ss_diag = self.kernel(x_star, x_star).diag()
            pred_cov = k_ss_diag - (k_xs * v).sum(dim=0)
            if return_var:
                pred_cov = pred_cov.clamp_min(0.0)
        return pred_mean.squeeze(-1), pred_cov

    def posterior_samples(self, x_star: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Draw samples from the posterior predictive distribution."""
        mean, cov = self.predict(x_star, return_cov=True)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return dist.rsample((n_samples,))

    def log_marginal_likelihood(self) -> float:
        """Return the log marginal likelihood under the current training data."""
        self._check_is_fit()
        assert self._y_train is not None and self._alpha is not None and self._chol is not None
        data_fit = -0.5 * torch.matmul(self._y_train.T, self._alpha)
        log_det = -torch.log(torch.diagonal(self._chol)).sum()
        constant = -0.5 * self._y_train.shape[0] * math.log(2.0 * math.pi)
        return float((data_fit + log_det).item() + constant)
