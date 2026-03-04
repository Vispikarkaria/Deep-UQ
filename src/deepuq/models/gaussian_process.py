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
from torch import nn, optim

from deepuq.types import UQResult


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

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ fields for exact GP regression."""
        mean, epistemic = self.predict(x_star, return_cov=False, return_var=True)
        aleatoric = torch.full_like(epistemic, float(self.noise))
        total = (epistemic + aleatoric).clamp_min(0.0)
        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=total,
            probs=None,
            probs_var=None,
            metadata={"method": "exact_gp"},
        )

    def log_marginal_likelihood(self) -> float:
        """Return the log marginal likelihood under the current training data."""
        self._check_is_fit()
        assert self._y_train is not None and self._alpha is not None and self._chol is not None
        data_fit = -0.5 * torch.matmul(self._y_train.T, self._alpha)
        log_det = -torch.log(torch.diagonal(self._chol)).sum()
        constant = -0.5 * self._y_train.shape[0] * math.log(2.0 * math.pi)
        return float((data_fit + log_det).item() + constant)


class SparseGaussianProcessRegressor:
    """Variational inducing-point GP regression with an RBF kernel.

    The implementation follows the variational free-energy formulation of
    Titsias (2009) and optimises kernel hyperparameters and inducing locations
    via Adam.  While intentionally lightweight, it scales to hundreds or
    thousands of points with modest numbers of inducing locations.
    """

    def __init__(
        self,
        num_inducing: int = 32,
        learning_rate: float = 5e-2,
        num_iterations: int = 500,
        kernel_jitter: float = 1e-6,
        min_noise: float = 1e-6,
        inducing_points: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        verbose: bool = False,
    ) -> None:
        self.num_inducing = num_inducing
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.kernel_jitter = kernel_jitter
        self.min_noise = min_noise
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self._init_inducing = inducing_points

        self._fitted = False
        self.elbo_history: list[float] = []

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def _rbf(self, x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor, outputscale: torch.Tensor) -> torch.Tensor:
        x1_scaled = x1 / lengthscale
        x2_scaled = x2 / lengthscale
        x1_sq = (x1_scaled ** 2).sum(dim=-1, keepdim=True)
        x2_sq = (x2_scaled ** 2).sum(dim=-1).unsqueeze(0)
        squared_dist = x1_sq + x2_sq - 2.0 * x1_scaled @ x2_scaled.t()
        cov = outputscale * torch.exp(-0.5 * squared_dist)
        return cov

    def _initialise_parameters(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        m = min(self.num_inducing, n)
        if self._init_inducing is not None:
            inducing = self._prepare(self._init_inducing)
            if inducing.ndim != 2:
                raise ValueError("inducing_points must have shape [M, D].")
            inducing = inducing[:m].clone()
        else:
            perm = torch.randperm(n, device=x.device)
            inducing = x[perm[:m]].clone()
        self.inducing = nn.Parameter(inducing)
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(0.5, device=x.device, dtype=x.dtype)))
        self.log_outputscale = nn.Parameter(torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(1e-2, device=x.device, dtype=x.dtype)))

    def _compute_elbo(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        m = self.inducing.shape[0]
        lengthscale = torch.exp(self.log_lengthscale)
        outputscale = torch.exp(self.log_outputscale)
        noise = torch.exp(self.log_noise) + self.min_noise

        Kmm = self._rbf(self.inducing, self.inducing, lengthscale, outputscale)
        Kmm = Kmm + (self.kernel_jitter + noise * 0.0) * torch.eye(m, device=x.device, dtype=x.dtype)
        Lmm = torch.linalg.cholesky(Kmm)

        Kmn = self._rbf(self.inducing, x, lengthscale, outputscale)
        Knm = Kmn.transpose(0, 1)
        Kmm_inv_Kmn = torch.cholesky_solve(Kmn, Lmm)
        Qnn = Knm @ Kmm_inv_Kmn

        eye_n = torch.eye(n, device=x.device, dtype=x.dtype)
        A = Qnn + noise * eye_n
        A = A + self.kernel_jitter * eye_n
        LA = torch.linalg.cholesky(A)

        log_det_A = 2.0 * torch.log(torch.diagonal(LA)).sum()
        solve_A_y = torch.cholesky_solve(y, LA)
        data_fit = torch.matmul(y.T, solve_A_y)

        trace_term = outputscale - (Kmn * Kmm_inv_Kmn).sum(dim=0)
        elbo = -0.5 * (n * math.log(2.0 * math.pi) + log_det_A + data_fit) - 0.5 * trace_term.sum() / noise
        return elbo.squeeze()

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "SparseGaussianProcessRegressor":
        """Optimise the variational objective and cache posterior statistics."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor of shape [N, D].")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        self._initialise_parameters(x)

        optimizer = optim.Adam(
            [self.inducing, self.log_lengthscale, self.log_outputscale, self.log_noise],
            lr=self.learning_rate,
        )

        self.elbo_history.clear()
        for it in range(self.num_iterations):
            optimizer.zero_grad()
            elbo = self._compute_elbo(x, y)
            loss = -elbo
            loss.backward()
            optimizer.step()
            self.elbo_history.append(float(elbo.detach()))
            if self.verbose and (it + 1) % max(1, self.num_iterations // 10) == 0:
                print(f"[SparseGP] Iter {it+1:04d}/{self.num_iterations}: ELBO={elbo.item():.4f}")

        with torch.no_grad():
            params = self._posterior_cache(x, y)
            self._beta = params["beta"]
            self._Sigma = params["Sigma"]
            self.inducing_points_ = params["inducing"]
            self.lengthscale_ = params["lengthscale"]
            self.outputscale_ = params["outputscale"]
            self.noise_ = params["noise"]

        self._fitted = True
        return self

    def _posterior_cache(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        lengthscale = torch.exp(self.log_lengthscale)
        outputscale = torch.exp(self.log_outputscale)
        noise = torch.exp(self.log_noise) + self.min_noise

        inducing = self.inducing.detach().clone()
        Kmm = self._rbf(inducing, inducing, lengthscale, outputscale)
        m = inducing.shape[0]
        Kmm = Kmm + self.kernel_jitter * torch.eye(m, device=inducing.device, dtype=inducing.dtype)
        Lmm = torch.linalg.cholesky(Kmm)

        Kmn = self._rbf(inducing, x, lengthscale, outputscale)
        Knm = Kmn.transpose(0, 1)
        Kmm_inv_Kmn = torch.cholesky_solve(Kmn, Lmm)
        Qnn = Knm @ Kmm_inv_Kmn

        n = x.shape[0]
        eye_n = torch.eye(n, device=x.device, dtype=x.dtype)
        A = Qnn + noise * eye_n + self.kernel_jitter * eye_n
        LA = torch.linalg.cholesky(A)

        alpha = torch.cholesky_solve(y, LA)
        beta = torch.cholesky_solve(Kmn @ alpha, Lmm)  # shape [m, 1]

        Kmm_inv = torch.cholesky_inverse(Lmm)
        A_inv = torch.cholesky_inverse(LA)
        Sigma = Kmm_inv - Kmm_inv_Kmn @ A_inv @ Kmm_inv_Kmn.transpose(0, 1)

        return {
            "beta": beta.detach(),
            "Sigma": Sigma.detach(),
            "inducing": inducing,
            "lengthscale": lengthscale.detach(),
            "outputscale": outputscale.detach(),
            "noise": noise.detach(),
        }

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit before predict.")

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the posterior mean and variance/covariance at new inputs."""
        self._ensure_fitted()
        x_star = self._prepare(x_star)
        inducing = self.inducing_points_
        lengthscale = self.lengthscale_
        outputscale = self.outputscale_
        noise = self.noise_

        K_sm = self._rbf(x_star, inducing, lengthscale, outputscale)
        mean = K_sm @ self._beta
        if return_cov:
            K_ss = self._rbf(x_star, x_star, lengthscale, outputscale)
            cov = K_ss - K_sm @ self._Sigma @ K_sm.transpose(0, 1)
            if include_noise:
                cov = cov + noise * torch.eye(x_star.shape[0], device=cov.device, dtype=cov.dtype)
            cov = cov + self.kernel_jitter * torch.eye(x_star.shape[0], device=cov.device, dtype=cov.dtype)
            return mean.squeeze(-1), cov

        tmp = K_sm @ self._Sigma
        var = outputscale - (tmp * K_sm).sum(dim=1)
        if include_noise:
            var = var + noise
        var = var.clamp_min(1e-10)
        return mean.squeeze(-1), var

    def posterior_samples(self, x_star: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Draw samples from the sparse GP posterior."""
        mean, cov = self.predict(x_star, return_cov=True, include_noise=True)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return dist.rsample((n_samples,))

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ fields for sparse variational GP regression."""
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
            metadata={"method": "sparse_gp", "num_inducing": int(self.inducing_points_.shape[0])},
        )
