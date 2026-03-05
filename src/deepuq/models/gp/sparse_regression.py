"""Sparse variational Gaussian process regression."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn, optim

from deepuq.types import UQResult

from .kernels import Kernel, RBFKernel
from .utils import stable_cholesky


class SparseGaussianProcessRegressor:
    """Variational inducing-point GP regression.

    Notes
    -----
    - By default the class uses an internal trainable RBF kernel (backward
      compatible behavior).
    - If a fixed ``kernel`` is provided, inducing points and noise are still
      optimized while kernel hyperparameters are treated as fixed.
    """

    def __init__(
        self,
        num_inducing: int = 32,
        learning_rate: float = 5e-2,
        num_iterations: int = 500,
        kernel_jitter: float = 1e-6,
        min_noise: float = 1e-6,
        inducing_points: Optional[torch.Tensor] = None,
        kernel: Optional[Kernel] = None,
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
        self.kernel = kernel
        self._use_fixed_kernel = kernel is not None

        self._fitted = False
        self.elbo_history: list[float] = []

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def _rbf(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        lengthscale: torch.Tensor,
        outputscale: torch.Tensor,
    ) -> torch.Tensor:
        x1_scaled = x1 / lengthscale
        x2_scaled = x2 / lengthscale
        x1_sq = (x1_scaled**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2_scaled**2).sum(dim=-1).unsqueeze(0)
        squared_dist = x1_sq + x2_sq - 2.0 * x1_scaled @ x2_scaled.t()
        return outputscale * torch.exp(-0.5 * squared_dist)

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self._use_fixed_kernel:
            assert self.kernel is not None
            return self.kernel(x1, x2)
        lengthscale = torch.exp(self.log_lengthscale)
        outputscale = torch.exp(self.log_outputscale)
        return self._rbf(x1, x2, lengthscale, outputscale)

    def _kernel_diag(self, x: torch.Tensor) -> torch.Tensor:
        kxx = self._kernel(x, x)
        return kxx.diag()

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
        self.log_noise = nn.Parameter(
            torch.log(torch.tensor(1e-2, device=x.device, dtype=x.dtype))
        )

        if not self._use_fixed_kernel:
            self.log_lengthscale = nn.Parameter(
                torch.log(torch.tensor(0.5, device=x.device, dtype=x.dtype))
            )
            self.log_outputscale = nn.Parameter(
                torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype))
            )

    def _compute_elbo(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        m = self.inducing.shape[0]
        noise = torch.exp(self.log_noise) + self.min_noise

        Kmm = self._kernel(self.inducing, self.inducing)
        Kmm = Kmm + self.kernel_jitter * torch.eye(m, device=x.device, dtype=x.dtype)
        Lmm, _ = stable_cholesky(
            Kmm,
            jitter_base=self.kernel_jitter,
            jitter_max=1e-2,
        )

        Kmn = self._kernel(self.inducing, x)
        Knm = Kmn.transpose(0, 1)
        Kmm_inv_Kmn = torch.cholesky_solve(Kmn, Lmm)
        Qnn = Knm @ Kmm_inv_Kmn

        eye_n = torch.eye(n, device=x.device, dtype=x.dtype)
        A = Qnn + noise * eye_n + self.kernel_jitter * eye_n
        LA, _ = stable_cholesky(
            A,
            jitter_base=self.kernel_jitter,
            jitter_max=1e-2,
        )

        log_det_A = 2.0 * torch.log(torch.diagonal(LA)).sum()
        solve_A_y = torch.cholesky_solve(y, LA)
        data_fit = torch.matmul(y.T, solve_A_y)

        Knn_diag = self._kernel_diag(x)
        trace_term = Knn_diag - (Kmn * Kmm_inv_Kmn).sum(dim=0)
        elbo = (
            -0.5 * (n * math.log(2.0 * math.pi) + log_det_A + data_fit)
            - 0.5 * trace_term.sum() / noise
        )
        return elbo.squeeze()

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "SparseGaussianProcessRegressor":
        """Optimise sparse variational objective and cache posterior state."""
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D tensor of shape [N, D].")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")

        self._initialise_parameters(x)

        params = [self.inducing, self.log_noise]
        if not self._use_fixed_kernel:
            params.extend([self.log_lengthscale, self.log_outputscale])

        optimizer = optim.Adam(params, lr=self.learning_rate)
        self.elbo_history.clear()

        for it in range(self.num_iterations):
            optimizer.zero_grad(set_to_none=True)
            elbo = self._compute_elbo(x, y)
            loss = -elbo
            loss.backward()
            optimizer.step()
            self.elbo_history.append(float(elbo.detach()))
            if self.verbose and (it + 1) % max(1, self.num_iterations // 10) == 0:
                print(
                    f"[SparseGP] Iter {it + 1:04d}/{self.num_iterations}: ELBO={elbo.item():.4f}"
                )

        with torch.no_grad():
            params_cache = self._posterior_cache(x, y)
            self._beta = params_cache["beta"]
            self._Sigma = params_cache["Sigma"]
            self.inducing_points_ = params_cache["inducing"]
            self.noise_ = params_cache["noise"]

        self._fitted = True
        return self

    def _posterior_cache(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        noise = torch.exp(self.log_noise).detach() + self.min_noise

        inducing = self.inducing.detach().clone()
        Kmm = self._kernel(inducing, inducing)
        m = inducing.shape[0]
        Kmm = Kmm + self.kernel_jitter * torch.eye(
            m,
            device=inducing.device,
            dtype=inducing.dtype,
        )
        Lmm, _ = stable_cholesky(
            Kmm,
            jitter_base=self.kernel_jitter,
            jitter_max=1e-2,
        )

        Kmn = self._kernel(inducing, x)
        Knm = Kmn.transpose(0, 1)
        Kmm_inv_Kmn = torch.cholesky_solve(Kmn, Lmm)
        Qnn = Knm @ Kmm_inv_Kmn

        n = x.shape[0]
        eye_n = torch.eye(n, device=x.device, dtype=x.dtype)
        A = Qnn + noise * eye_n + self.kernel_jitter * eye_n
        LA, _ = stable_cholesky(A, jitter_base=self.kernel_jitter, jitter_max=1e-2)

        alpha = torch.cholesky_solve(y, LA)
        beta = torch.cholesky_solve(Kmn @ alpha, Lmm)

        Kmm_inv = torch.cholesky_inverse(Lmm)
        A_inv = torch.cholesky_inverse(LA)
        Sigma = Kmm_inv - Kmm_inv_Kmn @ A_inv @ Kmm_inv_Kmn.transpose(0, 1)

        return {
            "beta": beta.detach(),
            "Sigma": Sigma.detach(),
            "inducing": inducing,
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
        """Predict posterior mean and variance/covariance for test inputs."""
        self._ensure_fitted()
        x_star = self._prepare(x_star)

        inducing = self.inducing_points_
        noise = self.noise_
        K_sm = self._kernel(x_star, inducing)
        mean = K_sm @ self._beta

        if return_cov:
            K_ss = self._kernel(x_star, x_star)
            cov = K_ss - K_sm @ self._Sigma @ K_sm.transpose(0, 1)
            if include_noise:
                cov = cov + noise * torch.eye(
                    x_star.shape[0],
                    device=cov.device,
                    dtype=cov.dtype,
                )
            cov = cov + self.kernel_jitter * torch.eye(
                x_star.shape[0],
                device=cov.device,
                dtype=cov.dtype,
            )
            cov = 0.5 * (cov + cov.t())
            return mean.squeeze(-1), cov

        tmp = K_sm @ self._Sigma
        k_ss_diag = self._kernel(x_star, x_star).diag()
        var = k_ss_diag - (tmp * K_sm).sum(dim=1)
        if include_noise:
            var = var + noise
        var = var.clamp_min(1e-10)
        return mean.squeeze(-1), var

    def posterior_samples(self, x_star: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Draw samples from sparse GP posterior predictive distribution."""
        mean, cov = self.predict(x_star, return_cov=True, include_noise=True)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        return dist.rsample((n_samples,))

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ fields for sparse GP regression."""
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
                "method": "sparse_gp",
                "num_inducing": int(self.inducing_points_.shape[0]),
            },
        )
