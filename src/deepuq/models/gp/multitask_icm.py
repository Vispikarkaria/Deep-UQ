"""Multi-task Gaussian process regression with ICM coregionalization."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from deepuq.types import UQResult

from .kernels import Kernel
from .utils import stable_cholesky


class MultiTaskGaussianProcessRegressor:
    """Multi-output GP regression using an intrinsic coregionalization model."""

    def __init__(
        self,
        num_tasks: int,
        kernel: Kernel | None = None,
        lr: float = 5e-2,
        opt_steps: int = 250,
        noise: float = 1e-3,
        jitter: float = 1e-6,
        full_max_points: int = 2500,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        verbose: bool = False,
    ) -> None:
        self.num_tasks = num_tasks
        self.kernel = kernel
        self.lr = lr
        self.opt_steps = opt_steps
        self.noise = noise
        self.jitter = jitter
        self.full_max_points = full_max_points
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        self._x_train: torch.Tensor | None = None
        self._y_train: torch.Tensor | None = None
        self._chol: torch.Tensor | None = None
        self._alpha: torch.Tensor | None = None
        self._B: torch.Tensor | None = None

        self._fitted = False

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def _rbf(
        self, x1: torch.Tensor, x2: torch.Tensor, ls: torch.Tensor, os: torch.Tensor
    ) -> torch.Tensor:
        x1_scaled = x1 / ls
        x2_scaled = x2 / ls
        x1_sq = (x1_scaled**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2_scaled**2).sum(dim=-1).unsqueeze(0)
        sqdist = x1_sq + x2_sq - 2.0 * x1_scaled @ x2_scaled.t()
        return os * torch.exp(-0.5 * sqdist)

    def _kernel(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        log_lengthscale: torch.Tensor | None = None,
        log_outputscale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.kernel is not None:
            return self.kernel(x1, x2)
        assert log_lengthscale is not None and log_outputscale is not None
        ls = torch.exp(log_lengthscale)
        os = torch.exp(log_outputscale)
        return self._rbf(x1, x2, ls, os)

    def _build_coregionalization(
        self,
        raw_L: torch.Tensor,
        raw_diag: torch.Tensor,
    ) -> torch.Tensor:
        lower = torch.tril(raw_L)
        diag = F.softplus(raw_diag) + 1e-6
        return lower @ lower.t() + torch.diag(diag)

    def fit(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> MultiTaskGaussianProcessRegressor:
        """Fit ICM GP on fully-observed multi-output targets ``y:[N, T]``."""
        x = self._prepare(x)
        y = self._prepare(y)

        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if y.ndim != 2:
            raise ValueError("y must have shape [N, T].")
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        if y.shape[1] != self.num_tasks:
            raise ValueError(
                f"Expected y with {self.num_tasks} tasks, got {y.shape[1]}."
            )

        n = x.shape[0]
        nt = n * self.num_tasks
        if nt > self.full_max_points:
            raise ValueError(
                f"N*T={nt} exceeds full_max_points={self.full_max_points}; "
                "reduce data size or increase full_max_points."
            )

        raw_L = nn.Parameter(torch.eye(self.num_tasks, device=x.device, dtype=x.dtype))
        raw_diag = nn.Parameter(
            torch.zeros(self.num_tasks, device=x.device, dtype=x.dtype)
        )
        log_noise = nn.Parameter(
            torch.log(torch.tensor(self.noise, device=x.device, dtype=x.dtype))
        )

        params = [raw_L, raw_diag, log_noise]
        if self.kernel is None:
            log_lengthscale = nn.Parameter(
                torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype))
            )
            log_outputscale = nn.Parameter(
                torch.log(torch.tensor(1.0, device=x.device, dtype=x.dtype))
            )
            params.extend([log_lengthscale, log_outputscale])
        else:
            log_lengthscale = None
            log_outputscale = None

        optimizer = torch.optim.Adam(params, lr=self.lr)

        y_vec = y.transpose(0, 1).reshape(-1, 1)

        for step in range(self.opt_steps):
            optimizer.zero_grad(set_to_none=True)
            B = self._build_coregionalization(raw_L, raw_diag)
            Kx = self._kernel(x, x, log_lengthscale, log_outputscale)
            Kx = Kx + self.jitter * torch.eye(n, device=x.device, dtype=x.dtype)

            K = torch.kron(B, Kx)
            noise_val = torch.exp(log_noise)
            K = K + noise_val * torch.eye(nt, device=x.device, dtype=x.dtype)
            chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)
            alpha = torch.cholesky_solve(y_vec, chol)

            nll = (
                0.5 * (y_vec.t() @ alpha)
                + torch.log(torch.diagonal(chol)).sum()
                + 0.5 * nt * math.log(2.0 * math.pi)
            )
            nll.squeeze().backward()
            optimizer.step()

            if self.verbose and (step + 1) % max(1, self.opt_steps // 10) == 0:
                print(
                    f"[MultiTaskGP] step {step + 1:04d}/{self.opt_steps} nll={nll.item():.4f}"
                )

        with torch.no_grad():
            B = self._build_coregionalization(raw_L, raw_diag)
            Kx = self._kernel(x, x, log_lengthscale, log_outputscale)
            Kx = Kx + self.jitter * torch.eye(n, device=x.device, dtype=x.dtype)
            noise_val = torch.exp(log_noise)
            K = torch.kron(B, Kx) + noise_val * torch.eye(
                nt, device=x.device, dtype=x.dtype
            )
            chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)
            alpha = torch.cholesky_solve(y_vec, chol)

        self._x_train = x
        self._y_train = y
        self._B = B.detach()
        self._noise = noise_val.detach()
        self._chol = chol.detach()
        self._alpha = alpha.detach()
        self._log_lengthscale = (
            log_lengthscale.detach() if log_lengthscale is not None else None
        )
        self._log_outputscale = (
            log_outputscale.detach() if log_outputscale is not None else None
        )
        self._fitted = True
        return self

    def _check_fit(self) -> None:
        if (
            not self._fitted
            or self._x_train is None
            or self._B is None
            or self._chol is None
            or self._alpha is None
        ):
            raise RuntimeError("Model must be fit before prediction.")

    def _kernel_predict(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self._kernel(x1, x2, self._log_lengthscale, self._log_outputscale)

    def predict(
        self,
        x_star: torch.Tensor,
        return_cov: bool = False,
        include_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-task means and variances/covariances at ``x_star``."""
        self._check_fit()
        x_star = self._prepare(x_star)
        if x_star.ndim != 2:
            raise ValueError("x_star must have shape [N*, D].")

        assert (
            self._x_train is not None
            and self._B is not None
            and self._chol is not None
            and self._alpha is not None
        )

        self._x_train.shape[0]
        n_test = x_star.shape[0]

        Ksx = self._kernel_predict(x_star, self._x_train).contiguous()  # [N*, N]
        Kxs = Ksx.t().contiguous()  # [N, N*]

        K_star_train = torch.kron(self._B, Ksx)  # [T*N*, T*N]
        mean_vec = K_star_train @ self._alpha
        mean = mean_vec.reshape(self.num_tasks, n_test).transpose(0, 1)

        if return_cov:
            Kss = self._kernel_predict(x_star, x_star)
            prior = torch.kron(self._B, Kss)
            K_train_star = torch.kron(self._B, Kxs)
            reduction = K_star_train @ torch.cholesky_solve(K_train_star, self._chol)
            cov = prior - reduction
            if include_noise:
                cov = cov + self._noise * torch.eye(
                    cov.shape[0],
                    device=cov.device,
                    dtype=cov.dtype,
                )
            cov = 0.5 * (cov + cov.t())
            return mean, cov

        K_train_star = torch.kron(self._B, Kxs)
        solved = torch.cholesky_solve(K_train_star, self._chol)
        reduction_diag = (K_train_star * solved).sum(dim=0)

        Kss_diag = self._kernel_predict(x_star, x_star).diag()
        prior_diag = (
            torch.diagonal(self._B).unsqueeze(1) * Kss_diag.unsqueeze(0)
        ).reshape(-1)
        var_vec = prior_diag - reduction_diag
        if include_noise:
            var_vec = var_vec + self._noise
        var = var_vec.reshape(self.num_tasks, n_test).transpose(0, 1).clamp_min(1e-10)
        return mean, var

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized UQ fields for multi-task GP regression."""
        mean, total = self.predict(x_star, return_cov=False, include_noise=True)
        _, epistemic = self.predict(x_star, return_cov=False, include_noise=False)
        aleatoric = (total - epistemic).clamp_min(0.0)
        assert self._B is not None
        corr = self._B / (
            torch.sqrt(torch.diagonal(self._B).unsqueeze(0))
            * torch.sqrt(torch.diagonal(self._B).unsqueeze(1))
            + 1e-10
        )
        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=total,
            probs=None,
            probs_var=None,
            metadata={
                "method": "multitask_icm_gp",
                "num_tasks": self.num_tasks,
                "task_correlation": corr.detach().cpu(),
            },
        )
