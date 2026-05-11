from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from ._base import _NativeLaplaceBase, _ensure_iterable_train_loader


class _SimpleDiagonalLaplace(_NativeLaplaceBase):
    """Diagonal Laplace approximation with optional last-layer restriction."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "regression",
        subset_of_weights: str = "last_layer",
        damping: float = 1e-6,
    ) -> None:
        super().__init__(
            model=model,
            likelihood=likelihood,
            subset_of_weights=subset_of_weights,
            damping=damping,
        )
        self.posterior_precision_diag: torch.Tensor | None = None
        self.posterior_variance_diag: torch.Tensor | None = None
        self.hessian_diag: torch.Tensor | None = None

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _SimpleDiagonalLaplace:
        _ensure_iterable_train_loader(train_loader)
        self.model.eval()

        params = self._parameter_modules
        ggn_diag = torch.zeros(self._param_dim, device=self.device)
        residual_sum_squares = 0.0
        count_outputs = 0
        n_data = 0

        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Each batch must be a tuple of (inputs, targets).")
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            if targets.dim() < outputs.dim():
                targets = targets.unsqueeze(-1)

            outputs_flat = outputs.reshape(outputs.shape[0], -1)
            n_batch = outputs_flat.shape[0]
            n_out = outputs_flat.shape[1]
            n_data += n_batch

            if self.likelihood == "regression":
                residual_sum_squares += torch.sum(
                    (outputs.detach() - targets.detach()) ** 2
                ).item()
                count_outputs += targets.numel()

            for i in range(n_batch):
                self.model.zero_grad(set_to_none=True)
                f_i = self.model(inputs[i : i + 1])
                f_i_flat = f_i.reshape(-1)
                for j in range(n_out):
                    grads = torch.autograd.grad(
                        f_i_flat[j], params, retain_graph=(j < n_out - 1),
                        create_graph=False,
                    )
                    j_vec = torch.cat([g.detach().reshape(-1) for g in grads])
                    ggn_diag += j_vec.pow(2)

        if n_data == 0:
            raise ValueError("train_loader produced zero batches.")

        param_vector = parameters_to_vector(params).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        # sigma_noise = 1.0 by default, matching laplace-torch convention.
        # The GGN H = sum(J^T J) and P = (1/sigma^2)*H + prior = H + prior.
        sigma_sq = 1.0

        hessian_diag = (1.0 / sigma_sq) * ggn_diag
        self.hessian_diag = hessian_diag

        self.posterior_precision_diag = hessian_diag + prior_tensor + self.damping
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag.clamp_min(
            1e-12
        )
        return self

    def optimize_prior_precision(self, value: float = 1.0) -> None:
        if self.hessian_diag is None:
            raise RuntimeError("Call fit() before optimising the prior precision.")

        prior_tensor = torch.full_like(self.hessian_diag, float(value))
        self.prior_precision = prior_tensor
        self.posterior_precision_diag = self.hessian_diag + prior_tensor + self.damping
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag.clamp_min(
            1e-12
        )

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.posterior_variance_diag is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        # Use GLM (linearized) predictive for proper OOD uncertainty divergence
        posterior_cov = torch.diag(self.posterior_variance_diag)
        return self._glm_predictive(x, posterior_cov)


class _EmpiricalFisherDiagonalLaplace(_SimpleDiagonalLaplace):
    """Fisher-diagonal variant using the same GGN diagonal as diag."""

    pass


class _LowRankDiagonalLaplace(_NativeLaplaceBase):
    """Low-rank plus diagonal precision approximation."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "regression",
        subset_of_weights: str = "last_layer",
        lowrank_rank: int = 20,
        damping: float = 1e-6,
    ) -> None:
        super().__init__(
            model=model,
            likelihood=likelihood,
            subset_of_weights=subset_of_weights,
            damping=damping,
        )
        self.lowrank_rank = int(max(lowrank_rank, 0))
        self.posterior_precision_diag: torch.Tensor | None = None
        self.posterior_variance_diag: torch.Tensor | None = None
        self.lowrank_u: torch.Tensor | None = None
        self.lowrank_lam: torch.Tensor | None = None

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _LowRankDiagonalLaplace:
        _ensure_iterable_train_loader(train_loader)
        self.model.eval()

        params = self._parameter_modules
        ggn_diag = torch.zeros(self._param_dim, device=self.device)
        jacobian_rows: list[torch.Tensor] = []
        residual_sum_squares = 0.0
        count_outputs = 0
        n_data = 0

        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Each batch must be a tuple of (inputs, targets).")
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            if targets.dim() < outputs.dim():
                targets = targets.unsqueeze(-1)

            outputs_flat = outputs.reshape(outputs.shape[0], -1)
            n_batch = outputs_flat.shape[0]
            n_out = outputs_flat.shape[1]
            n_data += n_batch

            if self.likelihood == "regression":
                residual_sum_squares += torch.sum(
                    (outputs.detach() - targets.detach()) ** 2
                ).item()
                count_outputs += targets.numel()

            for i in range(n_batch):
                self.model.zero_grad(set_to_none=True)
                f_i = self.model(inputs[i : i + 1])
                f_i_flat = f_i.reshape(-1)
                for j in range(n_out):
                    grads = torch.autograd.grad(
                        f_i_flat[j], params, retain_graph=(j < n_out - 1),
                        create_graph=False,
                    )
                    j_vec = torch.cat([g.detach().reshape(-1) for g in grads])
                    ggn_diag += j_vec.pow(2)
                    jacobian_rows.append(j_vec)

        if n_data == 0:
            raise ValueError("train_loader produced zero batches.")

        param_vector = parameters_to_vector(params).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        # sigma_noise = 1.0 by default, matching laplace-torch convention.
        # The GGN H = sum(J^T J) and P = (1/sigma^2)*H + prior = H + prior.
        sigma_sq = 1.0

        diag_total = (1.0 / sigma_sq) * ggn_diag
        jac_matrix = torch.stack(jacobian_rows, dim=0) / (sigma_sq ** 0.5)

        rank_cap = min(self.lowrank_rank, jac_matrix.shape[0], jac_matrix.shape[1])
        if rank_cap <= 0:
            self.lowrank_u = None
            self.lowrank_lam = None
            diag_residual = diag_total
        else:
            _, singular_vals, v_t = torch.linalg.svd(jac_matrix, full_matrices=False)
            lam = singular_vals[:rank_cap].pow(2)
            keep = lam > 1e-12

            if keep.any():
                self.lowrank_u = v_t[:rank_cap, :].transpose(0, 1)[:, keep]
                self.lowrank_lam = lam[keep]
                diag_lowrank = self.lowrank_u.pow(2).matmul(self.lowrank_lam)
                diag_residual = (diag_total - diag_lowrank).clamp_min(0.0)
            else:
                self.lowrank_u = None
                self.lowrank_lam = None
                diag_residual = diag_total

        self.posterior_precision_diag = prior_tensor + diag_residual + self.damping
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag.clamp_min(
            1e-12
        )
        return self

    def _sample_lowrank_noise(self, n_samples: int) -> torch.Tensor:
        assert self.posterior_precision_diag is not None
        assert self.mean_vector is not None

        d = self.posterior_precision_diag.clamp_min(1e-12)
        inv_sqrt_d = d.rsqrt()

        if (
            self.lowrank_u is None
            or self.lowrank_lam is None
            or self.lowrank_lam.numel() == 0
        ):
            z = torch.randn(n_samples, self._param_dim, device=self.device)
            return z * inv_sqrt_d.unsqueeze(0)

        u_scaled = self.lowrank_u * torch.sqrt(self.lowrank_lam).unsqueeze(0)
        b = inv_sqrt_d.unsqueeze(1) * u_scaled

        if b.numel() == 0 or b.shape[1] == 0:
            z = torch.randn(n_samples, self._param_dim, device=self.device)
            return z * inv_sqrt_d.unsqueeze(0)

        u_b, singular_vals, _ = torch.linalg.svd(b, full_matrices=False)
        coeff = 1.0 - 1.0 / torch.sqrt(1.0 + singular_vals.pow(2))

        z = torch.randn(n_samples, self._param_dim, device=self.device)
        proj = z @ u_b
        adjusted = z - (proj * coeff.unsqueeze(0)) @ u_b.transpose(0, 1)
        return cast(torch.Tensor, adjusted * inv_sqrt_d.unsqueeze(0))

    def _posterior_covariance(self) -> torch.Tensor:
        """Compute posterior covariance for low-rank + diagonal precision."""
        # Precision = diag(d) + U @ diag(lam) @ U^T
        # Use Woodbury: Sigma = D^{-1} - D^{-1} U (lam^{-1} + U^T D^{-1} U)^{-1} U^T D^{-1}
        d_inv = 1.0 / self.posterior_precision_diag.clamp_min(1e-12)

        if self.lowrank_u is None or self.lowrank_lam is None or self.lowrank_lam.numel() == 0:
            return torch.diag(d_inv)

        D_inv_U = d_inv.unsqueeze(1) * self.lowrank_u  # (p, r)
        inner = torch.diag(1.0 / self.lowrank_lam) + self.lowrank_u.transpose(0, 1) @ D_inv_U
        inner_inv = torch.linalg.inv(inner)
        correction = D_inv_U @ inner_inv @ D_inv_U.transpose(0, 1)
        return torch.diag(d_inv) - correction

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.posterior_precision_diag is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        posterior_cov = self._posterior_covariance()
        return self._glm_predictive(x, posterior_cov)
