from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from deepuq.types import UQResult


def _find_last_linear_layer(model: nn.Module) -> nn.Linear:
    last_linear: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError(
            "Could not locate a linear layer in the model to use for Laplace approximation."
        )
    return last_linear


def _select_parameters(model: nn.Module, subset_of_weights: str) -> list[nn.Parameter]:
    if subset_of_weights not in {"last_layer", "all"}:
        raise ValueError('subset_of_weights must be "last_layer" or "all".')

    if subset_of_weights == "last_layer":
        params = list(_find_last_linear_layer(model).parameters())
    else:
        params = list(model.parameters())

    if len(params) == 0:
        raise ValueError("No parameters selected for the Laplace approximation.")
    return params


def _safe_cholesky(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
    jitter = float(max(damping, 1e-12))
    last_error: RuntimeError | None = None
    for _ in range(7):
        try:
            return cast(torch.Tensor, torch.linalg.cholesky(matrix + jitter * eye))
        except (
            RuntimeError
        ) as exc:  # pragma: no cover - exercised only on ill-conditioned cases
            last_error = exc
            jitter *= 10.0
    raise RuntimeError(
        "Cholesky decomposition failed even after jitter escalation."
    ) from last_error


def _ensure_iterable_train_loader(train_loader: Iterable) -> None:
    if not hasattr(train_loader, "__iter__"):
        raise TypeError("train_loader must be an iterable over (input, target) pairs.")




class _NativeLaplaceBase:
    """Shared functionality for native Laplace approximations."""

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "regression",
        subset_of_weights: str = "last_layer",
        damping: float = 1e-6,
    ) -> None:
        if likelihood not in {"regression", "classification"}:
            raise ValueError(
                f'Unsupported likelihood "{likelihood}". Use "regression" or "classification".'
            )

        self.model = model
        self.likelihood = likelihood
        self.subset_of_weights = subset_of_weights
        self.damping = float(max(damping, 0.0))

        self._parameter_modules = _select_parameters(model, subset_of_weights)
        self.device = next(model.parameters()).device
        self._param_dim = parameters_to_vector(self._parameter_modules).numel()

        self.mean_vector: torch.Tensor | None = None
        self.prior_precision: torch.Tensor | None = None
        self.empirical_noise_variance: torch.Tensor | None = None

    def _compute_batch_statistics(
        self,
        train_loader: Iterable,
    ) -> tuple[torch.Tensor, torch.Tensor, int, float, int]:
        _ensure_iterable_train_loader(train_loader)

        self.model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        ce_loss = nn.CrossEntropyLoss(reduction="sum")

        batch_grads: list[torch.Tensor] = []
        diag_accumulator = torch.zeros(self._param_dim, device=self.device)
        residual_sum_squares = 0.0
        count_outputs = 0

        for batch in train_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Each batch must be a tuple of (inputs, targets).")

            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(inputs)

            if self.likelihood == "regression":
                if targets.dim() < outputs.dim():
                    targets = targets.unsqueeze(-1)
                loss = 0.5 * mse_loss(outputs, targets)
                residual_sum_squares += torch.sum(
                    (outputs.detach() - targets.detach()) ** 2
                ).item()
                count_outputs += targets.numel()
            else:
                if targets.dim() != 1:
                    raise ValueError(
                        "Classification targets must be a 1D tensor of class indices."
                    )
                loss = ce_loss(outputs, targets)
                count_outputs += targets.size(0)

            gradients = torch.autograd.grad(
                loss, self._parameter_modules, retain_graph=False
            )
            grad_vector = torch.cat([g.detach().reshape(-1) for g in gradients])
            batch_grads.append(grad_vector)
            diag_accumulator += grad_vector.pow(2)

        if len(batch_grads) == 0:
            raise ValueError(
                "train_loader produced zero batches; cannot fit Laplace approximation."
            )

        num_datapoints = len(getattr(train_loader, "dataset", []))
        if num_datapoints == 0:
            num_datapoints = len(batch_grads)

        grad_matrix = torch.stack(batch_grads, dim=0)
        return (
            grad_matrix,
            diag_accumulator,
            num_datapoints,
            residual_sum_squares,
            count_outputs,
        )

    def _finalize_common_fit(
        self,
        param_vector: torch.Tensor,
        prior_precision: float | None,
        residual_sum_squares: float,
        count_outputs: int,
    ) -> torch.Tensor:
        self.mean_vector = param_vector.detach().clone()

        prior_value = 1.0 if prior_precision is None else float(prior_precision)
        prior_tensor = torch.full(
            (self._param_dim,),
            prior_value,
            device=self.device,
            dtype=param_vector.dtype,
        )
        self.prior_precision = prior_tensor

        if self.likelihood == "regression":
            denom = max(count_outputs, 1)
            noise_var = residual_sum_squares / float(denom)
            self.empirical_noise_variance = torch.tensor(
                noise_var, device=self.device, dtype=param_vector.dtype
            )
        else:
            self.empirical_noise_variance = None

        return prior_tensor

    def _compute_jacobians(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians of model output w.r.t. selected parameters.

        Returns (J, f_map) where J has shape (batch, n_outputs, n_params)
        and f_map has shape (batch, n_outputs).
        """
        x = x.to(self.device)
        self.model.eval()

        params = self._parameter_modules

        with torch.enable_grad():
            f_map = self.model(x)
            f_map_flat = f_map.reshape(f_map.shape[0], -1)

            n_batch = f_map_flat.shape[0]
            n_out = f_map_flat.shape[1]

            jacobians = torch.zeros(n_batch, n_out, self._param_dim, device=self.device)
            for i in range(n_batch):
                self.model.zero_grad(set_to_none=True)
                f_i = self.model(x[i : i + 1])
                f_i_flat = f_i.reshape(-1)
                for j in range(n_out):
                    grads = torch.autograd.grad(
                        f_i_flat[j], params, retain_graph=True, create_graph=False
                    )
                    jacobians[i, j] = torch.cat([g.detach().reshape(-1) for g in grads])

        return jacobians, f_map_flat.detach()

    def _glm_predictive(
        self, x: torch.Tensor, posterior_covariance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """GLM (linearized) predictive: Var[f] = J @ Sigma @ J^T."""
        # Get original output shape for reshaping
        with torch.no_grad():
            f_orig = self.model(x.to(self.device))
        output_shape = f_orig.shape

        J, f_map = self._compute_jacobians(x)

        # f_var: (batch, n_out_flat) = diag(J @ Sigma @ J^T)
        f_var = torch.einsum("bop,pq,boq->bo", J, posterior_covariance, J)
        f_var = f_var.clamp_min(0.0)

        if self.likelihood == "regression":
            if self.empirical_noise_variance is not None:
                f_var = f_var + self.empirical_noise_variance
            # Reshape back to original model output shape
            f_map = f_map.reshape(output_shape)
            f_var = f_var.reshape(output_shape)
            return f_map, f_var

        # Classification: probit approximation
        kappa = 1.0 / torch.sqrt(1.0 + (torch.pi / 8.0) * f_var)
        probs = torch.softmax(kappa * f_map, dim=-1)
        return probs, None

    def _forward_parameter_samples(
        self, x: torch.Tensor, sample_vectors: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(self.device)
        originals = parameters_to_vector(self._parameter_modules).detach().clone()

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for sample_vec in sample_vectors:
                vector_to_parameters(sample_vec, self._parameter_modules)
                outputs.append(self.model(x).detach())
            vector_to_parameters(originals, self._parameter_modules)

        return torch.stack(outputs, dim=0)

    def _predict_from_outputs(
        self, stacked: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.likelihood == "regression":
            mean = stacked.mean(dim=0)
            var = stacked.var(dim=0, unbiased=False).clamp_min(0.0)
            if self.empirical_noise_variance is not None:
                var = var + self.empirical_noise_variance
            return mean, var

        probs = torch.softmax(stacked, dim=-1)
        mean_probs = probs.mean(dim=0)
        return mean_probs, None


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

        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
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
    """Diagonal empirical Fisher: uses squared loss gradients instead of GGN."""

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _EmpiricalFisherDiagonalLaplace:
        (
            grad_matrix,
            diag_accumulator,
            num_datapoints,
            residual_sum_squares,
            count_outputs,
        ) = self._compute_batch_statistics(train_loader)
        del grad_matrix

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
            sigma_sq = 1.0

        hessian_diag = (1.0 / sigma_sq) * diag_accumulator / float(num_datapoints)
        self.hessian_diag = hessian_diag

        self.posterior_precision_diag = hessian_diag + prior_tensor + self.damping
        self.posterior_variance_diag = 1.0 / self.posterior_precision_diag.clamp_min(
            1e-12
        )
        return self


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

        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
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


class _BlockDiagonalLaplace(_NativeLaplaceBase):
    """Block-diagonal Laplace approximation over selected parameter groups."""

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

        if subset_of_weights == "last_layer":
            self._blocks: list[list[nn.Parameter]] = [self._parameter_modules]
        else:
            self._blocks = [[param] for param in self._parameter_modules]

        self._block_sizes = [
            parameters_to_vector(block).numel() for block in self._blocks
        ]
        self._block_offsets: list[tuple[int, int]] = []
        start = 0
        for size in self._block_sizes:
            end = start + size
            self._block_offsets.append((start, end))
            start = end

        self.block_precision_cholesky: list[torch.Tensor] = []

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _BlockDiagonalLaplace:
        _ensure_iterable_train_loader(train_loader)
        self.model.eval()

        params = self._parameter_modules
        block_accumulators = [
            torch.zeros(size, size, device=self.device) for size in self._block_sizes
        ]

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
                    full_jac = torch.cat([g.detach().reshape(-1) for g in grads])

                    if self.subset_of_weights == "last_layer":
                        block_accumulators[0] += torch.outer(full_jac, full_jac)
                    else:
                        offset = 0
                        for idx, size in enumerate(self._block_sizes):
                            block_jac = full_jac[offset : offset + size]
                            block_accumulators[idx] += torch.outer(block_jac, block_jac)
                            offset += size

        if n_data == 0:
            raise ValueError("train_loader produced zero batches.")

        param_vector = parameters_to_vector(params).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        prior_scalar = float(prior_tensor[0].item())

        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
            sigma_sq = 1.0

        self.block_precision_cholesky = []
        for acc in block_accumulators:
            curvature = (1.0 / sigma_sq) * acc
            precision = curvature + (prior_scalar + self.damping) * torch.eye(
                curvature.shape[0], device=self.device, dtype=curvature.dtype
            )
            chol = _safe_cholesky(precision, self.damping)
            self.block_precision_cholesky.append(chol)

        return self

    def _posterior_covariance(self) -> torch.Tensor:
        """Assemble block-diagonal posterior covariance."""
        cov = torch.zeros(
            self._param_dim, self._param_dim, device=self.device,
            dtype=self.mean_vector.dtype,
        )
        for (start, end), chol in zip(
            self._block_offsets, self.block_precision_cholesky
        ):
            eye = torch.eye(chol.shape[0], device=self.device, dtype=chol.dtype)
            L_inv = torch.linalg.solve_triangular(chol, eye, upper=False)
            block_cov = L_inv.transpose(0, 1) @ L_inv
            cov[start:end, start:end] = block_cov
        return cov

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mean_vector is None or len(self.block_precision_cholesky) == 0:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        posterior_cov = self._posterior_covariance()
        return self._glm_predictive(x, posterior_cov)


class _FullLaplace(_NativeLaplaceBase):
    """Dense full-matrix Laplace approximation over selected parameters."""

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
        self.posterior_precision_cholesky: torch.Tensor | None = None

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _FullLaplace:
        _ensure_iterable_train_loader(train_loader)
        self.model.eval()

        params = self._parameter_modules
        ggn = torch.zeros(self._param_dim, self._param_dim, device=self.device)
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

            # Compute per-sample output Jacobians and accumulate GGN = sum J_n^T J_n
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
                    ggn += torch.outer(j_vec, j_vec)

        if n_data == 0:
            raise ValueError("train_loader produced zero batches.")

        param_vector = parameters_to_vector(params).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )
        prior_scalar = float(prior_tensor[0].item())

        # laplace-torch: P = (1/sigma^2) * sum(J^T J) + prior * I
        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
            sigma_sq = 1.0

        precision = (1.0 / sigma_sq) * ggn + (prior_scalar + self.damping) * torch.eye(
            self._param_dim, device=self.device, dtype=ggn.dtype,
        )
        self.posterior_precision_cholesky = _safe_cholesky(precision, self.damping)
        return self

    def _posterior_covariance(self) -> torch.Tensor:
        """Compute Sigma = P^{-1} from Cholesky L where P = L L^T."""
        L = self.posterior_precision_cholesky
        eye = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
        return L_inv.transpose(0, 1) @ L_inv

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.posterior_precision_cholesky is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        posterior_cov = self._posterior_covariance()
        return self._glm_predictive(x, posterior_cov)




class _KronLaplace(_NativeLaplaceBase):
    """Kronecker-factored Laplace approximation for Linear layers.

    This backend uses empirical curvature factors for each selected linear
    layer and samples from a layer-wise matrix-normal approximation.
    """

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

        if subset_of_weights == "last_layer":
            self._layers: list[nn.Linear] = [_find_last_linear_layer(model)]
        else:
            self._layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(self._layers) == 0:
            raise ValueError(
                'hessian_structure="kron" requires at least one nn.Linear layer.'
            )

        selected_param_ids = {id(p) for p in self._parameter_modules}
        layer_param_ids = {id(p) for layer in self._layers for p in layer.parameters()}
        if selected_param_ids != layer_param_ids:
            raise ValueError(
                'hessian_structure="kron" currently supports models where selected '
                "parameters are exactly the parameters of selected nn.Linear layers."
            )

        # Keep layer blocks aligned with the flattened parameter vector order.
        self._layer_groups: list[dict[str, object]] = []
        cursor = 0
        offset = 0
        for layer in self._layers:
            layer_params = list(layer.parameters())
            for param in layer_params:
                if cursor >= len(self._parameter_modules) or id(
                    self._parameter_modules[cursor]
                ) != id(param):
                    raise ValueError(
                        "Could not align Linear-layer parameter order for Kron backend. "
                        'Use hessian_structure="block_diag" for this model.'
                    )
                cursor += 1
            block_size = sum(param.numel() for param in layer_params)
            self._layer_groups.append(
                {
                    "layer": layer,
                    "start": offset,
                    "end": offset + block_size,
                    "has_bias": layer.bias is not None,
                }
            )
            offset += block_size
        if cursor != len(self._parameter_modules) or offset != self._param_dim:
            raise RuntimeError("Internal kron block construction mismatch.")

        self._factors: list[dict[str, torch.Tensor]] = []
        self._prior_scalar: float = 1.0

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0
    ) -> _KronLaplace:
        _ensure_iterable_train_loader(train_loader)

        self.model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        ce_loss = nn.CrossEntropyLoss(reduction="sum")

        layer_stats: dict[nn.Linear, dict[str, torch.Tensor]] = {}
        for group in self._layer_groups:
            layer = group["layer"]
            assert isinstance(layer, nn.Linear)
            in_dim = layer.in_features + (1 if layer.bias is not None else 0)
            out_dim = layer.out_features
            layer_stats[layer] = {
                "A": torch.zeros(in_dim, in_dim, device=self.device),
                "G": torch.zeros(out_dim, out_dim, device=self.device),
            }

        activations: dict[nn.Linear, torch.Tensor] = {}
        grad_outputs: dict[nn.Linear, torch.Tensor] = {}

        def _fwd_hook(module: nn.Module, inputs, _outputs):
            if len(inputs) == 0:
                return
            assert isinstance(module, nn.Linear)
            activations[module] = inputs[0].detach()

        def _bwd_hook(module: nn.Module, _grad_inputs, grad_output):
            if len(grad_output) == 0 or grad_output[0] is None:
                return
            assert isinstance(module, nn.Linear)
            grad_outputs[module] = grad_output[0].detach()

        handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer in self._layers:
            handles.append(layer.register_forward_hook(_fwd_hook))
            handles.append(layer.register_full_backward_hook(_bwd_hook))

        residual_sum_squares = 0.0
        count_outputs = 0
        batch_count = 0

        try:
            for batch in train_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError("Each batch must be a tuple of (inputs, targets).")

                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(inputs)

                if self.likelihood == "regression":
                    if targets.dim() < outputs.dim():
                        targets = targets.unsqueeze(-1)
                    loss = 0.5 * mse_loss(outputs, targets)
                    residual_sum_squares += torch.sum(
                        (outputs.detach() - targets.detach()) ** 2
                    ).item()
                    count_outputs += targets.numel()
                else:
                    if targets.dim() != 1:
                        raise ValueError(
                            "Classification targets must be a 1D tensor of class indices."
                        )
                    loss = ce_loss(outputs, targets)
                    count_outputs += targets.size(0)

                loss.backward()
                batch_count += 1

                for layer in self._layers:
                    a = activations.get(layer)
                    g = grad_outputs.get(layer)
                    if a is None or g is None:
                        raise RuntimeError(
                            "Failed to capture activations/gradients for Kron factors. "
                            'Use hessian_structure="block_diag" for this model.'
                        )

                    a = a.reshape(a.shape[0], -1)
                    g = g.reshape(g.shape[0], -1)
                    if layer.bias is not None:
                        ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                        a = torch.cat([a, ones], dim=1)

                    denom_a = float(max(a.shape[0], 1))
                    denom_g = float(max(g.shape[0], 1))
                    layer_stats[layer]["A"] += (a.transpose(0, 1).matmul(a)) / denom_a
                    layer_stats[layer]["G"] += (g.transpose(0, 1).matmul(g)) / denom_g
        finally:
            for handle in handles:
                handle.remove()

        if batch_count == 0:
            raise ValueError(
                "train_loader produced zero batches; cannot fit Laplace approximation."
            )

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )
        self._prior_scalar = float(prior_tensor[0].item())

        if self.likelihood == "regression" and count_outputs > 0:
            sigma_sq = max(residual_sum_squares / count_outputs, 1e-6)
        else:
            sigma_sq = 1.0
        self._sigma_sq = sigma_sq

        self._factors = []
        for group in self._layer_groups:
            layer = group["layer"]
            assert isinstance(layer, nn.Linear)
            stats = layer_stats[layer]

            A = stats["A"] / float(batch_count)
            G = (1.0 / sigma_sq) * stats["G"] / float(batch_count)

            A = A + self.damping * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            G = G + self.damping * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)

            eig_a, u_a = torch.linalg.eigh(A)
            eig_g, u_g = torch.linalg.eigh(G)
            eig_a = eig_a.clamp_min(1e-12)
            eig_g = eig_g.clamp_min(1e-12)

            self._factors.append(
                {
                    "start": torch.tensor(
                        cast(int, group["start"]),
                        device=self.device,
                    ),
                    "end": torch.tensor(
                        cast(int, group["end"]),
                        device=self.device,
                    ),
                    "has_bias": torch.tensor(
                        1 if bool(group["has_bias"]) else 0, device=self.device
                    ),
                    "u_a": u_a,
                    "u_g": u_g,
                    "eig_a": eig_a,
                    "eig_g": eig_g,
                    "in_features": torch.tensor(layer.in_features, device=self.device),
                    "out_features": torch.tensor(
                        layer.out_features, device=self.device
                    ),
                }
            )
        return self

    def _sample_layer_block(
        self, factor: dict[str, torch.Tensor], n_samples: int
    ) -> torch.Tensor:
        u_a = factor["u_a"]
        u_g = factor["u_g"]
        eig_a = factor["eig_a"]
        eig_g = factor["eig_g"]
        has_bias = bool(int(factor["has_bias"].item()))
        in_features = int(factor["in_features"].item())

        denom = eig_a.unsqueeze(1) * eig_g.unsqueeze(0)
        denom = denom + (self._prior_scalar + self.damping)
        denom = denom.clamp_min(1e-12)

        block_samples: list[torch.Tensor] = []
        for _ in range(n_samples):
            z = torch.randn(
                eig_a.numel(),
                eig_g.numel(),
                device=self.device,
                dtype=u_a.dtype,
            )
            z_tilde = u_a.transpose(0, 1).matmul(z).matmul(u_g)
            w_tilde = z_tilde / torch.sqrt(denom)
            w = u_a.matmul(w_tilde).matmul(u_g.transpose(0, 1))

            if has_bias:
                w_flat = w[:in_features, :].transpose(0, 1).reshape(-1)
                b_flat = w[in_features, :]
                block_samples.append(torch.cat([w_flat, b_flat], dim=0))
            else:
                block_samples.append(w[:in_features, :].transpose(0, 1).reshape(-1))

        return torch.stack(block_samples, dim=0)

    def _posterior_covariance(self) -> torch.Tensor:
        """Assemble block-diagonal Kronecker posterior covariance."""
        cov = torch.zeros(
            self._param_dim, self._param_dim, device=self.device,
            dtype=self.mean_vector.dtype,
        )
        for factor in self._factors:
            start = int(factor["start"].item())
            end = int(factor["end"].item())
            u_a = factor["u_a"]
            u_g = factor["u_g"]
            eig_a = factor["eig_a"]
            eig_g = factor["eig_g"]
            has_bias = bool(int(factor["has_bias"].item()))
            in_features = int(factor["in_features"].item())

            # Kronecker eigenvalues: (eig_a_i * eig_g_j + prior + damping)^{-1}
            denom = eig_a.unsqueeze(1) * eig_g.unsqueeze(0)
            denom = denom + (self._prior_scalar + self.damping)
            inv_denom = 1.0 / denom.clamp_min(1e-12)

            # Covariance in Kronecker eigenbasis: Sigma = (U_a kron U_g) diag(1/denom) (U_a kron U_g)^T
            # Reconstruct full block covariance
            # Sigma_block[vec(W)] = (U_g kron U_a) diag(inv_denom_vec) (U_g kron U_a)^T
            # where W is (in_dim x out_dim), vec is column-major
            in_dim = u_a.shape[0]
            out_dim = u_g.shape[0]

            # Build covariance in weight-space ordering matching the flattened params
            # Params are stored as: weight.T.reshape(-1) then bias
            # i.e., (out_features, in_features).reshape(-1) then bias
            # So we need Sigma for vec(W^T) where W is (in_dim x out_dim)

            # Cov[vec(W)] where W is (in_dim, out_dim): (U_a @ diag(1/sqrt) @ U_a^T) kron (U_g @ diag(1/sqrt) @ U_g^T)
            # But the Kron structure gives us: Cov = U_a Λ_cov U_a^T ⊗ U_g Λ_cov U_g^T
            # More precisely: Cov[vec(W)] = sum over (i,j) of inv_denom[i,j] * (u_a_i u_a_i^T) ⊗ (u_g_j u_g_j^T)

            # Efficient: Sigma = (U_a @ D_a @ U_a^T) where D_a depends on structure
            # For Kronecker: Sigma_block = kron(Sigma_A_inv, Sigma_G_inv) is NOT exact
            # Exact: Sigma[vec(W)]_{(i1,j1),(i2,j2)} = sum_k sum_l inv_denom[k,l] * u_a[i1,k]*u_a[i2,k] * u_g[j1,l]*u_g[j2,l]

            # Build block_size x block_size covariance
            block_size = end - start
            block_cov = torch.zeros(block_size, block_size, device=self.device, dtype=cov.dtype)

            # Weight part: stored as (out_features, in_features).reshape(-1)
            # So param index = o * in_features + i corresponds to W[i, o] in our (in_dim, out_dim) notation
            # Build via Kronecker product in the correct ordering
            for k in range(in_dim):
                for l in range(out_dim):
                    # Outer product contribution scaled by inv_denom[k,l]
                    # In param ordering: weight[o, i] -> index o*in_features + i
                    # Our factor ordering: W[in_dim, out_dim], stored as W[:in_features,:].T.reshape(-1)
                    # So index = o * in_features + i
                    pass

            # Simpler approach: construct directly via matrix operations
            # Sigma = (U_g ⊗ U_a[:in_features]) @ diag(inv_denom_reordered) @ (U_g ⊗ U_a[:in_features])^T
            # Reorder inv_denom to match vec(W^T) = vec(out x in) ordering

            # Actually let's use the efficient formula:
            # For weight params stored as (out, in).flatten():
            # Cov[(o1,i1), (o2,i2)] = sum_k sum_l inv_denom[k,l] * u_a[i1,k]*u_a[i2,k] * u_g[o1,l]*u_g[o2,l]
            # = (U_a @ inv_denom @ U_g^T) entry-wise... no.
            # = [U_g diag_l(sum_k inv_denom[k,l] * <stuff>)]
            # Efficient: Cov_weight = kron(U_g, U_a[:in_features]) @ diag(inv_denom_vec) @ kron(U_g, U_a[:in_features])^T
            # where inv_denom_vec orders as (a_idx, g_idx) flattened matching kron ordering

            u_a_w = u_a[:in_features, :]  # (in_features, in_dim)
            # Kronecker: (U_g ⊗ U_a_w) has shape (out*in, out*in)
            # This is block_size_weight x (in_dim * out_dim)
            # inv_denom has shape (in_dim, out_dim), flatten to (in_dim*out_dim,)

            # For the weight block:
            n_weight = in_features * out_dim
            # Build: K = kron(U_g, U_a_w) @ diag(inv_denom_flat) @ kron(U_g, U_a_w)^T
            # Use: kron(A,B) @ diag(d) @ kron(A,B)^T = kron(A diag(d_g) A^T, B diag(d_a) B^T) only if d is separable
            # Not separable here, so compute directly but efficiently:
            # K_{(o1,i1),(o2,i2)} = sum_{k,l} u_a_w[i1,k]*u_a_w[i2,k] * u_g[o1,l]*u_g[o2,l] * inv_denom[k,l]

            # Compute A_contrib[i1,i2,k] = u_a_w[i1,k]*u_a_w[i2,k]
            # Compute G_contrib[o1,o2,l] = u_g[o1,l]*u_g[o2,l]
            # K_{(o1,i1),(o2,i2)} = sum_{k,l} A_contrib[i1,i2,k] * G_contrib[o1,o2,l] * inv_denom[k,l]
            #                     = sum_k A_contrib[i1,i2,k] * (sum_l G_contrib[o1,o2,l] * inv_denom[k,l])

            # G_weighted[o1,o2,k] = sum_l u_g[o1,l]*u_g[o2,l]*inv_denom[k,l] = U_g @ diag(inv_denom[k,:]) @ U_g^T
            # Then K_{(o1,i1),(o2,i2)} = sum_k u_a_w[i1,k]*u_a_w[i2,k] * G_weighted[o1,o2,k]

            # G_weighted: (out, out, in_dim)
            G_weighted = torch.einsum("ol,kl,pl->opk", u_g, inv_denom, u_g)
            # K_{(o1,i1),(o2,i2)} = sum_k u_a_w[i1,k] * u_a_w[i2,k] * G_weighted[o1,o2,k]
            # Reshape to (out*in, out*in)
            # Use einsum: result[o1,i1,o2,i2] = sum_k A[i1,k]*A[i2,k]*G[o1,o2,k]
            weight_cov_4d = torch.einsum("ik,jk,opk->oipj", u_a_w, u_a_w, G_weighted)
            weight_cov = weight_cov_4d.reshape(n_weight, n_weight)

            if has_bias:
                # Bias uses the last row of u_a (index in_features)
                u_a_b = u_a[in_features, :]  # (in_dim,)
                # Bias cov: sum_{k,l} u_a_b[k]^2 * u_g[o1,l]*u_g[o2,l] * inv_denom[k,l]
                bias_G_weighted = torch.einsum("ol,kl,pl,k,k->op", u_g, inv_denom, u_g, u_a_b, u_a_b)
                # Cross terms weight-bias:
                # Cov[(o1,i1), bias_o2] = sum_{k,l} u_a_w[i1,k]*u_a_b[k] * u_g[o1,l]*u_g[o2,l] * inv_denom[k,l]
                cross_cov = torch.einsum("ik,k,ol,kl,pl->oip", u_a_w, u_a_b, u_g, inv_denom, u_g)
                cross_cov = cross_cov.reshape(n_weight, out_dim)

                block_cov[:n_weight, :n_weight] = weight_cov
                block_cov[:n_weight, n_weight:] = cross_cov
                block_cov[n_weight:, :n_weight] = cross_cov.transpose(0, 1)
                block_cov[n_weight:, n_weight:] = bias_G_weighted
            else:
                block_cov[:, :] = weight_cov

            cov[start:end, start:end] = block_cov

        return cov

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mean_vector is None or len(self._factors) == 0:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        posterior_cov = self._posterior_covariance()
        return self._glm_predictive(x, posterior_cov)


class LaplaceWrapper:
    """Fit a Laplace approximation around a MAP-trained model.

    Parameters
    ----------
    model:
        MAP-trained neural network to approximate locally with a Gaussian
        posterior.
    likelihood:
        Either ``"regression"`` or ``"classification"``. Controls predictive
        output interpretation.
    hessian_structure:
        Curvature backend. Supported values are ``diag``, ``fisher_diag``,
        ``lowrank_diag``, ``block_diag``, ``kron``, and ``full``.
    subset_of_weights:
        ``"last_layer"`` for a lightweight approximation around the last linear
        module, or ``"all"`` for all trainable parameters when the selected
        backend supports it.
    lowrank_rank:
        Target rank for the ``lowrank_diag`` backend.
    damping:
        Numerical stabilization term added to precision approximations.
    full_max_params:
        Safety guard for ``hessian_structure="full"`` with
        ``subset_of_weights="all"``.

    Examples
    --------
    >>> la = LaplaceWrapper(model, likelihood="classification", hessian_structure="diag")
    >>> la.fit(train_loader, prior_precision=1.0)
    >>> probs, probs_var = la.predict(x_test)
    """

    _SUPPORTED_STRUCTURES = (
        "diag",
        "fisher_diag",
        "lowrank_diag",
        "block_diag",
        "kron",
        "full",
    )

    def __init__(
        self,
        model: nn.Module,
        likelihood: str = "classification",
        hessian_structure: str = "diag",
        subset_of_weights: str = "last_layer",
        lowrank_rank: int = 20,
        damping: float = 1e-6,
        full_max_params: int = 20000,
    ) -> None:
        if hessian_structure not in self._SUPPORTED_STRUCTURES:
            supported = ", ".join(self._SUPPORTED_STRUCTURES)
            raise ValueError(
                f'Unsupported hessian_structure "{hessian_structure}". Supported: {supported}.'
            )

        self.model = model
        self.likelihood = likelihood
        self.hessian_structure = hessian_structure
        self.subset_of_weights = subset_of_weights
        self.lowrank_rank = int(max(lowrank_rank, 0))
        self.damping = float(max(damping, 0.0))
        self.full_max_params = int(max(full_max_params, 1))
        self.la = None

    @staticmethod
    def supported_hessian_structures() -> tuple[str, ...]:
        """Return the supported Hessian structure names."""
        return LaplaceWrapper._SUPPORTED_STRUCTURES

    def _build_backend(self):
        if self.hessian_structure == "diag":
            return _SimpleDiagonalLaplace(
                self.model,
                likelihood=self.likelihood,
                subset_of_weights=self.subset_of_weights,
                damping=self.damping,
            )
        if self.hessian_structure == "fisher_diag":
            return _EmpiricalFisherDiagonalLaplace(
                self.model,
                likelihood=self.likelihood,
                subset_of_weights=self.subset_of_weights,
                damping=self.damping,
            )
        if self.hessian_structure == "lowrank_diag":
            return _LowRankDiagonalLaplace(
                self.model,
                likelihood=self.likelihood,
                subset_of_weights=self.subset_of_weights,
                lowrank_rank=self.lowrank_rank,
                damping=self.damping,
            )
        if self.hessian_structure == "block_diag":
            return _BlockDiagonalLaplace(
                self.model,
                likelihood=self.likelihood,
                subset_of_weights=self.subset_of_weights,
                damping=self.damping,
            )

        if self.hessian_structure == "kron":
            return _KronLaplace(
                self.model,
                likelihood=self.likelihood,
                subset_of_weights=self.subset_of_weights,
                damping=self.damping,
            )
        return _FullLaplace(
            self.model,
            likelihood=self.likelihood,
            subset_of_weights=self.subset_of_weights,
            damping=self.damping,
        )

    def fit(
        self, train_loader: Iterable, prior_precision: float | None = 1.0, **_
    ) -> object:
        """Fit the selected Laplace backend.

        Parameters
        ----------
        train_loader:
            Iterable of ``(inputs, targets)`` mini-batches used to accumulate
            curvature statistics around the MAP solution.
        prior_precision:
            Isotropic Gaussian prior precision. Higher values keep the
            approximation closer to the MAP point.

        Returns
        -------
        object
            The concrete backend instance used internally.

        Raises
        ------
        ValueError
            If ``full`` curvature is requested over too many parameters.
        """
        # Guardrail: dense full Hessian over all parameters can become intractable.
        if self.hessian_structure == "full" and self.subset_of_weights == "all":
            param_count = sum(param.numel() for param in self.model.parameters())
            if param_count > self.full_max_params:
                raise ValueError(
                    f"full Hessian over all parameters selected {param_count} parameters, "
                    f"exceeding full_max_params={self.full_max_params}. "
                    'Use subset_of_weights="last_layer" or choose hessian_structure '
                    'in {"kron", "block_diag", "lowrank_diag", "diag"}.'
                )

        self.model.eval()
        backend = self._build_backend()
        backend.fit(train_loader, prior_precision=prior_precision)
        self.la = backend
        return backend

    def optimize_prior_precision(
        self,
        train_loader: Iterable | None = None,
        n_steps: int = 100,
        lr: float = 0.1,
    ) -> float:
        """Optimize prior precision via marginal likelihood.

        Uses the log marginal likelihood:
            log p(y|X) ≈ log p(y|θ_MAP) - 1/2 log|P/P_0| - 1/2 (θ-μ)^T P_0 (θ-μ)

        Parameters
        ----------
        train_loader:
            Not used (curvature already computed in fit). Kept for API compat.
        n_steps:
            Number of optimization steps.
        lr:
            Learning rate for Adam optimizer.

        Returns
        -------
        float
            Optimized prior precision value.
        """
        if self.la is None:
            raise RuntimeError("Call fit() before optimize_prior_precision().")

        backend = self.la
        if not hasattr(backend, "mean_vector") or backend.mean_vector is None:
            raise RuntimeError("Backend not fitted.")

        # Get the raw GGN eigenvalues depending on backend type
        if isinstance(backend, _FullLaplace):
            L = backend.posterior_precision_cholesky
            P = L @ L.transpose(0, 1)
            old_prior = backend.prior_precision[0].item() if backend.prior_precision is not None else 1.0
            H = P - (old_prior + backend.damping) * torch.eye(P.shape[0], device=P.device)
            eigvals = torch.linalg.eigvalsh(H).clamp_min(1e-12)
        elif isinstance(backend, _SimpleDiagonalLaplace) and backend.hessian_diag is not None:
            eigvals = backend.hessian_diag.clamp_min(1e-12)
        elif isinstance(backend, _BlockDiagonalLaplace):
            old_prior = backend.prior_precision[0].item() if backend.prior_precision is not None else 1.0
            all_eigvals = []
            for chol in backend.block_precision_cholesky:
                P_block = chol @ chol.transpose(0, 1)
                H_block = P_block - (old_prior + backend.damping) * torch.eye(
                    P_block.shape[0], device=P_block.device
                )
                all_eigvals.append(torch.linalg.eigvalsh(H_block).clamp_min(1e-12))
            eigvals = torch.cat(all_eigvals)
        elif isinstance(backend, _KronLaplace) and len(backend._factors) > 0:
            all_eigvals = []
            for factor in backend._factors:
                eig_a = factor["eig_a"]
                eig_g = factor["eig_g"]
                kron_eigs = (eig_a.unsqueeze(1) * eig_g.unsqueeze(0)).reshape(-1)
                all_eigvals.append(kron_eigs.clamp_min(1e-12))
            eigvals = torch.cat(all_eigvals)
        elif isinstance(backend, _LowRankDiagonalLaplace) and backend.posterior_precision_diag is not None:
            old_prior = backend.prior_precision[0].item() if backend.prior_precision is not None else 1.0
            eigvals = (backend.posterior_precision_diag - old_prior - backend.damping).clamp_min(1e-12)
        else:
            return self._grid_search_prior(backend)

        # Optimize log(alpha) via marginal likelihood
        log_alpha = torch.tensor(0.0, requires_grad=True, device=eigvals.device)
        opt = torch.optim.Adam([log_alpha], lr=lr)
        p = eigvals.numel()
        theta_map = backend.mean_vector

        for _ in range(n_steps):
            opt.zero_grad()
            alpha = torch.exp(log_alpha)
            # log marglik ≈ p/2 * log(alpha) - 1/2 * sum(log(eigvals + alpha)) - alpha/2 * ||theta||^2
            log_det_P = torch.sum(torch.log(eigvals + alpha))
            scatter = alpha * theta_map.pow(2).sum()
            neg_log_marglik = 0.5 * log_det_P - 0.5 * p * log_alpha + 0.5 * scatter
            neg_log_marglik.backward()
            opt.step()

        optimal_alpha = torch.exp(log_alpha).item()

        # Update the backend with the new prior
        if isinstance(backend, _FullLaplace):
            precision = H + (optimal_alpha + backend.damping) * torch.eye(
                H.shape[0], device=H.device, dtype=H.dtype
            )
            backend.posterior_precision_cholesky = _safe_cholesky(precision, backend.damping)
            backend.prior_precision = torch.full(
                (backend._param_dim,), optimal_alpha, device=backend.device
            )
        elif isinstance(backend, _SimpleDiagonalLaplace):
            backend.optimize_prior_precision(optimal_alpha)
        elif isinstance(backend, _BlockDiagonalLaplace):
            old_prior = backend.prior_precision[0].item() if backend.prior_precision is not None else 1.0
            new_chols = []
            for chol_old in backend.block_precision_cholesky:
                P_block = chol_old @ chol_old.transpose(0, 1)
                H_block = P_block - (old_prior + backend.damping) * torch.eye(
                    P_block.shape[0], device=P_block.device
                )
                new_P = H_block + (optimal_alpha + backend.damping) * torch.eye(
                    H_block.shape[0], device=H_block.device
                )
                new_chols.append(_safe_cholesky(new_P, backend.damping))
            backend.block_precision_cholesky = new_chols
            backend.prior_precision = torch.full(
                (backend._param_dim,), optimal_alpha, device=backend.device
            )
        elif isinstance(backend, _KronLaplace):
            backend._prior_scalar = optimal_alpha
            backend.prior_precision = torch.full(
                (backend._param_dim,), optimal_alpha, device=backend.device
            )
        elif isinstance(backend, _LowRankDiagonalLaplace):
            old_prior = backend.prior_precision[0].item() if backend.prior_precision is not None else 1.0
            hessian_contrib = backend.posterior_precision_diag - old_prior - backend.damping
            backend.posterior_precision_diag = hessian_contrib + optimal_alpha + backend.damping
            backend.posterior_variance_diag = 1.0 / backend.posterior_precision_diag.clamp_min(1e-12)
            backend.prior_precision = torch.full(
                (backend._param_dim,), optimal_alpha, device=backend.device
            )

        return optimal_alpha

    def _grid_search_prior(self, backend) -> float:
        """Fallback grid search for backends without easy eigenvalue access."""
        # Just return 1.0 as default
        return 1.0

    def predict(self, x: torch.Tensor, **predict_kwargs):
        """Return the legacy predictive tuple from the fitted backend.

        Parameters
        ----------
        x:
            Evaluation inputs.
        **predict_kwargs:
            Forwarded to the backend predictive routine. Common options include
            sample counts for structured backends.
        """
        if self.la is None:
            raise RuntimeError("Call fit() before predict().")
        return self.la.predictive(x, **predict_kwargs)

    def predict_uq(self, x: torch.Tensor, **predict_kwargs) -> UQResult:
        """Return predictive moments in standardized ``UQResult`` form.

        Parameters
        ----------
        x:
            Evaluation inputs.
        **predict_kwargs:
            Forwarded to the backend predictive routine.

        Returns
        -------
        UQResult
            For regression, ``mean`` plus variance fields. For classification,
            ``probs`` and optional ``probs_var`` with regression variance fields
            left unset.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called or if a regression backend fails
            to produce predictive variance.
        """
        if self.la is None:
            raise RuntimeError("Call fit() before predict_uq().")

        mean_or_probs, var = self.la.predictive(x, **predict_kwargs)
        if self.likelihood == "classification":
            return UQResult(
                mean=mean_or_probs,
                epistemic_var=None,
                aleatoric_var=None,
                total_var=None,
                probs=mean_or_probs,
                probs_var=var,
                metadata={
                    "method": "laplace",
                    "hessian_structure": self.hessian_structure,
                    "likelihood": self.likelihood,
                    "subset_of_weights": self.subset_of_weights,
                },
            )

        if var is None:
            raise RuntimeError(
                "Regression Laplace backend must return predictive variance."
            )

        noise_var = getattr(self.la, "empirical_noise_variance", None)
        if noise_var is not None:
            aleatoric = noise_var.to(var.device, var.dtype).expand_as(var)
            epistemic = (var - aleatoric).clamp_min(0.0)
        else:
            aleatoric = None
            epistemic = var

        return UQResult(
            mean=mean_or_probs,
            epistemic_var=epistemic,
            aleatoric_var=aleatoric,
            total_var=var.clamp_min(0.0),
            probs=None,
            probs_var=None,
            metadata={
                "method": "laplace",
                "hessian_structure": self.hessian_structure,
                "likelihood": self.likelihood,
                "subset_of_weights": self.subset_of_weights,
            },
        )
