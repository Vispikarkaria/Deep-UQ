from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from deepuq.types import UQResult


def _find_last_linear_layer(model: nn.Module) -> nn.Module:
    last_linear: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError(
            "Could not locate a linear layer in the model to use for Laplace approximation."
        )
    return last_linear


def _select_parameters(model: nn.Module, subset_of_weights: str) -> List[nn.Parameter]:
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
    last_error: Optional[RuntimeError] = None
    for _ in range(7):
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
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

        self.mean_vector: Optional[torch.Tensor] = None
        self.prior_precision: Optional[torch.Tensor] = None
        self.empirical_noise_variance: Optional[torch.Tensor] = None

    def _compute_batch_statistics(
        self,
        train_loader: Iterable,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, float, int]:
        _ensure_iterable_train_loader(train_loader)

        self.model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        ce_loss = nn.CrossEntropyLoss(reduction="sum")

        batch_grads: List[torch.Tensor] = []
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
        prior_precision: Optional[float],
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

    def _forward_parameter_samples(
        self, x: torch.Tensor, sample_vectors: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(self.device)
        originals = parameters_to_vector(self._parameter_modules).detach().clone()

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for sample_vec in sample_vectors:
                vector_to_parameters(sample_vec, self._parameter_modules)
                outputs.append(self.model(x).detach())
            vector_to_parameters(originals, self._parameter_modules)

        return torch.stack(outputs, dim=0)

    def _predict_from_outputs(
        self, stacked: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        self.posterior_precision_diag: Optional[torch.Tensor] = None
        self.posterior_variance_diag: Optional[torch.Tensor] = None
        self.hessian_diag: Optional[torch.Tensor] = None

    def fit(
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0
    ) -> "_SimpleDiagonalLaplace":
        (
            grad_matrix,
            diag_accumulator,
            num_datapoints,
            residual_sum_squares,
            count_outputs,
        ) = self._compute_batch_statistics(train_loader)
        del grad_matrix  # Only diagonal stats are needed for this backend.

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        hessian_diag = diag_accumulator / float(num_datapoints)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.posterior_variance_diag is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        std = torch.sqrt(self.posterior_variance_diag.clamp_min(1e-12))
        noise = torch.randn(n_samples, std.numel(), device=self.device)
        samples = self.mean_vector.unsqueeze(0) + noise * std.unsqueeze(0)

        outputs = self._forward_parameter_samples(x, samples)
        return self._predict_from_outputs(outputs)


class _EmpiricalFisherDiagonalLaplace(_SimpleDiagonalLaplace):
    """Explicit diagonal empirical Fisher variant (same estimator family as diag)."""


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
        self.posterior_precision_diag: Optional[torch.Tensor] = None
        self.posterior_variance_diag: Optional[torch.Tensor] = None
        self.lowrank_u: Optional[torch.Tensor] = None
        self.lowrank_lam: Optional[torch.Tensor] = None

    def fit(
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0
    ) -> "_LowRankDiagonalLaplace":
        (
            grad_matrix,
            diag_accumulator,
            num_datapoints,
            residual_sum_squares,
            count_outputs,
        ) = self._compute_batch_statistics(train_loader)

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        diag_total = diag_accumulator / float(num_datapoints)
        scaled_grads = grad_matrix / float(num_datapoints) ** 0.5

        rank_cap = min(self.lowrank_rank, scaled_grads.shape[0], scaled_grads.shape[1])
        if rank_cap <= 0:
            self.lowrank_u = None
            self.lowrank_lam = None
            diag_residual = diag_total
        else:
            _, singular_vals, v_t = torch.linalg.svd(scaled_grads, full_matrices=False)
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
        return adjusted * inv_sqrt_d.unsqueeze(0)

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.posterior_precision_diag is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        noise = self._sample_lowrank_noise(n_samples)
        samples = self.mean_vector.unsqueeze(0) + noise

        outputs = self._forward_parameter_samples(x, samples)
        return self._predict_from_outputs(outputs)


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
            self._blocks: List[List[nn.Parameter]] = [self._parameter_modules]
        else:
            self._blocks = [[param] for param in self._parameter_modules]

        self._block_sizes = [
            parameters_to_vector(block).numel() for block in self._blocks
        ]
        self._block_offsets: List[Tuple[int, int]] = []
        start = 0
        for size in self._block_sizes:
            end = start + size
            self._block_offsets.append((start, end))
            start = end

        self.block_precision_cholesky: List[torch.Tensor] = []

    def fit(
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0
    ) -> "_BlockDiagonalLaplace":
        _ensure_iterable_train_loader(train_loader)

        self.model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        ce_loss = nn.CrossEntropyLoss(reduction="sum")

        block_accumulators = [
            torch.zeros(size, size, device=self.device) for size in self._block_sizes
        ]

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

            grads = torch.autograd.grad(
                loss, self._parameter_modules, retain_graph=False
            )

            if self.subset_of_weights == "last_layer":
                grad_vec = torch.cat([g.detach().reshape(-1) for g in grads])
                block_accumulators[0] += torch.outer(grad_vec, grad_vec)
            else:
                for idx, grad in enumerate(grads):
                    grad_vec = grad.detach().reshape(-1)
                    block_accumulators[idx] += torch.outer(grad_vec, grad_vec)

        num_datapoints = len(getattr(train_loader, "dataset", []))
        if num_datapoints == 0:
            num_datapoints = 1

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )

        prior_scalar = float(prior_tensor[0].item())
        self.block_precision_cholesky = []
        for acc in block_accumulators:
            curvature = acc / float(num_datapoints)
            precision = curvature + (prior_scalar + self.damping) * torch.eye(
                curvature.shape[0], device=self.device, dtype=curvature.dtype
            )
            chol = _safe_cholesky(precision, self.damping)
            self.block_precision_cholesky.append(chol)

        return self

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mean_vector is None or len(self.block_precision_cholesky) == 0:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        samples = torch.zeros(
            n_samples, self._param_dim, device=self.device, dtype=self.mean_vector.dtype
        )
        for (start, end), chol in zip(
            self._block_offsets, self.block_precision_cholesky
        ):
            block_size = end - start
            z = torch.randn(
                block_size, n_samples, device=self.device, dtype=self.mean_vector.dtype
            )
            # Solve L^T x = z so Cov[x] = (L L^T)^-1.
            x_block = torch.linalg.solve_triangular(
                chol.transpose(-2, -1), z, upper=True
            ).transpose(0, 1)
            samples[:, start:end] = self.mean_vector[start:end].unsqueeze(0) + x_block

        outputs = self._forward_parameter_samples(x, samples)
        return self._predict_from_outputs(outputs)


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
        self.posterior_precision_cholesky: Optional[torch.Tensor] = None

    def fit(
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0
    ) -> "_FullLaplace":
        grad_matrix, _, num_datapoints, residual_sum_squares, count_outputs = (
            self._compute_batch_statistics(train_loader)
        )

        param_vector = parameters_to_vector(self._parameter_modules).detach().clone()
        prior_tensor = self._finalize_common_fit(
            param_vector, prior_precision, residual_sum_squares, count_outputs
        )
        prior_scalar = float(prior_tensor[0].item())

        curvature = grad_matrix.transpose(0, 1).matmul(grad_matrix) / float(
            num_datapoints
        )
        precision = curvature + (prior_scalar + self.damping) * torch.eye(
            curvature.shape[0],
            device=curvature.device,
            dtype=curvature.dtype,
        )
        self.posterior_precision_cholesky = _safe_cholesky(precision, self.damping)
        return self

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.posterior_precision_cholesky is None or self.mean_vector is None:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        dim = self.posterior_precision_cholesky.shape[0]
        z = torch.randn(
            dim, n_samples, device=self.device, dtype=self.mean_vector.dtype
        )
        noise = torch.linalg.solve_triangular(
            self.posterior_precision_cholesky.transpose(-2, -1),
            z,
            upper=True,
        ).transpose(0, 1)
        samples = self.mean_vector.unsqueeze(0) + noise

        outputs = self._forward_parameter_samples(x, samples)
        return self._predict_from_outputs(outputs)


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
            self._layers: List[nn.Linear] = [_find_last_linear_layer(model)]
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
        self._layer_groups: List[Dict[str, object]] = []
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

        self._factors: List[Dict[str, torch.Tensor]] = []
        self._prior_scalar: float = 1.0

    def fit(
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0
    ) -> "_KronLaplace":
        _ensure_iterable_train_loader(train_loader)

        self.model.eval()
        mse_loss = nn.MSELoss(reduction="sum")
        ce_loss = nn.CrossEntropyLoss(reduction="sum")

        layer_stats: Dict[nn.Linear, Dict[str, torch.Tensor]] = {}
        for group in self._layer_groups:
            layer = group["layer"]
            assert isinstance(layer, nn.Linear)
            in_dim = layer.in_features + (1 if layer.bias is not None else 0)
            out_dim = layer.out_features
            layer_stats[layer] = {
                "A": torch.zeros(in_dim, in_dim, device=self.device),
                "G": torch.zeros(out_dim, out_dim, device=self.device),
            }

        activations: Dict[nn.Linear, torch.Tensor] = {}
        grad_outputs: Dict[nn.Linear, torch.Tensor] = {}

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

        handles: List[torch.utils.hooks.RemovableHandle] = []
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

        self._factors = []
        for group in self._layer_groups:
            layer = group["layer"]
            assert isinstance(layer, nn.Linear)
            stats = layer_stats[layer]

            A = stats["A"] / float(batch_count)
            G = stats["G"] / float(batch_count)

            A = A + self.damping * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            G = G + self.damping * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)

            eig_a, u_a = torch.linalg.eigh(A)
            eig_g, u_g = torch.linalg.eigh(G)
            eig_a = eig_a.clamp_min(1e-12)
            eig_g = eig_g.clamp_min(1e-12)

            self._factors.append(
                {
                    "start": torch.tensor(int(group["start"]), device=self.device),
                    "end": torch.tensor(int(group["end"]), device=self.device),
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
        self, factor: Dict[str, torch.Tensor], n_samples: int
    ) -> torch.Tensor:
        u_a = factor["u_a"]
        u_g = factor["u_g"]
        eig_a = factor["eig_a"]
        eig_g = factor["eig_g"]
        has_bias = bool(int(factor["has_bias"].item()))
        in_features = int(factor["in_features"].item())
        out_features = int(factor["out_features"].item())

        denom = eig_a.unsqueeze(1) * eig_g.unsqueeze(0)
        denom = denom + (self._prior_scalar + self.damping)
        denom = denom.clamp_min(1e-12)

        block_samples: List[torch.Tensor] = []
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

    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mean_vector is None or len(self._factors) == 0:
            raise RuntimeError("Laplace approximation not fitted yet.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        samples = self.mean_vector.unsqueeze(0).repeat(n_samples, 1)
        for factor in self._factors:
            start = int(factor["start"].item())
            end = int(factor["end"].item())
            samples[:, start:end] = samples[:, start:end] + self._sample_layer_block(
                factor, n_samples
            )

        outputs = self._forward_parameter_samples(x, samples)
        return self._predict_from_outputs(outputs)


class LaplaceWrapper:
    """
    Fit a Laplace approximation around a MAP-trained model.

    Supported Hessian structures:
      - diag
      - fisher_diag
      - lowrank_diag
      - block_diag
      - kron
      - full

    Example:
        la = LaplaceWrapper(model, likelihood='classification', hessian_structure='diag')
        la.fit(dataloader, prior_precision=1.0)
        probs, var = la.predict(x)
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
    def supported_hessian_structures() -> Tuple[str, ...]:
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
        self, train_loader: Iterable, prior_precision: Optional[float] = 1.0, **_
    ) -> object:
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

    def predict(self, x: torch.Tensor, **predict_kwargs):
        if self.la is None:
            raise RuntimeError("Call fit() before predict().")
        return self.la.predictive(x, **predict_kwargs)

    def predict_uq(self, x: torch.Tensor, **predict_kwargs) -> UQResult:
        """Return standardized uncertainty fields without changing legacy predict()."""
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
