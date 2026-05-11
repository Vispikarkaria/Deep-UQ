from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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
            # Return epistemic (functional) variance only, matching laplace-torch.
            # Observation noise can be added externally if needed.
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
