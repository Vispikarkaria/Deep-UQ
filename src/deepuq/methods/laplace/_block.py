from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from ._base import _NativeLaplaceBase, _ensure_iterable_train_loader, _safe_cholesky


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

        # sigma_noise = 1.0 by default, matching laplace-torch convention.
        # The GGN H = sum(J^T J) and P = (1/sigma^2)*H + prior = H + prior.
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
