from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from ._base import _NativeLaplaceBase, _ensure_iterable_train_loader, _safe_cholesky


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
        # sigma_noise = 1.0 by default, matching laplace-torch convention.
        # The GGN H = sum(J^T J) and P = (1/sigma^2)*H + prior = H + prior.
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
