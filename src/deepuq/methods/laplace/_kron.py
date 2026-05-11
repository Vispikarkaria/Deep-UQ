from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from ._base import (
    _ensure_iterable_train_loader,
    _find_last_linear_layer,
    _NativeLaplaceBase,
)


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

        # sigma_noise = 1.0 by default, matching laplace-torch convention.
        # The GGN H = sum(J^T J) and P = (1/sigma^2)*H + prior = H + prior.
        sigma_sq = 1.0
        self._sigma_sq = sigma_sq

        self._factors = []
        for group in self._layer_groups:
            layer = group["layer"]
            assert isinstance(layer, nn.Linear)
            stats = layer_stats[layer]

            A = stats["A"] / float(batch_count)

            if self.likelihood == "regression":
                # For regression GGN-KFAC: G = (1/sigma^2) * I
                out_dim = stats["G"].shape[0]
                G = (1.0 / sigma_sq) * torch.eye(out_dim, device=self.device)
            else:
                G = stats["G"] / float(batch_count)

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
            self._param_dim,
            self._param_dim,
            device=self.device,
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
            block_cov = torch.zeros(
                block_size, block_size, device=self.device, dtype=cov.dtype
            )

            # Construct covariance via matrix operations
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
                bias_G_weighted = torch.einsum(
                    "ol,kl,pl,k,k->op", u_g, inv_denom, u_g, u_a_b, u_a_b
                )
                # Cross terms weight-bias:
                # Cov[(o1,i1), bias_o2] = sum_{k,l} u_a_w[i1,k]*u_a_b[k] * u_g[o1,l]*u_g[o2,l] * inv_denom[k,l]
                cross_cov = torch.einsum(
                    "ik,k,ol,kl,pl->oip", u_a_w, u_a_b, u_g, inv_denom, u_g
                )
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
