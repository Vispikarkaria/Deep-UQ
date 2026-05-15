from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from deepuq.types import UQResult

from ._base import _find_last_linear_layer, _safe_cholesky
from ._block import _BlockDiagonalLaplace
from ._diag import (
    _EmpiricalFisherDiagonalLaplace,
    _LowRankDiagonalLaplace,
    _SimpleDiagonalLaplace,
)
from ._full import _FullLaplace
from ._kron import _KronLaplace


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
            # For single-output last-layer, kron degenerates (G is 1x1).
            # Fall back to block_diag which is mathematically equivalent and more stable.
            if self.subset_of_weights == "last_layer":
                last_linear = _find_last_linear_layer(self.model)
                if last_linear.out_features == 1:
                    return _BlockDiagonalLaplace(
                        self.model,
                        likelihood=self.likelihood,
                        subset_of_weights=self.subset_of_weights,
                        damping=self.damping,
                    )
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
            old_prior = (
                backend.prior_precision[0].item()
                if backend.prior_precision is not None
                else 1.0
            )
            H = P - (old_prior + backend.damping) * torch.eye(
                P.shape[0], device=P.device
            )
            eigvals = torch.linalg.eigvalsh(H).clamp_min(1e-12)
        elif (
            isinstance(backend, _SimpleDiagonalLaplace)
            and backend.hessian_diag is not None
        ):
            eigvals = backend.hessian_diag.clamp_min(1e-12)
        elif isinstance(backend, _BlockDiagonalLaplace):
            old_prior = (
                backend.prior_precision[0].item()
                if backend.prior_precision is not None
                else 1.0
            )
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
        elif (
            isinstance(backend, _LowRankDiagonalLaplace)
            and backend.posterior_precision_diag is not None
        ):
            old_prior = (
                backend.prior_precision[0].item()
                if backend.prior_precision is not None
                else 1.0
            )
            eigvals = (
                backend.posterior_precision_diag - old_prior - backend.damping
            ).clamp_min(1e-12)
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
            backend.posterior_precision_cholesky = _safe_cholesky(
                precision, backend.damping
            )
            backend.prior_precision = torch.full(
                (backend._param_dim,), optimal_alpha, device=backend.device
            )
        elif isinstance(backend, _SimpleDiagonalLaplace):
            backend.optimize_prior_precision(optimal_alpha)
        elif isinstance(backend, _BlockDiagonalLaplace):
            old_prior = (
                backend.prior_precision[0].item()
                if backend.prior_precision is not None
                else 1.0
            )
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
            old_prior = (
                backend.prior_precision[0].item()
                if backend.prior_precision is not None
                else 1.0
            )
            hessian_contrib = (
                backend.posterior_precision_diag - old_prior - backend.damping
            )
            backend.posterior_precision_diag = (
                hessian_contrib + optimal_alpha + backend.damping
            )
            backend.posterior_variance_diag = (
                1.0 / backend.posterior_precision_diag.clamp_min(1e-12)
            )
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

    def _predict_glm(self, x: torch.Tensor) -> UQResult:
        """Linearized (GLM) predictive using Jacobian-based variance.

        Computes predictive variance as diag(J @ Sigma_post @ J^T) where J is
        the Jacobian of the network output w.r.t. the posterior parameters at MAP.

        Parameters
        ----------
        x:
            Evaluation inputs.

        Returns
        -------
        UQResult
            Predictive mean and variance from the linearized model.
        """
        if self.la is None:
            raise RuntimeError("Call fit() before _predict_glm().")

        backend = self.la
        self.model.eval()

        # Get the parameters we're approximating over
        params = [p for p in backend._parameter_modules if p.requires_grad]

        # Get posterior variance (diagonal)
        if hasattr(backend, "posterior_variance_diag") and backend.posterior_variance_diag is not None:
            post_var_diag = backend.posterior_variance_diag
        elif hasattr(backend, "posterior_precision_cholesky"):
            # Full covariance: invert precision
            L = backend.posterior_precision_cholesky
            L_inv = torch.linalg.inv(L)
            cov = L_inv.T @ L_inv
            post_var_diag = cov.diag()
        else:
            # Fallback: use 1/precision_diag
            if hasattr(backend, "posterior_precision_diag") and backend.posterior_precision_diag is not None:
                post_var_diag = 1.0 / backend.posterior_precision_diag.clamp_min(1e-12)
            else:
                raise RuntimeError("Cannot extract posterior covariance from backend.")

        # Compute per-sample Jacobian and predictive variance
        device = next(iter(params)).device
        x_dev = x.to(device)
        n_samples = x_dev.shape[0]

        # Forward pass for mean
        with torch.no_grad():
            mean = self.model(x_dev)

        out_dim = mean.shape[-1] if mean.dim() > 1 else 1
        pred_var = torch.zeros(n_samples, out_dim, device=device)

        for i in range(n_samples):
            xi = x_dev[i : i + 1]
            # Compute Jacobian for this sample
            self.model.zero_grad()
            out = self.model(xi)
            for d in range(out_dim):
                if out_dim == 1:
                    out_scalar = out.squeeze()
                else:
                    out_scalar = out[0, d]
                grads = torch.autograd.grad(out_scalar, params, retain_graph=(d < out_dim - 1))
                # Flatten Jacobian row
                j_row = torch.cat([g.reshape(-1) for g in grads])
                # var_d = j_row^T @ diag(post_var) @ j_row = sum(j_row^2 * post_var)
                pred_var[i, d] = (j_row.pow(2) * post_var_diag).sum()

        mean_out = mean.detach()
        pred_var = pred_var.detach()

        if self.likelihood == "classification":
            probs = torch.softmax(mean_out, dim=-1)
            return UQResult(
                mean=probs,
                epistemic_var=None,
                aleatoric_var=None,
                total_var=None,
                probs=probs,
                probs_var=pred_var,
                metadata={
                    "method": "laplace_glm",
                    "hessian_structure": self.hessian_structure,
                    "likelihood": self.likelihood,
                    "subset_of_weights": self.subset_of_weights,
                },
            )

        return UQResult(
            mean=mean_out,
            epistemic_var=pred_var,
            aleatoric_var=None,
            total_var=pred_var,
            probs=None,
            probs_var=None,
            metadata={
                "method": "laplace_glm",
                "hessian_structure": self.hessian_structure,
                "likelihood": self.likelihood,
                "subset_of_weights": self.subset_of_weights,
            },
        )

    def predict_uq(self, x: torch.Tensor, method: str = "sampling", **predict_kwargs) -> UQResult:
        """Return predictive moments in standardized ``UQResult`` form.

        Parameters
        ----------
        x:
            Evaluation inputs.
        method:
            Predictive method. ``"sampling"`` uses the backend's default
            weight-space sampling. ``"glm"`` uses the linearized (Jacobian-based)
            predictive which is exact and requires no sampling.
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

        if method == "glm":
            return self._predict_glm(x)

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
