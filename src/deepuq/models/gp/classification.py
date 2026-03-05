"""Gaussian process classification models."""

from __future__ import annotations

import math
from typing import Optional

import torch

from deepuq.types import UQResult

from .kernels import Kernel, RBFKernel
from .utils import stable_cholesky


class GaussianProcessClassifier:
    """Binary GP classifier using Laplace approximation."""

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        max_iter: int = 20,
        tol: float = 1e-4,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.kernel = kernel or RBFKernel()
        self.max_iter = max_iter
        self.tol = tol
        self.prior_variance = prior_variance
        self.jitter = jitter
        self.device = device
        self.dtype = dtype

        self._x_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._f_hat: Optional[torch.Tensor] = None
        self._K: Optional[torch.Tensor] = None
        self._K_chol: Optional[torch.Tensor] = None
        self._W: Optional[torch.Tensor] = None
        self._Kinv_f: Optional[torch.Tensor] = None

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype, copy=False)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianProcessClassifier":
        """Fit binary GP classifier.

        ``y`` must contain binary labels in {0, 1}.
        """
        x = self._prepare(x)
        y = self._prepare(y).reshape(-1)
        if x.ndim != 2:
            raise ValueError("x must have shape [N, D].")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape [N] matching x.")

        unique = torch.unique(y)
        if not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(
                "GaussianProcessClassifier expects binary labels in {0,1}."
            )

        n = x.shape[0]
        K = self.kernel(x, x) * self.prior_variance
        K = K + self.jitter * torch.eye(n, device=x.device, dtype=x.dtype)
        K_chol, _ = stable_cholesky(K, jitter_base=self.jitter, jitter_max=1e-2)

        f = torch.zeros(n, device=x.device, dtype=x.dtype)
        yb = y

        for _ in range(self.max_iter):
            pi = torch.sigmoid(f)
            W = (pi * (1.0 - pi)).clamp_min(1e-6)
            Kinv_f = torch.cholesky_solve(f.unsqueeze(-1), K_chol).squeeze(-1)
            grad = Kinv_f - yb + pi

            H = (
                torch.cholesky_inverse(K_chol)
                + torch.diag(W)
                + self.jitter * torch.eye(n, device=x.device, dtype=x.dtype)
            )
            H_chol, _ = stable_cholesky(H, jitter_base=self.jitter, jitter_max=1e-2)
            step = torch.cholesky_solve(grad.unsqueeze(-1), H_chol).squeeze(-1)
            f_next = f - step

            if torch.max(torch.abs(step)).item() < self.tol:
                f = f_next
                break
            f = f_next

        pi = torch.sigmoid(f)
        W = (pi * (1.0 - pi)).clamp_min(1e-6)
        Kinv_f = torch.cholesky_solve(f.unsqueeze(-1), K_chol).squeeze(-1)

        self._x_train = x
        self._y_train = yb
        self._f_hat = f
        self._K = K
        self._K_chol = K_chol
        self._W = W
        self._Kinv_f = Kinv_f
        return self

    def _check_fit(self) -> None:
        if (
            self._x_train is None
            or self._f_hat is None
            or self._K is None
            or self._K_chol is None
            or self._W is None
            or self._Kinv_f is None
        ):
            raise RuntimeError("Model must be fit before prediction.")

    def _latent_posterior(
        self, x_star: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_fit()
        x_star = self._prepare(x_star)
        assert (
            self._x_train is not None
            and self._K is not None
            and self._K_chol is not None
            and self._W is not None
            and self._Kinv_f is not None
        )

        k_xs = self.kernel(self._x_train, x_star) * self.prior_variance
        mean_f = k_xs.transpose(0, 1) @ self._Kinv_f.unsqueeze(-1)

        w_inv = 1.0 / self._W
        A = self._K + torch.diag(w_inv)
        A_chol, _ = stable_cholesky(A, jitter_base=self.jitter, jitter_max=1e-2)
        v = torch.cholesky_solve(k_xs, A_chol)
        k_ss_diag = (self.kernel(x_star, x_star) * self.prior_variance).diag()
        var_f = (k_ss_diag - (k_xs * v).sum(dim=0)).clamp_min(1e-10)
        return mean_f.squeeze(-1), var_f

    def predict_proba(self, x_star: torch.Tensor) -> torch.Tensor:
        """Return p(y=1|x*) and p(y=0|x*) for test points."""
        mean_f, var_f = self._latent_posterior(x_star)
        scale = torch.sqrt(1.0 + (math.pi / 8.0) * var_f)
        p1 = torch.sigmoid(mean_f / scale)
        p0 = 1.0 - p1
        return torch.stack([p0, p1], dim=-1)

    def predict(self, x_star: torch.Tensor) -> torch.Tensor:
        """Return hard class labels for binary GP classification."""
        probs = self.predict_proba(x_star)
        return torch.argmax(probs, dim=1)

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized uncertainty output for binary classification."""
        probs = self.predict_proba(x_star)
        p1 = probs[:, 1]
        p1_var = (p1 * (1.0 - p1)).clamp_min(0.0)
        probs_var = torch.stack([p1_var, p1_var], dim=-1)
        return UQResult(
            mean=probs,
            epistemic_var=None,
            aleatoric_var=None,
            total_var=None,
            probs=probs,
            probs_var=probs_var,
            metadata={"method": "gp_classification_binary"},
        )


class OneVsRestGaussianProcessClassifier:
    """Multiclass GP classification via one-vs-rest binary classifiers."""

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        max_iter: int = 20,
        tol: float = 1e-4,
        prior_variance: float = 1.0,
        jitter: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.prior_variance = prior_variance
        self.jitter = jitter
        self.device = device
        self.dtype = dtype

        self.classes_: Optional[torch.Tensor] = None
        self.models_: list[GaussianProcessClassifier] = []

    def fit(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> "OneVsRestGaussianProcessClassifier":
        """Fit one binary GP classifier per class label."""
        y = y.reshape(-1)
        classes = torch.unique(y).sort().values
        self.classes_ = classes
        self.models_ = []

        for cls in classes:
            y_bin = (y == cls).to(dtype=torch.float32)
            model = GaussianProcessClassifier(
                kernel=self.kernel or RBFKernel(),
                max_iter=self.max_iter,
                tol=self.tol,
                prior_variance=self.prior_variance,
                jitter=self.jitter,
                device=self.device,
                dtype=self.dtype,
            )
            model.fit(x, y_bin)
            self.models_.append(model)

        return self

    def _check_fit(self) -> None:
        if self.classes_ is None or not self.models_:
            raise RuntimeError("Model must be fit before prediction.")

    def predict_proba(self, x_star: torch.Tensor) -> torch.Tensor:
        """Return multiclass probabilities by normalized OvR scores."""
        self._check_fit()
        per_class = []
        for model in self.models_:
            p = model.predict_proba(x_star)[:, 1]
            per_class.append(p)
        stacked = torch.stack(per_class, dim=1)
        norm = stacked.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return stacked / norm

    def predict(self, x_star: torch.Tensor) -> torch.Tensor:
        """Return predicted multiclass labels."""
        probs = self.predict_proba(x_star)
        assert self.classes_ is not None
        idx = torch.argmax(probs, dim=1)
        return self.classes_[idx]

    def predict_uq(self, x_star: torch.Tensor) -> UQResult:
        """Return standardized uncertainty output for multiclass GP classifier."""
        probs = self.predict_proba(x_star)
        probs_var = (probs * (1.0 - probs)).clamp_min(0.0)
        return UQResult(
            mean=probs,
            epistemic_var=None,
            aleatoric_var=None,
            total_var=None,
            probs=probs,
            probs_var=probs_var,
            metadata={
                "method": "gp_classification_ovr",
                "num_classes": int(probs.shape[1]),
            },
        )
