"""Deep ensemble wrappers for regression and classification uncertainty.

The ensemble variants in this module aggregate independently trained
deterministic models. ``predict_uq`` returns a :class:`deepuq.types.UQResult`
with either regression moments or class-probability moments.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from deepuq.types import UQResult


def _set_seed(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)


def _move_batch(
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        return xb, yb
    return xb.to(device), yb.to(device)


class _BaseDeepEnsemble(nn.Module):
    """Shared training and aggregation logic for deep ensembles.

    Parameters
    ----------
    models:
        Sequence of pre-instantiated PyTorch models. All members must share the
        same input/output contract.
    """

    method_name = "deep_ensemble"

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        if len(models) == 0:
            raise ValueError(f"{self.__class__.__name__} requires at least one model.")
        self.models = nn.ModuleList(models)

    def _default_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _prediction_for_loss(self, model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
        return model(xb)

    def fit(
        self,
        train_loader,
        *,
        epochs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> _BaseDeepEnsemble:
        """Train each ensemble member independently.

        The dataloader is expected to yield ``(xb, yb)`` tensor batches. Each
        member is re-seeded with ``seed + member_index`` when ``seed`` is
        provided.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        active_loss = loss_fn or self._default_loss
        for model_idx, model in enumerate(self.models):
            _set_seed(None if seed is None else seed + model_idx)
            if device is not None:
                model.to(device)
            optimizer = optimizer_cls(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                seen = 0
                for xb, yb in train_loader:
                    xb, yb = _move_batch(xb, yb, device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self._prediction_for_loss(model, xb)
                    loss = active_loss(prediction, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.detach()) * int(xb.shape[0])
                    seen += int(xb.shape[0])
                if verbose and seen > 0:
                    avg = running_loss / seen
                    print(
                        f"{self.__class__.__name__} member={model_idx + 1} "
                        f"epoch={epoch + 1} loss={avg:.6f}"
                    )
        return self


class DeepEnsembleRegressor(_BaseDeepEnsemble):
    """Deterministic regression ensemble with epistemic uncertainty.

    Shape contract
    --------------
    - input: any tensor accepted by the wrapped regressor
    - member output: ``[batch, ...]``
    - ``predict`` returns ``(mean, variance)`` with the same prediction shape

    Example
    -------
    ```python
    ensemble = DeepEnsembleRegressor([model_a, model_b, model_c])
    ensemble.fit(train_loader, epochs=50, lr=1e-3)
    uq = ensemble.predict_uq(x_test)
    ```
    """

    method_name = "deep_ensemble_regressor"

    def _default_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(prediction, target)

    @torch.inference_mode()
    def predict_members(self, x: torch.Tensor) -> torch.Tensor:
        """Return stacked member predictions with shape ``[n_members, batch, ...]``."""
        preds = []
        for model in self.models:
            model.eval()
            preds.append(model(x).unsqueeze(0))
        return torch.cat(preds, dim=0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.predict_members(x)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0, unbiased=False)
        return mean, var

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return predictive mean and epistemic variance in a ``UQResult``."""
        mean, var = self.predict(x)
        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "n_members": len(self.models),
            },
        )


class HeteroscedasticDeepEnsembleRegressor(_BaseDeepEnsemble):
    """Regression ensemble with predicted aleatoric noise per member.

    Each member must output concatenated mean and log-variance tensors. For
    vector outputs the concatenation is along the last dimension. For field
    outputs it is along channel dimension ``1``.
    """

    method_name = "heteroscedastic_deep_ensemble_regressor"

    def __init__(self, models: Sequence[nn.Module], min_variance: float = 1e-6):
        super().__init__(models)
        self.min_variance = float(min_variance)

    def _split_prediction(
        self,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        split_dim = 1 if prediction.dim() > 2 else -1
        split_size = int(prediction.shape[split_dim])
        if split_size % 2 != 0:
            raise ValueError(
                "Heteroscedastic ensemble models must output mean and log-variance "
                "concatenated along the channel dimension for field outputs or "
                "the last dimension for vector outputs."
            )
        mean, log_var = torch.chunk(prediction, 2, dim=split_dim)
        var = torch.exp(log_var).clamp_min(self.min_variance)
        return mean, var

    def _default_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        mean, var = self._split_prediction(prediction)
        return 0.5 * (((target - mean) ** 2) / var + torch.log(var)).mean()

    @torch.inference_mode()
    def predict_members(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means = []
        aleatoric_vars = []
        for model in self.models:
            model.eval()
            mean, var = self._split_prediction(model(x))
            means.append(mean.unsqueeze(0))
            aleatoric_vars.append(var.unsqueeze(0))
        return torch.cat(means, dim=0), torch.cat(aleatoric_vars, dim=0)

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _, _, total_var = self.predict_components(x)
        return mean, total_var

    @torch.inference_mode()
    def predict_components(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        means, aleatoric = self.predict_members(x)
        mean = means.mean(dim=0)
        epistemic_var = means.var(dim=0, unbiased=False)
        aleatoric_var = aleatoric.mean(dim=0)
        total_var = epistemic_var + aleatoric_var
        return mean, epistemic_var, aleatoric_var, total_var

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return predictive mean plus epistemic, aleatoric, and total variance."""
        mean, epistemic_var, aleatoric_var, total_var = self.predict_components(x)
        return UQResult(
            mean=mean,
            epistemic_var=epistemic_var,
            aleatoric_var=aleatoric_var,
            total_var=total_var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "n_members": len(self.models),
                "min_variance": self.min_variance,
            },
        )


class DeepEnsembleClassifier(_BaseDeepEnsemble):
    """Classification ensemble using member-wise logits and probability averaging.

    Shape contract
    --------------
    - input: any tensor accepted by the wrapped classifier
    - member output: logits with shape ``[batch, n_classes]``
    - ``predict`` returns ``(mean_probs, probs_var)``
    """

    method_name = "deep_ensemble_classifier"

    def _default_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if target.dim() > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        return F.cross_entropy(prediction, target.long())

    @torch.inference_mode()
    def predict_members(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for model in self.models:
            model.eval()
            logits = model(x)
            probs.append(torch.softmax(logits, dim=-1).unsqueeze(0))
        return torch.cat(probs, dim=0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        member_probs = self.predict_members(x)
        mean_probs = member_probs.mean(dim=0)
        probs_var = member_probs.var(dim=0, unbiased=False)
        return mean_probs, probs_var

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return mean class probabilities and probability variance."""
        probs, probs_var = self.predict(x)
        return UQResult(
            mean=probs,
            epistemic_var=probs_var,
            aleatoric_var=None,
            total_var=probs_var,
            probs=probs,
            probs_var=probs_var,
            metadata={
                "method": self.method_name,
                "n_members": len(self.models),
            },
        )


class MultiOutputDeepEnsembleRegressor(DeepEnsembleRegressor):
    """Multi-output regression ensemble with epistemic uncertainty.

    Member predictions are vector-valued, typically with shape
    ``[batch, n_outputs]``.
    """

    method_name = "multi_output_deep_ensemble_regressor"


class HeteroscedasticMultiOutputDeepEnsembleRegressor(
    HeteroscedasticDeepEnsembleRegressor
):
    """Multi-output regression ensemble with epistemic and aleatoric uncertainty."""

    method_name = "heteroscedastic_multi_output_deep_ensemble_regressor"


class DeepEnsembleWrapper(DeepEnsembleRegressor):
    """Backward-compatible alias for the original regression-first ensemble wrapper."""

    method_name = "deep_ensemble"
