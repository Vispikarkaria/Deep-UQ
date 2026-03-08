from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn

from deepuq.types import UQResult


class DeepEnsembleWrapper(nn.Module):
    """Regression-first deep ensemble baseline.

    The wrapper owns multiple independently initialized model copies and provides
    simple supervised fitting plus predictive uncertainty aggregation.
    """

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        if len(models) == 0:
            raise ValueError("DeepEnsembleWrapper requires at least one model.")
        self.models = nn.ModuleList(models)

    def fit(
        self,
        train_loader,
        *,
        epochs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> "DeepEnsembleWrapper":
        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        for model_idx, model in enumerate(self.models):
            if seed is not None:
                torch.manual_seed(seed + model_idx)
            if device is not None:
                model.to(device)
            optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    if device is not None:
                        xb = xb.to(device)
                        yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()
        return self

    @torch.inference_mode()
    def predict_members(self, x: torch.Tensor) -> torch.Tensor:
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
        mean, var = self.predict(x)
        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": "deep_ensemble",
                "n_members": len(self.models),
            },
        )
