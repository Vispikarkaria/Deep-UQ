"""Monte-Carlo dropout inference wrappers."""

import torch
import torch.nn as nn

from deepuq.types import UQResult


class MCDropoutWrapper(nn.Module):
    """
    Wrap any model with dropout to perform MC Dropout at inference.

    Args:
        model: torch.nn.Module with Dropout layers
        n_mc: number of stochastic forward passes
        apply_softmax: whether to convert logits to probabilities
    """

    def __init__(self, model: nn.Module, n_mc: int = 20, apply_softmax: bool = True):
        super().__init__()
        self.model = model
        self.n_mc = n_mc
        self.apply_softmax = apply_softmax

    def train(self, mode: bool = True):
        # Override: we want to be able to force dropout at eval-time
        self.model.train(mode)
        return super().train(mode)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        """Run stochastic dropout passes and return predictive mean/variance."""
        self.model.train(True)  # enable dropout
        preds = []
        for _ in range(self.n_mc):
            out = self.model(x)
            if self.apply_softmax:
                out = torch.softmax(out, dim=-1)
            preds.append(out.unsqueeze(0))
        preds = torch.cat(preds, dim=0)  # [K,B,C]
        mean = preds.mean(dim=0)
        var = preds.var(dim=0, unbiased=False)
        self.model.eval()
        return mean, var

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return standardized uncertainty fields."""
        mean, var = self.predict(x)
        probs = mean if self.apply_softmax else None
        probs_var = var if self.apply_softmax else None
        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=probs,
            probs_var=probs_var,
            metadata={
                "method": "mc_dropout",
                "n_mc": int(self.n_mc),
                "apply_softmax": bool(self.apply_softmax),
            },
        )
