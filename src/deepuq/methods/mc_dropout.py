"""Monte-Carlo dropout inference wrappers.

The wrapper in this module keeps dropout layers active at inference time,
collects repeated stochastic forward passes, and summarizes their spread as a
predictive uncertainty estimate.
"""

import torch
import torch.nn as nn

from deepuq.types import UQResult


class MCDropoutWrapper(nn.Module):
    """Wrap a dropout-enabled model to perform MC Dropout at inference.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` that already contains dropout layers.
    n_mc:
        Number of stochastic forward passes used at prediction time.
    apply_softmax:
        If ``True``, interpret model outputs as logits and return
        probability-space moments.
    """

    def __init__(self, model: nn.Module, n_mc: int = 20, apply_softmax: bool = True):
        super().__init__()
        self.model = model
        self.n_mc = n_mc
        self.apply_softmax = apply_softmax

    def train(self, mode: bool = True):
        """Mirror the wrapped model's train/eval state.

        MC Dropout still forces dropout-active behavior inside ``predict`` and
        ``predict_uq`` by temporarily calling ``self.model.train(True)``.
        """
        # Override: we want to be able to force dropout at eval-time
        self.model.train(mode)
        return super().train(mode)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        """Run stochastic dropout passes and return predictive mean/variance.

        Parameters
        ----------
        x:
            Input batch with shape accepted by the wrapped model.

        Returns
        -------
        (mean, var):
            Tensors with the same trailing shape as a single model prediction.
            For classification with ``apply_softmax=True``, the last dimension
            is class probability.
        """
        self.model.train(True)  # enable dropout
        pred_samples = []
        for _ in range(self.n_mc):
            out = self.model(x)
            if self.apply_softmax:
                out = torch.softmax(out, dim=-1)
            pred_samples.append(out.unsqueeze(0))
        pred_tensor = torch.cat(pred_samples, dim=0)  # [K,B,C]
        mean = pred_tensor.mean(dim=0)
        var = pred_tensor.var(dim=0, unbiased=False)
        self.model.eval()
        return mean, var

    @torch.inference_mode()
    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return predictive moments as a :class:`deepuq.types.UQResult`.

        The wrapper reports dropout spread as ``epistemic_var``. No explicit
        aleatoric component is modeled.
        """
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
