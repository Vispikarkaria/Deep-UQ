import torch
import torch.nn as nn

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
