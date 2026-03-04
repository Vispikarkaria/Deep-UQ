"""MCMC utilities based on Stochastic Gradient Langevin Dynamics (SGLD)."""

import torch
from torch import nn

from deepuq.types import UQResult


class SGLDOptimizer(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    This optimizer performs an SGD-like update with additive Gaussian noise
    calibrated by the step size, following Welling & Teh (2011).
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Apply one SGLD parameter update in-place."""
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad + wd * p
                noise = torch.randn_like(p) * (2 * lr)**0.5
                p.add_(-lr * grad + noise)

def collect_posterior_samples(
    model: nn.Module,
    data_loader,
    n_steps=1000,
    lr=1e-4,
    weight_decay=1e-4,
    burn_in=0.2,
    loss_fn=None,
    device="cpu",
):
    """Run SGLD and collect posterior parameter snapshots.

    Parameters
    ----------
    model:
        Neural network to sample.
    data_loader:
        Iterable of mini-batches.
    n_steps:
        Total SGLD updates.
    burn_in:
        Fraction of updates to skip before collecting snapshots.
    """
    model.train()
    opt = SGLDOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    samples = []
    step = 0
    for epoch in range(10**6):  # loop until enough steps
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            step += 1
            if step > int(burn_in * n_steps):
                # store a copy of parameters
                samples.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if step >= n_steps:
                return samples
    return samples

@torch.inference_mode()
def predict_with_samples(model: nn.Module, samples, x, apply_softmax=True, device="cpu"):
    """Predictive mean and variance from stored parameter samples."""
    preds = []
    for s in samples:
        model.load_state_dict(s, strict=True)
        out = model(x.to(device))
        if apply_softmax:
            out = torch.softmax(out, dim=-1)
        preds.append(out.unsqueeze(0).cpu())
    preds = torch.cat(preds, dim=0)
    return preds.mean(0), preds.var(0, unbiased=False)


@torch.inference_mode()
def predict_with_samples_uq(model: nn.Module, samples, x, apply_softmax=True, device="cpu") -> UQResult:
    """Predictive uncertainty summary from posterior samples."""
    mean, var = predict_with_samples(
        model=model,
        samples=samples,
        x=x,
        apply_softmax=apply_softmax,
        device=device,
    )
    return UQResult(
        mean=mean,
        epistemic_var=var,
        aleatoric_var=None,
        total_var=var,
        probs=mean if apply_softmax else None,
        probs_var=var if apply_softmax else None,
        metadata={
            "method": "sgld",
            "num_samples": int(len(samples)),
            "apply_softmax": bool(apply_softmax),
        },
    )
