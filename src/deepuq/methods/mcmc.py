import torch
from torch import nn

class SGLDOptimizer(torch.optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011)

    Performs SGD with additive Gaussian noise calibrated by the step size.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
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
                p.add_( -lr * grad + noise )

def collect_posterior_samples(model: nn.Module, data_loader, n_steps=1000, lr=1e-4, weight_decay=1e-4, burn_in=0.2, device="cpu"):
    model.train()
    opt = SGLDOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    samples = []
    step = 0
    for epoch in range(10**6):  # loop until enough steps
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
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
    preds = []
    for s in samples:
        model.load_state_dict(s, strict=True)
        out = model(x.to(device))
        if apply_softmax:
            out = torch.softmax(out, dim=-1)
        preds.append(out.unsqueeze(0).cpu())
    preds = torch.cat(preds, dim=0)
    return preds.mean(0), preds.var(0, unbiased=False)
