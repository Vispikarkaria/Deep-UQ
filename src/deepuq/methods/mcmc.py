"""MCMC utilities based on Stochastic Gradient Langevin Dynamics (SGLD) and HMC."""

import math

import torch
from torch import nn

from deepuq.types import UQResult


class SGLDOptimizer(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.

    This optimizer performs an SGD-like update with additive Gaussian noise
    calibrated by the step size, following Welling & Teh (2011).

    Parameters
    ----------
    params:
        Iterable of parameters to optimize.
    lr:
        SGLD step size.
    weight_decay:
        Optional L2 penalty added to the stochastic gradient.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Apply one SGLD parameter update in-place.

        Returns
        -------
        None
            The update is applied directly to the optimizer parameters.
        """
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad + wd * p
                noise = torch.randn_like(p) * (2 * lr) ** 0.5
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
    loss_fn:
        Loss used to compute stochastic gradients. Defaults to cross-entropy.
    device:
        Device on which optimization runs.

    Returns
    -------
    list[dict[str, torch.Tensor]]
        State-dict snapshots collected after burn-in. Each element can be fed
        into ``predict_with_samples`` or ``predict_with_samples_uq``.
    """
    model.train()
    opt = SGLDOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    samples = []
    step = 0
    for _epoch in range(10**6):  # loop until enough steps
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
                samples.append(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                )
            if step >= n_steps:
                return samples
    return samples


@torch.inference_mode()
def predict_with_samples(
    model: nn.Module, samples, x, apply_softmax=True, device="cpu"
):
    """Predictive mean and variance from stored parameter samples.

    Parameters
    ----------
    model:
        Model architecture compatible with the saved state dicts.
    samples:
        Posterior parameter snapshots, typically from
        ``collect_posterior_samples``.
    x:
        Evaluation inputs.
    apply_softmax:
        If ``True``, convert logits into probabilities before aggregation.
    device:
        Device used for model evaluation.

    Returns
    -------
    (mean, var):
        Predictive mean and variance over the posterior sample dimension.
    """
    pred_samples = []
    for s in samples:
        model.load_state_dict(s, strict=True)
        out = model(x.to(device))
        if apply_softmax:
            out = torch.softmax(out, dim=-1)
        pred_samples.append(out.unsqueeze(0).cpu())
    pred_tensor = torch.cat(pred_samples, dim=0)
    return pred_tensor.mean(0), pred_tensor.var(0, unbiased=False)


@torch.inference_mode()
def predict_with_samples_uq(
    model: nn.Module, samples, x, apply_softmax=True, device="cpu"
) -> UQResult:
    """Return posterior-sample predictive moments in ``UQResult`` form.

    ``epistemic_var`` stores the variance across posterior samples. No separate
    aleatoric component is estimated.
    """
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


class SGHMCOptimizer(torch.optim.Optimizer):
    """Stochastic Gradient Hamiltonian Monte Carlo optimizer.

    Maintains a velocity buffer per parameter and applies the SGHMC update:
        v = (1 - momentum_decay) * v - lr * grad + N(0, 2*momentum_decay*lr) * noise_scale
        theta = theta + v

    Parameters
    ----------
    params:
        Iterable of parameters to optimize.
    lr:
        Step size.
    momentum_decay:
        Friction coefficient for the velocity.
    noise_scale:
        Scaling factor for the injected noise.
    num_training_samples:
        Number of training samples (used for gradient scaling context).
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        momentum_decay=0.01,
        noise_scale=1.0,
        num_training_samples=1000,
    ):
        defaults = dict(
            lr=lr,
            momentum_decay=momentum_decay,
            noise_scale=noise_scale,
            num_training_samples=num_training_samples,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """Apply one SGHMC parameter update in-place."""
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["momentum_decay"]
            noise_scale = group["noise_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)
                v = state["velocity"]
                noise_std = math.sqrt(2 * alpha * lr) * noise_scale
                noise = torch.randn_like(p) * noise_std
                v.mul_(1 - alpha).add_(-lr * p.grad + noise)
                p.add_(v)


class CyclicalSGMCMC:
    """Cyclical Stochastic Gradient MCMC for posterior sampling.

    Uses cosine annealing within each cycle and collects samples at the end
    of each cycle (low LR region).

    Parameters
    ----------
    model:
        Neural network to sample from.
    base_optimizer_cls:
        Optimizer class (e.g. SGHMCOptimizer or SGLDOptimizer).
    cycle_length:
        Number of training steps per cycle.
    n_cycles:
        Number of full cycles to run.
    samples_per_cycle:
        Number of posterior samples to collect at the end of each cycle.
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer_cls,
        cycle_length: int = 50,
        n_cycles: int = 4,
        samples_per_cycle: int = 3,
    ):
        self.model = model
        self.base_optimizer_cls = base_optimizer_cls
        self.cycle_length = cycle_length
        self.n_cycles = n_cycles
        self.samples_per_cycle = samples_per_cycle

    def run(self, train_loader, loss_fn) -> list[dict[str, torch.Tensor]]:
        """Execute cyclical SGMCMC and return collected posterior samples.

        Parameters
        ----------
        train_loader:
            Iterable of (inputs, targets) mini-batches.
        loss_fn:
            Loss function for computing gradients.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            Collected state-dict snapshots.
        """
        self.model.train()
        optimizer = self.base_optimizer_cls(self.model.parameters())
        base_lr = optimizer.param_groups[0]["lr"]

        samples: list[dict[str, torch.Tensor]] = []
        # Steps within each cycle where we collect samples
        collect_start = self.cycle_length - self.samples_per_cycle

        step = 0
        data_iter = iter(train_loader)
        for _cycle in range(self.n_cycles):
            for t in range(self.cycle_length):
                # Cosine annealing within cycle
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * t / self.cycle_length))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Get batch
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                # Collect at end of cycle (low LR region)
                if t >= collect_start:
                    samples.append(
                        {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                    )

                step += 1

        return samples
