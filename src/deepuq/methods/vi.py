"""Variational-inference primitives for Bayes-by-Backprop.

The implementation here follows a mean-field setup:
- each Bayesian parameter tensor has an independent Gaussian posterior,
- Bayesian layers sample weights during stochastic forward passes,
- KL(q || p) against a Gaussian prior is computed analytically,
- the ELBO helper combines data-fit loss + scaled KL term.
"""

import math
import warnings
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepuq.types import UQResult


class GaussianPosterior(nn.Module):
    """Diagonal Gaussian variational posterior over one parameter tensor.

    The posterior is parameterized by ``mu`` and ``rho``. We transform ``rho``
    with ``softplus`` to obtain ``sigma`` and guarantee strictly positive
    standard deviations.
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

    @property
    def sigma(self) -> torch.Tensor:
        # Softplus keeps sigma > 0 and avoids manual clamping.
        return torch.log1p(torch.exp(self.rho))

    def sample(self) -> torch.Tensor:
        # Reparameterization trick: w = mu + sigma * eps, eps ~ N(0, I).
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """Return log q(w) for a given sample ``w``."""
        return cast(
            torch.Tensor,
            (
                -0.5 * ((w - self.mu) / self.sigma).pow(2)
                - torch.log(self.sigma)
                - 0.5 * math.log(2 * math.pi)
            ).sum(),
        )


class GaussianPrior:
    """Isotropic Gaussian prior used by Bayesian layers."""

    def __init__(self, mu: float = 0.0, sigma: float = 0.1):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """Return log p(w) under the isotropic Gaussian prior."""
        return (
            -0.5 * ((w - self.mu) / self.sigma).pow(2)
            - math.log(self.sigma)
            - 0.5 * math.log(2 * math.pi)
        ).sum()


class BayesianLinear(nn.Module):
    """Fully-connected layer with Bayesian weights and biases.

    During ``sample=True`` forward passes, weights are sampled from the
    posterior. During ``sample=False`` passes, posterior means are used.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_posterior = GaussianPosterior(
            torch.zeros(out_features, in_features),
            torch.full((out_features, in_features), -3.0),
        )
        self.bias_posterior = GaussianPosterior(
            torch.zeros(out_features),
            torch.full((out_features,), -3.0),
        )
        self.prior = GaussianPrior(0.0, prior_sigma)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        # ``sample=False`` is useful for deterministic warmup or debugging.
        w = self.weight_posterior.sample() if sample else self.weight_posterior.mu
        b = self.bias_posterior.sample() if sample else self.bias_posterior.mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        """Analytic KL(q || p) for diagonal Gaussian posterior vs Gaussian prior.

        For each scalar component:
            KL = log(sigma_p / sigma_q)
                 + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2)
                 - 1/2
        Here ``mu_p = 0`` so the squared-mean term becomes ``mu_q^2``.
        """

        qw_mu, qw_sigma = self.weight_posterior.mu, self.weight_posterior.sigma
        qb_mu, qb_sigma = self.bias_posterior.mu, self.bias_posterior.sigma
        pw_sigma = self.prior.sigma
        pb_sigma = self.prior.sigma

        kl_w = (
            torch.log(pw_sigma / qw_sigma)
            + (qw_sigma**2 + qw_mu**2) / (2 * pw_sigma**2)
            - 0.5
        ).sum()
        kl_b = (
            torch.log(pb_sigma / qb_sigma)
            + (qb_sigma**2 + qb_mu**2) / (2 * pb_sigma**2)
            - 0.5
        ).sum()
        return cast(torch.Tensor, kl_w + kl_b)


class BayesByBackpropMLP(nn.Module):
    """Convenience MLP composed from ``BayesianLinear`` layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        prior_sigma: float = 0.1,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [BayesianLinear(dims[i], dims[i + 1], prior_sigma), nn.ReLU()]
        layers += [BayesianLinear(dims[-2], dims[-1], prior_sigma)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = x
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                h = layer(h, sample=sample)
            else:
                h = layer(h)
        return h

    def kl(self) -> torch.Tensor:
        """Sum KL terms from all Bayesian layers."""
        kl_terms = [m.kl() for m in self.layers if isinstance(m, BayesianLinear)]
        if not kl_terms:
            return torch.zeros((), device=next(self.parameters()).device)

        total = kl_terms[0]
        for term in kl_terms[1:]:
            total = total + term
        return total


def vi_elbo_step(
    model,
    x,
    y,
    num_batches: Optional[int] = None,
    n_batches: Optional[int] = None,
    criterion=None,
    kl_weight: float = 1.0,
    mc_samples: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one Bayes-by-Backprop ELBO step.

    Parameters
    ----------
    model:
        Bayesian model exposing ``forward(sample=True)`` and ``kl()``.
    x, y:
        Mini-batch inputs and targets.
    num_batches:
        Canonical number of optimizer steps per epoch, usually
        ``len(train_loader)``. KL is scaled as ``KL / num_batches``.
    n_batches:
        Deprecated alias for ``num_batches`` kept for backward compatibility.
    criterion:
        Data-fit loss. Defaults to ``nn.CrossEntropyLoss(reduction='mean')``.
    kl_weight:
        Multiplicative weight for the scaled KL term.
    mc_samples:
        Number of stochastic forward passes used to Monte Carlo-average NLL and
        KL for a lower-variance ELBO estimate.

    Returns
    -------
    (loss, nll, kl):
        ``loss`` keeps graph for backprop.
        ``nll`` and ``kl`` are detached tensors intended for logging.
    """
    if num_batches is None:
        if n_batches is None:
            raise ValueError("num_batches must be provided and > 0.")
        warnings.warn(
            "Argument 'n_batches' is deprecated; use 'num_batches' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        num_batches = n_batches
    elif n_batches is not None and n_batches != num_batches:
        raise ValueError("Provide only one batch-count value or make them equal.")

    if not isinstance(num_batches, int) or num_batches <= 0:
        raise ValueError(
            f"num_batches must be a positive integer, got {num_batches!r}."
        )
    if not isinstance(mc_samples, int) or mc_samples <= 0:
        raise ValueError(f"mc_samples must be a positive integer, got {mc_samples!r}.")

    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    nll_acc = torch.zeros((), device=x.device)
    kl_acc = torch.zeros((), device=x.device)
    for _ in range(mc_samples):
        # Each pass samples a new weight realization through model.forward(..., sample=True).
        logits = model(x, sample=True)
        nll_acc = nll_acc + criterion(logits, y)
        # KL is normalized by steps/epoch so objective scale is batch-size-stable.
        kl_acc = kl_acc + (model.kl() / float(num_batches))

    nll = nll_acc / float(mc_samples)
    kl = kl_acc / float(mc_samples)
    loss = nll + kl_weight * kl
    return loss, nll.detach(), kl.detach()


@torch.inference_mode()
def predict_vi_uq(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 50,
    apply_softmax: bool = False,
    aleatoric_var: Optional[torch.Tensor] = None,
) -> UQResult:
    """Monte Carlo predictive summary for Bayes-by-Backprop models.

    Parameters
    ----------
    model:
        Bayesian model supporting ``forward(sample=True)``.
    x:
        Inputs.
    n_samples:
        Number of stochastic weight samples.
    apply_softmax:
        If True, treat outputs as logits and return probability moments.
    aleatoric_var:
        Optional additive aleatoric variance term for regression.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")

    model.eval()
    draws = []
    for _ in range(n_samples):
        out = model(x, sample=True)
        if apply_softmax:
            out = torch.softmax(out, dim=-1)
        draws.append(out.unsqueeze(0))

    stacked = torch.cat(draws, dim=0)
    mean = stacked.mean(dim=0)
    epistemic = stacked.var(dim=0, unbiased=False).clamp_min(0.0)

    if apply_softmax:
        return UQResult(
            mean=mean,
            epistemic_var=epistemic,
            aleatoric_var=None,
            total_var=epistemic,
            probs=mean,
            probs_var=epistemic,
            metadata={
                "method": "vi",
                "n_samples": int(n_samples),
                "task": "classification",
            },
        )

    if aleatoric_var is not None:
        aleatoric = aleatoric_var.to(mean.device, mean.dtype)
        total_var = (epistemic + aleatoric).clamp_min(0.0)
    else:
        aleatoric = None
        total_var = epistemic

    return UQResult(
        mean=mean,
        epistemic_var=epistemic,
        aleatoric_var=aleatoric,
        total_var=total_var,
        probs=None,
        probs_var=None,
        metadata={"method": "vi", "n_samples": int(n_samples), "task": "regression"},
    )
