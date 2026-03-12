"""Variational-inference primitives and wrappers for Deep-UQ.

This module centers on mean-field Bayes-by-Backprop layers and extends them to
three additional regression variants plus a scalable last-layer VI wrapper.
The public surface is designed to stay small:

- Bayesian layers are built from :class:`BayesianLinear`.
- ``vi_elbo_step(...)`` remains the shared training helper.
- ``predict_vi_uq(...)`` remains the shared Monte Carlo predictive helper.

The new model classes make task-specific behavior explicit through public
attributes such as ``task_type``, ``heteroscedastic``, and ``output_dim``.
That lets the training and prediction helpers implement current behavior
without inventing parallel APIs per VI variant.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepuq.types import UQResult

ActivationFn = Callable[[torch.Tensor], torch.Tensor]


def _resolve_activation(name: str) -> ActivationFn:
    """Return the activation function used by Bayesian MLP variants.

    Parameters
    ----------
    name:
        Activation name. Supported values are ``"relu"``, ``"gelu"``,
        ``"tanh"``, and ``"silu"``.
    """

    normalized = name.lower()
    if normalized == "relu":
        return F.relu
    if normalized == "gelu":
        return F.gelu
    if normalized == "tanh":
        return cast(ActivationFn, torch.tanh)
    if normalized == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation {name!r}.")


def _variance_from_raw(raw_var: torch.Tensor) -> torch.Tensor:
    """Convert an unconstrained noise parameter into a positive variance."""

    return F.softplus(raw_var) + 1e-6


class GaussianPosterior(nn.Module):
    """Diagonal Gaussian variational posterior over one parameter tensor.

    The posterior is parameterized by ``mu`` and ``rho``. We transform ``rho``
    with ``softplus`` to obtain ``sigma`` and guarantee strictly positive
    standard deviations.
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor):
        """Create a diagonal Gaussian posterior.

        Parameters
        ----------
        mu:
            Initial posterior mean tensor.
        rho:
            Unconstrained scale parameter. ``softplus(rho)`` defines the
            posterior standard deviation.
        """
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
        """Return ``log q(w)`` for a given sample ``w``."""

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
        """Initialize the Gaussian prior.

        Parameters
        ----------
        mu:
            Prior mean.
        sigma:
            Prior standard deviation.
        """

        self.mu = mu
        self.sigma = sigma

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        """Return ``log p(w)`` under the isotropic Gaussian prior."""

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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 0.1,
        prior_mu: float = 0.0,
    ):
        """Build a Bayesian fully connected layer.

        Parameters
        ----------
        in_features:
            Input feature dimension.
        out_features:
            Output feature dimension.
        prior_sigma:
            Standard deviation of the isotropic Gaussian prior.
        prior_mu:
            Mean of the isotropic Gaussian prior.
        """

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
        self.prior = GaussianPrior(prior_mu, prior_sigma)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Apply the Bayesian affine transform.

        Parameters
        ----------
        x:
            Input tensor whose trailing feature dimension equals
            ``in_features``.
        sample:
            If ``True``, sample weights and biases from the variational
            posterior. If ``False``, use posterior means.
        """

        w = self.weight_posterior.sample() if sample else self.weight_posterior.mu
        b = self.bias_posterior.sample() if sample else self.bias_posterior.mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        """Analytic ``KL(q || p)`` for diagonal Gaussian posterior vs prior."""

        qw_mu, qw_sigma = self.weight_posterior.mu, self.weight_posterior.sigma
        qb_mu, qb_sigma = self.bias_posterior.mu, self.bias_posterior.sigma
        pw_mu = self.prior.mu
        pw_sigma = self.prior.sigma

        kl_w = (
            torch.log(torch.full_like(qw_sigma, pw_sigma) / qw_sigma)
            + (qw_sigma**2 + (qw_mu - pw_mu) ** 2) / (2 * pw_sigma**2)
            - 0.5
        ).sum()
        kl_b = (
            torch.log(torch.full_like(qb_sigma, pw_sigma) / qb_sigma)
            + (qb_sigma**2 + (qb_mu - pw_mu) ** 2) / (2 * pw_sigma**2)
            - 0.5
        ).sum()
        return cast(torch.Tensor, kl_w + kl_b)


class _BayesianMLPBase(nn.Module):
    """Internal helper for Bayesian MLP-style VI models."""

    task_type = "regression"
    heteroscedastic = False

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        raw_output_dim: int,
        *,
        output_dim: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.raw_output_dim = raw_output_dim
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self.activation_name = activation
        self._activation = _resolve_activation(activation)

        dims = [input_dim] + list(hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [
                BayesianLinear(
                    dims[i],
                    dims[i + 1],
                    prior_sigma=prior_sigma,
                    prior_mu=prior_mu,
                )
                for i in range(len(dims) - 1)
            ]
        )
        last_in = dims[-1]
        self.output_layer = BayesianLinear(
            last_in,
            raw_output_dim,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
        )

    def _hidden_forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = x
        for layer in self.hidden_layers:
            h = self._activation(layer(h, sample=sample))
        return h

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Return the raw predictive tensor for input ``x``."""

        return self.output_layer(self._hidden_forward(x, sample=sample), sample=sample)

    def kl(self) -> torch.Tensor:
        """Sum KL terms from every Bayesian layer in the network."""

        total = self.output_layer.kl()
        for layer in self.hidden_layers:
            total = total + layer.kl()
        return total


class BayesByBackpropMLP(_BayesianMLPBase):
    """Convenience MLP composed from :class:`BayesianLinear` layers.

    This is the original mean-field Bayes-by-Backprop baseline in Deep-UQ. It
    remains intentionally generic and can be used for regression or
    classification depending on the chosen output dimension and criterion.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        prior_sigma: float = 0.1,
        prior_mu: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            output_dim,
            output_dim=output_dim,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            activation=activation,
        )
        self.task_type = "generic"
        self.heteroscedastic = False


class HeteroscedasticBayesByBackpropRegressor(_BayesianMLPBase):
    """Mean-field Bayesian regressor with input-dependent noise.

    The network predicts two values per target dimension: a predictive mean and
    an unconstrained variance parameter. The variance parameter is transformed
    with ``softplus`` so that the Gaussian likelihood remains valid.
    """

    task_type = "regression"
    heteroscedastic = True

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 64),
        output_dim: int = 1,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            2 * output_dim,
            output_dim=output_dim,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            activation=activation,
        )

    def split_prediction(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split raw output into predictive mean and raw variance tensor."""

        mean, raw_var = prediction.split(self.output_dim, dim=-1)
        return cast(torch.Tensor, mean), cast(torch.Tensor, raw_var)

    def predictive_variance(self, raw_var: torch.Tensor) -> torch.Tensor:
        """Return positive predictive variance from the raw network output."""

        return _variance_from_raw(raw_var)

    def nll(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return heteroscedastic Gaussian negative log likelihood."""

        mean, raw_var = self.split_prediction(prediction)
        var = self.predictive_variance(raw_var)
        per_output = 0.5 * (((target - mean) ** 2) / var + torch.log(var))
        return per_output.sum(dim=-1).mean()


class MultiOutputBayesByBackpropRegressor(_BayesianMLPBase):
    """Mean-field Bayesian regressor for vector-valued targets."""

    task_type = "regression"
    heteroscedastic = False

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(64, 64),
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            output_dim,
            output_dim=output_dim,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            activation=activation,
        )


class HeteroscedasticMultiOutputBayesByBackpropRegressor(_BayesianMLPBase):
    """Multi-output Bayesian regressor with per-output noise prediction."""

    task_type = "regression"
    heteroscedastic = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(64, 64),
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__(
            input_dim,
            hidden_dims,
            2 * output_dim,
            output_dim=output_dim,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            activation=activation,
        )

    def split_prediction(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split raw output into predictive mean and raw variance tensor."""

        mean, raw_var = prediction.split(self.output_dim, dim=-1)
        return cast(torch.Tensor, mean), cast(torch.Tensor, raw_var)

    def predictive_variance(self, raw_var: torch.Tensor) -> torch.Tensor:
        """Return positive predictive variance from the raw network output."""

        return _variance_from_raw(raw_var)

    def nll(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return heteroscedastic Gaussian NLL for vector-valued outputs."""

        mean, raw_var = self.split_prediction(prediction)
        var = self.predictive_variance(raw_var)
        per_output = 0.5 * (((target - mean) ** 2) / var + torch.log(var))
        return per_output.sum(dim=-1).mean()


class LastLayerVariationalInference(nn.Module):
    """Deterministic feature extractor with a Bayesian linear output head.

    This wrapper is the scalable VI path for larger backbones. The feature
    extractor stays deterministic and only the final affine map is treated with
    a variational posterior.

    Parameters
    ----------
    feature_extractor:
        Deterministic module that returns a tensor whose trailing dimension is
        ``feature_dim``.
    feature_dim:
        Size of the final feature dimension produced by ``feature_extractor``.
    output_dim:
        Number of regression outputs or classes.
    task:
        ``"regression"`` or ``"classification"``.
    heteroscedastic:
        If ``True`` and ``task='regression'``, the Bayesian head predicts both
        mean and input-dependent variance for each output dimension.
    prior_mu, prior_sigma:
        Gaussian prior parameters for the Bayesian head.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        output_dim: int,
        task: str = "regression",
        heteroscedastic: bool = False,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
    ):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self.feature_extractor = feature_extractor
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.task_type = task
        self.heteroscedastic = bool(heteroscedastic) and task == "regression"
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        raw_output_dim = output_dim * 2 if self.heteroscedastic else output_dim
        self.head = BayesianLinear(
            self.feature_dim,
            raw_output_dim,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
        )

    def _flatten_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                "feature_extractor must return a tensor whose trailing dimension "
                f"matches feature_dim={self.feature_dim}; got shape {tuple(features.shape)}."
            )
        leading_shape = tuple(features.shape[:-1])
        return features.reshape(-1, self.feature_dim), leading_shape

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Apply the deterministic backbone and Bayesian head.

        The wrapper preserves all leading dimensions returned by the feature
        extractor and applies the Bayesian linear head over the last dimension.
        """

        features = self.feature_extractor(x)
        flat_features, leading_shape = self._flatten_features(features)
        flat_output = self.head(flat_features, sample=sample)
        return flat_output.reshape(*leading_shape, flat_output.shape[-1])

    def kl(self) -> torch.Tensor:
        """Return the KL contribution of the Bayesian output head only."""

        return self.head.kl()

    def split_prediction(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split raw regression output into mean and raw variance tensors."""

        if not self.heteroscedastic:
            raise RuntimeError(
                "split_prediction is only valid for heteroscedastic regression."
            )
        mean, raw_var = prediction.split(self.output_dim, dim=-1)
        return cast(torch.Tensor, mean), cast(torch.Tensor, raw_var)

    def predictive_variance(self, raw_var: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained regression noise output into positive variance."""

        return _variance_from_raw(raw_var)

    def nll(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the heteroscedastic Gaussian NLL for regression wrappers."""

        if not self.heteroscedastic:
            raise RuntimeError("nll() is only defined for heteroscedastic regression.")
        mean, raw_var = self.split_prediction(prediction)
        var = self.predictive_variance(raw_var)
        per_output = 0.5 * (((target - mean) ** 2) / var + torch.log(var))
        return per_output.sum(dim=-1).mean()


# Public alias kept because downstream docs and examples refer to the generic
# Bayesian weight posterior as a regression-capable baseline.
BayesByBackpropRegressor = BayesByBackpropMLP


def vi_elbo_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_batches: int | None = None,
    n_batches: int | None = None,
    criterion: nn.Module | None = None,
    kl_weight: float = 1.0,
    mc_samples: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one Bayes-by-Backprop ELBO step.

    Parameters
    ----------
    model:
        Bayesian model exposing ``forward(sample=True)`` and ``kl()``.
        Heteroscedastic regression wrappers also expose ``nll(prediction,
        target)``; that internal likelihood term is used automatically when
        ``model.heteroscedastic`` is ``True``.
    x, y:
        Minibatch inputs and targets.
    num_batches:
        Canonical number of optimizer steps per epoch, usually
        ``len(train_loader)``. KL is scaled as ``KL / num_batches``.
    n_batches:
        Deprecated alias for ``num_batches`` kept for backward compatibility.
    criterion:
        Data-fit loss used when the model does not implement ``nll(...)``.
        Defaults to mean-squared error for models with ``task_type='regression'``
        and cross-entropy otherwise.
    kl_weight:
        Multiplicative weight for the scaled KL term.
    mc_samples:
        Number of stochastic forward passes used to Monte Carlo-average NLL and
        KL for a lower-variance ELBO estimate.

    Returns
    -------
    (loss, nll, kl):
        ``loss`` keeps graph for backprop. ``nll`` and ``kl`` are detached
        scalar tensors intended for logging.
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

    model_task = getattr(model, "task_type", "classification")
    use_model_nll = callable(getattr(model, "nll", None)) and bool(
        getattr(model, "heteroscedastic", False)
    )

    if criterion is None and not use_model_nll:
        if model_task == "regression":
            criterion = nn.MSELoss(reduction="mean")
        else:
            criterion = nn.CrossEntropyLoss(reduction="mean")

    nll_acc = torch.zeros((), device=x.device)
    kl_acc = torch.zeros((), device=x.device)
    for _ in range(mc_samples):
        prediction = model(x, sample=True)
        if use_model_nll:
            sample_nll = cast(torch.Tensor, model.nll(prediction, y))
        else:
            assert criterion is not None
            sample_nll = cast(torch.Tensor, criterion(prediction, y))
        nll_acc = nll_acc + sample_nll
        kl_acc = kl_acc + (cast(torch.Tensor, model.kl()) / float(num_batches))

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
    aleatoric_var: torch.Tensor | None = None,
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
        If ``True``, treat outputs as logits and return probability moments.
        Classification wrappers also enable this automatically.
    aleatoric_var:
        Optional additive aleatoric variance term for plain regression models.

    Returns
    -------
    UQResult
        Regression calls populate ``mean`` and variance fields. Classification
        calls populate ``mean``/``probs`` and ``probs_var`` after softmax
        averaging.
    """

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")

    model.eval()
    task_type = getattr(model, "task_type", "regression")
    is_classification = apply_softmax or task_type == "classification"
    is_heteroscedastic = bool(getattr(model, "heteroscedastic", False))

    draws: list[torch.Tensor] = []
    noise_draws: list[torch.Tensor] = []
    for _ in range(n_samples):
        raw_output = model(x, sample=True)
        if is_classification:
            draws.append(torch.softmax(raw_output, dim=-1).unsqueeze(0))
            continue

        if is_heteroscedastic:
            if not hasattr(model, "split_prediction") or not hasattr(
                model, "predictive_variance"
            ):
                raise ValueError(
                    "Heteroscedastic VI models must define split_prediction(...) "
                    "and predictive_variance(...)."
                )
            mean_draw, raw_var = model.split_prediction(raw_output)
            var_draw = model.predictive_variance(raw_var)
            draws.append(mean_draw.unsqueeze(0))
            noise_draws.append(var_draw.unsqueeze(0))
        else:
            draws.append(raw_output.unsqueeze(0))

    stacked = torch.cat(draws, dim=0)
    mean = stacked.mean(dim=0)
    epistemic = stacked.var(dim=0, unbiased=False).clamp_min(0.0)

    if is_classification:
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

    if is_heteroscedastic:
        aleatoric = torch.cat(noise_draws, dim=0).mean(dim=0).clamp_min(0.0)
        total_var = (epistemic + aleatoric).clamp_min(0.0)
    elif aleatoric_var is not None:
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
        metadata={
            "method": "vi",
            "n_samples": int(n_samples),
            "task": "regression",
            "heteroscedastic": is_heteroscedastic,
        },
    )
