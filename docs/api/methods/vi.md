# Variational Inference API

This page documents the public variational-inference classes and helpers behind
Bayes by Backprop in `deepuq`. Use this page for the concrete API contract:
constructor arguments, tensor shapes, return semantics, and workflow
requirements. Use the method guide for the mathematical derivations and broader
tradeoffs.

## Public objects

- `GaussianPosterior`
- `GaussianPrior`
- `BayesianLinear`
- `BayesByBackpropMLP`
- `BayesByBackpropRegressor`
- `HeteroscedasticBayesByBackpropRegressor`
- `MultiOutputBayesByBackpropRegressor`
- `HeteroscedasticMultiOutputBayesByBackpropRegressor`
- `LastLayerVariationalInference`
- `vi_elbo_step`
- `predict_vi_uq`

## Workflow summary

The VI workflow in `deepuq` is consistent across all public VI models:

1. Instantiate a VI model.
2. Train it with `vi_elbo_step(...)` inside the optimizer loop.
3. Draw Monte Carlo predictive samples with `predict_vi_uq(...)`.
4. Read predictive moments through the returned `UQResult`.

The important distinction between the model variants is the structure of the raw
model output:

- plain regression returns a mean only
- heteroscedastic regression returns mean and raw log-variance channels
- classification returns logits
- multi-output regression returns one mean per output dimension
- last-layer VI wraps a deterministic feature extractor and Bayesian linear head

## Parameter and variable conventions

| Name | Meaning | Where it appears |
|---|---|---|
| `prior_mu` | Mean of the Gaussian prior placed on Bayesian weights. | Bayesian layers and VI model constructors |
| `prior_sigma` | Standard deviation of the Gaussian prior. Lower values tighten the posterior around zero. | Bayesian layers and VI model constructors |
| `input_dim` | Size of the final feature dimension expected by dense VI models. | `BayesByBackpropMLP` and regression variants |
| `hidden_dims` | Width of each hidden Bayesian MLP layer. | Dense VI models |
| `output_dim` | Number of regression outputs or classes. | Multi-output models and last-layer VI |
| `activation` | Hidden-layer nonlinearity. Supported values are documented by the source docstrings. | Dense VI models |
| `feature_extractor` | Deterministic backbone whose final features feed a Bayesian head. | `LastLayerVariationalInference` |
| `feature_dim` | Size of the feature dimension produced by `feature_extractor`. | `LastLayerVariationalInference` |
| `task` | Either `"regression"` or `"classification"`. Controls likelihood semantics and prediction formatting. | `LastLayerVariationalInference` |
| `heteroscedastic` | Whether the regression head also predicts observation noise. | `LastLayerVariationalInference` |
| `num_batches` | Number of optimizer steps per epoch. Used to scale the KL term in `vi_elbo_step(...)`. | `vi_elbo_step(...)` |
| `n_batches` | Deprecated alias for `num_batches`. | `vi_elbo_step(...)` |
| `kl_weight` | Multiplicative factor applied to the scaled KL term. Often written as `$\beta$` in the ELBO. | `vi_elbo_step(...)` |
| `mc_samples` | Number of stochastic forward passes used to estimate the training ELBO for one minibatch. | `vi_elbo_step(...)` |
| `n_samples` | Number of stochastic predictive forward passes used for Monte Carlo uncertainty estimates. | `predict_vi_uq(...)` |
| `apply_softmax` | Interpret raw outputs as class logits and average probabilities instead of raw outputs. | `predict_vi_uq(...)` |
| `aleatoric` | Optional additive variance term passed into `predict_vi_uq(...)` for plain regression models. | `predict_vi_uq(...)` |

## Input and output shapes

### Dense Bayes-by-Backprop models

- `BayesianLinear.forward(x, sample=True)` expects the trailing feature
dimension of `x` to equal `in_features`.
- `BayesByBackpropMLP` and the dense regression variants typically consume
`x` with shape `[batch, input_dim]` and return:
  - plain regression: `[batch, output_dim]`
  - heteroscedastic regression: `[batch, 2 * output_dim]`
  - classification: `[batch, num_classes]`

### Last-layer VI

`LastLayerVariationalInference` is more general. Its deterministic feature
extractor may emit either:

- `[batch, feature_dim]` for dense features, or
- `[batch, ..., feature_dim]` for spatial or sequence-style features.

The Bayesian head is applied over the last dimension and preserves the leading
dimensions. Examples:

- dense regression: `[batch, feature_dim] -> [batch, output_dim]`
- dense classification: `[batch, feature_dim] -> [batch, num_classes]`
- spatial regression head: `[batch, height, width, feature_dim] -> [batch, height, width, output_dim]`

### Training helper

`vi_elbo_step(model, x, y, ...)` expects `x` and `y` to be compatible with one
stochastic forward pass of `model`. If the model implements `nll(prediction,
target)`, that method is used. Otherwise the helper falls back to the provided
`criterion`, or to a default likelihood based on `model.task_type`.

### Prediction helper

`predict_vi_uq(model, x, n_samples=...)` stacks `n_samples` stochastic forward
passes and returns moments with the same leading shape as one model output.
Examples:

- scalar regression: `mean.shape == [batch, 1]`
- multi-output regression: `mean.shape == [batch, output_dim]`
- classification: `probs.shape == [batch, num_classes]`
- spatial last-layer VI: `mean.shape == [batch, height, width, output_dim]`

## `UQResult` mapping

`predict_vi_uq(...)` returns a `UQResult` with semantics determined by the model
family:

| Variant | Populated fields |
|---|---|
| Plain regression | `mean`, `epistemic_var`, `total_var`, `metadata` |
| Heteroscedastic regression | `mean`, `epistemic_var`, `aleatoric_var`, `total_var`, `metadata` |
| Multi-output regression | same as plain or heteroscedastic regression, but with an extra output dimension |
| Classification | `mean`, `probs`, `probs_var`, `epistemic_var`, `metadata` |
| Last-layer VI | follows the same rules as its configured `task` and `heteroscedastic` setting |

Interpretation details:

- `mean` is always the Monte Carlo predictive mean.
- `epistemic_var` is the variance across stochastic weight samples.
- `aleatoric_var` is populated only when the model predicts observation noise,
  or when an explicit `aleatoric=` tensor is supplied for plain regression.
- `total_var` is `epistemic_var + aleatoric_var` when aleatoric variance exists;
  otherwise it equals `epistemic_var`.
- `probs` and `probs_var` are classification-specific convenience views.

For field-by-field details, see [Types API](../types.md).

## Common preconditions and failure modes

- `num_batches` must be a positive integer. Omitting it or setting it to zero
  raises `ValueError`.
- `n_batches` is accepted only as a deprecated alias for backward compatibility.
- `mc_samples` and `n_samples` must both be positive.
- VI models used with `vi_elbo_step(...)` must expose `kl()`.
- `predict_vi_uq(...)` assumes `forward(sample=True)` is supported.
- Classification tasks require logits and integer class labels compatible with
  cross-entropy.
- Heteroscedastic regression assumes the output can be split into mean and raw
  variance channels.
- `LastLayerVariationalInference` requires the deterministic backbone to return
  a trailing feature dimension equal to `feature_dim`.

## Minimal examples

### Plain Bayes by Backprop regression

```python
import torch
from deepuq.methods import BayesByBackpropMLP, predict_vi_uq, vi_elbo_step

model = BayesByBackpropMLP(input_dim=8, hidden_dims=(32, 32), output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for xb, yb in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss, nll, kl = vi_elbo_step(
        model,
        xb,
        yb,
        num_batches=len(train_loader),
        criterion=torch.nn.MSELoss(),
        kl_weight=0.01,
        mc_samples=4,
    )
    loss.backward()
    optimizer.step()

uq = predict_vi_uq(model, x_test, n_samples=32)
print(uq.mean.shape, uq.total_var.shape)
```

### Heteroscedastic regression

```python
import torch
from deepuq.methods import (
    HeteroscedasticBayesByBackpropRegressor,
    predict_vi_uq,
    vi_elbo_step,
)

model = HeteroscedasticBayesByBackpropRegressor(
    input_dim=6,
    hidden_dims=(64, 64),
    prior_sigma=0.2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for xb, yb in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = vi_elbo_step(model, xb, yb, num_batches=len(train_loader))
    loss.backward()
    optimizer.step()

uq = predict_vi_uq(model, x_test, n_samples=64)
print(uq.mean.shape, uq.epistemic_var.shape, uq.aleatoric_var.shape)
```

### Last-layer VI classification

```python
import torch
import torch.nn as nn
from deepuq.methods import LastLayerVariationalInference, predict_vi_uq, vi_elbo_step

backbone = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
model = LastLayerVariationalInference(
    feature_extractor=backbone,
    feature_dim=16,
    output_dim=3,
    task="classification",
    prior_sigma=0.1,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for xb, yb in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = vi_elbo_step(model, xb, yb, num_batches=len(train_loader))
    loss.backward()
    optimizer.step()

uq = predict_vi_uq(model, x_test, n_samples=32, apply_softmax=True)
print(uq.probs.shape, uq.probs_var.shape)
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [Variational Inference method guide](../../methods/variational-inference.md)
- [Bayes by Backprop tutorial](../../tutorials/bayes-by-backprop.md)
- [Heteroscedastic Bayes by Backprop + ADR1D](../../tutorials/vi-heteroscedastic-bbb-adr1d.md)
- [Multi-Output Bayes by Backprop + Elastic Bar](../../tutorials/vi-multioutput-bbb-elastic-bar.md)
- [Heteroscedastic Multi-Output Bayes by Backprop + Transport2D](../../tutorials/vi-heteroscedastic-multioutput-bbb-transport2d.md)
- [Last-Layer VI + Heat2D Classification](../../tutorials/vi-last-layer-heat2d-classification.md)

::: deepuq.methods.vi
