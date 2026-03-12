# VI API

This page documents the public variational-inference helpers behind Bayes by Backprop.
Use the generated section below for signatures and source-linked docstrings, and the notes here for workflow and shape expectations.

## Public objects

- `GaussianPosterior`
- `GaussianPrior`
- `BayesianLinear`
- `BayesByBackpropMLP`
- `vi_elbo_step`
- `predict_vi_uq`

## Parameter and variable conventions

| Name | Meaning |
|---|---|
| `prior_sigma` | standard deviation of the Gaussian prior |
| `num_batches` | optimizer steps per epoch used to scale the KL term |
| `n_batches` | deprecated alias for `num_batches` |
| `kl_weight` | multiplicative KL weight, often interpreted as `eta` |
| `mc_samples` | number of stochastic forward passes used to estimate the ELBO |
| `n_samples` | number of stochastic predictive samples in `predict_vi_uq(...)` |
| `apply_softmax` | interpret outputs as logits and return probability moments |

## Input and output shapes

- `BayesianLinear.forward(x, sample=True)` expects the trailing feature dimension to equal `in_features`.
- `BayesByBackpropMLP.forward(x, sample=True)` typically uses `x` with shape `[batch, input_dim]`.
- `vi_elbo_step(model, x, y, ...)` expects a minibatch `x` and targets `y` compatible with the provided `criterion`.
- `predict_vi_uq(...)` returns tensors with the same trailing shape as one stochastic model output.

## `UQResult` mapping

`predict_vi_uq(...)` populates:

- regression: `mean`, `epistemic_var`, optional `aleatoric_var`, `total_var`
- classification (`apply_softmax=True`): `mean`, `probs`, `probs_var`, and `epistemic_var`

## Common preconditions and failure modes

- `num_batches` must be a positive integer; omitting it raises `ValueError`
- `n_batches` is accepted only as a backward-compatible alias
- `mc_samples` and `n_samples` must be positive integers
- the model passed to `vi_elbo_step(...)` and `predict_vi_uq(...)` must support `forward(sample=True)` and `kl()`
- classification and regression target shapes must match the provided `criterion`

## Minimal example

```python
model = BayesByBackpropMLP(input_dim=8, hidden_dims=[32, 32], output_dim=1)
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
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [Variational Inference theory](../../methods/variational-inference.md)

::: deepuq.methods.vi
