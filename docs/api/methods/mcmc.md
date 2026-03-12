# MCMC API

This page documents the SGLD-based MCMC helpers provided by `deepuq.methods.mcmc`.
These helpers expose a lower-level workflow than the wrapper-style APIs, so the notes here focus on sample collection and predictive aggregation.

## Public objects

- `SGLDOptimizer`
- `collect_posterior_samples`
- `predict_with_samples`
- `predict_with_samples_uq`

## Parameter and variable conventions

| Name | Meaning |
|---|---|
| `lr` | SGLD step size |
| `weight_decay` | L2 penalty added to the stochastic gradient |
| `n_steps` | total SGLD updates |
| `burn_in` | fraction of early updates discarded before collecting samples |
| `loss_fn` | loss used to compute stochastic gradients |
| `samples` | list of state-dict snapshots collected after burn-in |
| `apply_softmax` | convert logits to probabilities before aggregating |
| `device` | device used for optimization or evaluation |

## Workflow expectations

1. instantiate a deterministic model
2. call `collect_posterior_samples(...)` with a training loader and loss
3. reuse the returned `samples` with `predict_with_samples(...)` or `predict_with_samples_uq(...)`

## Input and output shapes

- `collect_posterior_samples(...)` expects minibatches `(x, y)` from `data_loader`.
- `predict_with_samples(...)` returns tensors with the same trailing shape as one model forward pass.
- classification helpers typically use outputs shaped `[batch, n_classes]`.

## `UQResult` mapping

`predict_with_samples_uq(...)` populates:

- regression: `mean`, `epistemic_var`, `total_var`
- classification (`apply_softmax=True`): `mean`, `probs`, `probs_var`, and `epistemic_var`

## Common preconditions and failure modes

- the architecture used for prediction must match the architecture used to collect `samples`
- `burn_in` should be in `[0, 1)` to keep a meaningful number of posterior samples
- `loss_fn` must match the task; the default is cross-entropy
- `apply_softmax=True` should only be used when the model emits logits

## Minimal example

```python
samples = collect_posterior_samples(
    model,
    train_loader,
    n_steps=500,
    lr=1e-4,
    loss_fn=torch.nn.CrossEntropyLoss(),
    device="cuda",
)
uq = predict_with_samples_uq(model, samples, x_test, apply_softmax=True)
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [MCMC (SGLD) theory](../../methods/mcmc-sgld.md)

::: deepuq.methods.mcmc
