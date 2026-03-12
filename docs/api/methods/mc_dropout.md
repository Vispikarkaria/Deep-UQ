# MC Dropout API

This page documents the wrapper used to keep dropout active at inference time and summarize predictive spread.

## Public objects

- `MCDropoutWrapper`

## Parameter and variable conventions

| Name | Meaning |
|---|---|
| `model` | dropout-enabled base model |
| `n_mc` | number of stochastic forward passes |
| `apply_softmax` | convert logits to probabilities before aggregation |

## Workflow expectations

1. train the base model normally with dropout layers present
2. wrap it with `MCDropoutWrapper(model, n_mc=..., apply_softmax=...)`
3. call `predict(...)` or `predict_uq(...)`

## Input and output shapes

- `x` can have any shape accepted by the wrapped model.
- `predict(...)` returns `(mean, var)` with the same trailing shape as a single forward pass.
- if `apply_softmax=True`, the last dimension is interpreted as class probability.

## `UQResult` mapping

`predict_uq(...)` populates `mean`, `epistemic_var`, and `total_var`. When `apply_softmax=True`, it additionally populates `probs` and `probs_var`.

## Common preconditions and failure modes

- the wrapped model must already contain dropout layers for MC Dropout to have any effect
- `n_mc` should be positive and large enough to stabilize the variance estimate
- `apply_softmax=True` should only be used when the wrapped model emits logits

## Minimal example

```python
mc_model = MCDropoutWrapper(model, n_mc=50, apply_softmax=False)
uq = mc_model.predict_uq(x_test)
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [MC Dropout theory](../../methods/mc-dropout.md)

::: deepuq.methods.mc_dropout
