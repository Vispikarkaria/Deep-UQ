# Deep Ensembles API

This page documents the regression and classification ensemble wrappers in `deepuq.methods.ensembles`.
The manual notes here explain the public variants and their tensor contracts; the generated section below contains the exact signatures and source docstrings.

## Public objects

- `DeepEnsembleRegressor`
- `HeteroscedasticDeepEnsembleRegressor`
- `DeepEnsembleClassifier`
- `MultiOutputDeepEnsembleRegressor`
- `HeteroscedasticMultiOutputDeepEnsembleRegressor`
- `DeepEnsembleWrapper` (backward-compatible alias of `DeepEnsembleRegressor`)

## Parameter and variable conventions

| Name | Meaning |
|---|---|
| `models` | sequence of independently trained deterministic members |
| `epochs` | number of optimization epochs per member in `fit(...)` |
| `optimizer_cls` | optimizer factory used to train each member |
| `lr` | member learning rate |
| `weight_decay` | member weight decay |
| `device` | optional training device |
| `seed` | base seed; member `i` uses `seed + i` |
| `min_variance` | lower bound for heteroscedastic predicted variances |

## Input and output shapes

- ensemble members must share the same input/output contract
- `predict_members(...)` returns a stacked tensor with leading axis `[n_members, ...]`
- classification members must emit logits with shape `[batch, n_classes]`
- heteroscedastic regressors must emit mean and log-variance concatenated in a single tensor
- multi-output regressors preserve the full output shape and report per-output uncertainty

## `UQResult` mapping

- `DeepEnsembleRegressor`: `mean`, `epistemic_var`, `total_var`
- `HeteroscedasticDeepEnsembleRegressor`: `mean`, `epistemic_var`, `aleatoric_var`, `total_var`
- `DeepEnsembleClassifier`: `mean`, `probs`, `probs_var`, `epistemic_var`
- `MultiOutputDeepEnsembleRegressor`: same fields as plain regression, but vector-valued
- `HeteroscedasticMultiOutputDeepEnsembleRegressor`: same fields as heteroscedastic regression, but vector-valued

## Common preconditions and failure modes

- `models` must contain at least one member
- all members must accept the same `x` shape and emit the same output shape
- heteroscedastic members must concatenate mean and log-variance correctly; odd channel counts raise `ValueError`
- classification targets are cast to integer labels inside `DeepEnsembleClassifier`

## Minimal example

```python
ensemble = DeepEnsembleRegressor([model_a, model_b, model_c])
ensemble.fit(train_loader, epochs=50, lr=1e-3)
uq = ensemble.predict_uq(x_test)
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [Deep Ensembles theory](../../methods/deep-ensembles.md)

::: deepuq.methods.ensembles
