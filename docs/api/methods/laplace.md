# Laplace API

This page documents the public Laplace approximation wrapper used throughout Deep-UQ.
The notes below explain the public control variables and workflow; the generated section contains the exact signatures and source docstrings.

## Public objects

- `LaplaceWrapper`

## Parameter and variable conventions

| Name | Meaning |
|---|---|
| `likelihood` | either `"regression"` or `"classification"` |
| `hessian_structure` | curvature backend: `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full` |
| `subset_of_weights` | `"last_layer"` or `"all"` |
| `lowrank_rank` | target rank for `lowrank_diag` |
| `damping` | numerical stabilization added to the precision approximation |
| `full_max_params` | guardrail for dense full-Hessian fitting |
| `prior_precision` | Gaussian prior precision used during `fit(...)` |
| `predict_kwargs` | backend-specific predictive options forwarded unchanged |

## Workflow expectations

1. train the base model to a MAP solution with ordinary optimization
2. construct `LaplaceWrapper(model, ...)`
3. call `fit(train_loader, prior_precision=...)`
4. call `predict(...)` or `predict_uq(...)`

`predict_uq(...)` cannot be called before `fit(...)`.

## Input and output shapes

- `fit(...)` expects an iterable of `(inputs, targets)` minibatches compatible with the wrapped model.
- regression `predict(...)` returns `(mean, var)` with the same trailing shape as one model prediction.
- classification `predict(...)` returns `(probs, probs_var_or_none)` with shape `[batch, n_classes]`.

## `UQResult` mapping

`predict_uq(...)` returns:

- regression: `mean`, `epistemic_var`, optional `aleatoric_var`, `total_var`
- classification: `probs`, optional `probs_var`, and metadata describing the chosen backend

## Common preconditions and failure modes

- unsupported `hessian_structure` raises `ValueError`
- `full` curvature over `subset_of_weights="all"` may be rejected if `full_max_params` is exceeded
- calling `predict(...)` or `predict_uq(...)` before `fit(...)` raises `RuntimeError`
- regression backends must return predictive variance; otherwise `predict_uq(...)` raises `RuntimeError`

## Minimal example

```python
la = LaplaceWrapper(
    model,
    likelihood="regression",
    hessian_structure="block_diag",
    subset_of_weights="last_layer",
)
la.fit(train_loader, prior_precision=10.0)
uq = la.predict_uq(x_test, n_samples=32)
```

## Related docs

- [UQ API Conventions](../uq-conventions.md)
- [Types API](../types.md)
- [Laplace Approximation theory](../../methods/laplace.md)

::: deepuq.methods.laplace
