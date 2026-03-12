# Types API

`deepuq.types` contains the shared container types used across the UQ APIs.
The most important one is `UQResult`, which gives every `predict_uq(...)` helper a common return shape.

## `UQResult` at a glance

| Field | Type | Meaning |
|---|---|---|
| `mean` | `torch.Tensor` | predictive mean; for classifiers this often mirrors `probs` |
| `epistemic_var` | `torch.Tensor \| None` | uncertainty due to model/posterior uncertainty |
| `aleatoric_var` | `torch.Tensor \| None` | uncertainty due to observation noise |
| `total_var` | `torch.Tensor \| None` | total predictive variance |
| `probs` | `torch.Tensor \| None` | predictive class probabilities |
| `probs_var` | `torch.Tensor \| None` | probability-space disagreement/variance |
| `metadata` | `dict[str, Any]` | backend, sample-count, or likelihood metadata |

## Usage notes

- Regression methods should treat `mean` as the primary prediction and populate variance fields where available.
- Classification methods should prefer `probs` and `probs_var`; `mean` may mirror `probs` for convenience.
- `metadata` is intentionally open-ended. Use it for non-tensor method details such as `n_members`, `n_mc`, `likelihood`, or `hessian_structure`.

## Related docs

- [UQ API Conventions](uq-conventions.md)

::: deepuq.types
