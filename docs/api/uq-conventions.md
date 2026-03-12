# UQ API Conventions

This page defines the shared API conventions for all public objects under `deepuq.methods`.
Use it as the reference for tensor shapes, `UQResult` fields, and common workflow assumptions.

## Common workflow

Most UQ methods follow the same high-level lifecycle:

1. instantiate the model or wrapper
2. train a MAP or deterministic base model when required
3. fit the UQ approximation if the method has a separate fitting step
4. call `predict(...)` for the legacy tuple-style output or `predict_uq(...)` for the standardized container

Typical patterns:

- `MCDropoutWrapper`: wrap an already-trained model, then call `predict(...)` or `predict_uq(...)`
- `LaplaceWrapper`: construct, call `fit(train_loader, ...)`, then call `predict(...)` or `predict_uq(...)`
- `DeepEnsemble*`: construct, call `fit(train_loader, ...)`, then call `predict(...)` or `predict_uq(...)`
- `vi_elbo_step`: used inside a manual training loop; `predict_vi_uq(...)` is the standardized prediction helper
- `SGLDOptimizer`: used inside a manual sampler loop, often through `collect_posterior_samples(...)`

## Shape conventions

All public methods assume a leading batch dimension unless explicitly noted otherwise.

| Convention | Meaning |
|---|---|
| `[batch, ...]` | one prediction per input example |
| `[n_members, batch, ...]` | stacked deep-ensemble member predictions |
| `[n_samples, batch, ...]` | stacked Monte Carlo or posterior-sample predictions |
| `[batch, n_classes]` | classification logits or probabilities |
| `[batch, n_outputs]` | multi-output regression predictions |

Method-specific sample axes:

| Method | Extra leading axis |
|---|---|
| Deep ensembles | `n_members` |
| MC Dropout | `n_mc` stochastic forward passes |
| VI prediction | `n_samples` stochastic weight samples |
| SGLD / MCMC | number of stored posterior snapshots |

## `UQResult` semantics

All standardized prediction helpers return [`UQResult`](types.md).

| Field | Meaning | Typical methods |
|---|---|---|
| `mean` | predictive mean tensor | all regression methods; often mirrors `probs` for classification |
| `epistemic_var` | model/posterior uncertainty | ensembles, Laplace, VI, MC Dropout, SGLD |
| `aleatoric_var` | observation-noise uncertainty | heteroscedastic ensembles, some GP/Laplace regression backends |
| `total_var` | total predictive variance | most regression APIs |
| `probs` | averaged class probabilities | classifiers, MC Dropout with `apply_softmax=True`, VI classification |
| `probs_var` | probability-space disagreement/variance | classifiers |
| `metadata` | method/backend metadata | all methods |

Regression APIs usually populate `mean`, `epistemic_var`, and `total_var`.
Classification APIs usually populate `probs`, may mirror `probs` into `mean`, and often leave regression-only variance fields unset.

## Device and dtype expectations

- Inputs should already be on the device expected by the wrapped model unless the API explicitly moves batches during training.
- Returned tensors are usually on the same device as the underlying model evaluation.
- `predict_uq(...)` does not change dtypes; if your model runs in `float32`, returned moments are also `float32`.
- SGLD helper functions move stored parameter snapshots to CPU when collecting them, then reload them onto the requested evaluation device.

## Common API variables

| API variable | Meaning | Related theory symbol |
|---|---|---|
| `n_mc` | number of dropout forward passes | `K` |
| `n_samples` | number of stochastic predictive draws | `S` |
| `num_batches` | optimizer steps per epoch used to scale the VI KL term | batch-count normalization |
| `kl_weight` | weight on the KL contribution in VI | `eta` |
| `prior_precision` | Gaussian prior precision in Laplace | `\lambda` |
| `hessian_structure` | curvature approximation family in Laplace | approximation to `H` or posterior precision `\Lambda` |
| `subset_of_weights` | parameter subset used in Laplace | subset of `	heta` |
| `burn_in` | fraction of SGLD steps discarded before collecting samples | warm-up / transient phase |

## Common failure modes

- Calling `predict_uq(...)` before `fit(...)` for methods with a separate fitting phase, especially `LaplaceWrapper`
- Passing classification targets to regression losses or vice versa
- Using an ensemble member architecture whose output shape does not match the wrapper's expectations
- Asking classification helpers for regression-style outputs or assuming `aleatoric_var` exists when the method does not model observation noise
- Reusing state-dict samples with a different model architecture in SGLD helpers

## Related docs

- [Types API](types.md)
- [Deep Ensembles theory](../methods/deep-ensembles.md)
- [Variational Inference theory](../methods/variational-inference.md)
- [Laplace theory](../methods/laplace.md)
- [MCMC (SGLD) theory](../methods/mcmc-sgld.md)
- [MC Dropout theory](../methods/mc-dropout.md)
