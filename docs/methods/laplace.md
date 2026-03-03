# Laplace Approximation

`LaplaceWrapper` builds a Gaussian posterior around a MAP solution using multiple curvature structures.

## Supported Hessian Structures

- `diag`: diagonal approximation
- `fisher_diag`: explicit empirical Fisher diagonal variant
- `lowrank_diag`: low-rank + diagonal residual approximation
- `block_diag`: block-wise curvature approximation
- `kron`: Kronecker-factored approximation for selected `nn.Linear` layers
- `full`: dense full Hessian backend

## Subset of Weights

- `last_layer`: Laplace on final linear layer only (fast and stable)
- `all`: Laplace on all trainable parameters (higher cost)

## Design

- Native implementations for all supported structures
- Predictive contract:
  - regression: `(mean, var)`
  - classification: `(mean_probs, None)`

## Practical Guidance

- Start with `last_layer` for robust behavior.
- Tune `prior_precision` on validation NLL.
- Use `full` only for small parameter subsets.

## References

- [Laplace Comparison Tutorial](../tutorials/laplace-comparison.md)
- [Laplace API](../api/methods/laplace.md)
