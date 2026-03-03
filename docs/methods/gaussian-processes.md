# Gaussian Processes

Deep-UQ provides both exact and sparse GP regression implementations in PyTorch.

## Exact GP

- `RBFKernel`
- `GaussianProcessRegressor`

Features:
- closed-form posterior mean/variance
- posterior sampling
- log marginal likelihood

## Sparse Variational GP

- `SparseGaussianProcessRegressor`

Features:
- inducing points for scalability
- ELBO optimization with Adam
- practical for larger datasets than exact GP

## Practical Guidance

- Exact GP is best for small/medium datasets.
- Sparse GP is preferable when memory/runtime of exact GP becomes limiting.

## References

- [Gaussian Process Tutorial Guide](../tutorials/gp.md)
- [Gaussian Process API](../api/models/gaussian_process.md)
