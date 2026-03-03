<div class="hero-panel reveal">

# Deep-UQ Documentation

Unified uncertainty quantification toolkit in PyTorch.
Build and compare **Bayes by Backprop**, **Laplace**, **SGLD**, **MC Dropout**, and **Gaussian Process** methods in one package.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }

</div>

<div class="card-grid">
  <div class="card reveal">
    <h3>Developer First</h3>
    <p>Clear wrappers, examples, notebooks, and API-level primitives for direct integration into existing PyTorch code.</p>
  </div>
  <div class="card reveal">
    <h3>Method Breadth</h3>
    <p>Deep Bayesian VI, Laplace backends, MCMC sampling, stochastic dropout, and exact/sparse Gaussian processes.</p>
  </div>
  <div class="card reveal">
    <h3>Reproducible Workflows</h3>
    <p>Tutorial-driven structure with runnable scripts under <code>examples/</code> and notebooks under <code>notebooks/</code>.</p>
  </div>
</div>

## Method Summary

| Method Family | Implemented Variants | Main Wrapper / Class | Tutorial |
|---|---|---|---|
| Variational Inference | Bayes by Backprop | `BayesianLinear`, `vi_elbo_step` | `notebooks/BayesByBackprop_Tutorial.ipynb` |
| Laplace Approximation | `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full` | `LaplaceWrapper` | `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| MCMC | Stochastic Gradient Langevin Dynamics | `SGLDOptimizer` | `notebooks/SGLD_Tutorial.ipynb` |
| MC Dropout | Monte Carlo dropout inference | `MCDropoutWrapper` | `notebooks/MC_Dropout_Tutorial.ipynb` |
| Gaussian Process | Exact GP (`RBFKernel`) | `GaussianProcessRegressor` | `notebooks/GaussianProcess_Tutorial.ipynb` |
| Sparse GP | Variational inducing-point GP | `SparseGaussianProcessRegressor` | `notebooks/SparseGaussianProcess_Tutorial.ipynb` |

## Quick Install

```bash
pip install uqdeepnn
```

For Laplace `kron` and `full` backends:

```bash
pip install "laplace-torch>=0.1.7"
```

## Start Here

- [Installation guide](getting-started/installation.md)
- [Quickstart examples](getting-started/quickstart.md)
- [Method docs](methods/variational-inference.md)
- [Tutorial guides](tutorials/index.md)
- [API reference](api/index.md)
