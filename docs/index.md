---
hide:
  - title
---

<div class="hero-panel reveal">
  <h1>Deep-UQ Documentation</h1>
  <p>Unified uncertainty quantification toolkit in PyTorch. Build and compare <strong>Bayes by Backprop</strong>, <strong>Laplace</strong>, <strong>SGLD</strong>, <strong>MC Dropout</strong>, and <strong>Gaussian Process</strong> methods in one package.</p>
  <p>
    <a href="getting-started/installation/" class="md-button md-button--primary">Get Started</a>
    <a href="api/" class="md-button">API Reference</a>
  </p>
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

## Start Here

- [Installation guide](getting-started/installation.md)
- [Quickstart examples](getting-started/quickstart.md)
- [Method docs](methods/variational-inference.md)
- [Tutorial guides](tutorials/index.md)
- [API reference](api/index.md)
