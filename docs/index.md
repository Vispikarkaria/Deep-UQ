---
hide:
  - title
---

<div class="hero-panel reveal">
  <h1>Deep-UQ Documentation</h1>
  <p><strong>Purpose:</strong> Deep-UQ helps you train predictive models that know when they are uncertain, so decisions can use both predictions and confidence.</p>
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

## Why This Library Exists

Deep-UQ is built for engineers and researchers who need uncertainty-aware modeling without stitching together multiple libraries. It provides one package with shared workflows for:

- model training,
- posterior or predictive uncertainty estimation,
- method-to-method comparison,
- tutorials and examples that run from the same codebase.

## What You Get

- Five UQ families in one interface surface.
- Consistent regression/classification uncertainty outputs through `UQResult`.
- Native Laplace backends for `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, and `full`.
- Full Gaussian Process suite: exact, sparse, classification, heteroscedastic, multitask, spectral, and deep-kernel variants.
- Reproducible tutorials, examples, and benchmark scripts.

## Method Summary

| Method Family | Implemented Variants | Main Wrapper / Class | Tutorial |
|---|---|---|---|
| Variational Inference | Bayes by Backprop | `BayesianLinear`, `vi_elbo_step`, `predict_vi_uq` | `notebooks/BayesByBackprop_Tutorial.ipynb` |
| Laplace Approximation | `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full` | `LaplaceWrapper`, `predict_uq` | `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| MCMC | Stochastic Gradient Langevin Dynamics | `SGLDOptimizer`, `predict_with_samples_uq` | `notebooks/SGLD_Tutorial.ipynb` |
| MC Dropout | Monte Carlo dropout inference | `MCDropoutWrapper`, `predict_uq` | `notebooks/MC_Dropout_Tutorial.ipynb` |
| Gaussian Process | Exact GP + kernel composition (`RBF`, Matérn, RQ, Periodic, Linear) | `GaussianProcessRegressor`, kernels, `predict_uq` | `notebooks/gp/GP_Exact_Tutorial.ipynb` |
| Sparse GP | Variational inducing-point GP | `SparseGaussianProcessRegressor`, `predict_uq` | `notebooks/gp/GP_Sparse_Tutorial.ipynb` |
| GP Classification | Binary + OvR multiclass | `GaussianProcessClassifier`, `OneVsRestGaussianProcessClassifier`, `predict_uq` | `notebooks/gp/GP_Classification_Tutorial.ipynb` |
| GP Heteroscedastic | Input-dependent noise | `HeteroscedasticGaussianProcessRegressor`, `predict_uq` | `notebooks/gp/GP_Heteroscedastic_Tutorial.ipynb` |
| GP Multi-task | ICM coregionalization | `MultiTaskGaussianProcessRegressor`, `predict_uq` | `notebooks/gp/GP_MultiTask_ICM_Tutorial.ipynb` |
| GP Spectral + DKL | Spectral mixture + deep kernel learning | `SpectralMixtureGaussianProcessRegressor`, `DeepKernelGaussianProcessRegressor`, `predict_uq` | `notebooks/gp/GP_SpectralMixture_Tutorial.ipynb`, `notebooks/gp/GP_DeepKernel_Tutorial.ipynb` |

## Choosing a Method

| Goal | Recommended Start | Why |
|---|---|---|
| Fast UQ baseline for deep nets | MC Dropout | Minimal training changes, simple inference |
| Better local posterior around MAP | Laplace (`diag` or `kron`) | Strong uncertainty quality vs cost |
| Full Bayesian weight posterior approximation | VI (Bayes by Backprop) | End-to-end posterior learning |
| Posterior sampling perspective | SGLD | Direct sample-based uncertainty |
| Calibration-oriented nonparametric baseline | Exact/Sparse GP | Strong uncertainty behavior with kernel priors |

## Unified Output (`UQResult`)

All major methods provide a standardized uncertainty output with fields:

- `mean`
- `epistemic_var`
- `aleatoric_var`
- `total_var`
- `probs`, `probs_var` (classification)
- `metadata`

```python
from deepuq.methods import LaplaceWrapper

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
la.fit(train_loader)
uq = la.predict_uq(x_test, n_samples=100)
print(uq.mean.shape, uq.total_var.shape)
```

## Quick Install

```bash
pip install uqdeepnn
```

## Benchmarks

Deep-UQ includes a multi-dataset benchmark runner for regression metrics and runtime comparisons:

```bash
python benchmarks/run_benchmarks.py --preset quick
```

Outputs:

- `benchmarks/results/results.csv`
- `benchmarks/results/summary.md`

## Start Here

- [Installation guide](getting-started/installation.md)
- [Quickstart examples](getting-started/quickstart.md)
- [Method docs](methods/variational-inference.md)
- [Tutorial guides](tutorials/index.md)
- [API reference](api/index.md)
