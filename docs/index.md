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

## Method Families

The website is the canonical reading surface for the method guides below. Each
family section here gives a compact comparison table, then links to the full
method page, API reference, and tutorial guide.

<div class="family-table-block reveal" markdown="1">
### Variational Inference

Bayes by Backprop is Deep-UQ's end-to-end Bayesian neural network family. Use
it when you want weight uncertainty learned directly during training.

<p class="family-table-readmore"><strong>Read more:</strong> <a href="methods/variational-inference/">Variational Inference method guide</a></p>

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Bayes by Backprop | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | End-to-end Bayesian neural-network training | `BayesianLinear`, `BayesByBackpropMLP`, `vi_elbo_step`, `predict_vi_uq` | [Guide](methods/variational-inference.md)<br>[API](api/methods/vi.md) | [Guide](tutorials/bayes-by-backprop.md)<br>`notebooks/BayesByBackprop_Tutorial.ipynb` |
</div>

<div class="family-table-block reveal" markdown="1">
### Laplace Approximation

Laplace methods wrap a trained MAP model with a Gaussian posterior defined by a
chosen curvature structure. They are a good fit when you want strong post-hoc
uncertainty with less retraining cost than full Bayesian neural nets.

<p class="family-table-readmore"><strong>Read more:</strong> <a href="methods/laplace/">Laplace method guide</a></p>

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Diagonal Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Cheapest Laplace baseline around a trained MAP network | `LaplaceWrapper(hessian_structure="diag")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_Tutorial.ipynb` |
| Fisher-Diagonal Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Diagonal empirical-Fisher style approximation | `LaplaceWrapper(hessian_structure="fisher_diag")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Low-Rank + Diagonal Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Capture dominant coupled directions cheaply | `LaplaceWrapper(hessian_structure="lowrank_diag")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Block-Diagonal Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Preserve within-block coupling at moderate memory cost | `LaplaceWrapper(hessian_structure="block_diag")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Kronecker-Factored Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Better fidelity/cost trade-off for layerwise structures | `LaplaceWrapper(hessian_structure="kron")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Full-Hessian Laplace | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Epistemic</span> | Dense curvature for small enough models | `LaplaceWrapper(hessian_structure="full")` | [Guide](methods/laplace.md)<br>[API](api/methods/laplace.md) | [Guide](tutorials/laplace-comparison.md)<br>`notebooks/laplace/Laplace_FullHessian_Tutorial.ipynb` |
</div>

<div class="family-table-block reveal" markdown="1">
### MCMC / SGLD

SGLD is the package's posterior-sampling method for deep networks. Use it when
you want sampled parameter trajectories and Monte Carlo predictive uncertainty.

<p class="family-table-readmore"><strong>Read more:</strong> <a href="methods/mcmc-sgld/">MCMC / SGLD method guide</a></p>

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Stochastic Gradient Langevin Dynamics | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Posterior Samples</span> <span class="family-chip family-chip--accent">Epistemic</span> | SGD-like posterior sampling with predictive averaging | `SGLDOptimizer`, `collect_posterior_samples`, `predict_with_samples_uq` | [Guide](methods/mcmc-sgld.md)<br>[API](api/methods/mcmc.md) | [Guide](tutorials/sgld.md)<br>`notebooks/SGLD_Tutorial.ipynb` |
</div>

<div class="family-table-block reveal" markdown="1">
### MC Dropout

MC Dropout is the fastest neural-network UQ baseline in the package. Use it
when you want uncertainty estimates with minimal changes to an existing
dropout-enabled model.

<p class="family-table-readmore"><strong>Read more:</strong> <a href="methods/mc-dropout/">MC Dropout method guide</a></p>

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| MC Dropout | <span class="family-chip">Regression</span> <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Approx. Epistemic</span> | Minimal-code-change uncertainty baseline for dropout models | `MCDropoutWrapper`, `predict_uq` | [Guide](methods/mc-dropout.md)<br>[API](api/methods/mc_dropout.md) | [Guide](tutorials/mc-dropout.md)<br>`notebooks/MC_Dropout_Tutorial.ipynb` |
</div>

<div class="family-table-block reveal" markdown="1">
### Gaussian Processes

The GP family covers the broadest set of structured nonparametric models in the
package: exact and sparse regression, GP classification, heteroscedastic noise,
multi-task coupling, spectral kernels, and deep kernel learning.

<p class="family-table-readmore"><strong>Read more:</strong> <a href="methods/gaussian-processes/">Gaussian Processes method guide</a></p>

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Exact GP Regression | <span class="family-chip">Regression</span> | <span class="family-chip family-chip--accent">Epistemic + Aleatoric</span> | Strong small-data calibration baseline | `GaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_Exact_Tutorial.ipynb` |
| Sparse Variational GP | <span class="family-chip">Regression</span> | <span class="family-chip family-chip--accent">Epistemic + Aleatoric</span> | Scalable inducing-point GP regression | `SparseGaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_Sparse_Tutorial.ipynb` |
| GP Classifier | <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Class Probability</span> | Binary decision-boundary uncertainty | `GaussianProcessClassifier` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_Classification_Tutorial.ipynb` |
| OvR GP Classifier | <span class="family-chip">Classification</span> | <span class="family-chip family-chip--accent">Class Probability</span> | Multiclass GP classification via one-vs-rest | `OneVsRestGaussianProcessClassifier` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_Classification_Tutorial.ipynb` |
| Heteroscedastic GP | <span class="family-chip">Regression</span> | <span class="family-chip family-chip--accent">Epistemic + Aleatoric</span> | Input-dependent noise decomposition | `HeteroscedasticGaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_Heteroscedastic_Tutorial.ipynb` |
| Multi-task ICM GP | <span class="family-chip">Multi-output Regression</span> | <span class="family-chip family-chip--accent">Shared-output Epistemic + Aleatoric</span> | Correlated multi-output regression | `MultiTaskGaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_MultiTask_ICM_Tutorial.ipynb` |
| Spectral Mixture GP | <span class="family-chip">Regression</span> | <span class="family-chip family-chip--accent">Epistemic + Aleatoric</span> | Multi-frequency signal structure and extrapolation | `SpectralMixtureGaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_SpectralMixture_Tutorial.ipynb` |
| Deep Kernel GP | <span class="family-chip">Regression</span> | <span class="family-chip family-chip--accent">Epistemic + Aleatoric</span> | Learn a feature space before GP inference | `DeepKernelGaussianProcessRegressor` | [Guide](methods/gaussian-processes.md)<br>[API](api/models/gaussian_process.md) | [Guide](tutorials/gp.md)<br>`notebooks/gp/GP_DeepKernel_Tutorial.ipynb` |
</div>

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
