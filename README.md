# deepuq

Unified deep learning uncertainty quantification (UQ) toolkit in PyTorch.

![tests](https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/tests.yml/badge.svg)
![lint](https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/lint.yml/badge.svg)
![docs](https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/docs.yml/badge.svg)

Implements **five** widely used UQ families with multiple variants:

1. **Variational Inference (VI)** — Bayes by Backprop with BayesianLinear layers.
2. **Laplace Approximation** — native backends for all supported structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`).
3. **MCMC (SGLD)** — Stochastic Gradient Langevin Dynamics sampler for NN posteriors.
4. **MC Dropout** — Keep dropout active at test-time and aggregate Monte Carlo predictions.
5. **Gaussian Processes (GPs)** — Exact, sparse, classification, heteroscedastic, multitask, spectral-mixture, and deep-kernel GP variants.

Examples and tutorials focus on a synthetic Euler-Bernoulli beam deflection regression task to illustrate confidence bounds.

## Method Summary

| Method Family | Implemented Variants | Main Wrapper / Class | Tutorial |
|---|---|---|---|
| Variational Inference | Bayes by Backprop | `BayesianLinear`, `vi_elbo_step`, `predict_vi_uq` | `notebooks/BayesByBackprop_Tutorial.ipynb` |
| Laplace Approximation | `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full` | `LaplaceWrapper`, `predict_uq` | `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| MCMC | Stochastic Gradient Langevin Dynamics | `SGLDOptimizer`, `collect_posterior_samples`, `predict_with_samples_uq` | `notebooks/SGLD_Tutorial.ipynb` |
| MC Dropout | Monte Carlo dropout inference | `MCDropoutWrapper`, `predict_uq` | `notebooks/MC_Dropout_Tutorial.ipynb` |
| Gaussian Processes | Exact, sparse, classification (binary + OvR), heteroscedastic, multi-task ICM, spectral mixture, deep kernel, and kernel composition (`RBF`, Matérn, RQ, Periodic, Linear) | `GaussianProcessRegressor`, `SparseGaussianProcessRegressor`, `GaussianProcessClassifier`, `OneVsRestGaussianProcessClassifier`, `HeteroscedasticGaussianProcessRegressor`, `MultiTaskGaussianProcessRegressor`, `SpectralMixtureGaussianProcessRegressor`, `DeepKernelGaussianProcessRegressor`, kernel classes, `predict_uq` | `notebooks/gp/GP_Model_Comparison.ipynb` (see all under `notebooks/gp/`) |

## Documentation Website

- Docs home: https://vispikarkaria.github.io/Deep-UQ/
- Tutorials: https://vispikarkaria.github.io/Deep-UQ/tutorials/
- API reference: https://vispikarkaria.github.io/Deep-UQ/api/

## Install (local)

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git
cd Deep-UQ
pip install -e .
```

## Install (PyPI)

```bash
pip install uqdeepnn
```

## Publish / Update PyPI Release

Releases are tag-driven via `.github/workflows/release.yml` and PyPI trusted publishing.

1. Bump version in `pyproject.toml`.
2. Commit and push to `master`.
3. Tag and push:
```bash
git tag v0.1.5
git push origin v0.1.5
```
4. Verify package on PyPI:
```bash
pip install -U uqdeepnn
python -c "import deepuq; print(deepuq.__version__)"
```

Detailed release checklist: `RELEASING.md`.

## Quickstart

```python
import torch
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

# Beam deflection regression input grid
L = 2.0
x = torch.linspace(0.0, L, 200).unsqueeze(-1)

# After training an MLP, enable MC Dropout for uncertainty estimates
model = MLP(input_dim=1, hidden_dims=[128, 128], output_dim=1, p_drop=0.15)
uq = MCDropoutWrapper(model, n_mc=200, apply_softmax=False)
result = uq.predict_uq(x)
print(result.mean.shape, result.total_var.shape)
```

See the **examples/** folder for end-to-end regression scripts on the Euler-Bernoulli beam deflection problem.

## Common UQResult API

All methods now expose a standardized uncertainty container through `predict_uq(...)` (or method-specific helper functions):

- `mean`: predictive mean (or class probability mean for classification)
- `epistemic_var`: model uncertainty
- `aleatoric_var`: noise/data uncertainty when available
- `total_var`: combined predictive uncertainty
- `probs`, `probs_var`: classification probability moments
- `metadata`: method-specific context

```python
from deepuq.methods import LaplaceWrapper

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
la.fit(train_loader)
uq = la.predict_uq(x_batch, n_samples=100)
print(uq.mean.shape, uq.total_var.shape)
```

## Methods

- **VI**: Place Gaussian posteriors over weights with reparameterization trick and KL regularization.
- **Laplace**: Fit a Gaussian around a MAP solution using one of multiple curvature structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`) and calibrate with a prior precision.
- **MCMC (SGLD)**: Inject Gaussian noise into SGD steps to sample from the posterior.
- **MC Dropout**: Use dropout at inference; Monte Carlo average for mean and variance.
- **Gaussian Processes**: Full GP suite for regression and classification, including richer kernels and advanced structured variants.

For Laplace users:
- All supported structures are implemented natively in `deepuq`.
- Scientific details and equations: `https://vispikarkaria.github.io/Deep-UQ/methods/laplace/`

## Tutorials

- `notebooks/BayesByBackprop_Tutorial.ipynb`: Variational Inference (Bayes by Backprop) for regression with predictive uncertainty.
- `notebooks/MC_Dropout_Tutorial.ipynb`: MC Dropout tutorial on a nonlinear beam-style regression case.
- `notebooks/laplace/Laplace_Tutorial.ipynb`: Core Laplace workflow around a MAP model.
- `notebooks/laplace/Laplace_FullHessian_Tutorial.ipynb`: Full-Hessian Laplace example.
- `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb`: Side-by-side comparison of all Hessian structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`) using shared MAP weights and common metrics (RMSE, NLL, coverage, interval width, ID/OOD uncertainty ratio).
- `notebooks/SGLD_Tutorial.ipynb`: MCMC posterior sampling with SGLD.
- `notebooks/gp/GP_Exact_Tutorial.ipynb`: Exact Gaussian Process regression on an engineering-style deflection task.
- `notebooks/gp/GP_Sparse_Tutorial.ipynb`: Sparse variational GP with inducing points and ELBO trend.
- `notebooks/gp/GP_Kernel_Zoo_Tutorial.ipynb`: Kernel misspecification and calibration comparison.
- `notebooks/gp/GP_Classification_Tutorial.ipynb`: Binary and OvR GP classification uncertainty.
- `notebooks/gp/GP_Heteroscedastic_Tutorial.ipynb`: Input-dependent noise decomposition.
- `notebooks/gp/GP_MultiTask_ICM_Tutorial.ipynb`: Correlated multi-output ICM GP.
- `notebooks/gp/GP_SpectralMixture_Tutorial.ipynb`: Multi-frequency spectral mixture GP.
- `notebooks/gp/GP_DeepKernel_Tutorial.ipynb`: Deep kernel GP for representation-rich inputs.
- `notebooks/gp/GP_Model_Comparison.ipynb`: Calibration and runtime comparison across GP families.

### Gaussian Processes

The module `deepuq.models.gaussian_process` provides a lightweight, pure-PyTorch GP suite so everything runs on CPU or GPU without extra GP dependencies.

#### Exact GP + Kernel Variants

`GaussianProcessRegressor` supports exact regression with interchangeable kernels (`RBFKernel`, `MaternKernel`, `RationalQuadraticKernel`, `PeriodicKernel`, `LinearKernel`) and kernel composition via `+` and `*`.

```python
import torch
from deepuq.models import (
    GaussianProcessRegressor,
    RBFKernel,
    PeriodicKernel,
    LinearKernel,
)

# Training data
x = torch.linspace(-1.0, 1.0, 40).unsqueeze(-1)
y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)

# Model setup
kernel = PeriodicKernel(lengthscale=0.6, outputscale=0.8, period=2.0) + LinearKernel(variance=0.05)
gp = GaussianProcessRegressor(kernel=kernel, noise=0.02)
gp.fit(x, y)

# Posterior predictions
x_star = torch.linspace(-1.5, 1.5, 200).unsqueeze(-1)
mean, var = gp.predict(x_star)
samples = gp.posterior_samples(x_star, n_samples=5)
```

#### Sparse Variational GP

`SparseGaussianProcessRegressor` follows the variational inducing-point
approach of Titsias (2009), optimising kernel hyperparameters and inducing
locations with Adam for scalability.

```python
import torch
from deepuq.models import SparseGaussianProcessRegressor

x = torch.linspace(-2.0, 2.0, 500).unsqueeze(-1)
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn_like(x)

sparse_gp = SparseGaussianProcessRegressor(num_inducing=40, num_iterations=800)
sparse_gp.fit(x, y)
mean, var = sparse_gp.predict(x[:50])
```

#### Additional GP Families

- Classification: `GaussianProcessClassifier`, `OneVsRestGaussianProcessClassifier`
- Heteroscedastic regression: `HeteroscedasticGaussianProcessRegressor`
- Multi-task regression (ICM): `MultiTaskGaussianProcessRegressor`
- Spectral mixture regression: `SpectralMixtureGaussianProcessRegressor`
- Deep kernel regression: `DeepKernelGaussianProcessRegressor`

Explore the full suite under `notebooks/gp/`.
Legacy notebook paths are kept as compatibility stubs:
- `notebooks/GaussianProcess_Tutorial.ipynb`
- `notebooks/SparseGaussianProcess_Tutorial.ipynb`

## Documentation

- API docs are in each module and the README sections below.
- Run `pydoc deepuq.methods.vi` etc., or open the examples.

## Benchmarks

Multi-dataset benchmark scripts are under `benchmarks/`.

```bash
python benchmarks/run_benchmarks.py --preset quick
```

Outputs:

- `benchmarks/results/results.csv`
- `benchmarks/results/summary.md`

## Data Policy

- Tracked tutorial datasets remain under `data/` and `notebooks/data/`.
- Provenance and usage notes are documented in:
  - `data/README.md`
  - `notebooks/data/README.md`

## Contributing

PRs welcome. Please add tests under `tests/` and run `pytest`.

## License

MIT
