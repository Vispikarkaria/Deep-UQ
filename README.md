# deepuq

Unified deep learning uncertainty quantification (UQ) toolkit in PyTorch.

Implements **five** widely used methods:

1. **Variational Inference (VI)** — Bayes by Backprop with BayesianLinear layers.
2. **Laplace Approximation** — native backends for all supported structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`).
3. **MCMC (SGLD)** — Stochastic Gradient Langevin Dynamics sampler for NN posteriors.
4. **MC Dropout** — Keep dropout active at test-time and aggregate Monte Carlo predictions.
5. **Gaussian Processes (GPs)** — Exact regression and sparse inducing-point approximations with RBF kernels.

Examples and tutorials focus on a synthetic Euler-Bernoulli beam deflection regression task to illustrate confidence bounds.

## Method Summary

| Method Family | Implemented Variants | Main Wrapper / Class | Tutorial |
|---|---|---|---|
| Variational Inference | Bayes by Backprop | `BayesianLinear`, `vi_elbo_step` | `notebooks/BayesByBackprop_Tutorial.ipynb` |
| Laplace Approximation | `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full` | `LaplaceWrapper` | `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| MCMC | Stochastic Gradient Langevin Dynamics | `SGLDOptimizer`, `collect_posterior_samples` | `notebooks/SGLD_Tutorial.ipynb` |
| MC Dropout | Monte Carlo dropout inference | `MCDropoutWrapper` | `notebooks/MC_Dropout_Tutorial.ipynb` |
| Gaussian Process | Exact GP (`RBFKernel`) | `GaussianProcessRegressor` | `notebooks/GaussianProcess_Tutorial.ipynb` |
| Sparse GP | Variational inducing-point GP | `SparseGaussianProcessRegressor` | `notebooks/SparseGaussianProcess_Tutorial.ipynb` |

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

Use this flow whenever you want to publish a new pip version.

1. Bump version in `pyproject.toml`:
```toml
[project]
version = "0.1.4"
```
2. Commit and push the version bump:
```bash
git add pyproject.toml
git commit -m "Bump version to 0.1.4"
git push
```
3. Build distributions:
```bash
python -m pip install --upgrade build twine
python -m build
```
4. Validate package metadata:
```bash
python -m twine check dist/*
```
5. Upload to TestPyPI (recommended first):
```bash
python -m twine upload --repository testpypi dist/*
```
6. Upload to PyPI:
```bash
python -m twine upload dist/*
```
7. Verify installation:
```bash
pip install -U uqdeepnn
python -c "import deepuq; print('deepuq import ok')"
```

Notes:
- Prefer API tokens over passwords for Twine auth.
- Revoke and rotate any token immediately if it is ever exposed.

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
mean, var = uq.predict(x)
print(mean.shape, var.shape)
```

See the **examples/** folder for end-to-end regression scripts on the Euler-Bernoulli beam deflection problem.

## Methods

- **VI**: Place Gaussian posteriors over weights with reparameterization trick and KL regularization.
- **Laplace**: Fit a Gaussian around a MAP solution using one of multiple curvature structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`) and calibrate with a prior precision.
- **MCMC (SGLD)**: Inject Gaussian noise into SGD steps to sample from the posterior.
- **MC Dropout**: Use dropout at inference; Monte Carlo average for mean and variance.
- **Gaussian Processes**: Closed-form posterior inference with RBF kernels for regression and uncertainty-aware interpolation.

For Laplace users:
- All supported structures are implemented natively in `deepuq`.

## Tutorials

- `notebooks/BayesByBackprop_Tutorial.ipynb`: Variational Inference (Bayes by Backprop) for regression with predictive uncertainty.
- `notebooks/MC_Dropout_Tutorial.ipynb`: MC Dropout tutorial on a nonlinear beam-style regression case.
- `notebooks/laplace/Laplace_Tutorial.ipynb`: Core Laplace workflow around a MAP model.
- `notebooks/laplace/Laplace_FullHessian_Tutorial.ipynb`: Full-Hessian Laplace example.
- `notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb`: Side-by-side comparison of all Hessian structures (`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`) using shared MAP weights and common metrics (RMSE, NLL, coverage, interval width, ID/OOD uncertainty ratio).
- `notebooks/SGLD_Tutorial.ipynb`: MCMC posterior sampling with SGLD.
- `notebooks/GaussianProcess_Tutorial.ipynb`: Exact Gaussian Process regression.
- `notebooks/SparseGaussianProcess_Tutorial.ipynb`: Sparse variational GP with inducing points.

### Gaussian Processes

The module `deepuq.models.gaussian_process` provides lightweight GP utilities
implemented entirely in PyTorch so everything can run on CPU or GPU.

#### Exact GP

`GaussianProcessRegressor` implements closed-form inference with an RBF kernel.
The API mirrors scikit-learn while keeping tensors on the chosen device.

```python
import torch
from deepuq.models import GaussianProcessRegressor, RBFKernel

# Training data
x = torch.linspace(-1.0, 1.0, 40).unsqueeze(-1)
y = torch.sin(2 * torch.pi * x) + 0.05 * torch.randn_like(x)

# Model setup
kernel = RBFKernel(lengthscale=0.5, outputscale=1.0)
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

Explore both flavours in the notebooks
`notebooks/GaussianProcess_Tutorial.ipynb` (exact GP) and
`notebooks/SparseGaussianProcess_Tutorial.ipynb` (sparse GP), each of which
visualises posterior means, credible intervals, and posterior samples on toy
datasets.

## Documentation

- API docs are in each module and the README sections below.
- Run `pydoc deepuq.methods.vi` etc., or open the examples.

## Contributing

PRs welcome. Please add tests under `tests/` and run `pytest`.

## License

MIT
