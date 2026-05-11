# Deep-UQ

<p align="center">
  <img src="docs/assets/images/logo.svg" alt="Deep-UQ Logo" width="150"/>
</p>

<p align="center">
  <strong>Unified deep learning uncertainty quantification toolkit in PyTorch.</strong>
</p>

<p align="center">
  <a href="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/tests.yml"><img src="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/tests.yml/badge.svg" alt="tests"/></a>
  <a href="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/lint.yml"><img src="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/lint.yml/badge.svg" alt="lint"/></a>
  <a href="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/docs.yml"><img src="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/docs.yml/badge.svg" alt="docs"/></a>
  <a href="https://pypi.org/project/uqdeepnn/"><img src="https://img.shields.io/pypi/v/uqdeepnn" alt="PyPI"/></a>
</p>

---

## Why Deep-UQ?

Most UQ libraries focus on one method. Deep-UQ provides **six method families** behind a single `predict_uq()` API, works with any PyTorch model, and includes scientific ML architectures (FNO, DeepONet, GNO, PINNs) as first-class citizens.

```python
from deepuq.models import MLP
from deepuq.methods import LaplaceWrapper

model = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1)
# ... train model ...

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="full")
la.fit(train_loader)
la.optimize_prior_precision()

result = la.predict_uq(x_test)
print(result.mean.shape, result.epistemic_var.shape)
```

---

## Install

```bash
pip install uqdeepnn          # from PyPI
pip install -e ".[dev,tests]" # from source
```

---

## UQ Methods

| Method | Key Classes | What it does |
|--------|-------------|--------------|
| **Deep Ensembles** | `DeepEnsembleRegressor`, `HeteroscedasticDeepEnsembleRegressor` | Train N models, aggregate mean/variance |
| **Variational Inference** | `BayesByBackpropMLP`, `LastLayerVariationalInference` | Gaussian weight posteriors via ELBO |
| **Laplace Approximation** | `LaplaceWrapper` | Post-hoc Gaussian posterior (6 Hessian structures) |
| **MCMC (SGLD)** | `SGLDOptimizer`, `collect_posterior_samples` | Posterior samples via noisy SGD |
| **MC Dropout** | `MCDropoutWrapper` | Dropout at test-time for uncertainty |
| **Gaussian Processes** | `GaussianProcessRegressor`, `SparseGaussianProcessRegressor` | Exact/sparse/multi-task/deep-kernel GPs |

All methods return a **`UQResult`** with `.mean`, `.epistemic_var`, `.aleatoric_var`, `.total_var`.

---

## Model Architectures

| Family | Classes | UQ Compatibility |
|--------|---------|-----------------|
| Dense | `MLP`, `PINN1D`, `PINN2D` | All methods |
| Spatial | `CNNRegressor2D`, `ResNetRegressor2D`, `UNet2D` | Ensembles, MC Dropout |
| Operators | `DeepONet1D`, `DeepONet2D`, `FNO2D`, `FNO3D` | Laplace, Ensembles |
| Graph | `GraphNeuralOperator2D` | Ensembles, Laplace |
| GPs | All GP classes | Native Bayesian |

---

## Laplace Backends

All implemented natively (no external dependencies):

| Structure | Approximation | Best for |
|-----------|--------------|----------|
| `diag` | Diagonal GGN | Fast, large models |
| `fisher_diag` | Diagonal GGN (alias) | Same as diag |
| `lowrank_diag` | Low-rank + diagonal | Strong OOD detection |
| `block_diag` | Block-diagonal GGN | Good accuracy/speed tradeoff |
| `kron` | Kronecker-factored | Multi-output layers |
| `full` | Dense GGN | Reference quality (small models) |

---

## Tutorials

36 executable notebooks covering all methods. See the [tutorial index](https://vispikarkaria.github.io/Deep-UQ/tutorials/).

**Core UQ**: MC Dropout, SGLD, Bayes by Backprop, Laplace (full/comparison)

**Scientific ML**: DeepONet + Burgers/Poisson, FNO + Darcy/Heat, PINN + Poisson, CNN/UNet surrogates, Graph operators, Conditional diffusion

**Gaussian Processes**: Exact, sparse, classification, heteroscedastic, multi-task, spectral mixture, deep kernel

---

## Documentation

| Resource | Link |
|----------|------|
| Full docs | https://vispikarkaria.github.io/Deep-UQ/ |
| API reference | https://vispikarkaria.github.io/Deep-UQ/api/ |
| Method guides | https://vispikarkaria.github.io/Deep-UQ/methods/ |
| Tutorials | https://vispikarkaria.github.io/Deep-UQ/tutorials/ |

---

## Contributing

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git && cd Deep-UQ
pip install -e ".[dev,tests]" && pre-commit install
pytest -q && ruff check . && black --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Citation

```bibtex
@software{deep_uq,
  title = {Deep-UQ: Unified Deep Learning Uncertainty Quantification Toolkit},
  author = {Karkaria, Vispi Nevile},
  url = {https://github.com/Vispikarkaria/Deep-UQ},
  license = {MIT}
}
```

---

## License

MIT
