# deepuq — Usage Guide

This guide shows end-to-end training and uncertainty estimation for the core methods provided by `deepuq`.

## Common UQResult API

Alongside legacy `predict(...)` outputs, Deep-UQ now supports standardized `predict_uq(...)` outputs through `deepuq.UQResult`.

`UQResult` fields:

- `mean`
- `epistemic_var`
- `aleatoric_var`
- `total_var`
- `probs`, `probs_var`
- `metadata`

## Data
The examples use a synthetic Euler-Bernoulli beam deflection dataset. Replace with your dataset of choice; you only need tensors `(X, y)`.

## Method Family Reference

The website is the canonical place to read the full method documentation. This
usage guide keeps a compact family-level reference and links outward to the
detailed method pages.

Legend: `✓` direct support, `✗` not a primary capability for that method.

### Variational Inference

Bayes by Backprop is the package's end-to-end Bayesian neural network option.
Use it when you want to train uncertainty into the weights from the start.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/variational-inference/

| Method | Reg. | Cls. | Multi | Model UQ | Noise UQ | Main Interface | Learn More |
|---|---|---|---|---|---|---|---|
| Bayes by Backprop | ✓ | ✓ | ✗ | ✓ | ✗ | `BayesianLinear`, `BayesByBackpropMLP`, `vi_elbo_step`, `predict_vi_uq` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/variational-inference/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/bayes-by-backprop/) |

### Laplace Approximation

Laplace methods build a Gaussian posterior around a trained MAP solution. Use
them when you want stronger post-hoc uncertainty without retraining a Bayesian
neural network.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/laplace/

| Method | Reg. | Cls. | Multi | Model UQ | Noise UQ | Main Interface | Learn More |
|---|---|---|---|---|---|---|---|
| Diagonal Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="diag")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |
| Fisher-Diagonal Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="fisher_diag")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |
| Low-Rank + Diagonal Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="lowrank_diag")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |
| Block-Diagonal Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="block_diag")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |
| Kronecker-Factored Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="kron")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |
| Full-Hessian Laplace | ✓ | ✓ | ✗ | ✓ | ✗ | `LaplaceWrapper(hessian_structure="full")` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/) |

### MCMC / SGLD

SGLD is the package's sampling-based deep-learning UQ method. Use it when you
want posterior samples directly rather than a closed-form approximation.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/mcmc-sgld/

| Method | Reg. | Cls. | Multi | Model UQ | Noise UQ | Main Interface | Learn More |
|---|---|---|---|---|---|---|---|
| Stochastic Gradient Langevin Dynamics | ✓ | ✓ | ✗ | ✓ | ✗ | `SGLDOptimizer`, `collect_posterior_samples`, `predict_with_samples_uq` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/mcmc-sgld/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/sgld/) |

### MC Dropout

MC Dropout is the lowest-friction uncertainty baseline in the library. Use it
when you already have dropout in your model and want uncertainty estimates with
very little retraining effort.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/mc-dropout/

| Method | Reg. | Cls. | Multi | Model UQ | Noise UQ | Main Interface | Learn More |
|---|---|---|---|---|---|---|---|
| MC Dropout | ✓ | ✓ | ✗ | ✓ | ✗ | `MCDropoutWrapper`, `predict_uq` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/mc-dropout/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/mc-dropout/) |

### Gaussian Processes

The GP family covers the most structured nonparametric uncertainty models in the
package, from exact regression to deep kernel learning.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/

| Method | Reg. | Cls. | Multi | Model UQ | Noise UQ | Main Interface | Learn More |
|---|---|---|---|---|---|---|---|
| Exact GP Regression | ✓ | ✗ | ✗ | ✓ | ✓ | `GaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| Sparse Variational GP | ✓ | ✗ | ✗ | ✓ | ✓ | `SparseGaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| GP Classifier | ✗ | ✓ | ✗ | ✓ | ✗ | `GaussianProcessClassifier` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| OvR GP Classifier | ✗ | ✓ | ✗ | ✓ | ✗ | `OneVsRestGaussianProcessClassifier` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| Heteroscedastic GP | ✓ | ✗ | ✗ | ✓ | ✓ | `HeteroscedasticGaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| Multi-task ICM GP | ✓ | ✗ | ✓ | ✓ | ✓ | `MultiTaskGaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| Spectral Mixture GP | ✓ | ✗ | ✗ | ✓ | ✓ | `SpectralMixtureGaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |
| Deep Kernel GP | ✓ | ✗ | ✗ | ✓ | ✓ | `DeepKernelGaussianProcessRegressor` | [Guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/)<br>[Tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/) |

## 1) MC Dropout
```python
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

model = MLP(1, [128,128], 1, p_drop=0.15)
# ...train with MSE loss...
uq = MCDropoutWrapper(model, n_mc=200, apply_softmax=False)
mean, var = uq.predict(x_batch)  # [B,1] each
uq_result = uq.predict_uq(x_batch)
```
Interpretation: `var` is predictive variance from Monte Carlo stochasticity. Increase `n_mc` to smooth the uncertainty estimate.

## 2) Variational Inference (Bayes by Backprop)
```python
import torch.nn as nn
from deepuq.methods import BayesByBackpropMLP, vi_elbo_step

model = BayesByBackpropMLP(1, [128,128], 1, prior_sigma=0.2)
criterion = nn.MSELoss(reduction='mean')
num_batches = len(train_loader)
kl_weight = 0.01
for x,y in train_loader:
    loss, nll, kl = vi_elbo_step(
        model,
        x,
        y,
        num_batches=num_batches,
        criterion=criterion,
        kl_weight=kl_weight,
        mc_samples=1,
    )
    loss.backward(); opt.step()
```
At inference, sample multiple weight draws and average to obtain predictive mean/variance.
For epoch-to-epoch ELBO comparisons, keep `kl_weight` fixed. Raw ELBO can wiggle due to stochastic sampling; use an EMA trend for reporting.

## 3) Laplace Approximation
```python
from deepuq.methods import LaplaceWrapper

# Available hessian_structure values:
#   'diag', 'fisher_diag', 'lowrank_diag', 'block_diag', 'kron', 'full'
la = LaplaceWrapper(
    trained_model,
    likelihood='regression',
    hessian_structure='diag',       # swap for 'kron' or 'full' if needed
    subset_of_weights='last_layer', # or 'all'
    lowrank_rank=20,                # used by 'lowrank_diag'
    damping=1e-6,
)
la.fit(train_loader, prior_precision=1.0)
mean, var = la.predict(x_batch, n_samples=200)
uq_result = la.predict_uq(x_batch, n_samples=200)
```
Notes:
- `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, and `full` are native backends in `deepuq`.
- `full` with `subset_of_weights='all'` is guarded by `full_max_params` to avoid infeasible memory usage.
- Theory reference (equations and citations): `https://vispikarkaria.github.io/Deep-UQ/methods/laplace/`

## 4) MCMC (SGLD)
```python
import torch.nn as nn
from deepuq.methods import collect_posterior_samples, predict_with_samples
from deepuq.methods import predict_with_samples_uq

loss_fn = nn.MSELoss(reduction='mean')
samples = collect_posterior_samples(model, train_loader, n_steps=200, lr=1e-4, loss_fn=loss_fn)
mean, var = predict_with_samples(model, samples, x_batch, apply_softmax=False)
uq_result = predict_with_samples_uq(model, samples, x_batch, apply_softmax=False)
```
Tune `lr`, `weight_decay`, and `burn_in` for better mixing. Save samples to disk if needed.

## 5) Gaussian Processes
```python
from deepuq.models import GaussianProcessRegressor

gp = GaussianProcessRegressor(noise=1e-4)
gp.fit(x_train, y_train)
mean, var = gp.predict(x_batch)
uq_result = gp.predict_uq(x_batch)
```

For sparse GP:
```python
from deepuq.models import SparseGaussianProcessRegressor

sgp = SparseGaussianProcessRegressor(num_inducing=32, num_iterations=200)
sgp.fit(x_train, y_train)
uq_result = sgp.predict_uq(x_batch)
```

Kernel variants and composition:
```python
from deepuq.models import (
    GaussianProcessRegressor,
    MaternKernel,
    PeriodicKernel,
    LinearKernel,
)

kernel = MaternKernel(lengthscale=0.8, outputscale=1.1, nu=2.5) + PeriodicKernel(
    lengthscale=0.6, outputscale=0.7, period=2.2
) * LinearKernel(variance=0.02)
gp = GaussianProcessRegressor(kernel=kernel, noise=0.01)
```

GP classification:
```python
from deepuq.models import GaussianProcessClassifier, OneVsRestGaussianProcessClassifier

gpc = GaussianProcessClassifier()
gpc.fit(x_train, y_binary)
uq_binary = gpc.predict_uq(x_batch)

ovr = OneVsRestGaussianProcessClassifier()
ovr.fit(x_train, y_multiclass)
uq_multi = ovr.predict_uq(x_batch)
```

Advanced GP regressors:
```python
from deepuq.models import (
    DeepKernelGaussianProcessRegressor,
    HeteroscedasticGaussianProcessRegressor,
    MultiTaskGaussianProcessRegressor,
    SpectralMixtureGaussianProcessRegressor,
)

het = HeteroscedasticGaussianProcessRegressor().fit(x_train, y_train)
sm = SpectralMixtureGaussianProcessRegressor(num_mixtures=4).fit(x_train, y_train)
dkl = DeepKernelGaussianProcessRegressor().fit(x_train_features, y_train)
mt = MultiTaskGaussianProcessRegressor(num_tasks=3).fit(x_train, y_train_multi)
```

GP tutorials are organized under `notebooks/gp/`.

## Calibration and Metrics
- For regression, track **RMSE**, **MAE**, and 95% interval coverage.
- Adjust `n_mc` or posterior sample counts to stabilize uncertainty bands.
- For reproducible multi-dataset comparisons, run `python benchmarks/run_benchmarks.py --preset quick`.
