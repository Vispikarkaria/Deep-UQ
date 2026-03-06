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

### Variational Inference

Bayes by Backprop is the package's end-to-end Bayesian neural network option.
Use it when you want to train uncertainty into the weights from the start.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/variational-inference/

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Bayes by Backprop | Regression, Classification | Epistemic | End-to-end Bayesian neural-network training | `BayesianLinear`, `BayesByBackpropMLP`, `vi_elbo_step`, `predict_vi_uq` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/variational-inference/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/bayes-by-backprop/)<br>`notebooks/BayesByBackprop_Tutorial.ipynb` |

### Laplace Approximation

Laplace methods build a Gaussian posterior around a trained MAP solution. Use
them when you want stronger post-hoc uncertainty without retraining a Bayesian
neural network.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/laplace/

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Diagonal Laplace | Regression, Classification | Epistemic | Fast local posterior around a trained model | `LaplaceWrapper(hessian_structure="diag")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Laplace tutorial](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_Tutorial.ipynb` |
| Fisher-Diagonal Laplace | Regression, Classification | Epistemic | Diagonal empirical-Fisher style approximation | `LaplaceWrapper(hessian_structure="fisher_diag")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Comparison guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Low-Rank + Diagonal Laplace | Regression, Classification | Epistemic | Capture dominant coupled directions cheaply | `LaplaceWrapper(hessian_structure="lowrank_diag")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Comparison guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Block-Diagonal Laplace | Regression, Classification | Epistemic | Preserve within-block coupling at moderate memory cost | `LaplaceWrapper(hessian_structure="block_diag")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Comparison guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Kronecker-Factored Laplace | Regression, Classification | Epistemic | Better fidelity/cost trade-off for layerwise structures | `LaplaceWrapper(hessian_structure="kron")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Comparison guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb` |
| Full-Hessian Laplace | Regression, Classification | Epistemic | Dense curvature for small enough models | `LaplaceWrapper(hessian_structure="full")` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/laplace/) | [Full-Hessian guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/laplace-comparison/)<br>`notebooks/laplace/Laplace_FullHessian_Tutorial.ipynb` |

### MCMC / SGLD

SGLD is the package's sampling-based deep-learning UQ method. Use it when you
want posterior samples directly rather than a closed-form approximation.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/mcmc-sgld/

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Stochastic Gradient Langevin Dynamics | Regression, Classification | Posterior samples, Epistemic | SGD-like posterior sampling with predictive averaging | `SGLDOptimizer`, `collect_posterior_samples`, `predict_with_samples_uq` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/mcmc-sgld/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/sgld/)<br>`notebooks/SGLD_Tutorial.ipynb` |

### MC Dropout

MC Dropout is the lowest-friction uncertainty baseline in the library. Use it
when you already have dropout in your model and want uncertainty estimates with
very little retraining effort.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/mc-dropout/

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| MC Dropout | Regression, Classification | Approx. Epistemic | Fast uncertainty baseline for dropout-enabled models | `MCDropoutWrapper`, `predict_uq` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/mc-dropout/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/mc-dropout/)<br>`notebooks/MC_Dropout_Tutorial.ipynb` |

### Gaussian Processes

The GP family covers the most structured nonparametric uncertainty models in the
package, from exact regression to deep kernel learning.

Read more:
- https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/

| Method | Primary Task | Uncertainty Type | Best Use Case | Main Interface | Docs | Tutorial |
|---|---|---|---|---|---|---|
| Exact GP Regression | Regression | Epistemic + Aleatoric | Strong small-data calibration baseline | `GaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_Exact_Tutorial.ipynb` |
| Sparse Variational GP | Regression | Epistemic + Aleatoric | Scalable inducing-point GP regression | `SparseGaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_Sparse_Tutorial.ipynb` |
| GP Classifier | Classification | Class-probability uncertainty | Binary decision-boundary uncertainty | `GaussianProcessClassifier` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_Classification_Tutorial.ipynb` |
| OvR GP Classifier | Classification | Class-probability uncertainty | Multiclass GP classification via one-vs-rest | `OneVsRestGaussianProcessClassifier` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_Classification_Tutorial.ipynb` |
| Heteroscedastic GP | Regression | Epistemic + Aleatoric | Input-dependent noise decomposition | `HeteroscedasticGaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_Heteroscedastic_Tutorial.ipynb` |
| Multi-task ICM GP | Multi-output Regression | Shared-output Epistemic + Aleatoric | Correlated multi-output regression | `MultiTaskGaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_MultiTask_ICM_Tutorial.ipynb` |
| Spectral Mixture GP | Regression | Epistemic + Aleatoric | Multi-frequency signal structure and extrapolation | `SpectralMixtureGaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_SpectralMixture_Tutorial.ipynb` |
| Deep Kernel GP | Regression | Epistemic + Aleatoric | Learn a feature space before GP inference | `DeepKernelGaussianProcessRegressor` | [Method guide](https://vispikarkaria.github.io/Deep-UQ/methods/gaussian-processes/) | [Tutorial guide](https://vispikarkaria.github.io/Deep-UQ/tutorials/gp/)<br>`notebooks/gp/GP_DeepKernel_Tutorial.ipynb` |

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
