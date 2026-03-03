# deepuq — Usage Guide

This guide shows end-to-end training and uncertainty estimation for the four methods provided by `deepuq`.

## Data
The examples use a synthetic Euler-Bernoulli beam deflection dataset. Replace with your dataset of choice; you only need tensors `(X, y)`.

## 1) MC Dropout
```python
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

model = MLP(1, [128,128], 1, p_drop=0.15)
# ...train with MSE loss...
uq = MCDropoutWrapper(model, n_mc=200, apply_softmax=False)
mean, var = uq.predict(x_batch)  # [B,1] each
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
```
Notes:
- `diag`, `fisher_diag`, `lowrank_diag`, `block_diag` are native backends in `deepuq`.
- `kron` and `full` require `laplace-torch`.
- `full` with `subset_of_weights='all'` is guarded by `full_max_params` to avoid infeasible memory usage.

## 4) MCMC (SGLD)
```python
import torch.nn as nn
from deepuq.methods import collect_posterior_samples, predict_with_samples

loss_fn = nn.MSELoss(reduction='mean')
samples = collect_posterior_samples(model, train_loader, n_steps=200, lr=1e-4, loss_fn=loss_fn)
mean, var = predict_with_samples(model, samples, x_batch, apply_softmax=False)
```
Tune `lr`, `weight_decay`, and `burn_in` for better mixing. Save samples to disk if needed.

## Calibration and Metrics
- For regression, track **RMSE**, **MAE**, and 95% interval coverage.
- Adjust `n_mc` or posterior sample counts to stabilize uncertainty bands.
