# deepuq — Usage Guide

This guide shows end-to-end training and uncertainty estimation for the four methods provided by `deepuq`.

## Data
The example scripts use MNIST from `torchvision`. Replace with your dataset of choice; you only need tensors `(X, y)`.

## 1) MC Dropout
```python
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

model = MLP(784, [512,256], 10, p_drop=0.2)
# ...train with standard CE loss...
uq = MCDropoutWrapper(model, n_mc=30)
mean, var = uq.predict(x_batch)  # [B,C] each
```
Interpretation: `var` is predictive variance from Monte Carlo stochasticity. Calibrate with temperature scaling or increase `n_mc`.

## 2) Variational Inference (Bayes by Backprop)
```python
from deepuq.methods import BayesByBackpropMLP, vi_elbo_step
model = BayesByBackpropMLP(784, [400,200], 10, prior_sigma=0.1)
for x,y in train_loader:
    loss, nll, kl = vi_elbo_step(model, x, y, n_batches=len(train_loader))
    loss.backward(); opt.step()
```
At inference, sample multiple forward passes and average to obtain predictive mean/variance.

## 3) Laplace Approximation
```python
from deepuq.methods import LaplaceWrapper
la = LaplaceWrapper(trained_model, 'classification', 'diag')
la.fit(train_loader, prior_precision=1.0)
probs, _ = la.predict(x_batch)
```
`laplace-torch` handles posterior predictive. You can switch `hessian_structure` to `'kron'` or `'full'` if memory allows.

## 4) MCMC (SGLD)
```python
from deepuq.methods import collect_posterior_samples, predict_with_samples
samples = collect_posterior_samples(model, train_loader, n_steps=200, lr=1e-4)
mean, var = predict_with_samples(model, samples, x_batch)
```
Tune `lr`, `weight_decay`, and `burn_in` for better mixing. Save samples to disk if needed.

## Calibration and Metrics
- **NLL**, **Brier score**, **ECE** (expected calibration error) are recommended. Add temperature scaling on validation.
- For regression, modify losses to MSE and output Gaussian parameters.

