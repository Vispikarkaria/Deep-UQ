# Tutorial: Bayes by Backprop

Notebook: [BayesByBackprop_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/vi/BayesByBackprop_Tutorial.ipynb)

## Purpose

Train a mean-field Bayesian neural network with Bayes by Backprop and analyze predictive uncertainty in interpolation and extrapolation settings.

## Data Setup

- synthetic 1D nonlinear regression
- in-domain and OOD evaluation ranges
- controlled observation noise to separate model uncertainty from the noise floor

## Core Logic

- `BayesianLinear` layers sample weights during each forward pass
- `vi_elbo_step(...)` optimizes the ELBO with mini-batch KL scaling
- `predict_vi_uq(...)` aggregates Monte Carlo predictive draws into a `UQResult`

## Expected Outputs

- ELBO / NLL / KL curves
- fit quality on train and held-out ranges
- wider uncertainty outside the training support
