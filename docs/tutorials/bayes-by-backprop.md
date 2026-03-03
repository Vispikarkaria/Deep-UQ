# Tutorial: Bayes by Backprop

Notebook: [BayesByBackprop_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/BayesByBackprop_Tutorial.ipynb)

## Purpose

Train a Bayesian neural network with variational inference and analyze predictive uncertainty behavior in interpolation and extrapolation settings.

## Data Setup

- Synthetic 1D nonlinear regression
- In-distribution and OOD evaluation ranges
- Controlled noise level to separate epistemic and aleatoric effects

## Core Logic

- Bayesian linear layers with sampled weights
- ELBO optimization using `vi_elbo_step`
- MC-based predictive mean and confidence intervals

## Expected Outputs

- ELBO/NLL/KL training curves
- fit quality on train/test
- uncertainty widening in OOD regions
