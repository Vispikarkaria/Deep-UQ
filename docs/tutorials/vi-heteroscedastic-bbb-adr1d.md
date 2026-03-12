# Tutorial: Heteroscedastic Bayes by Backprop on ADR1D

Notebook: [Heteroscedastic_BayesByBackprop_ADR1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/vi/Heteroscedastic_BayesByBackprop_ADR1D_Tutorial.ipynb)

## Scientific problem

A manufactured 1D advection-diffusion-reaction field with spatially varying observation noise.

## Input and output

- input: coordinate `x` plus ADR/source parameters
- output: scalar field value `u(x)` with predicted observation variance

## UQ method

- `HeteroscedasticBayesByBackpropRegressor`
- `predict_vi_uq(...)` returns `mean`, `epistemic_var`, `aleatoric_var`, and `total_var`

## Why this notebook exists

This is the cleanest VI example for separating model uncertainty from input-dependent noise in a scientific regression setting.
