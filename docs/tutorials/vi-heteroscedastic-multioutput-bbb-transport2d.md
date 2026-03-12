# Tutorial: Heteroscedastic Multi-Output Bayes by Backprop on Transport2D

Notebook: [HeteroscedasticMultiOutput_BayesByBackprop_Transport2D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/vi/HeteroscedasticMultiOutput_BayesByBackprop_Transport2D_Tutorial.ipynb)

## Scientific problem

A pointwise 2D transport field with two outputs: concentration and flux magnitude, both observed with input-dependent noise.

## Input and output

- input: `(x, y)` plus transport parameters
- output: `[c(x, y), |\nabla c(x, y)|]` and corresponding predicted variances

## UQ method

- `HeteroscedasticMultiOutputBayesByBackpropRegressor`
- `predict_vi_uq(...)` returns multi-output epistemic and aleatoric terms

## Why this notebook exists

This is the most complete regression VI example in the package: multi-output, Bayesian weight uncertainty, and input-dependent noise in one workflow.
