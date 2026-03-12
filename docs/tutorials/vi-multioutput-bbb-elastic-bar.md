# Tutorial: Multi-Output Bayes by Backprop on a 1D Elastic Bar

Notebook: [MultiOutput_BayesByBackprop_ElasticBar1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/vi/MultiOutput_BayesByBackprop_ElasticBar1D_Tutorial.ipynb)

## Scientific problem

A 1D elastic-bar surrogate with two coupled outputs: displacement and axial stress.

## Input and output

- input: coordinate `x` plus material/load parameters
- output: `[u(x), \sigma(x)]`

## UQ method

- `MultiOutputBayesByBackpropRegressor`
- `predict_vi_uq(...)` returns vector-valued predictive moments for both outputs

## Why this notebook exists

This notebook demonstrates how one Bayesian MLP can quantify uncertainty over multiple physically coupled targets.
