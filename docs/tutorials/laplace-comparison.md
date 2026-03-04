# Tutorial: Laplace Hessian Comparison

Notebook: [Laplace_HessianComparison_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/laplace/Laplace_HessianComparison_Tutorial.ipynb)

## Purpose

Compare all Laplace Hessian structures on one shared MAP model and dataset.

## Data Setup

- Nonlinear 1D regression
- Train/validation/test split in the same support
- OOD plotting range outside train support

## Core Logic

- Train one MAP MLP checkpoint
- Fit each structure with `LaplaceWrapper`
- Tune prior precision by validation NLL
- Compare metrics and uncertainty profiles

## Metrics

- RMSE / MAE / NLL
- 95% coverage and interval width
- ID vs OOD predictive standard deviation ratio

## Scientific Details

- Equations, derivations, and references for all six Hessian structures:
  - [Laplace method scientific reference](../methods/laplace.md)

## Related Notebooks

- [Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/laplace/Laplace_Tutorial.ipynb)
- [Laplace_FullHessian_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/laplace/Laplace_FullHessian_Tutorial.ipynb)
