# Tutorial: Gaussian Processes

Notebooks:

- [GaussianProcess_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/GaussianProcess_Tutorial.ipynb)
- [SparseGaussianProcess_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/SparseGaussianProcess_Tutorial.ipynb)

## Purpose

Compare exact GP and sparse variational GP behavior on regression tasks.

## Data Setup

- smooth synthetic regression functions
- varying sample density and noise

## Core Logic

- exact posterior inference with RBF kernels
- sparse inducing-point approximation optimized with ELBO

## Expected Outputs

- posterior mean and variance plots
- posterior samples
- training objective trends for sparse GP
