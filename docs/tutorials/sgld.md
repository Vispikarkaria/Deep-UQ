# Tutorial: SGLD

Notebook: [SGLD_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/SGLD_Tutorial.ipynb)

## Purpose

Approximate Bayesian posterior sampling through stochastic gradient Langevin dynamics.

## Data Setup

- Supervised learning setup with mini-batches
- Burn-in and sampling windows

## Core Logic

- run `SGLDOptimizer` updates
- collect model parameter snapshots
- estimate predictive mean/variance from sampled models

## Expected Outputs

- posterior sample trajectories
- uncertainty-aware predictions from sample aggregation
