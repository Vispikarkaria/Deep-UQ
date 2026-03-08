# Tutorial: DeepONet + Burgers + Laplace

Notebook: [DeepONet_Burgers_Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/DeepONet_Burgers_Laplace_Tutorial.ipynb)

## Purpose

Train a DeepONet-style surrogate for a small 2D viscous Burgers problem and
quantify spatial epistemic uncertainty with the existing Laplace approximation
tooling in Deep-UQ.

## Data Setup

- Scalar 2D viscous Burgers evolution on `[0, 1]^2`
- Periodic boundary conditions
- Smooth random initial conditions sampled from low-frequency Fourier modes
- Operator target: map the initial field to the solution field at a fixed final time

## Core Logic

- Reusable `DeepONet2D` model with:
  - branch network for sampled input functions
  - trunk network for spatial coordinates
  - linear readout over fused branch/trunk features so last-layer Laplace is well-defined
- MAP training with field-wise MSE loss
- Last-layer Laplace on the trained model using `LaplaceWrapper`
- Optional `diag` vs `kron` curvature comparison on the same fixed query grid

## Expected Outputs

- Burgers initial-condition and final-solution field plots
- MAP prediction field and absolute error heatmap
- Laplace predictive mean field
- Epistemic uncertainty / standard-deviation heatmap over the 2D domain
- Optional comparison of `diag` and `kron` uncertainty maps
