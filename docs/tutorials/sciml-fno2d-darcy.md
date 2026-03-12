# FNO2D + Darcy Flow + Laplace

Notebook: [FNO2D_Darcy_Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/operators/FNO2D_Darcy_Laplace_Tutorial.ipynb)

This tutorial builds a 2D scientific machine learning example around the
Darcy-flow equation

$$
-\nabla \cdot \left(k(x,y)\nabla u(x,y)\right) = q(x,y),
$$

with Dirichlet boundary condition

$$
u|_{\partial \Omega} = g(x,y).
$$

The notebook uses three input fields and one output field:

- permeability `k(x,y)`,
- source / sink field `q(x,y)`,
- boundary-condition field `g(x,y)`,
- pressure / hydraulic head `u(x,y)`.

What the notebook covers:

- physical interpretation of Darcy flow and the role of each input field,
- notebook-local dataset generation with a variable-coefficient finite-difference
  solver,
- a reusable `FNO2D` model from `deepuq.models`,
- residual training around a deterministic Darcy baseline,
- last-layer Laplace approximation with `block_diag`,
- optional comparison against `lowrank_diag`,
- uncertainty maps comparing in-domain and OOD permeability / forcing regimes.

Why this tutorial matters:

- it is the package's first 2D Fourier-neural-operator example,
- it uses a realistic elliptic PDE rather than only time-evolution examples,
- it shows how uncertainty increases when permeability contrast and forcing
  patterns move outside the training distribution.

Primary references cited in the notebook:

- Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*.
- Lu Lu et al., *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*.
- David J. C. MacKay, *A Practical Bayesian Framework for Backpropagation Networks*.
- Hippolyt Ritter et al., *A Scalable Laplace Approximation for Neural Networks*.
- Erik Daxberger et al., *Laplace Redux - Effortless Bayesian Deep Learning*.
