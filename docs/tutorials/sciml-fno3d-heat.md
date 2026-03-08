# FNO3D + 3D Heat Equation + Laplace

Notebook: [FNO3D_Heat_Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/FNO3D_Heat_Laplace_Tutorial.ipynb)

This tutorial builds a 3D scientific machine learning example around the periodic heat equation

$$
\partial_t u = \kappa \Delta u, \qquad (x,y,z)\in[0,1)^3,
$$

and learns the operator

$$
\mathcal{G}: u_0(x,y,z) \mapsto u(x,y,z,T)
$$

with a reusable `FNO3D` model from `deepuq.models`.

What the notebook covers:

- physical meaning of the 3D heat-diffusion problem,
- exact spectral data generation for train/test/OOD splits,
- a 3D Fourier Neural Operator with spectral convolution layers,
- a residual physics baseline that keeps the CPU tutorial accurate,
- last-layer Laplace approximation with `block_diag`,
- predictive mean and epistemic-uncertainty slice plots across the 3D grid.

Why this notebook is useful:

- it is the package's first 3D operator-learning example,
- the PDE has an exact spectral solution, so the tutorial can focus on learning and UQ rather than numerical solver noise,
- uncertainty can be inspected on central `xy`, `xz`, and `yz` slices and compared between in-domain and OOD inputs.

Executed quick-mode result in this repository:

- MAP test RMSE: about `1.0e-5`
- Laplace test RMSE: about `7.9e-4`
