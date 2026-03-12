# DeepONet + 1D Poisson + Laplace

Notebook: [DeepONet_Poisson1D_Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/operators/DeepONet_Poisson1D_Laplace_Tutorial.ipynb)

This tutorial builds a 1D scientific machine learning example around the boundary value problem

$$
-u''(x) = f(x), \qquad x \in [0, 1], \qquad u(0)=u(1)=0.
$$

It uses a `DeepONet1D` surrogate to learn the field-to-field map from a forcing field $f(x)$ to the equilibrium response field $u(x)$, then adds a last-layer Laplace approximation to quantify predictive uncertainty.

What the notebook covers:

- physical meaning of the Poisson problem in diffusion / elasticity / electrostatics language,
- dataset generation from random smooth forcing fields,
- DeepONet branch/trunk structure for 1D operator learning,
- MAP training diagnostics,
- uncertainty bands over the 1D spatial coordinate,
- in-domain vs OOD comparisons for uncertainty calibration.

Why this notebook is useful:

- both the input and the output are 1D fields, so the neural-operator map is easy to interpret,
- uncertainty is visible as a standard mean-curve plus confidence-band plot,
- the Poisson solver is stable and inexpensive, which keeps the tutorial focused on operator learning rather than PDE debugging.
