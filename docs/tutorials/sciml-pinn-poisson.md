# PINN + Poisson + Laplace

Notebook: [PINN_Poisson_Laplace_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/PINN_Poisson_Laplace_Tutorial.ipynb)

This tutorial introduces `PINN1D` and `PINN2D` on analytic Poisson problems.
The networks are trained with physics-informed residuals, and uncertainty is
quantified with last-layer Laplace over sparse supervised anchor data.

Primary references:

- Raissi, Perdikaris, & Karniadakis (2019), *Physics-informed neural networks*
- MacKay (1992), *A Practical Bayesian Framework for Backpropagation Networks*
- Ritter et al. (2018), *A Scalable Laplace Approximation for Neural Networks*
