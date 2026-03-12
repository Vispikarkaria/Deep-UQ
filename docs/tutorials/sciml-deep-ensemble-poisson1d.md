# Deep Ensembles + Parametric Poisson 1D

Notebook: [DeepEnsemble_ParametricPoisson1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/ensembles/DeepEnsemble_ParametricPoisson1D_Tutorial.ipynb)

This tutorial uses an ensemble of `MLP` models to learn a parametric 1D Poisson
response map. Inputs are the coordinate and source parameters, output is the
solution value, and uncertainty is visualized as a band that widens on OOD
forcing settings. The notebook now explicitly distinguishes this
parameter-space OOD effect from a separate physical-space sparse-data story.

Primary references:

- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
- Rasmussen & Williams (2006) for the broader uncertainty calibration context
