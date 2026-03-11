# Scientific Deep Ensemble for 1D Advection-Diffusion-Reaction

Notebook: [DeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/DeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb)

This tutorial uses `DeepEnsembleRegressor` on a steady 1D advection-diffusion-reaction boundary-value problem. The ensemble learns a pointwise surrogate for the PDE response and visualizes epistemic uncertainty bands over the spatial coordinate.

Key ideas:
- coordinate-plus-parameter regression for a scientific PDE surrogate,
- parameter-space OOD shift through sharper forcing and stronger transport,
- predictive mean and ensemble spread as the uncertainty signal.

Primary references:
- Lakshminarayanan, Pritzel, Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
- Hansen, Salamon (1990), *Neural Network Ensembles*. DOI: [10.1109/34.58871](https://doi.org/10.1109/34.58871)
