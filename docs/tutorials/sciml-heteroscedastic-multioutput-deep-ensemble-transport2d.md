# Heteroscedastic Multi-Output Deep Ensemble for 2D Transport

Notebook: [HeteroscedasticMultiOutputDeepEnsemble_Transport2D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/HeteroscedasticMultiOutputDeepEnsemble_Transport2D_Tutorial.ipynb)

This tutorial uses `HeteroscedasticMultiOutputDeepEnsembleRegressor` on a 2D advection-diffusion transport problem. The ensemble predicts both concentration and flux magnitude fields, together with per-pixel aleatoric noise.

Key ideas:
- convolutional backbones inside a deep ensemble,
- multi-output field prediction,
- uncertainty increasing for high-frequency OOD plumes.

Primary references:
- Nix, Weigend (1994), *Estimating the Mean and Variance of the Target Probability Distribution*. DOI: [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138)
- Kendall, Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* [NeurIPS proceedings](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need)
- Lakshminarayanan, Pritzel, Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
