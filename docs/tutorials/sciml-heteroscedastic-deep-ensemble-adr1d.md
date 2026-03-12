# Heteroscedastic Deep Ensemble for 1D Advection-Diffusion-Reaction

Notebook: [HeteroscedasticDeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/ensembles/HeteroscedasticDeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb)

This tutorial extends the 1D ADR problem with spatially varying observation noise. Each ensemble member predicts both a mean response and a local noise variance, allowing the notebook to separate epistemic and aleatoric uncertainty.

Key ideas:
- heteroscedastic Gaussian outputs,
- aleatoric noise highest near the center of the domain,
- ensemble spread vs predicted noise comparison.

Primary references:
- Nix, Weigend (1994), *Estimating the Mean and Variance of the Target Probability Distribution*. DOI: [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138)
- Kendall, Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* [NeurIPS proceedings](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need)
- Lakshminarayanan, Pritzel, Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
