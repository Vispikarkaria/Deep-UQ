# Multi-Output Deep Ensemble for a 1D Elastic Bar

Notebook: [MultiOutputDeepEnsemble_ElasticBar1D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/MultiOutputDeepEnsemble_ElasticBar1D_Tutorial.ipynb)

This tutorial uses `MultiOutputDeepEnsembleRegressor` on a 1D elastic bar. The model predicts two coupled scientific outputs at once: displacement and stress.

Key ideas:
- multi-output regression under parameter variation,
- simultaneous uncertainty bands for displacement and stress,
- OOD cases with stronger loading and stiffness gradients.

Primary references:
- Lakshminarayanan, Pritzel, Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
- Nix, Weigend (1994), *Estimating the Mean and Variance of the Target Probability Distribution*. DOI: [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138)
