# Deep Ensemble Classifier for Elasticity Failure Maps

Notebook: [DeepEnsemble_Elasticity2D_Classification_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/ensembles/DeepEnsemble_Elasticity2D_Classification_Tutorial.ipynb)

This tutorial uses `DeepEnsembleClassifier` on a notch-mechanics-inspired failure map. The input space is scientific parameter space rather than physical coordinates, and the ensemble highlights uncertainty near the safe/failure boundary.

Key ideas:
- probability averaging across independent classifiers,
- failure classification under parameter-space distribution shift,
- ensemble disagreement near the decision boundary.

Primary references:
- Lakshminarayanan, Pritzel, Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
- Hansen, Salamon (1990), *Neural Network Ensembles*. DOI: [10.1109/34.58871](https://doi.org/10.1109/34.58871)
