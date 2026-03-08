# CNN / ResNet + Heat2D UQ

Notebook: [CNN_ResNet_Heat2D_UQ_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/CNN_ResNet_Heat2D_UQ_Tutorial.ipynb)

This tutorial trains `CNNRegressor2D` and `ResNetRegressor2D` on a 2D heat
source-to-solution problem. MC Dropout is the primary uncertainty path, with an
optional ensemble comparison for stronger calibration.

Primary references:

- He et al. (2016), *Deep Residual Learning for Image Recognition*
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
