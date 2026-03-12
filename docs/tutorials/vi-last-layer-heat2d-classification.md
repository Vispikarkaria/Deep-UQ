# Tutorial: Last-Layer VI for Heat2D Failure Classification

Notebook: [LastLayerVI_Heat2D_Classification_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/vi/LastLayerVI_Heat2D_Classification_Tutorial.ipynb)

## Scientific problem

Classify 2D heat-source maps into safe and failure regimes based on whether the peak temperature exceeds a fixed threshold.

## Input and output

- input: 2D source field
- output: binary class probabilities for `safe` / `failure`

## UQ method

- deterministic convolutional feature extractor
- `LastLayerVariationalInference(..., task="classification")`
- `predict_vi_uq(...)` returns class probabilities and probability variance

## Why this notebook exists

This notebook is the scalable VI example for larger deterministic backbones. Only the final linear head is Bayesian, which keeps training and sampling practical.
