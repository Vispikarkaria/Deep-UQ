# Deep Ensembles

Deep ensembles estimate predictive uncertainty by training multiple independent
models from different random initializations and aggregating their predictions.
In Deep-UQ, this is provided by `DeepEnsembleWrapper` and is positioned as a
strong regression-first baseline for deterministic backbones.

## Why Use Deep Ensembles

Deep ensembles are often the most pragmatic uncertainty baseline for neural
networks:

- no posterior approximation assumptions are required
- they work naturally with convolutional backbones
- predictive spread is easy to interpret as model uncertainty

For regression with models $f_{\theta^{(m)}}(x)$, the predictive mean is

$$
\mu(x)=\frac{1}{M}\sum_{m=1}^M f_{\theta^{(m)}}(x)
$$

and the ensemble epistemic variance is

$$
\sigma^2_{\mathrm{epi}}(x)=\frac{1}{M}\sum_{m=1}^M \left(f_{\theta^{(m)}}(x)-\mu(x)\right)^2.
$$

## Deep-UQ Interface

Use `deepuq.methods.DeepEnsembleWrapper` with a list of independently
initialized models.

## Recommended Model Pairings

- `MLP` for parametric regression
- `CNNRegressor2D` / `ResNetRegressor2D` for image-like scientific fields
- `UNet2D` / `UNet3D` for multi-scale field-to-field surrogates

## References

1. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS. [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
2. Wilson, A. G., & Izmailov, P. (2020). *Bayesian Deep Learning and a Probabilistic Perspective of Generalization*. NeurIPS tutorial survey. [arXiv](https://arxiv.org/abs/2002.08791)
