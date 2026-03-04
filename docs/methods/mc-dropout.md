# MC Dropout

`deepuq` exposes Monte Carlo Dropout through `MCDropoutWrapper`.

## 1) Motivation

Dropout is widely used for regularization during training. MC Dropout reuses stochastic dropout masks at inference time and interprets repeated forward passes as approximate Bayesian model averaging.

This provides a low-friction uncertainty baseline for deep models.

## 2) What Uncertainty Is Quantified

MC Dropout primarily approximates **epistemic uncertainty** through stochastic subnetworks.

With \(T\) stochastic forward passes:

$$
\mu(x)=\frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

$$
\sigma^2_{\mathrm{epi}}(x)=
\frac{1}{T}\sum_{t=1}^{T}\left(f_t(x)-\mu(x)\right)^2
$$

Classification predictive probability estimate:

$$
\bar p(y\mid x)=
\frac{1}{T}\sum_{t=1}^{T} p_t(y\mid x)
$$

## 3) Mathematical Setup / Notation

For layer \(\ell\), let mask entries follow Bernoulli sampling:

$$
m_{\ell,j}\sim\mathrm{Bernoulli}(1-p_{\ell})
$$

One stochastic network realization is

$$
f_t(x)=f\!\left(x;\theta,m^{(t)}\right)
$$

where \(m^{(t)}\) is the mask sample at pass \(t\).

A standard variational interpretation uses a Bernoulli family over effective weights and approximates model averaging via Monte Carlo over masks.

## 4) Core Method Equations

Predictive first and second moments:

$$
\hat\mu(x)=\frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

$$
\widehat{\mathbb E}[f(x)^2]=\frac{1}{T}\sum_{t=1}^{T} f_t(x)^2
$$

So predictive variance estimate is:

$$
\widehat{\mathrm{Var}}[f(x)]
=
\widehat{\mathbb E}[f(x)^2]-\hat\mu(x)^2
$$

For class probabilities \(P_t(x)\in[0,1]^C\):

$$
\hat P(x)=\frac{1}{T}\sum_{t=1}^{T} P_t(x)
$$

and componentwise variance:

$$
\widehat{\mathrm{Var}}[P_c(x)]
=
\frac{1}{T}\sum_{t=1}^{T}\left(P_{t,c}(x)-\hat P_c(x)\right)^2
$$

## 5) Inference / Prediction Equations

In `deepuq`, prediction keeps dropout active and returns Monte Carlo summaries.

Regression-style output:

$$
\left(\hat\mu(x),\widehat{\mathrm{Var}}[f(x)]\right)
$$

Classification-style output:

$$
\left(\hat P(x),\widehat{\mathrm{Var}}[P(x)]\right)
$$

## 6) Practical Implications

- `n_mc` controls Monte Carlo error; larger values stabilize uncertainty estimates.
- Dropout rate \(p_{\ell}\) affects both fit and uncertainty amplitude.
- MC Dropout is computationally light relative to many full Bayesian alternatives.
- It is an approximation and may miss multimodal posterior behavior.

## 7) References

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15, 1929-1958. [JMLR](https://jmlr.org/papers/v15/srivastava14a.html)
2. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML (PMLR 48). [Proceedings](https://proceedings.mlr.press/v48/gal16.html)
3. Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS. [Proceedings](https://papers.nips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)

## Related docs

- [MC Dropout Tutorial Guide](../tutorials/mc-dropout.md)
- [MC Dropout API](../api/methods/mc_dropout.md)
