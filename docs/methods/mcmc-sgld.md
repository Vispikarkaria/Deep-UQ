# MCMC (SGLD)

`deepuq` provides SGLD-based posterior sampling via `SGLDOptimizer` and prediction utilities such as `collect_posterior_samples`.

## 1) Motivation

Deterministic training yields a point estimate of parameters. For uncertainty-aware prediction, we want samples from a posterior distribution over parameters. Stochastic Gradient Langevin Dynamics (SGLD) approximates this by combining stochastic gradients with Langevin noise.

## 2) What Uncertainty Is Quantified

SGLD quantifies **epistemic uncertainty** by sampling multiple plausible parameter settings from an approximate posterior trajectory.

Predictive distribution:

$$
p(y_\*\mid x_\*,\mathcal D)
\approx
\frac{1}{S}\sum_{s=1}^{S} p(y_\*\mid x_\*,\theta^{(s)})
$$

## 3) Mathematical Setup / Notation

Define posterior energy:

$$
U(\theta)= -\log p(\mathcal D\mid\theta)-\log p(\theta)
$$

Then:

$$
p(\theta\mid\mathcal D)\propto\exp\!\left(-U(\theta)\right)
$$

Continuous-time Langevin diffusion:

$$
d\theta_t=-\nabla_{\theta}U(\theta_t)\,dt+\sqrt{2}\,dW_t
$$

where $W_t$ is standard Brownian motion.

## 4) Core Method Equations

SGLD Euler-Maruyama discretization with stochastic gradient $\widehat\nabla U(\theta_t)$:

$$
\theta_{t+1}=
\theta_t
-\eta_t\widehat\nabla U(\theta_t)
+\sqrt{2\eta_t}\,\xi_t,
\qquad
\xi_t\sim\mathcal N(0,I)
$$

Posterior sampling uses:

- burn-in period before retaining samples,
- optional thinning to reduce autocorrelation,
- multiple retained states $\{\theta^{(s)}\}_{s=1}^{S}$ for prediction.

A useful efficiency diagnostic is effective sample size:

$$
\mathrm{ESS}\approx
\frac{S}{1+2\sum_{k=1}^{\infty}\rho_k}
$$

where $\rho_k$ is lag-$k$ autocorrelation.

## 5) Inference / Prediction Equations

Regression predictive mean and variance:

$$
\mu(x)=\frac{1}{S}\sum_{s=1}^{S} f(x;\theta^{(s)})
$$

$$
\sigma^2_{\mathrm{epi}}(x)=
\frac{1}{S}\sum_{s=1}^{S}\left(f(x;\theta^{(s)})-\mu(x)\right)^2
$$
Classification predictive probabilities:
$$
\bar p(y\mid x)=
\frac{1}{S}\sum_{s=1}^{S}
\mathrm{softmax}\!\left(z(x;\theta^{(s)})\right)
$$

## 6) Practical Implications

- Step-size schedule $\eta_t$ controls the bias-variance tradeoff of samples.
- Too-short burn-in yields biased uncertainty estimates.
- Strong sample autocorrelation reduces effective posterior sample quality.
- Compared with VI/Laplace, SGLD can represent richer posterior geometry but usually costs more wall-clock time.

## UQResult Field Mapping

`predict_with_samples_uq(...)` returns:

| Field | Regression | Classification (`apply_softmax=True`) |
|---|---|---|
| `mean` | Predictive mean | Mean class probabilities |
| `epistemic_var` | Variance across posterior samples | Probability variance across samples |
| `aleatoric_var` | `None` | `None` |
| `total_var` | Same as `epistemic_var` | Same as `epistemic_var` |
| `probs` | `None` | Mean class probabilities |
| `probs_var` | `None` | Probability variance |
| `metadata` | Method/sample/task info | Method/sample/task info |

## 7) References

1. Welling, M., & Teh, Y. W. (2011). *Bayesian Learning via Stochastic Gradient Langevin Dynamics*. ICML. [Paper](https://www.cs.utoronto.ca/~duvenaud/courses/csc2541/readings/welling-teh-2011.pdf)
2. Teh, Y. W., Thiery, A. H., & Vollmer, S. J. (2016). *Consistency and Fluctuations for Stochastic Gradient Langevin Dynamics*. Journal of Machine Learning Research, 17(7), 1-33. [JMLR](https://jmlr.org/papers/v17/14-136.html)
3. Vollmer, S. J., Zygalakis, K. C., & Teh, Y. W. (2016). *Exploration of the (Non-)Asymptotic Bias and Variance of Stochastic Gradient Langevin Dynamics*. Journal of Machine Learning Research, 17(159), 1-48. [JMLR](https://jmlr.org/papers/v17/15-334.html)
4. Ma, Y.-A., Chen, T., & Fox, E. B. (2015). *A Complete Recipe for Stochastic Gradient MCMC*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/2015/hash/9a4400501febb2a95e79248486a5f6d3-Abstract.html)

## Related docs

- [SGLD Tutorial Guide](../tutorials/sgld.md)
- [MCMC API](../api/methods/mcmc.md)
