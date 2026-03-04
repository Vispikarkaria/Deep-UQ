# Variational Inference (Bayes by Backprop)

`deepuq` implements Bayes by Backprop through variational layers and `vi_elbo_step`.

## 1) Motivation

Exact Bayesian inference for neural-network weights is generally intractable. Variational inference replaces the true posterior with a tractable family and turns inference into optimization.

This gives a practical path to uncertainty-aware deep learning while keeping stochastic-gradient training workflows.

## 2) What Uncertainty Is Quantified

Variational inference in `deepuq` quantifies **epistemic uncertainty** via a learned distribution over weights.

Posterior predictive distribution:

$$
p(y_\*\mid x_\*,\mathcal D)=\int p(y_\*\mid x_\*,w)\,q_{\phi}(w)\,dw
$$

Monte Carlo approximation:

$$
p(y_\*\mid x_\*,\mathcal D)
\approx
\frac{1}{S}\sum_{s=1}^{S} p(y_\*\mid x_\*,w^{(s)}),
\qquad
w^{(s)}\sim q_{\phi}(w)
$$

## 3) Mathematical Setup / Notation

Let $\mathcal D=\{(x_i,y_i)\}_{i=1}^N$, prior $p(w)$, and variational family $q_{\phi}(w)$.

Mean-field Gaussian parameterization:

$$
q_{\phi}(w)=\mathcal N\!\left(w;\mu,\mathrm{diag}(\sigma^2)\right)
$$

A common unconstrained scale parameterization is:

$$
\sigma=\log\!\left(1+e^{\rho}\right)
$$

Reparameterization trick:

$$
w=\mu+\sigma\odot\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I)
$$

## 4) Core Method Equations

Canonical ELBO (maximization form):

$$
\mathcal F(\phi)=
\mathbb E_{q_{\phi}(w)}\left[\log p(\mathcal D\mid w)\right]
-
\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Equivalent minimization form used in training:

$$
\mathcal L_{\mathrm{ELBO}}(\phi)=
\mathbb E_{q_{\phi}(w)}\left[-\log p(\mathcal D\mid w)\right]
+
\beta\,\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Mini-batch objective with $N_b$ optimizer steps per epoch:

$$
\widehat{\mathcal L}_{\mathrm{ELBO}}=
\widehat{\mathcal L}_{\mathrm{NLL}}
+
\beta\,\frac{1}{N_b}
\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Relationship to posterior KL:

$$
\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w\mid\mathcal D)\right)
=
\mathcal L_{\mathrm{ELBO}}(\phi)+\log p(\mathcal D)
$$

## 5) Inference / Prediction Equations

Regression predictive moments:

$$
\mu(x)=\frac{1}{S}\sum_{s=1}^{S} f(x;w^{(s)})
$$

$$
\sigma^2_{\mathrm{epi}}(x)=
\frac{1}{S}\sum_{s=1}^{S}\left(f(x;w^{(s)})-\mu(x)\right)^2
$$

Classification predictive probabilities:

$$
\bar p(y\mid x)=
\frac{1}{S}\sum_{s=1}^{S}
\mathrm{softmax}\!\left(z(x;w^{(s)})\right)
$$

## 6) Practical Implications

- Fixed $\beta$ is useful when comparing ELBO trends across epochs.
- Larger Monte Carlo sample counts reduce estimator variance but increase compute.
- Mean-field VI is scalable but cannot represent full posterior correlations.
- Monitoring NLL and KL separately helps diagnose underfitting vs over-regularization.

## UQResult Field Mapping

`predict_vi_uq(...)` returns:

| Field | Regression | Classification (`apply_softmax=True`) |
|---|---|---|
| `mean` | Predictive mean | Mean class probabilities |
| `epistemic_var` | MC variance across weight samples | Probability variance across samples |
| `aleatoric_var` | Optional user-supplied additive term | `None` |
| `total_var` | `epistemic_var + aleatoric_var` (if provided) | Probability variance |
| `probs` | `None` | Mean class probabilities |
| `probs_var` | `None` | Probability variance |
| `metadata` | Method/sample/task info | Method/sample/task info |

## 7) References

1. Graves, A. (2011). *Practical Variational Inference for Neural Networks*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)
2. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). *Weight Uncertainty in Neural Networks*. ICML (PMLR 37). [Proceedings](https://proceedings.mlr.press/v37/blundell15.html)
3. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [OpenReview](https://openreview.net/forum?id=33X9fd2-9FyZd)
4. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). *An Introduction to Variational Methods for Graphical Models*. Machine Learning, 37, 183-233. DOI: [10.1023/A:1007665907178](https://doi.org/10.1023/A:1007665907178)
5. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). *Variational Inference: A Review for Statisticians*. Journal of the American Statistical Association, 112(518), 859-877. DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)

## Related docs

- [Bayes by Backprop Tutorial](../tutorials/bayes-by-backprop.md)
- [VI API](../api/methods/vi.md)
