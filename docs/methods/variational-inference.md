# Variational Inference

`deepuq` implements a practical family of mean-field variational inference (VI)
methods built around Bayes by Backprop. The package now covers plain
regression/classification, heteroscedastic regression, multi-output regression,
and scalable last-layer VI on deterministic feature extractors.

## 1) Motivation

Exact Bayesian inference for neural-network weights is generally intractable.
Variational inference replaces the true posterior with a tractable family and
optimizes that approximation with stochastic gradients.

This gives a practical path to uncertainty-aware deep learning while keeping
standard PyTorch training loops.

## 2) What uncertainty is quantified

VI in `deepuq` primarily quantifies **epistemic uncertainty** through a
posterior approximation over network weights.

Posterior predictive distribution:

$$
p(y_\*\mid x_\*,\mathcal D)=\int p(y_\*\mid x_\*,w)\,q_{\phi}(w)\,dw
$$

Monte Carlo approximation:

$$
p(y_\*\mid x_\*,\mathcal D)\approx\frac{1}{S}\sum_{s=1}^{S} p(y_\*\mid x_\*,w^{(s)}),
\qquad w^{(s)}\sim q_{\phi}(w)
$$

For heteroscedastic regression variants, the predictive distribution also
includes a learned data-noise term, so the returned uncertainty decomposes into:

$$
\sigma^2_{\mathrm{total}}(x)=\sigma^2_{\mathrm{epi}}(x)+\sigma^2_{\mathrm{alea}}(x)
$$

## 3) Mathematical setup / notation

Let

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^N
$$

with prior $p(w)$ and variational family $q_{\phi}(w)$.

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

## 4) Core method equations

Canonical ELBO (maximization form):

$$
\mathcal F(\phi)=\mathbb E_{q_{\phi}(w)}\left[\log p(\mathcal D\mid w)\right]-\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Equivalent minimization form used in training:

$$
\mathcal L_{\mathrm{ELBO}}(\phi)=\mathbb E_{q_{\phi}(w)}\left[-\log p(\mathcal D\mid w)\right]+\beta\,\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Mini-batch objective with $N_b$ optimizer steps per epoch:

$$
\widehat{\mathcal L}_{\mathrm{ELBO}}=\widehat{\mathcal L}_{\mathrm{NLL}}+\beta\,\frac{1}{N_b}\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w)\right)
$$

Relationship to posterior KL:

$$
\mathrm{KL}\!\left(q_{\phi}(w)\,\|\,p(w\mid\mathcal D)\right)=\mathcal L_{\mathrm{ELBO}}(\phi)+\log p(\mathcal D)
$$

## 5) Implemented VI variants in Deep-UQ

### Bayes by Backprop

The baseline mean-field Bayesian MLP predicts either a scalar/vector regression
output or classification logits.

Regression predictive moments:

$$
\mu(x)=\frac{1}{S}\sum_{s=1}^{S} f(x;w^{(s)})
$$

$$
\sigma^2_{\mathrm{epi}}(x)=\frac{1}{S}\sum_{s=1}^{S}\left(f(x;w^{(s)})-\mu(x)\right)^2
$$

Classification predictive probabilities:

$$
\bar p(y\mid x)=\frac{1}{S}\sum_{s=1}^{S}\mathrm{softmax}\!\left(z(x;w^{(s)})\right)
$$

### Heteroscedastic Bayes by Backprop

The heteroscedastic regressor predicts both a mean and an observation variance.
For one output dimension:

$$
\mathcal L_{\mathrm{NLL}}(x,y)=\frac{1}{2}\frac{(y-\mu_w(x))^2}{\sigma_w^2(x)}+\frac{1}{2}\log \sigma_w^2(x)
$$

The returned uncertainty decomposes into Monte Carlo variance across sampled
means plus the Monte Carlo average of the predicted observation variance.

### Multi-output Bayes by Backprop

The multi-output regressor predicts a vector-valued response
$\mu(x)\in\mathbb R^m$. The ELBO stays the same, but the likelihood is summed or
averaged across output dimensions.

### Heteroscedastic multi-output Bayes by Backprop

This combines the previous two ideas: a vector mean and a vector of predicted
noise variances. It is the most complete regression VI variant in the package,
covering both multi-output predictions and explicit aleatoric noise.

### Last-layer variational inference

For larger backbones, Deep-UQ supports VI only in the final linear head. Let a
deterministic feature extractor produce $h=\phi(x)$ and a Bayesian head predict:

$$
y=W h+b,\qquad q(W,b)
$$

This keeps the feature extractor deterministic and scales VI to CNN, operator,
and other feature-based architectures while retaining uncertainty in the final
mapping.

## 6) Practical implications

- Fixed $\beta$ is useful when comparing ELBO trends across epochs.
- Larger Monte Carlo sample counts reduce estimator variance but increase
  compute.
- Mean-field VI is scalable but cannot represent full posterior correlations.
- Heteroscedastic regression variants are the correct choice when the data-noise
  level itself changes with the input.
- Last-layer VI is the most practical VI route for spatial backbones and
  operator models.
- Monitoring NLL and KL separately helps diagnose underfitting versus excessive
  regularization.

## UQResult field mapping

`predict_vi_uq(...)` returns:

| Variant | Populated fields |
|---|---|
| Plain regression | `mean`, `epistemic_var`, `total_var`, `metadata` |
| Heteroscedastic regression | `mean`, `epistemic_var`, `aleatoric_var`, `total_var`, `metadata` |
| Multi-output regression | same fields as regression, with an extra output dimension |
| Classification | `mean`, `probs`, `probs_var`, `epistemic_var`, `metadata` |
| Last-layer VI | follows the configured head task (`regression` or `classification`) |

## Related tutorials

- [Bayes by Backprop Tutorial](../tutorials/bayes-by-backprop.md)
- [Heteroscedastic Bayes by Backprop + ADR1D](../tutorials/vi-heteroscedastic-bbb-adr1d.md)
- [Multi-Output Bayes by Backprop + Elastic Bar](../tutorials/vi-multioutput-bbb-elastic-bar.md)
- [Heteroscedastic Multi-Output Bayes by Backprop + Transport2D](../tutorials/vi-heteroscedastic-multioutput-bbb-transport2d.md)
- [Last-Layer VI + Heat2D Classification](../tutorials/vi-last-layer-heat2d-classification.md)
- [VI API](../api/methods/vi.md)

## 7) References

1. Graves, A. (2011). *Practical Variational Inference for Neural Networks*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)
2. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). *Weight Uncertainty in Neural Networks*. ICML (PMLR 37). [Proceedings](https://proceedings.mlr.press/v37/blundell15.html)
3. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [OpenReview](https://openreview.net/forum?id=33X9fd2-9FyZd)
4. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). *An Introduction to Variational Methods for Graphical Models*. Machine Learning, 37, 183-233. DOI: [10.1023/A:1007665907178](https://doi.org/10.1023/A:1007665907178)
5. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). *Variational Inference: A Review for Statisticians*. Journal of the American Statistical Association, 112(518), 859-877. DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
