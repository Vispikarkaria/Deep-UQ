# Laplace Approximation for Deep UQ: A Technical Story

This page is written as a research-style narrative for the Laplace/LLA methods implemented in `deepuq`.
It explains the motivation, the uncertainty semantics, the mathematical foundations, and the exact implementation choices behind:

- `diag`
- `fisher_diag`
- `lowrank_diag`
- `block_diag`
- `kron`
- `full`

For API signatures, see the [Laplace API](../api/methods/laplace.md).  
For experiments, see the [Laplace comparison tutorial](../tutorials/laplace-comparison.md).

## Abstract

Modern neural networks often produce accurate point predictions while remaining overconfident far from training support.  
Laplace approximation offers a practical middle path between deterministic deep learning and full Bayesian inference: train a model to a MAP point, then locally approximate the posterior with a Gaussian around that optimum.

In `deepuq`, this idea is made operational through six curvature structures that trade off fidelity and scalability. The resulting predictive uncertainty is primarily **epistemic** (parameter uncertainty), with an additive residual-noise term for regression outputs.

## 1) Why This Method Exists

### The practical gap

A deterministic network outputs one weight vector \(\hat\theta\), so it cannot represent uncertainty over plausible parameter settings. In many scientific and engineering contexts, this is insufficient.

### Why UQ matters

Uncertainty quantification is essential when decisions depend on confidence, not only on expected value:

- sensor-driven control and digital twins
- model extrapolation outside measured regimes
- scientific inverse problems
- risk-sensitive optimization

### Why Laplace is attractive

Full posterior sampling is often expensive in deep nets. Laplace approximation is attractive because it reuses standard training pipelines:

1. train to MAP with familiar optimizers,
2. estimate local curvature,
3. sample from a Gaussian posterior approximation.

This preserves much of the engineering simplicity of standard deep learning while adding uncertainty-awareness.

## 2) What Uncertainty This Quantifies

In `deepuq` Laplace backends, uncertainty is interpreted as follows.

### Epistemic uncertainty

The main quantity captured is uncertainty induced by limited data under the model class:

$$
\theta \sim q(\theta \mid \mathcal D)
$$

and predictive spread comes from parameter draws \(\theta^{(s)}\).

### Aleatoric uncertainty

For regression only, `deepuq` adds an empirical residual noise estimate to predictive variance:

$$
\sigma^2_{\text{pred}}(x)
=
\sigma^2_{\text{epi}}(x) + \hat\sigma^2_{\text{noise}}
$$

This means returned regression variance is epistemic plus a global observation-noise floor estimated from fit residuals.

### Not captured

This approach does not provide a fully nonlocal posterior over parameters; it is a local Gaussian approximation around one mode.

## 3) Bayesian Setup and Laplace Derivation

Let \(\mathcal D = \{(x_i,y_i)\}_{i=1}^N\) and \(\theta \in \mathbb R^P\).

We start from Bayes' rule:

$$
p(\theta\mid\mathcal D)
\propto
p(\mathcal D\mid\theta)\,p(\theta)
$$

Assume an isotropic Gaussian prior:

$$
p(\theta)=\mathcal N(0,\lambda^{-1}I),\quad \lambda>0
$$

Define the MAP point:

$$
\theta^* = \arg\min_\theta \mathcal J(\theta),
\quad
\mathcal J(\theta) = -\log p(\mathcal D\mid\theta)-\log p(\theta)
$$

Laplace approximation uses a second-order Taylor expansion of \(\mathcal J\) at \(\theta^*\):

$$
\mathcal J(\theta)
\approx
\mathcal J(\theta^*)
+
\tfrac{1}{2}(\theta-\theta^*)^\top H(\theta^*)(\theta-\theta^*)
$$

which yields:

$$
q(\theta\mid\mathcal D)
=
\mathcal N\!\left(\theta^*,\,\Lambda^{-1}\right),
\quad
\Lambda\approx H(\theta^*)+\lambda I
$$

In deep learning, exact \(H\) is often replaced by positive-semidefinite surrogates such as empirical Fisher/GGN-like quantities.

## 4) From Theory to `deepuq`

`LaplaceWrapper` implements this story with structure-specific approximations to \(\Lambda\), balancing statistical fidelity and computation.

Common implementation elements across backends:

- curvature statistics from mini-batch gradients
- prior precision term \(\lambda I\)
- damping term \(\epsilon I\) for numerical stability
- posterior sampling for Monte Carlo prediction

For regression, `deepuq` uses batch gradients of \(\tfrac{1}{2}\sum(f_\theta(x)-y)^2\); for classification, summed cross-entropy gradients.

## 5) The Six LLA Backends as One Continuum

Think of the six methods as one continuum from cheapest to most expressive curvature.

## `diag`

### Canonical equation

$$
\Lambda_{\text{diag}} = \operatorname{diag}(H)+\lambda I
$$

### Implementation-faithful equation

With batch gradients \(g_b\):

$$
d = \frac{1}{N}\sum_b g_b\odot g_b,
\qquad
\Lambda_{\text{diag}} = d + \lambda\mathbf 1 + \epsilon\mathbf 1
$$

### Narrative interpretation

This is the minimal Bayesianization step: each parameter gets an independent variance, no cross-parameter coupling.

### Compute profile

- fit: \(O(BP)\)
- memory: \(O(P)\)

## `fisher_diag`

### Canonical equation

$$
\Lambda_{\text{fdiag}} = \operatorname{diag}(F_{\text{emp}})+\lambda I
$$

### Implementation-faithful equation

In `deepuq`, `fisher_diag` is an explicit semantic alias in the same estimator family as `diag`.

### Narrative interpretation

Use this option when you want explicit empirical-Fisher semantics without changing the computational behavior.

### Compute profile

Same as `diag`.

## `lowrank_diag`

### Canonical equation

$$
H \approx U_r\Sigma_rU_r^\top + \operatorname{diag}(r),
\quad
\Lambda \approx \lambda I + U_r\Sigma_rU_r^\top + \operatorname{diag}(r)
$$

### Implementation-faithful equation

Let \(\widetilde G = G/\sqrt{N}\) with SVD \(\widetilde G = USV^\top\):

$$
U_r\leftarrow V_{:,1:r},\qquad \Lambda_r\leftarrow S_{1:r}^2
$$

$$
d_{\text{total}}=\frac{1}{N}\sum_b g_b\odot g_b,
\quad
d_{\text{lr}}=(U_r\odot U_r)\Lambda_r,
\quad
d_{\text{res}}=\max(d_{\text{total}}-d_{\text{lr}},0)
$$

$$
D = \lambda I + \operatorname{diag}(d_{\text{res}})+\epsilon I
$$

Sampling uses a Woodbury-style transform combining \(D\) with low-rank factors.

### Narrative interpretation

This captures dominant global curvature directions while remaining scalable compared to full dense curvature.

### Compute profile

- fit: SVD-dominated, \(O(\min(B^2P, BP^2))\)
- memory: \(O(BP + Pr)\)

## `block_diag`

### Canonical equation

$$
\Lambda \approx \operatorname{blockdiag}(\Lambda_1,\dots,\Lambda_K),
\qquad
\Lambda_k \approx H_k + \lambda I_k
$$

### Implementation-faithful equation

For each block gradient \(g_{b,k}\):

$$
C_k = \frac{1}{N}\sum_b g_{b,k}g_{b,k}^\top,
\qquad
\Lambda_k = C_k + (\lambda+\epsilon)I_k
$$

Samples are drawn by block-wise triangular solves using Cholesky factors.

### Narrative interpretation

A middle ground: keep local parameter couplings within blocks, ignore couplings across blocks.

### Compute profile

- fit: \(O\!\left(B\sum_k p_k^2\right)\)
- memory: \(O\!\left(\sum_k p_k^2\right)\)

## `kron`

### Canonical equation

For layer \(l\):

$$
H_l \approx A_l \otimes G_l
$$

### Implementation-faithful equation

For selected `nn.Linear` layers, hooks capture activations \(a\) and output gradients \(g\). With bias augmentation \(\bar a=[a;1]\):

$$
A_l = \frac{1}{B}\sum_b \frac{\bar a_b^\top\bar a_b}{m_b},
\qquad
G_l = \frac{1}{B}\sum_b \frac{g_b^\top g_b}{m_b}
$$

Then eigendecompose:

$$
A_l=U_a\operatorname{diag}(s_a)U_a^\top,
\qquad
G_l=U_g\operatorname{diag}(s_g)U_g^\top
$$

Sampling uses Kronecker eigen-spectrum denominator:

$$
s_a\otimes s_g + (\lambda+\epsilon)
$$

### Narrative interpretation

This preserves layerwise structure and typically gives a much richer approximation than diagonal methods while remaining practical for larger models.

### Compute profile

- factor accumulation: \(O\!\left(B\sum_l(n_{in,l}'^2+n_{out,l}^2)\right)\)
- eigendecomposition: \(O\!\left(\sum_l(n_{in,l}'^3+n_{out,l}^3)\right)\)
- memory: \(O\!\left(\sum_l(n_{in,l}'^2+n_{out,l}^2)\right)\)

where \(n_{in,l}'=n_{in,l}+1\) when bias is used.

## `full`

### Canonical equation

$$
\Lambda_{\text{full}} = H + \lambda I
$$

### Implementation-faithful equation

With stacked gradient matrix \(G\in\mathbb R^{B\times P}\):

$$
C = \frac{1}{N}G^\top G,
\qquad
\Lambda_{\text{full}} = C + (\lambda+\epsilon)I
$$

Sampling uses Cholesky \(\Lambda_{\text{full}}=LL^\top\) and triangular solves:

$$
\theta^{(s)} = \theta^* + L^{-\top}\xi^{(s)},
\quad
\xi^{(s)}\sim\mathcal N(0,I)
$$

### Narrative interpretation

Highest local fidelity, highest cost. This is the reference approximation when parameter dimension is small enough.

### Compute profile

- fit: \(O(BP^2 + P^3)\)
- memory: \(O(P^2)\)

## 6) Predictive Story: From Weight Posterior to Output Uncertainty

Given posterior samples \(\{\theta^{(s)}\}_{s=1}^S\):

$$
\mu(x) = \frac{1}{S}\sum_s f(x;\theta^{(s)})
$$

$$
\sigma^2_{\text{epi}}(x)
=
\frac{1}{S}\sum_s\left(f(x;\theta^{(s)})-\mu(x)\right)^2
$$

For regression, `deepuq` returns:

$$
\sigma^2_{\text{pred}}(x) = \sigma^2_{\text{epi}}(x)+\hat\sigma^2_{\text{noise}}
$$

For classification, `deepuq` returns Monte Carlo-averaged class probabilities:

$$
\bar p(y\mid x) = \frac{1}{S}\sum_s \operatorname{softmax}(z^{(s)}(x))
$$

## 7) Practical Method Selection Guide

| Structure | Geometry captured | Cost level | Typical use |
|---|---|---|---|
| `diag` | parameter-wise only | very low | fast baseline UQ |
| `fisher_diag` | parameter-wise empirical Fisher family | very low | explicit Fisher-diagonal choice |
| `lowrank_diag` | dominant global directions + residual diagonal | medium | better geometry with bounded memory |
| `block_diag` | within-block coupling | medium | structured approximation without full dense cost |
| `kron` | layerwise Kronecker structure | medium-high | scalable structured deep-net curvature |
| `full` | full local coupling | high | small models / last-layer high-fidelity analysis |

## 8) Limitations and Failure Modes

- Laplace is local around one MAP basin; multimodal posteriors are not explicitly represented.
- Too-small damping can destabilize Cholesky-based sampling.
- `full` quickly becomes intractable; `deepuq` protects this with `full_max_params` for `subset_of_weights='all'`.
- `kron` requires compatible `nn.Linear` parameter grouping.
- Diagonal variants can underestimate correlations and thus understate uncertainty in coupled directions.

## 9) Scientific References

### Foundations of Laplace Approximation

1. MacKay, D. J. C. (1992). *A Practical Bayesian Framework for Backpropagation Networks*. Neural Computation, 4(3), 448–472. DOI: [10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
2. Tierney, L., & Kadane, J. B. (1986). *Accurate Approximations for Posterior Moments and Marginal Densities*. JASA, 81(393), 82–86. DOI: [10.1080/01621459.1986.10478240](https://doi.org/10.1080/01621459.1986.10478240)

### Fisher / Curvature Perspective

3. Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR, 21(146), 1–76. [JMLR](https://jmlr.org/papers/v21/17-678.html)
4. Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. NeurIPS 2019. [Proceedings](https://papers.nips.cc/paper/8669-limitations-of-the-empirical-fisher-approximation-for-natural-gradient-descent)

### Structured and Scalable Approximations

5. Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML 2015 (PMLR 37). [PMLR](https://proceedings.mlr.press/v37/martens15.html)
6. Botev, A., Ritter, H., & Barber, D. (2017). *Practical Gauss-Newton Optimisation for Deep Learning*. ICML 2017 (PMLR 70). [PMLR](https://proceedings.mlr.press/v70/botev17a.html)
7. Ritter, H., Botev, A., & Barber, D. (2018). *A Scalable Laplace Approximation for Neural Networks*. ICLR 2018. [Conference entry](https://iclr.cc/virtual/2018/poster/224)
8. Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). *Laplace Redux — Effortless Bayesian Deep Learning*. NeurIPS 2021. [Proceedings](https://papers.nips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)

### Low-Rank + Diagonal Posterior Context

9. Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). *A Simple Baseline for Bayesian Uncertainty in Deep Learning*. NeurIPS 2019. [Proceedings](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning)

## Related Project Documentation

- [Laplace API](../api/methods/laplace.md)
- [Laplace Hessian Comparison Tutorial](../tutorials/laplace-comparison.md)
