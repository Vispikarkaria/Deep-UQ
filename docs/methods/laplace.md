# Laplace Approximation (LLA) in `deepuq`

This page documents all Laplace/LLA backends implemented by `LaplaceWrapper`:

- `diag`
- `fisher_diag`
- `lowrank_diag`
- `block_diag`
- `kron`
- `full`

The goal is practical: what problem this solves, what uncertainty it gives you, and exactly how each backend is formed and sampled.

## Why this method is useful

Neural networks usually return point predictions without reliable confidence.  
Laplace approximation adds a posterior over weights around a trained MAP model, so predictions include uncertainty.

This is useful when:

- data are sparse or noisy,
- predictions are used for decisions,
- you care about out-of-distribution behavior,
- you need confidence bounds, not only mean predictions.

## What uncertainty it quantifies

`deepuq` Laplace methods mainly quantify **epistemic uncertainty** (uncertainty in parameters due to finite data).

For regression, `deepuq` returns:

$$
\sigma^2_{\mathrm{pred}}(x)
=
\sigma^2_{\mathrm{epi}}(x) + \hat\sigma^2_{\mathrm{noise}}
$$

where:

- \(\sigma^2_{\mathrm{epi}}\): spread from posterior weight samples,
- \(\hat\sigma^2_{\mathrm{noise}}\): empirical residual-noise term estimated during fit.

For classification, `deepuq` returns Monte Carlo averaged class probabilities.

## Notation

- Dataset: \(\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N\)
- Parameters: \(\theta\in\mathbb{R}^P\)
- MAP point: \(\theta^*\)
- Prior precision: \(\lambda>0\) with \(p(\theta)=\mathcal{N}(0,\lambda^{-1}I)\)
- Damping: \(\epsilon>0\)
- Posterior precision approximation: \(\Lambda\)

## Canonical Laplace formulation

Posterior approximation around \(\theta^*\):

$$
q(\theta\mid\mathcal D)
=
\mathcal N\left(\theta^*,\Lambda^{-1}\right)
$$

with precision approximately:

$$
\Lambda \approx H(\theta^*) + \lambda I
$$

where \(H(\theta^*)\) is a Hessian-like local curvature matrix (or PSD surrogate such as empirical Fisher/GGN family).

## How `deepuq` builds curvature

Across backends, `deepuq` uses batch gradients from:

- regression objective: \(\frac{1}{2}\sum (f_\theta(x)-y)^2\),
- classification objective: summed cross-entropy.

Then each backend builds a different structured approximation of \(\Lambda\), always adding prior precision and damping.

## Backend-by-backend details

## `diag`

### Canonical equation

$$
\Lambda_{\mathrm{diag}} = \mathrm{diag}(H) + \lambda I
$$

### Implementation-faithful equation

With batch gradients \(g_b\):

$$
d = \frac{1}{N}\sum_b g_b\odot g_b
$$

$$
\Lambda_{\mathrm{diag}} = d + \lambda\mathbf{1} + \epsilon\mathbf{1}
$$

Sampling:

$$
\theta^{(s)} = \theta^* + \xi^{(s)}\odot\Lambda_{\mathrm{diag}}^{-1/2},
\quad
\xi^{(s)}\sim\mathcal N(0,I)
$$

### Cost

- Fit: \(O(BP)\)
- Memory: \(O(P)\)

## `fisher_diag`

### Canonical equation

$$
\Lambda_{\mathrm{fdiag}} = \mathrm{diag}(F_{\mathrm{emp}}) + \lambda I
$$

### Implementation-faithful equation

In `deepuq`, `fisher_diag` is an explicit diagonal empirical-Fisher option in the same estimator family as `diag`.

### Cost

Same as `diag`.

## `lowrank_diag`

### Canonical equation

$$
H \approx U_r\Sigma_rU_r^\top + \mathrm{diag}(r)
$$

$$
\Lambda \approx \lambda I + U_r\Sigma_rU_r^\top + \mathrm{diag}(r)
$$

### Implementation-faithful equation

With scaled gradient matrix \(\widetilde G = G/\sqrt{N}\) and SVD \(\widetilde G = USV^\top\):

$$
U_r\leftarrow V_{:,1:r},
\quad
\Lambda_r\leftarrow S_{1:r}^2
$$

$$
d_{\mathrm{total}} = \frac{1}{N}\sum_b g_b\odot g_b,
\quad
d_{\mathrm{lr}} = (U_r\odot U_r)\Lambda_r,
\quad
d_{\mathrm{res}} = \max(d_{\mathrm{total}} - d_{\mathrm{lr}},0)
$$

$$
D = \lambda I + \mathrm{diag}(d_{\mathrm{res}}) + \epsilon I
$$

Sampling uses a Woodbury-style low-rank-plus-diagonal transform.

### Cost

- Fit: approximately \(O(\min(B^2P, BP^2))\)
- Memory: \(O(BP + Pr)\)

## `block_diag`

### Canonical equation

$$
\Lambda \approx \mathrm{blockdiag}(\Lambda_1,\dots,\Lambda_K),
\quad
\Lambda_k \approx H_k + \lambda I_k
$$

### Implementation-faithful equation

For each block \(k\):

$$
C_k = \frac{1}{N}\sum_b g_{b,k}g_{b,k}^\top
$$

$$
\Lambda_k = C_k + (\lambda+\epsilon)I_k
$$

Sampling per block using Cholesky \(\Lambda_k=L_kL_k^\top\):

$$
\theta_k^{(s)} = \theta_k^* + L_k^{-\top}\xi_k^{(s)},
\quad
\xi_k^{(s)}\sim\mathcal N(0,I_k)
$$

### Cost

- Fit: \(O\!\left(B\sum_k p_k^2\right)\)
- Memory: \(O\!\left(\sum_k p_k^2\right)\)

## `kron`

### Canonical equation

For layer \(l\):

$$
H_l \approx A_l\otimes G_l
$$

### Implementation-faithful equation

For selected `nn.Linear` layers, `deepuq` captures activations \(a\) and output gradients \(g\):

$$
A_l = \frac{1}{B}\sum_b \frac{\bar a_b^\top\bar a_b}{m_b},
\quad
G_l = \frac{1}{B}\sum_b \frac{g_b^\top g_b}{m_b}
$$

where \(\bar a\) includes a bias-augmentation term when bias is present.

Then:

$$
A_l=U_a\,\mathrm{diag}(s_a)\,U_a^\top,
\quad
G_l=U_g\,\mathrm{diag}(s_g)\,U_g^\top
$$

Sampling uses denominator:

$$
s_a\otimes s_g + (\lambda+\epsilon)
$$

### Cost

- Factor accumulation: \(O\!\left(B\sum_l(n'_{\mathrm{in},l}{}^2+n_{\mathrm{out},l}^2)\right)\)
- Eigendecomposition: \(O\!\left(\sum_l(n'_{\mathrm{in},l}{}^3+n_{\mathrm{out},l}^3)\right)\)
- Memory: \(O\!\left(\sum_l(n'_{\mathrm{in},l}{}^2+n_{\mathrm{out},l}^2)\right)\)

## `full`

### Canonical equation

$$
\Lambda_{\mathrm{full}} = H + \lambda I
$$

### Implementation-faithful equation

With stacked gradient matrix \(G\in\mathbb R^{B\times P}\):

$$
C = \frac{1}{N}G^\top G
$$

$$
\Lambda_{\mathrm{full}} = C + (\lambda+\epsilon)I
$$

If \(\Lambda_{\mathrm{full}}=LL^\top\), sampling is:

$$
\theta^{(s)} = \theta^* + L^{-\top}\xi^{(s)},
\quad
\xi^{(s)}\sim\mathcal N(0,I)
$$

### Cost

- Fit: \(O(BP^2 + P^3)\)
- Memory: \(O(P^2)\)

## Predictive formulas used in `deepuq`

For posterior samples \(\{\theta^{(s)}\}_{s=1}^S\):

$$
\mu(x)=\frac{1}{S}\sum_s f(x;\theta^{(s)})
$$

$$
\sigma^2_{\mathrm{epi}}(x)=\frac{1}{S}\sum_s\left(f(x;\theta^{(s)})-\mu(x)\right)^2
$$

Regression return:

$$
\sigma^2_{\mathrm{pred}}(x)=\sigma^2_{\mathrm{epi}}(x)+\hat\sigma^2_{\mathrm{noise}}
$$

Classification return:

$$
\bar p(y\mid x)=\frac{1}{S}\sum_s \mathrm{softmax}(z^{(s)}(x))
$$

## Practical comparison table

| Structure | Curvature captured | Runtime/Memory | Typical usage |
|---|---|---|---|
| `diag` | parameter-wise only | lowest | fast baseline UQ |
| `fisher_diag` | parameter-wise empirical Fisher family | lowest | explicit Fisher-diagonal choice |
| `lowrank_diag` | dominant directions + residual diagonal | medium | better geometry with bounded memory |
| `block_diag` | within-block coupling | medium | richer than diagonal, cheaper than full |
| `kron` | layerwise Kronecker structure | medium-high | scalable structured curvature |
| `full` | full local coupling | highest | small models / last-layer high fidelity |

## Stability and guardrails in this package

- `damping` is added before inversion/Cholesky.
- `full_max_params` guards expensive `full` + `subset_of_weights='all'` settings.
- `kron` checks that selected parameters match selected `nn.Linear` groups.

## References

1. MacKay, D. J. C. (1992). *A Practical Bayesian Framework for Backpropagation Networks*. Neural Computation, 4(3), 448–472. DOI: [10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
2. Tierney, L., & Kadane, J. B. (1986). *Accurate Approximations for Posterior Moments and Marginal Densities*. JASA, 81(393), 82–86. DOI: [10.1080/01621459.1986.10478240](https://doi.org/10.1080/01621459.1986.10478240)
3. Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR, 21(146), 1–76. [JMLR](https://jmlr.org/papers/v21/17-678.html)
4. Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. NeurIPS 2019. [Proceedings](https://papers.nips.cc/paper/8669-limitations-of-the-empirical-fisher-approximation-for-natural-gradient-descent)
5. Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML 2015 (PMLR 37). [PMLR](https://proceedings.mlr.press/v37/martens15.html)
6. Botev, A., Ritter, H., & Barber, D. (2017). *Practical Gauss-Newton Optimisation for Deep Learning*. ICML 2017 (PMLR 70). [PMLR](https://proceedings.mlr.press/v70/botev17a.html)
7. Ritter, H., Botev, A., & Barber, D. (2018). *A Scalable Laplace Approximation for Neural Networks*. ICLR 2018. [Conference entry](https://iclr.cc/virtual/2018/poster/224)
8. Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). *Laplace Redux — Effortless Bayesian Deep Learning*. NeurIPS 2021. [Proceedings](https://papers.nips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)
9. Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). *A Simple Baseline for Bayesian Uncertainty in Deep Learning*. NeurIPS 2019. [Proceedings](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning)

## Related docs

- [Laplace API](../api/methods/laplace.md)
- [Laplace Hessian Comparison Tutorial](../tutorials/laplace-comparison.md)
