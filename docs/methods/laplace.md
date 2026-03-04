# Laplace Approximation (LLA) in `deepuq`

This page explains the six Laplace/LLA backends implemented in `LaplaceWrapper`:

- `diag`
- `fisher_diag`
- `lowrank_diag`
- `block_diag`
- `kron`
- `full`

It is written in stable plain-text equation format so it renders correctly on all clients.

## Why this method is useful

Neural networks usually output one prediction without confidence.
Laplace approximation adds uncertainty by building a Gaussian posterior around a MAP solution.

This helps when:

- training data are limited,
- decisions require confidence bounds,
- extrapolation / OOD behavior matters,
- model-risk awareness is important.

## What uncertainty it quantifies

`deepuq` Laplace mainly captures epistemic uncertainty (uncertainty in parameters).

For regression, returned variance includes:

```text
sigma_pred^2(x) = sigma_epi^2(x) + sigma_noise_hat^2
```

where:

- `sigma_epi^2`: spread from posterior weight samples
- `sigma_noise_hat^2`: residual noise estimate from fit

For classification, `deepuq` returns MC-averaged class probabilities.

## Notation

- Dataset: `D = {(x_i, y_i)}_{i=1}^N`
- Parameters: `theta in R^P`
- MAP point: `theta*`
- Prior precision: `lambda > 0`, prior `p(theta) = N(0, lambda^-1 I)`
- Damping: `epsilon > 0`
- Posterior precision approximation: `Lambda`

## Canonical Laplace formulation

```text
q(theta | D) = N(theta*, Lambda^-1)
Lambda ~ H(theta*) + lambda I
```

`H(theta*)` is a local curvature matrix (or PSD surrogate such as empirical Fisher / GGN family).

## How `deepuq` builds curvature

Across backends, `deepuq` uses batch gradients from:

- regression objective: `0.5 * sum((f_theta(x) - y)^2)`
- classification objective: summed cross-entropy

Then each backend builds a structured approximation of `Lambda`, always adding prior precision and damping.

## Backend-by-backend details

## `diag`

### Canonical equation

```text
Lambda_diag = diag(H) + lambda I
```

### Implementation-faithful equation

With batch gradients `g_b`:

```text
d = (1/N) * sum_b (g_b ⊙ g_b)
Lambda_diag = d + lambda*1 + epsilon*1
```

Sampling:

```text
theta^(s) = theta* + xi^(s) ⊙ Lambda_diag^(-1/2)
xi^(s) ~ N(0, I)
```

### Cost

- Fit: `O(BP)`
- Memory: `O(P)`

## `fisher_diag`

### Canonical equation

```text
Lambda_fdiag = diag(F_emp) + lambda I
```

### Implementation-faithful equation

In `deepuq`, `fisher_diag` is an explicit diagonal empirical-Fisher option in the same estimator family as `diag`.

### Cost

Same as `diag`.

## `lowrank_diag`

### Canonical equation

```text
H ~ U_r * Sigma_r * U_r^T + diag(r)
Lambda ~ lambda I + U_r * Sigma_r * U_r^T + diag(r)
```

### Implementation-faithful equation

With scaled gradient matrix `G_tilde = G / sqrt(N)` and SVD `G_tilde = U S V^T`:

```text
U_r <- V[:, 1:r]
Lambda_r <- S[1:r]^2

d_total = (1/N) * sum_b (g_b ⊙ g_b)
d_lr    = (U_r ⊙ U_r) * Lambda_r
d_res   = max(d_total - d_lr, 0)

D = lambda I + diag(d_res) + epsilon I
```

Sampling uses a Woodbury-style low-rank-plus-diagonal transform.

### Cost

- Fit: approx `O(min(B^2 P, B P^2))`
- Memory: `O(BP + Pr)`

## `block_diag`

### Canonical equation

```text
Lambda ~ blockdiag(Lambda_1, ..., Lambda_K)
Lambda_k ~ H_k + lambda I_k
```

### Implementation-faithful equation

For each block `k`:

```text
C_k = (1/N) * sum_b (g_{b,k} g_{b,k}^T)
Lambda_k = C_k + (lambda + epsilon) I_k
```

Sampling with Cholesky `Lambda_k = L_k L_k^T`:

```text
theta_k^(s) = theta_k* + L_k^{-T} xi_k^(s)
xi_k^(s) ~ N(0, I_k)
```

### Cost

- Fit: `O(B * sum_k p_k^2)`
- Memory: `O(sum_k p_k^2)`

## `kron`

### Canonical equation

```text
For layer l:
H_l ~ A_l ⊗ G_l
```

### Implementation-faithful equation

For selected `nn.Linear` layers, `deepuq` captures activations `a` and output gradients `g`:

```text
A_l = (1/B) * sum_b ((a_bar_b^T a_bar_b) / m_b)
G_l = (1/B) * sum_b ((g_b^T g_b) / m_b)
```

`a_bar` includes bias augmentation when bias exists.

Then:

```text
A_l = U_a diag(s_a) U_a^T
G_l = U_g diag(s_g) U_g^T
```

Sampling denominator:

```text
s_a ⊗ s_g + (lambda + epsilon)
```

### Cost

- Factor accumulation: `O(B * sum_l (n_in'_l^2 + n_out_l^2))`
- Eigendecomposition: `O(sum_l (n_in'_l^3 + n_out_l^3))`
- Memory: `O(sum_l (n_in'_l^2 + n_out_l^2))`

## `full`

### Canonical equation

```text
Lambda_full = H + lambda I
```

### Implementation-faithful equation

With stacked gradient matrix `G in R^(B x P)`:

```text
C = (1/N) * G^T G
Lambda_full = C + (lambda + epsilon) I
```

If `Lambda_full = L L^T`, sampling is:

```text
theta^(s) = theta* + L^{-T} xi^(s)
xi^(s) ~ N(0, I)
```

### Cost

- Fit: `O(BP^2 + P^3)`
- Memory: `O(P^2)`

## Predictive formulas used in `deepuq`

For posterior samples `{theta^(s)}_{s=1}^S`:

```text
mu(x) = (1/S) * sum_s f(x; theta^(s))
sigma_epi^2(x) = (1/S) * sum_s (f(x; theta^(s)) - mu(x))^2
```

Regression return:

```text
sigma_pred^2(x) = sigma_epi^2(x) + sigma_noise_hat^2
```

Classification return:

```text
p_bar(y|x) = (1/S) * sum_s softmax(z^(s)(x))
```

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
