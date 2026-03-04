# Laplace Approximation (LLA) in `deepuq`

This page explains all six Laplace/LLA backends implemented in `LaplaceWrapper`:

- `diag`
- `fisher_diag`
- `lowrank_diag`
- `block_diag`
- `kron`
- `full`

The style here is practical: motivation, uncertainty meaning, and equations that match the implementation.

## Why this method is useful

A deterministic neural network gives a single prediction, but not a reliable confidence estimate.
Laplace approximation adds uncertainty by building a Gaussian posterior around the MAP weights.

This is useful when:

- data are limited,
- predictions affect decisions,
- OOD behavior matters,
- confidence bounds are required.

## What uncertainty it quantifies

For Laplace in `deepuq`, the dominant uncertainty is epistemic (parameter uncertainty).

Regression variance returned by the package:

$$
sigma_pred^2(x) = sigma_epi^2(x) + sigma_noise_hat^2
$$

where:

- `sigma_epi^2`: uncertainty from posterior weight samples
- `sigma_noise_hat^2`: residual noise estimate from training data

For classification, `deepuq` returns Monte Carlo averaged class probabilities.

## Notation

- `D = {(x_i, y_i)}_{i=1}^N`
- `theta in R^P`
- `theta_star`: MAP point
- `lambda > 0`: prior precision
- `epsilon > 0`: damping
- `Lambda`: posterior precision approximation

## Canonical Laplace formulation

Posterior approximation around MAP:

$$
q(theta | D) = N(theta_star, Lambda^{-1})
$$

Local precision model:

$$
Lambda approx H(theta_star) + lambda I
$$

`H(theta_star)` is a local curvature matrix (or PSD surrogate such as empirical Fisher / GGN family).

## How `deepuq` builds curvature

Across backends, `deepuq` uses batch gradients from:

- regression loss: `0.5 * sum((f_theta(x) - y)^2)`
- classification loss: summed cross-entropy

Then each backend builds a structured approximation of `Lambda`, and adds prior + damping.

## Backend-by-backend details

## `diag`

### Canonical equation

$$
Lambda_diag = diag(H) + lambda I
$$

### Implementation-faithful equation

With batch gradients `g_b`:

$$
d = (1/N) * sum_b (g_b elementwise_square)
$$

$$
Lambda_diag = d + lambda*1 + epsilon*1
$$

Sampling:

$$
theta^(s) = theta_star + xi^(s) * Lambda_diag^(-1/2)
$$

## `fisher_diag`

### Canonical equation

$$
Lambda_fdiag = diag(F_emp) + lambda I
$$

### Implementation-faithful equation

In `deepuq`, this is an explicit empirical-Fisher diagonal option in the same estimator family as `diag`.

## `lowrank_diag`

### Canonical equation

$$
H approx U_r * Sigma_r * U_r^T + diag(r)
$$

$$
Lambda approx lambda I + U_r * Sigma_r * U_r^T + diag(r)
$$

### Implementation-faithful equation

Using `G_tilde = G / sqrt(N)` and SVD `G_tilde = U S V^T`:

$$
U_r = V[:, 1:r],   Lambda_r = S[1:r]^2
$$

$$
d_total = (1/N) * sum_b (g_b elementwise_square)
$$

$$
d_lr = (U_r elementwise_square) * Lambda_r
$$

$$
d_res = max(d_total - d_lr, 0)
$$

$$
D = lambda I + diag(d_res) + epsilon I
$$

Sampling uses a Woodbury-style low-rank + diagonal transform.

## `block_diag`

### Canonical equation

$$
Lambda approx blockdiag(Lambda_1, ..., Lambda_K)
$$

$$
Lambda_k approx H_k + lambda I_k
$$

### Implementation-faithful equation

For each block `k`:

$$
C_k = (1/N) * sum_b (g_{b,k} * g_{b,k}^T)
$$

$$
Lambda_k = C_k + (lambda + epsilon) I_k
$$

Sampling uses Cholesky per block:

$$
if   Lambda_k = L_k L_k^T,
then theta_k^(s) = theta_k_star + L_k^{-T} xi_k^(s)
$$

## `kron`

### Canonical equation

For layer `l`:

$$
H_l approx A_l kron G_l
$$

### Implementation-faithful equation

For selected `nn.Linear` layers:

$$
A_l = (1/B) * sum_b ((a_bar_b^T a_bar_b)/m_b)
$$

$$
G_l = (1/B) * sum_b ((g_b^T g_b)/m_b)
$$

Then eigendecompose:

$$
A_l = U_a diag(s_a) U_a^T,
G_l = U_g diag(s_g) U_g^T
$$

Sampling denominator:

$$
(s_a kron s_g) + (lambda + epsilon)
$$

## `full`

### Canonical equation

$$
Lambda_full = H + lambda I
$$

### Implementation-faithful equation

With stacked gradient matrix `G in R^(B x P)`:

$$
C = (1/N) * G^T G
$$

$$
Lambda_full = C + (lambda + epsilon) I
$$

Sampling:

$$
if   Lambda_full = L L^T,
then theta^(s) = theta_star + L^{-T} xi^(s)
$$

## Predictive equations used in `deepuq`

For posterior samples `{theta^(s)}_{s=1}^S`:

$$
mu(x) = (1/S) * sum_s f(x; theta^(s))
$$

$$
sigma_epi^2(x) = (1/S) * sum_s (f(x; theta^(s)) - mu(x))^2
$$

Regression return:

$$
sigma_pred^2(x) = sigma_epi^2(x) + sigma_noise_hat^2
$$

Classification return:

$$
p_bar(y|x) = (1/S) * sum_s softmax(z^(s)(x))
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
- `full_max_params` guards expensive `full` + `subset_of_weights='all'`.
- `kron` checks selected parameters match selected `nn.Linear` groups.

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
