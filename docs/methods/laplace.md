# Laplace Approximation (LLA Backends in `deepuq`)

This page documents the scientific details of all Laplace/LLA backends implemented in `LaplaceWrapper`:

- `diag`
- `fisher_diag`
- `lowrank_diag`
- `block_diag`
- `kron`
- `full`

The goal is to show both:

- the **canonical** equations used in the literature, and
- the **implementation-faithful** equations used in `src/deepuq/methods/laplace.py`.

For API details, see the [Laplace API](../api/methods/laplace.md).  
For experiments, see the [Laplace comparison tutorial](../tutorials/laplace-comparison.md).

## Notation

- Dataset: \(\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N\)
- Parameters: \(\theta \in \mathbb{R}^P\)
- MAP solution: \(\theta^\*\)
- Negative log-posterior objective:
  \[
  \mathcal{J}(\theta)
  =
  - \log p(\mathcal{D}\mid\theta)
  - \log p(\theta)
  \]
- Prior precision (isotropic Gaussian prior): \(\lambda > 0\), so \(p(\theta)=\mathcal{N}(0,\lambda^{-1}I)\)
- Damping/jitter used for numerical stability: \(\epsilon > 0\)
- Posterior precision approximation: \(\Lambda\)

## Laplace Approximation: Canonical Form

The posterior is locally approximated around \(\theta^\*\) by a Gaussian:

\[
p(\theta \mid \mathcal{D})
\approx
q(\theta)
=
\mathcal{N}(\theta^\*, \Lambda^{-1})
\]

with

\[
\Lambda
\approx
H(\theta^\*) + \lambda I
\]

where \(H(\theta^\*)\) is typically the Hessian (or a positive-semidefinite surrogate such as GGN/Fisher).

In practice, `deepuq` uses structure-specific approximations to \(\Lambda\) for scalability.

## Likelihood Details Used in `deepuq`

The implementation computes batch gradients of:

- Regression: \( \frac{1}{2}\sum (f_\theta(x)-y)^2 \)
- Classification: summed cross-entropy over the batch

This yields empirical curvature surrogates built from gradient outer products or squares.

## Backend-by-Backend Scientific Details

## `diag`

### Canonical

Diagonal Laplace keeps only per-parameter curvature:

\[
\Lambda_{\text{diag}}
=
\mathrm{diag}(H) + \lambda I
\]

Common PSD substitute:

\[
\mathrm{diag}(H)
\approx
\mathrm{diag}(F)
\]

where \(F\) is the Fisher/GGN-style curvature.

### Implementation in `deepuq`

Let \(g_b = \nabla_\theta \ell_b(\theta^\*)\) for each batch \(b\). The code accumulates:

\[
d
=
\frac{1}{N}
\sum_b g_b \odot g_b
\]

then constructs:

\[
\Lambda_{\text{diag}}
=
d + \lambda \mathbf{1} + \epsilon \mathbf{1}
\]

and samples each parameter independently:

\[
\theta^{(s)}
=
\theta^\* + \xi^{(s)} \odot \Lambda_{\text{diag}}^{-1/2},
\quad
\xi^{(s)}\sim\mathcal{N}(0,I)
\]

### Complexity

- Fit time: \(O(BP)\)
- Memory: \(O(P)\)

### Safeguards

- Positive clamps on precision terms
- Damping \(\epsilon\) added to avoid near-singular precision

## `fisher_diag`

### Canonical

Diagonal empirical Fisher variant:

\[
\Lambda_{\text{fdiag}}
=
\mathrm{diag}(F_{\text{emp}}) + \lambda I
\]

### Implementation in `deepuq`

`fisher_diag` is an explicit alias to the same estimator family as `diag` in this package.  
It uses the same batch gradient-square accumulation and diagonal Gaussian sampling pipeline.

### Complexity and safeguards

Same as `diag`.

## `lowrank_diag`

### Canonical

Low-rank + diagonal decomposition:

\[
H \approx U_r \Sigma_r U_r^\top + \mathrm{diag}(r)
\]

\[
\Lambda
\approx
\lambda I + U_r \Sigma_r U_r^\top + \mathrm{diag}(r)
\]

This preserves dominant curvature directions while keeping memory manageable.

### Implementation in `deepuq`

1. Build gradient matrix \(G \in \mathbb{R}^{B\times P}\), scaled as \(\widetilde{G}=G/\sqrt{N}\).
2. Compute SVD: \(\widetilde{G}=U S V^\top\).
3. Keep top rank \(r\) singular values and set \(U_r \leftarrow V_{:,1:r}\), \(\Lambda_r \leftarrow S_{1:r}^2\).
4. Diagonal total curvature from gradient squares: \(d_{\text{total}} = \frac{1}{N}\sum_b g_b\odot g_b\).
5. Low-rank diagonal contribution: \(d_{\text{lr}} = (U_r \odot U_r)\Lambda_r\).
6. Residual diagonal (clipped): \(d_{\text{res}} = \max(d_{\text{total}} - d_{\text{lr}}, 0)\).
7. Stored diagonal precision part: \(D = \lambda I + \mathrm{diag}(d_{\text{res}}) + \epsilon I\).

Sampling uses a Woodbury-style transform with \(D\) and low-rank factors.

### Complexity

- Fit time: SVD-dominated, approximately \(O(\min(B^2P,\;BP^2))\)
- Memory: \(O(BP + Pr)\)

### Safeguards

- Rank clipping to feasible rank
- Degenerate-rank fallback to diagonal-like behavior
- Residual diagonal clamped non-negative

## `block_diag`

### Canonical

Partition parameters into blocks \(\{\theta_k\}\) and assume:

\[
\Lambda
\approx
\mathrm{blockdiag}(\Lambda_1,\dots,\Lambda_K)
\]

with

\[
\Lambda_k \approx H_k + \lambda I_k
\]

### Implementation in `deepuq`

Blocks are chosen as:

- `last_layer`: one block containing all selected last-layer parameters
- `all`: one block per selected parameter tensor

For each block \(k\), accumulate batch outer products:

\[
C_k = \frac{1}{N}\sum_b g_{b,k}g_{b,k}^\top
\]

then:

\[
\Lambda_k = C_k + (\lambda+\epsilon)I_k
\]

Cholesky factors \(L_k\) of \(\Lambda_k\) are used for block-wise sampling:

\[
\theta_k^{(s)} = \theta_k^\* + L_k^{-\top}\xi_k^{(s)},\quad \xi_k^{(s)}\sim\mathcal{N}(0,I_k)
\]

### Complexity

- Fit time: \(O\!\left(B\sum_k p_k^2\right)\)
- Memory: \(O\!\left(\sum_k p_k^2\right)\)

### Safeguards

- Stable Cholesky with escalating jitter via `_safe_cholesky`

## `kron`

### Canonical

For layer \(l\), approximate curvature as Kronecker-factored:

\[
H_l \approx A_l \otimes G_l
\]

where:

- \(A_l\): input/activation covariance factor
- \(G_l\): output-gradient covariance factor

### Implementation in `deepuq`

For selected `nn.Linear` layers only:

1. Capture layer inputs \(a\) and output gradients \(g\) via hooks.
2. If bias exists, augment input with ones: \(\bar{a}=[a;1]\).
3. Per-layer factor estimates: \(A_l = \frac{1}{B}\sum_b \frac{\bar{a}_b^\top \bar{a}_b}{m_b}\), \(G_l = \frac{1}{B}\sum_b \frac{g_b^\top g_b}{m_b}\).
4. Add damping and eigendecompose: \(A_l = U_a \operatorname{diag}(s_a) U_a^\top\), \(G_l = U_g \operatorname{diag}(s_g) U_g^\top\).
5. Use Kronecker eigenbasis sampling denominator \(s_a \otimes s_g + (\lambda + \epsilon)\).

This yields layer-wise matrix-normal style weight perturbations.

### Complexity

- Factor accumulation: \(O\!\left(B\sum_l(n_{in,l}'^2+n_{out,l}^2)\right)\)
- Eigendecomposition: \(O\!\left(\sum_l(n_{in,l}'^3+n_{out,l}^3)\right)\)
- Memory: \(O\!\left(\sum_l(n_{in,l}'^2+n_{out,l}^2)\right)\)

where \(n_{in,l}'=n_{in,l}+1\) when bias is included.

### Safeguards / constraints

- Requires selected parameters to match selected `nn.Linear` layers exactly
- Raises informative errors if layer-parameter alignment fails
- Falls back recommendation: use `block_diag` when constraints are violated

## `full`

### Canonical

Dense precision:

\[
\Lambda_{\text{full}}
=
H + \lambda I
\]

### Implementation in `deepuq`

With stacked batch gradients \(G \in \mathbb{R}^{B\times P}\):

\[
C = \frac{1}{N}G^\top G
\]

\[
\Lambda_{\text{full}}
=
C + (\lambda+\epsilon)I
\]

Sampling uses Cholesky:

\[
\Lambda_{\text{full}} = LL^\top,\quad
\theta^{(s)}=\theta^\* + L^{-\top}\xi^{(s)},\;\xi^{(s)}\sim\mathcal{N}(0,I)
\]

### Complexity

- Fit time: \(O(BP^2 + P^3)\)
- Memory: \(O(P^2)\)

### Safeguards

- `full_max_params` guard in `LaplaceWrapper.fit()` for `subset_of_weights='all'`
- `_safe_cholesky` jitter escalation

## Predictive Uncertainty in `deepuq`

## Regression

From posterior samples \(\{\theta^{(s)}\}_{s=1}^S\):

\[
\mu(x) = \frac{1}{S}\sum_s f(x;\theta^{(s)})
\]

\[
\sigma_{\text{epi}}^2(x)
=
\frac{1}{S}\sum_s \left(f(x;\theta^{(s)})-\mu(x)\right)^2
\]

The returned variance is:

\[
\sigma_{\text{pred}}^2(x)
=
\sigma_{\text{epi}}^2(x) + \hat{\sigma}_{\text{noise}}^2
\]

where \(\hat{\sigma}_{\text{noise}}^2\) is estimated during fit from MAP residuals.

## Classification

For sampled logits \(z^{(s)}(x)\):

\[
\bar{p}(y\mid x)
=
\frac{1}{S}\sum_s \operatorname{softmax}(z^{(s)}(x))
\]

`deepuq` returns `(mean_probs, None)` for classification.

## Practical Comparison Matrix

| Structure | Curvature form | Time / Memory profile | Typical use |
|---|---|---|---|
| `diag` | Diagonal only | Fastest, \(O(P)\) memory | Baseline uncertainty with minimal overhead |
| `fisher_diag` | Diagonal empirical Fisher family | Same as `diag` | Explicit Fisher-diagonal semantic choice |
| `lowrank_diag` | Low-rank + diagonal residual | Medium-to-high SVD cost | Capture dominant directions beyond diagonal |
| `block_diag` | Independent dense blocks | Medium, block-size dependent | Better local coupling without full \(P\times P\) |
| `kron` | Layerwise Kronecker factors | Efficient for large linear layers | Scalable structured curvature with richer geometry |
| `full` | Dense \(P\times P\) precision | Most expensive | Small models or last-layer-only high fidelity |

## Failure Modes and Stability Notes

- Very small damping can destabilize Cholesky; keep `damping > 0` in hard cases.
- `full` on large `all`-parameter models is intractable; use `last_layer`, `kron`, or `block_diag`.
- `kron` is constrained to compatible `nn.Linear` parameter groupings.
- `fisher_diag`/`diag` can under-represent cross-parameter curvature.
- Low-rank methods depend on useful gradient subspace rank.

## Scientific References

## Foundations of Laplace Approximation

1. MacKay, D. J. C. (1992). *A Practical Bayesian Framework for Backpropagation Networks*. Neural Computation, 4(3), 448–472. DOI: [10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
2. Tierney, L., & Kadane, J. B. (1986). *Accurate Approximations for Posterior Moments and Marginal Densities*. JASA, 81(393), 82–86. DOI: [10.1080/01621459.1986.10478240](https://doi.org/10.1080/01621459.1986.10478240)

## Fisher / Curvature Perspective

3. Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. JMLR, 21(146), 1–76. [JMLR link](https://jmlr.org/papers/v21/17-678.html)
4. Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. NeurIPS 2019. [Proceedings link](https://papers.nips.cc/paper/8669-limitations-of-the-empirical-fisher-approximation-for-natural-gradient-descent)

## Structured and Scalable Approximations

5. Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML 2015 (PMLR 37). [PMLR link](https://proceedings.mlr.press/v37/martens15.html)
6. Botev, A., Ritter, H., & Barber, D. (2017). *Practical Gauss-Newton Optimisation for Deep Learning*. ICML 2017 (PMLR 70). [PMLR link](https://proceedings.mlr.press/v70/botev17a.html)
7. Ritter, H., Botev, A., & Barber, D. (2018). *A Scalable Laplace Approximation for Neural Networks*. ICLR 2018. [ICLR link](https://iclr.cc/virtual/2018/poster/224)
8. Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). *Laplace Redux — Effortless Bayesian Deep Learning*. NeurIPS 2021. [Proceedings link](https://papers.nips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)

## Low-rank + Diagonal Posterior Context

9. Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). *A Simple Baseline for Bayesian Uncertainty in Deep Learning*. NeurIPS 2019. [Proceedings link](https://papers.nips.cc/paper/9472-a-simple-baseline-for-bayesian-uncertainty-in-deep-learning)

## Related Project Documentation

- [Laplace API](../api/methods/laplace.md)
- [Laplace Hessian Comparison Tutorial](../tutorials/laplace-comparison.md)
