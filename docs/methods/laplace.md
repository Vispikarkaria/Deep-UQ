# Laplace Approximation

`deepuq` exposes Laplace methods through `LaplaceWrapper` with six Hessian structures:
`diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, and `full`.

## 1) Motivation

Modern neural networks are often accurate in-domain but can be confidently wrong away from training support. Laplace approximation adds a Bayesian posterior layer on top of a trained MAP network, so predictions include both central tendency and confidence.

The practical idea is simple: optimize once for a MAP point, then approximate the local posterior geometry around that point.

## 2) What Uncertainty Is Quantified

The method primarily quantifies **epistemic uncertainty** via posterior weight uncertainty.

With parameter samples $\{\theta^{(s)}\}_{s=1}^S$:

$$ \mu(x)=\frac{1}{S}\sum_{s=1}^S f(x;\theta^{(s)}) $$

$$ \sigma^2_{\mathrm{epi}}(x)= \frac{1}{S}\sum_{s=1}^S\left(f(x;\theta^{(s)})-\mu(x)\right)^2 $$

For regression, the predictive variance returned by `deepuq` is:

$$ \sigma^2_{\mathrm{pred}}(x)=\sigma^2_{\mathrm{epi}}(x)+\hat\sigma^2_{\varepsilon} $$

where $\hat\sigma^2_{\varepsilon}$ is an empirical residual-noise estimate.

For classification, predictive probabilities are Monte Carlo averaged:

$$ \bar p(y\mid x)= \frac{1}{S}\sum_{s=1}^S p(y\mid x,\theta^{(s)}) $$

## 3) Mathematical Setup / Notation

Dataset and parameters:

$$ \mathcal D=\{(x_i,y_i)\}_{i=1}^N, \qquad \theta\in\mathbb R^P $$

MAP estimator:

$$ \theta^*=\arg\min_{\theta}\,\mathcal J(\theta), \qquad \mathcal J(\theta)=-\log p(\mathcal D\mid\theta)-\log p(\theta) $$

With isotropic Gaussian prior:

$$ p(\theta)=\mathcal N(0,\lambda^{-1}I),\qquad \lambda>0 $$

Canonical Laplace posterior:

$$ q(\theta\mid\mathcal D)=\mathcal N\!\left(\theta^*,\Lambda^{-1}\right) $$

$$ \Lambda\approx H(\theta^*)+\lambda I+\epsilon I $$

where $H(\theta^*)$ is a local curvature surrogate and $\epsilon>0$ is damping.

## 4) Core Method Equations

### 4.1 Diagonal (`diag`)

$$ \Lambda_{\mathrm{diag}}=\mathrm{diag}(H)+\lambda I+\epsilon I $$

Using empirical batch gradients $g_b=\nabla_{\theta}\ell_b(\theta^*)$:

$$ d=\frac{1}{N}\sum_b g_b\odot g_b, \qquad \Lambda_{\mathrm{diag}}=\mathrm{diag}(d)+(\lambda+\epsilon)I $$

### 4.2 Empirical Fisher Diagonal (`fisher_diag`)

$$ \Lambda_{\mathrm{fdiag}}=\mathrm{diag}(F_{\mathrm{emp}})+(\lambda+\epsilon)I $$

with

$$ F_{\mathrm{emp}}\approx\frac{1}{N}\sum_b g_b g_b^{\top} $$

and only the diagonal retained.

### 4.3 Low-Rank + Diagonal (`lowrank_diag`)

Curvature decomposition:

$$ H\approx U_r\Sigma_r U_r^{\top}+D_r $$

Posterior precision:

$$ \Lambda\approx U_r\Sigma_r U_r^{\top}+D_r+(\lambda+\epsilon)I $$

If $\widetilde G=G/\sqrt N$ with SVD $\widetilde G=USV^{\top}$, then

$$ U_r=V_{:,1:r}, \qquad \Sigma_r=\mathrm{diag}(S_{1:r}^2) $$

and a diagonal residual form is

$$ D_r=\mathrm{diag}\!\left(\max\left(d_{\mathrm{tot}}-d_{\mathrm{lr}},0\right)\right) $$

### 4.4 Block Diagonal (`block_diag`)

Partition parameters into $K$ blocks:

$$ \Lambda\approx\mathrm{blkdiag}(\Lambda_1,\ldots,\Lambda_K) $$

Block curvature and precision:

$$ C_k=\frac{1}{N}\sum_b g_{b,k}g_{b,k}^{\top}, \qquad \Lambda_k=C_k+(\lambda+\epsilon)I_k $$

### 4.5 Kronecker-Factored (`kron`)

For linear layer $\ell$:

$$ H_{\ell}\approx A_{\ell}\otimes G_{\ell} $$

with activation and output-gradient factors:

$$ A_{\ell}=\mathbb E\left[a_{\ell}a_{\ell}^{\top}\right], \qquad G_{\ell}=\mathbb E\left[g_{\ell}g_{\ell}^{\top}\right] $$

A standard eigenbasis view is

$$ A_{\ell}=U_A S_A U_A^{\top}, \qquad G_{\ell}=U_G S_G U_G^{\top} $$

so the layer precision spectrum is approximated by

$$ S_A\otimes S_G+(\lambda+\epsilon)I $$

### 4.6 Full (`full`)

$$ \Lambda_{\mathrm{full}}=H+(\lambda+\epsilon)I $$

With stacked gradients $G\in\mathbb R^{B\times P}$:

$$ C=\frac{1}{N}G^{\top}G, \qquad \Lambda_{\mathrm{full}}=C+(\lambda+\epsilon)I $$

## 5) Inference / Prediction Equations

Given $\theta\sim q(\theta\mid\mathcal D)$, Monte Carlo prediction uses:

$$ \mu(x)\approx\frac{1}{S}\sum_{s=1}^S f(x;\theta^{(s)}) $$

$$ \mathrm{Var}[f(x)]\approx \frac{1}{S}\sum_{s=1}^S\left(f(x;\theta^{(s)})-\mu(x)\right)^2 $$

Regression total predictive variance:

$$ \sigma^2_{\mathrm{pred}}(x)=\mathrm{Var}[f(x)]+\hat\sigma^2_{\varepsilon} $$

Classification predictive probability:

$$ \bar p(y\mid x)\approx \frac{1}{S}\sum_{s=1}^S \mathrm{softmax}\!\left(z(x;\theta^{(s)})\right) $$

## 6) Practical Implications

Curvature expressivity increases from `diag` to `full`, and cost rises accordingly.

- `diag` / `fisher_diag`: memory $\mathcal O(P)$, cheapest, weakest coupling.
- `lowrank_diag`: memory $\mathcal O(Pr)$, captures dominant directions.
- `block_diag`: memory $\mathcal O(\sum_k m_k^2)$, captures within-block coupling.
- `kron`: layerwise factorized coupling with favorable scaling for linear layers.
- `full`: memory $\mathcal O(P^2)$, highest fidelity and highest cost.

Numerical and safety controls in `deepuq` include:

- damping $\epsilon$ before inversion/factorization,
- parameter-count guard for expensive full-structure settings,
- structure checks for Kronecker-factorized assumptions.

## UQResult Field Mapping

`LaplaceWrapper.predict_uq(...)` returns:

| Field | Regression | Classification |
|---|---|---|
| `mean` | Predictive mean | Mean class probabilities |
| `epistemic_var` | Posterior-sampling variance (noise removed when available) | `None` |
| `aleatoric_var` | Empirical residual-noise term (if estimated) | `None` |
| `total_var` | Predictive variance | `None` |
| `probs` | `None` | Mean class probabilities |
| `probs_var` | `None` | Optional probability variance (backend-dependent) |
| `metadata` | Method/structure/likelihood details | Method/structure/likelihood details |

## 7) References

1. MacKay, D. J. C. (1992). *A Practical Bayesian Framework for Backpropagation Networks*. Neural Computation, 4(3), 448-472. DOI: [10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
2. Tierney, L., & Kadane, J. B. (1986). *Accurate Approximations for Posterior Moments and Marginal Densities*. Journal of the American Statistical Association, 81(393), 82-86. DOI: [10.1080/01621459.1986.10478240](https://doi.org/10.1080/01621459.1986.10478240)
3. Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML (PMLR 37). [Proceedings](https://proceedings.mlr.press/v37/martens15.html)
4. Botev, A., Ritter, H., & Barber, D. (2017). *Practical Gauss-Newton Optimisation for Deep Learning*. ICML (PMLR 70). [Proceedings](https://proceedings.mlr.press/v70/botev17a.html)
5. Ritter, H., Botev, A., & Barber, D. (2018). *A Scalable Laplace Approximation for Neural Networks*. ICLR. [Conference entry](https://iclr.cc/virtual/2018/poster/224)
6. Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). *Laplace Redux: Effortless Bayesian Deep Learning*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)
7. Kunstner, F., Hennig, P., & Balles, L. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/8669-limitations-of-the-empirical-fisher-approximation-for-natural-gradient-descent)

## Related docs

- [Laplace Hessian Comparison Tutorial](../tutorials/laplace-comparison.md)
- [Laplace API](../api/methods/laplace.md)
