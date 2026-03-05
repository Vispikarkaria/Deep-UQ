# Gaussian Processes

`deepuq` now includes a full GP family:

- exact GP regression (`GaussianProcessRegressor`)
- sparse variational GP regression (`SparseGaussianProcessRegressor`)
- binary GP classification (`GaussianProcessClassifier`)
- multiclass OvR GP classification (`OneVsRestGaussianProcessClassifier`)
- heteroscedastic GP regression (`HeteroscedasticGaussianProcessRegressor`)
- multi-task ICM GP regression (`MultiTaskGaussianProcessRegressor`)
- spectral mixture GP regression (`SpectralMixtureGaussianProcessRegressor`)
- deep kernel GP regression (`DeepKernelGaussianProcessRegressor`)

## 1) Motivation

Gaussian processes provide Bayesian function-space inference. They are a strong UQ baseline because posterior uncertainty expands in regions with weak data support.

In Deep-UQ, the GP suite is designed to cover:

- calibrated interpolation baselines (exact/sparse)
- classification boundary uncertainty
- input-dependent noise modeling
- correlated multi-output regression
- rich spectral structure
- learned representations through deep kernels

## 2) What Uncertainty Is Quantified

For regression:

$$
y = f(x) + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

Predictive variance is decomposed as:

$$
\sigma_{\mathrm{pred}}^2(x)
=
\sigma_{\mathrm{epi}}^2(x)
+
\sigma_{\mathrm{alea}}^2(x)
$$

For classification, GP classifiers return class probabilities and probability spread near boundaries.

## 3) Core Models and Equations

### 3.1 Exact GP regression

Prior:

$$
f(\cdot) \sim \mathcal{GP}(0, k(\cdot, \cdot))
$$

Posterior mean and covariance:

$$
\mu_* = K_{*X}(K_{XX} + \sigma_\varepsilon^2 I)^{-1} y
$$

$$
\Sigma_* = K_{**} - K_{*X}(K_{XX} + \sigma_\varepsilon^2 I)^{-1}K_{X*}
$$

### 3.2 Sparse variational GP regression

Inducing variables $u = f(Z)$ with $M \ll N$ and variational posterior $q(u)$.

A common ELBO form is:

$$
\mathcal{F}
=
\log \mathcal{N}(y \mid 0, Q_{NN} + \sigma_\varepsilon^2 I)
-
\frac{1}{2\sigma_\varepsilon^2}\mathrm{tr}(K_{NN} - Q_{NN})
$$

with:

$$
Q_{NN} = K_{NM}K_{MM}^{-1}K_{MN}
$$

### 3.3 GP classification (binary + OvR multiclass)

Binary latent function with Bernoulli likelihood:

$$
p(y_i=1 \mid f_i) = \sigma(f_i)
$$

where $\sigma(\cdot)$ is logistic sigmoid.

Deep-UQ uses Laplace approximation for the latent posterior around its mode. OvR multiclass fits one binary GP per class and normalizes scores.

### 3.4 Heteroscedastic GP regression

Noise depends on input:

$$
\varepsilon(x) \sim \mathcal{N}(0, \sigma_\varepsilon^2(x))
$$

The implementation alternates between:

- mean GP fit
- noise GP fit on $\log((y-\hat{f})^2 + \delta)$

### 3.5 Multi-task ICM GP regression

Intrinsic coregionalization uses:

$$
K\big((x,t), (x',t')\big)
=
K_x(x,x') B_{tt'}
$$

Equivalent matrix form:

$$
K = B \otimes K_x + \sigma_\varepsilon^2 I
$$

where $B$ is learned PSD task covariance.

### 3.6 Spectral mixture GP regression

Spectral mixture kernel approximates stationary kernels using Gaussian mixtures in spectral domain:

$$
k(\tau)
=
\sum_{q=1}^{Q}
 w_q
 \prod_{d=1}^{D}
 \exp\!\left(-2\pi^2 \tau_d^2 v_{qd}\right)
 \cos\!\left(2\pi \tau_d \mu_{qd}\right)
$$

### 3.7 Deep kernel GP regression

Feature map $\phi_\psi(x)$ from an MLP is composed with an RBF GP head:

$$
k_{\mathrm{DKL}}(x,x')
=
k_{\mathrm{RBF}}\big(\phi_\psi(x), \phi_\psi(x')\big)
$$

Parameters of $\phi_\psi$ and GP hyperparameters are optimized jointly by marginal likelihood.

## 4) Kernel Support

Deep-UQ GP kernels include:

- `RBFKernel` (scalar or ARD lengthscale)
- `MaternKernel` (`nu=1.5` or `2.5`)
- `RationalQuadraticKernel`
- `PeriodicKernel`
- `LinearKernel`
- `SpectralMixtureKernel`
- `SumKernel` via `k1 + k2`
- `ProductKernel` via `k1 * k2`

## 5) UQResult Field Mapping

| Model Type | `mean` | `epistemic_var` | `aleatoric_var` | `total_var` | `probs` | `probs_var` |
|---|---|---|---|---|---|---|
| Regression GPs | Posterior mean | Latent posterior variance | Noise term (constant or input-dependent) | Sum of epi + alea | `None` | `None` |
| Classification GPs | Probability mean tensor | `None` | `None` | `None` | Class probabilities | Probability variance proxy |

## 6) Practical Notes

- Exact GP gives strongest calibration for small/medium datasets.
- Sparse GP is preferred when $N$ grows and exact $\mathcal{O}(N^3)$ cost is too high.
- Heteroscedastic GP is useful when sensor noise varies by operating regime.
- Multi-task ICM helps when outputs are correlated.
- Spectral mixture kernels help with multi-frequency or quasi-periodic signals.
- Deep kernel GP helps when raw input space is not kernel-friendly.

## 7) References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Book](https://gaussianprocess.org/gpml/)
2. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS (PMLR 5). [Proceedings](https://proceedings.mlr.press/v5/titsias09a.html)
3. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). *Gaussian Processes for Big Data*. UAI. [Paper](https://arxiv.org/abs/1309.6835)
4. Williams, C. K. I., & Barber, D. (1998). *Bayesian Classification with Gaussian Processes*. IEEE TPAMI, 20(12), 1342-1351. DOI: [10.1109/34.735807](https://doi.org/10.1109/34.735807)
5. Álvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review*. Foundations and Trends in ML, 4(3), 195-266. DOI: [10.1561/2200000036](https://doi.org/10.1561/2200000036)
6. Wilson, A. G., & Adams, R. P. (2013). *Gaussian Process Kernels for Pattern Discovery and Extrapolation*. ICML (PMLR). [Proceedings](https://proceedings.mlr.press/v28/wilson13.html)
7. Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). *Deep Kernel Learning*. AISTATS (PMLR). [Proceedings](https://proceedings.mlr.press/v51/wilson16.html)

## Related docs

- [Gaussian Process Tutorial Guide](../tutorials/gp.md)
- [Gaussian Process API](../api/models/gaussian_process.md)
