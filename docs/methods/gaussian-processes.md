# Gaussian Processes

`deepuq` includes a full Gaussian Process (GP) family:

- exact GP regression (`GaussianProcessRegressor`)
- sparse variational GP regression (`SparseGaussianProcessRegressor`)
- binary GP classification (`GaussianProcessClassifier`)
- one-vs-rest multiclass GP classification (`OneVsRestGaussianProcessClassifier`)
- heteroscedastic GP regression (`HeteroscedasticGaussianProcessRegressor`)
- multi-task ICM GP regression (`MultiTaskGaussianProcessRegressor`)
- spectral mixture GP regression (`SpectralMixtureGaussianProcessRegressor`)
- deep kernel GP regression (`DeepKernelGaussianProcessRegressor`)

## 1) Motivation

Gaussian Processes are Bayesian models over functions. They are important in UQ because they produce both a predictive mean and a predictive uncertainty estimate that grows when data support is weak.

In `deepuq`, the GP suite is designed to cover:

- calibrated interpolation baselines (exact and sparse)
- uncertainty-aware classification boundaries
- input-dependent noise modeling
- correlated multi-output regression
- spectral and periodic structure
- learned representations through deep kernels

## 2) What Uncertainty Is Quantified

For regression, `deepuq` reports:

- `epistemic_var`: uncertainty in the latent function estimate
- `aleatoric_var`: observation noise uncertainty
- `total_var`: sum of epistemic and aleatoric components

Regression observation model:

$$
y = f(x) + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$

Predictive variance decomposition:

$$
\sigma_{\mathrm{pred}}^2(x) = \sigma_{\mathrm{epi}}^2(x) + \sigma_{\mathrm{alea}}^2(x)
$$

For classification, GP classifiers return class probabilities (`probs`) and probability spread proxies (`probs_var`).

## 3) Core Models and Equations

### 3.1 Exact GP regression

Prior:

$$
f(\cdot) \sim \mathcal{GP}(0, k(\cdot, \cdot))
$$

With training inputs $X$, targets $y$, and test inputs $X_*$:

$$
\mu_* = K_{*X}\left(K_{XX} + \sigma_\varepsilon^2 I\right)^{-1} y
$$

$$
\Sigma_* = K_{**} - K_{*X}\left(K_{XX} + \sigma_\varepsilon^2 I\right)^{-1} K_{X*}
$$

Log marginal likelihood:

$$
\log p(y \mid X) =
-\frac{1}{2} y^\top \left(K_{XX} + \sigma_\varepsilon^2 I\right)^{-1} y
- \frac{1}{2} \log\left|K_{XX} + \sigma_\varepsilon^2 I\right|
- \frac{N}{2}\log(2\pi)
$$

### 3.2 Sparse variational GP regression

Inducing variables $u = f(Z)$ with $M \ll N$ and variational posterior $q(u)$ are used for scalability.

Approximate covariance term:

$$
Q_{NN} = K_{NM} K_{MM}^{-1} K_{MN}
$$

Common collapsed ELBO form:

$$
\mathcal{F} = \log \mathcal{N}\left(y \mid 0, Q_{NN} + \sigma_\varepsilon^2 I\right)
- \frac{1}{2\sigma_\varepsilon^2}\mathrm{tr}\left(K_{NN} - Q_{NN}\right)
$$

### 3.3 GP classification (binary + OvR multiclass)

Binary GP classification uses:

$$
p(y_i=1 \mid f_i) = \sigma(f_i)
$$

where $\sigma(\cdot)$ is the logistic sigmoid.

`deepuq` uses a Laplace approximation around the latent mode.

A common logistic-Gaussian predictive approximation is:

$$
p(y=1 \mid x) \approx \sigma\left(\frac{\mu_f(x)}{\sqrt{1 + \frac{\pi}{8}\sigma_f^2(x)}}\right)
$$

For multiclass classification, one binary GP is fit per class (OvR), then class scores are normalized into probabilities.

### 3.4 Heteroscedastic GP regression

Noise depends on input:

$$
\varepsilon(x) \sim \mathcal{N}\left(0, \sigma_\varepsilon^2(x)\right)
$$

`deepuq` alternates between:

- a mean GP fit
- a noise GP fit on transformed residual targets

Residual-noise target used in practice:

$$
\log\left((y - \hat{f})^2 + \delta\right)
$$

### 3.5 Multi-task ICM GP regression

For task indices $t, t'$:

$$
K\big((x,t), (x',t')\big) = K_x(x,x')\,B_{tt'}
$$

Equivalent matrix form:

$$
K = B \otimes K_x + \sigma_\varepsilon^2 I
$$

Task covariance is constrained PSD, e.g.:

$$
B = L L^\top + \mathrm{diag}(d)
$$

### 3.6 Spectral mixture GP regression

Spectral mixture kernel (for lag $\tau = x-x'$):

$$
k(\tau) = \sum_{q=1}^{Q} w_q \prod_{d=1}^{D}
\exp\left(-2\pi^2 \tau_d^2 v_{qd}\right)\cos\left(2\pi \tau_d \mu_{qd}\right)
$$

### 3.7 Deep kernel GP regression

A learned feature map $\phi_\psi(x)$ is composed with a GP kernel:

$$
k_{\mathrm{DKL}}(x,x') = k\big(\phi_\psi(x), \phi_\psi(x')\big)
$$

The feature extractor parameters and GP hyperparameters are optimized jointly.

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
| Regression GP family | Posterior mean | Latent posterior variance | Noise term (constant or input-dependent) | Sum of epi + alea | `None` | `None` |
| Classification GP family | `None` | `None` | `None` | `None` | Class probabilities | Probability spread proxy |

## 6) Practical Notes

- Exact GP is strongest for calibration on small/medium datasets.
- Sparse GP is preferred when $N$ grows and exact $\mathcal{O}(N^3)$ cost is too high.
- Heteroscedastic GP is useful when noise level changes with operating regime.
- Multi-task ICM helps when outputs are correlated.
- Spectral mixture kernels help for multi-frequency and quasi-periodic signals.
- Deep kernel GP helps when raw input space is not kernel-friendly.

## 7) References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Book](https://gaussianprocess.org/gpml/)
2. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS (PMLR 5). [Proceedings](https://proceedings.mlr.press/v5/titsias09a.html)
3. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). *Gaussian Processes for Big Data*. UAI. [Paper](https://arxiv.org/abs/1309.6835)
4. Williams, C. K. I., & Barber, D. (1998). *Bayesian Classification with Gaussian Processes*. IEEE TPAMI, 20(12), 1342-1351. DOI: [10.1109/34.735807](https://doi.org/10.1109/34.735807)
5. Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review*. Foundations and Trends in ML, 4(3), 195-266. DOI: [10.1561/2200000036](https://doi.org/10.1561/2200000036)
6. Wilson, A. G., & Adams, R. P. (2013). *Gaussian Process Kernels for Pattern Discovery and Extrapolation*. ICML (PMLR). [Proceedings](https://proceedings.mlr.press/v28/wilson13.html)
7. Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). *Deep Kernel Learning*. AISTATS (PMLR). [Proceedings](https://proceedings.mlr.press/v51/wilson16.html)

## Related docs

- [Gaussian Process Tutorial Guide](../tutorials/gp.md)
- [Gaussian Process API](../api/models/gaussian_process.md)
