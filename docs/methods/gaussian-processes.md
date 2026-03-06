# Gaussian Processes

`deepuq` exposes Gaussian Process methods through
`GaussianProcessRegressor`, `SparseGaussianProcessRegressor`,
`GaussianProcessClassifier`, `OneVsRestGaussianProcessClassifier`,
`HeteroscedasticGaussianProcessRegressor`,
`MultiTaskGaussianProcessRegressor`,
`SpectralMixtureGaussianProcessRegressor`, and
`DeepKernelGaussianProcessRegressor`.

## 1) Motivation

Gaussian Processes place a Bayesian prior directly over functions, which makes
them a strong baseline when the goal is not only prediction accuracy but also
calibrated uncertainty. Instead of fitting one function and treating it as
certain, the model infers a posterior distribution over plausible latent
functions after observing the dataset.

This is why GP methods remain important in uncertainty quantification:

- uncertainty grows naturally away from observed data,
- observation noise can be modeled explicitly,
- posterior covariance reveals coupling across inputs and tasks,
- kernel design gives direct control over smoothness, periodicity, and shared
  structure.

In `deepuq`, the GP family covers exact regression, sparse variational
regression, GP classification, heteroscedastic regression, multi-task ICM,
spectral mixture kernels, and deep kernel learning.

## 2) What Uncertainty Is Quantified

For regression, Gaussian Processes quantify posterior uncertainty in the latent
function and, when the observation model includes noise, aleatoric uncertainty
in the measurements.

With the standard regression model,

$$ y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\sigma_n^2), $$

the predictive variance decomposes as

$$ \mathrm{Var}[y_* \mid x_*, \mathcal{D}] = \mathrm{Var}[f_* \mid x_*, \mathcal{D}] + \sigma_n^2. $$

For classification, the main quantity is the predictive class probability. In
the binary setting,

$$ p(y_* = 1 \mid x_*, \mathcal{D}) = \int p(y_* = 1 \mid f_*) \, p(f_* \mid x_*, \mathcal{D}) \, df_*. $$

So the regression models quantify both latent uncertainty and noise structure,
while the classification models quantify uncertainty in the class-probability
surface, especially near boundaries and away from training support.

## 3) Mathematical Setup / Notation

Throughout this page, let

$$ \mathcal{D} = \{(x_i,y_i)\}_{i=1}^{N}, \qquad X = [x_1,\ldots,x_N]^{\top}, \qquad y = [y_1,\ldots,y_N]^{\top}. $$

Kernel matrices follow the standard GP notation:

$$ K_{XX} = k(X,X), \qquad K_{*X} = k(X_*,X), \qquad K_{X*} = K_{*X}^{\top}, \qquad K_{**} = k(X_*,X_*). $$

Additional symbols used below:

- $f$ is the latent function,
- $u = f(Z)$ are inducing variables at inducing inputs $Z$,
- $\sigma_n^2$ is homoscedastic observation-noise variance,
- $\sigma_n^2(x)$ is input-dependent noise variance,
- $B$ is the task covariance matrix in the intrinsic coregionalization model,
- $\phi_{\psi}$ is a trainable feature extractor in deep kernel learning.

## 4) Core Models and Equations

### 4.1 Exact GP Regression

The exact GP prior is

$$ f(\cdot) \sim \mathcal{GP}\!\left(0, k(\cdot,\cdot)\right). $$

Conditioning on observed data gives the predictive mean

$$ \mu_* = K_{*X}\left(K_{XX} + \sigma_n^2 I\right)^{-1}y, $$

and predictive covariance

$$ \Sigma_* = K_{**} - K_{*X}\left(K_{XX} + \sigma_n^2 I\right)^{-1}K_{X*}. $$

The log marginal likelihood used for kernel learning is

$$ \log p(y \mid X) = -\frac{1}{2} y^{\top}\left(K_{XX} + \sigma_n^2 I\right)^{-1}y - \frac{1}{2}\log\left|K_{XX} + \sigma_n^2 I\right| - \frac{N}{2}\log(2\pi). $$

### 4.2 Sparse Variational GP Regression

Sparse variational GP regression introduces inducing variables

$$ u = f(Z), \qquad q(u) = \mathcal{N}(m,S), \qquad M \ll N. $$

The projected covariance is

$$ Q_{XX} = K_{XZ}K_{ZZ}^{-1}K_{ZX}. $$

A common collapsed evidence lower bound is

$$ \mathcal{F} = \log \mathcal{N}\!\left(y \mid 0, Q_{XX} + \sigma_n^2 I\right) - \frac{1}{2\sigma_n^2}\mathrm{tr}\!\left(K_{XX} - Q_{XX}\right). $$

This is the standard sparse-GP approximation used to preserve posterior
uncertainty structure while reducing the cost of exact $N \times N$ kernel
algebra.

### 4.3 GP Classification (Binary + OvR Multiclass)

For binary classification with latent score $f_i$, the Bernoulli likelihood
is

$$ p(y_i = 1 \mid f_i) = \sigma(f_i), $$

where $\sigma(\cdot)$ is the logistic sigmoid.

`deepuq` uses a Laplace approximation in latent-function space. A standard
logistic-Gaussian predictive approximation is

$$ p(y_* = 1 \mid x_*, \mathcal{D}) \approx \sigma\!\left(\frac{\mu_*}{\sqrt{1 + \frac{\pi}{8}\sigma_*^2}}\right). $$

For multiclass classification, one binary GP is fit per class and the resulting
scores are normalized as

$$ \tilde p_c(x) = \frac{p_c(x)}{\sum_{j=1}^{C} p_j(x)}. $$

### 4.4 Heteroscedastic GP Regression

When the measurement noise depends on the input, the observation model becomes

$$ y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}\!\left(0,\sigma_n^2(x_i)\right). $$

In practice, `deepuq` alternates between:

- a GP for the latent mean,
- a GP for the log noise process.

The transformed residual target used for the noise model is

$$ \log\!\left((y_i - \hat f(x_i))^2 + \delta\right), \qquad \delta > 0. $$

### 4.5 Multi-task ICM GP Regression

For task indices $t$ and $t'$, the intrinsic coregionalization kernel is

$$ k\!\left((x,t),(x',t')\right) = k_x(x,x') \, B_{tt'}. $$

The corresponding covariance matrix is

$$ K = B \otimes K_x + \sigma_n^2 I. $$

To guarantee positive semidefiniteness, the task covariance is parameterized as

$$ B = LL^{\top} + \mathrm{diag}(d). $$

This allows information sharing across correlated outputs while keeping
task-specific uncertainty.

### 4.6 Spectral Mixture GP Regression

For lag $\tau = x - x'$, the spectral mixture kernel takes the form

$$ k(\tau) = \sum_{q=1}^{Q} w_q \prod_{d=1}^{D} \exp\!\left(-2\pi^2\tau_d^2 v_{qd}\right)\cos\!\left(2\pi\tau_d\mu_{qd}\right). $$

This kernel is useful when the latent signal contains several frequency bands or
requires extrapolation of oscillatory structure.

### 4.7 Deep Kernel GP Regression

Deep kernel learning composes a trainable feature map with a base kernel:

$$ k_{\mathrm{DKL}}(x,x') = k_{\mathrm{base}}\!\left(\phi_{\psi}(x), \phi_{\psi}(x')\right). $$

This lets the model learn a representation in which the GP prior is better
matched to the observed data.

## 5) Inference / Prediction Equations

For regression, the posterior predictive moments are

$$ \mathbb{E}[y_* \mid x_*, \mathcal{D}] = \mu_*, $$

$$ \sigma_{\mathrm{epi}}^2(x_*) = \mathrm{diag}(\Sigma_*), $$

$$ \sigma_{\mathrm{pred}}^2(x_*) = \sigma_{\mathrm{epi}}^2(x_*) + \sigma_{\mathrm{alea}}^2(x_*). $$

For exact homoscedastic regression, $\sigma_{\mathrm{alea}}^2(x_*) = \sigma_n^2$.
For heteroscedastic regression, $\sigma_{\mathrm{alea}}^2(x_*) = \sigma_n^2(x_*)$.

For classification, the predictive probability is obtained by integrating over
the latent posterior:

$$ p(y_* \mid x_*, \mathcal{D}) = \int p(y_* \mid f_*) \, p(f_* \mid x_*, \mathcal{D}) \, df_*. $$

In one-vs-rest classification, classwise probabilities are estimated
independently and then renormalized into a multiclass simplex.

## 6) Practical Implications

- Exact GP regression is the strongest calibrated baseline when the dataset is
  small or medium sized.
- Sparse variational GP regression trades exactness for scalability while
  keeping a principled probabilistic objective.
- GP classification is useful when uncertainty near failure boundaries matters.
- Heteroscedastic GP regression separates model uncertainty from input-dependent
  noise.
- Multi-task ICM helps when multiple outputs are correlated and should share
  information.
- Spectral mixture kernels are appropriate for multi-frequency or oscillatory
  structure.
- Deep kernel learning is useful when a fixed kernel in raw input space is too
  restrictive.

## UQResult Field Mapping

`predict_uq(...)` outputs map as follows:

| Model Type | `mean` | `epistemic_var` | `aleatoric_var` | `total_var` | `probs` | `probs_var` |
|---|---|---|---|---|---|---|
| Regression GP family | Posterior mean | Latent posterior variance | Noise term (constant or input-dependent) | Sum of epistemic and aleatoric variance | `None` | `None` |
| Classification GP family | `None` | `None` | `None` | `None` | Class probabilities | Probability spread proxy |

## 7) References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Book](https://gaussianprocess.org/gpml/)
2. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS (PMLR 5). [Proceedings](https://proceedings.mlr.press/v5/titsias09a.html)
3. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). *Gaussian Processes for Big Data*. UAI. [Paper](https://arxiv.org/abs/1309.6835)
4. Williams, C. K. I., & Barber, D. (1998). *Bayesian Classification with Gaussian Processes*. IEEE TPAMI, 20(12), 1342-1351. DOI: [10.1109/34.735807](https://doi.org/10.1109/34.735807)
5. Le, H., Smola, A., & Canu, S. (2005). *Heteroscedastic Gaussian Process Regression*. ICML Workshop. [Paper](https://www.researchgate.net/publication/228827935_Heteroscedastic_Gaussian_process_regression)
6. Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review*. Foundations and Trends in Machine Learning, 4(3), 195-266. DOI: [10.1561/2200000036](https://doi.org/10.1561/2200000036)
7. Wilson, A. G., & Adams, R. P. (2013). *Gaussian Process Kernels for Pattern Discovery and Extrapolation*. ICML (PMLR). [Proceedings](https://proceedings.mlr.press/v28/wilson13.html)
8. Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). *Deep Kernel Learning*. AISTATS (PMLR). [Proceedings](https://proceedings.mlr.press/v51/wilson16.html)

## Related docs

- [Gaussian Process Tutorial Guide](../tutorials/gp.md)
- [Gaussian Process API](../api/models/gaussian_process.md)
