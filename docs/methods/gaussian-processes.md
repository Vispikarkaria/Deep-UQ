# Gaussian Processes

`deepuq` exposes Gaussian Process methods through:
`GaussianProcessRegressor`, `SparseGaussianProcessRegressor`,
`GaussianProcessClassifier`, `OneVsRestGaussianProcessClassifier`,
`HeteroscedasticGaussianProcessRegressor`,
`MultiTaskGaussianProcessRegressor`,
`SpectralMixtureGaussianProcessRegressor`, and
`DeepKernelGaussianProcessRegressor`.

## 1) Motivation

Gaussian Processes place a Bayesian prior directly over functions rather than over
finite-dimensional weights. This makes them a strong uncertainty quantification
baseline because posterior uncertainty expands naturally when observations are
sparse, noisy, or out of distribution.

In `deepuq`, the GP family is designed to cover:

- exact regression for calibrated small-data baselines,
- sparse variational regression for larger datasets,
- classification with uncertainty near decision boundaries,
- input-dependent noise models,
- correlated multi-output regression,
- spectral structure and learned feature maps.

## 2) What Uncertainty Is Quantified

For regression, GP models quantify posterior uncertainty in the latent function
and, when appropriate, observation noise.

With the standard noisy-observation model:

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma_n^2)
$$

the predictive variance decomposes as:

$$
\mathrm{Var}[y_*\mid x_*,\mathcal D]=\mathrm{Var}[f_*\mid x_*,\mathcal D]+\sigma_n^2
$$

For classification, the object of interest is the predictive class probability.
In the binary case:

$$
p(y_*=1\mid x_*,\mathcal D)=\int p(y_*=1\mid f_*)\,p(f_*\mid x_*,\mathcal D)\,df_*
$$

## 3) Mathematical Setup / Notation

Dataset and kernel notation:

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^N,
\qquad
X=[x_1,\ldots,x_N]^{\top}
$$

$$
K_{XX}=k(X,X),
\qquad
K_{*X}=k(X_*,X),
\qquad
K_{**}=k(X_*,X_*)
$$

Throughout the page:

- $f$ denotes the latent function,
- $u=f(Z)$ denotes inducing variables at inducing inputs $Z$,
- $\sigma_n^2$ denotes observation-noise variance,
- $B$ denotes the task covariance matrix in the multi-task model.

## 4) Core Method Equations

### 4.1 Exact GP Regression

Prior:

$$
f(\cdot)\sim\mathcal{GP}(0,k(\cdot,\cdot))
$$

Posterior predictive mean and covariance:

$$
\mu_*=K_{*X}\left(K_{XX}+\sigma_n^2 I\right)^{-1}y
$$

$$
\Sigma_*=K_{**}-K_{*X}\left(K_{XX}+\sigma_n^2 I\right)^{-1}K_{X*}
$$

Log marginal likelihood:

$$
\log p(y\mid X)=-\frac{1}{2}y^{\top}\left(K_{XX}+\sigma_n^2 I\right)^{-1}y-\frac{1}{2}\log\left|K_{XX}+\sigma_n^2 I\right|-\frac{N}{2}\log(2\pi)
$$

### 4.2 Sparse Variational GP Regression

Introduce inducing variables:

$$
u=f(Z),
\qquad
q(u)=\mathcal N(m,S),
\qquad
M\ll N
$$

Projected covariance:

$$
Q_{XX}=K_{XZ}K_{ZZ}^{-1}K_{ZX}
$$

Common collapsed ELBO form:

$$
\mathcal F=\log\mathcal N\left(y\mid 0,Q_{XX}+\sigma_n^2 I\right)-\frac{1}{2\sigma_n^2}\mathrm{tr}\left(K_{XX}-Q_{XX}\right)
$$

### 4.3 GP Classification (Binary + OvR Multiclass)

Binary likelihood:

$$
p(y_i=1\mid f_i)=\sigma(f_i)
$$

where $\sigma(\cdot)$ is the logistic sigmoid.

`deepuq` uses a Laplace approximation in the latent space. A common predictive
approximation is:

$$
p(y_*=1\mid x_*,\mathcal D)\approx\sigma\left(\frac{\mu_*}{\sqrt{1+\frac{\pi}{8}\sigma_*^2}}\right)
$$

For multiclass classification, one binary GP is fit per class and the resulting
scores are normalized into class probabilities.

### 4.4 Heteroscedastic GP Regression

Input-dependent noise model:

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N\left(0,\sigma_n^2(x_i)\right)
$$

`deepuq` alternates between a mean GP and a noise GP. The transformed
residual target used for the noise process is:

$$
\log\left((y_i-\hat f(x_i))^2+\delta\right)
$$

### 4.5 Multi-task ICM GP Regression

For task indices $t$ and $t'$:

$$
k\big((x,t),(x',t')\big)=k_x(x,x')\,B_{tt'}
$$

Equivalent covariance form:

$$
K=B\otimes K_x+\sigma_n^2 I
$$

with positive semidefinite task covariance:

$$
B=LL^{\top}+\mathrm{diag}(d)
$$

### 4.6 Spectral Mixture GP Regression

For lag $\tau=x-x'$:

$$
k(\tau)=\sum_{q=1}^{Q}w_q\prod_{d=1}^{D}\exp\left(-2\pi^2\tau_d^2 v_{qd}\right)\cos\left(2\pi\tau_d\mu_{qd}\right)
$$

### 4.7 Deep Kernel GP Regression

Learned representation with GP head:

$$
k_{\mathrm{DKL}}(x,x')=k_{\mathrm{base}}\big(\phi_{\psi}(x),\phi_{\psi}(x')\big)
$$

where $\phi_{\psi}$ is a trainable feature extractor.

## 5) Inference / Prediction Equations

For regression, the posterior predictive moments are:

$$
\mathbb E[y_*\mid x_*,\mathcal D]=\mu_*
$$

$$
\mathrm{Var}[f_*\mid x_*,\mathcal D]=\mathrm{diag}(\Sigma_*)
$$

$$
\mathrm{Var}[y_*\mid x_*,\mathcal D]=\mathrm{diag}(\Sigma_*)+\sigma_n^2
$$

For classification, predictive probabilities are estimated from the latent
posterior. In the one-vs-rest setting, classwise probabilities are computed
independently and then normalized.

## 6) Practical Implications

- Exact GP gives the strongest calibrated baseline when $N$ is modest.
- Sparse GP reduces cost from cubic training in $N$ to inducing-point scaling.
- Classification GPs are useful when boundary uncertainty matters more than raw
  point accuracy.
- Heteroscedastic GP separates epistemic and aleatoric structure when noise is
  regime-dependent.
- Multi-task ICM helps when outputs are correlated and can share information.
- Spectral mixture and deep kernel variants are useful when simple stationary
  kernels are too restrictive.

## UQResult Field Mapping

`predict_uq(...)` outputs map as follows:

| Model Type | `mean` | `epistemic_var` | `aleatoric_var` | `total_var` | `probs` | `probs_var` |
|---|---|---|---|---|---|---|
| Regression GP family | Posterior mean | Latent posterior variance | Noise term (constant or input-dependent) | Sum of epi + alea | `None` | `None` |
| Classification GP family | `None` | `None` | `None` | `None` | Class probabilities | Probability spread proxy |

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
