# Gaussian Processes

`deepuq` includes both exact GP regression (`GaussianProcessRegressor`) and sparse variational GP regression (`SparseGaussianProcessRegressor`).

## 1) Motivation

Gaussian processes provide a nonparametric Bayesian model over functions. They are a strong uncertainty baseline because posterior variance is tied directly to kernel structure and data support.

Exact GPs are statistically elegant but cubic in training-set size; sparse variational GPs trade exactness for scalability.

## 2) What Uncertainty Is Quantified

GPs quantify posterior uncertainty over latent functions $f$. For noisy regression,

$$
y=f(x)+\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,\sigma_n^2)
$$

Predictive uncertainty combines latent-function uncertainty and observation noise.

## 3) Mathematical Setup / Notation

Prior process:

$$
f(\cdot)\sim\mathcal{GP}\!\left(m(\cdot),k(\cdot,\cdot)\right)
$$

For training inputs $X$, targets $y$, test inputs $X_\*$:

$$
K=K(X,X),
\quad
K_\*=K(X,X_\*),
\quad
K_{\*\*}=K(X_\*,X_\*)
$$

Noisy covariance matrix:

$$
K_y=K+\sigma_n^2 I
$$

## 4) Core Method Equations

### 4.1 Exact GP

Posterior mean:

$$
m_\*(X_\*)=K_\*^{\top}K_y^{-1}y
$$

Posterior covariance:

$$
\Sigma_\*(X_\*)=K_{\*\*}-K_\*^{\top}K_y^{-1}K_\*
$$

Log marginal likelihood:

$$
\log p(y\mid X)=
-\frac{1}{2}y^{\top}K_y^{-1}y
-\frac{1}{2}\log\lvert K_y\rvert
-\frac{N}{2}\log(2\pi)
$$

### 4.2 Sparse Variational GP

Introduce inducing inputs $Z\in\mathbb R^{M\times d}$, inducing variables $u=f(Z)$, and variational posterior:

$$
q(u)=\mathcal N(m,S)
$$

Variational ELBO:

$$
\mathcal F=
\mathbb E_{q(f)}\left[\log p(y\mid f)\right]
-
\mathrm{KL}\!\left(q(u)\,\|\,p(u)\right)
$$

Titsias-style covariance term:

$$
Q_{NN}=K_{NM}K_{MM}^{-1}K_{MN}
$$

and a common collapsed objective form:

$$
\mathcal F=
\log\mathcal N\!\left(y\mid 0,Q_{NN}+\sigma_n^2 I\right)
-
\frac{1}{2\sigma_n^2}\mathrm{tr}\!\left(K_{NN}-Q_{NN}\right)
$$

## 5) Inference / Prediction Equations

### Exact GP prediction

For a test point $x_\*$ with $k_\*=k(X,x_\*)$:

$$
\mu_\*(x_\*)=k_\*^{\top}K_y^{-1}y
$$

$$
\sigma_{f,\*}^2(x_\*)=k(x_\*,x_\*)-k_\*^{\top}K_y^{-1}k_\*
$$

Noisy predictive variance adds $\sigma_n^2$.

### Sparse variational GP prediction

Using $q(u)=\mathcal N(m,S)$:

$$
\mu_\*(x_\*)=k_{\*M}K_{MM}^{-1}m
$$

$$
\sigma_{f,\*}^2(x_\*)=
 k_{\*\*}
 +k_{\*M}K_{MM}^{-1}(S-K_{MM})K_{MM}^{-1}k_{M\*}
$$

## 6) Practical Implications

- Exact GP: strong calibration for small/medium $N$, cost roughly $\mathcal O(N^3)$ training and $\mathcal O(N^2)$ memory.
- Sparse GP: training cost depends on inducing count $M$, typically around $\mathcal O(NM^2)$, with memory $\mathcal O(NM)$.
- Kernel choice determines smoothness, extrapolation behavior, and uncertainty shape.
- Inducing locations and $M$ strongly affect sparse-GP fidelity.

## 7) References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [Book](https://gaussianprocess.org/gpml/)
2. Quiñonero-Candela, J., & Rasmussen, C. E. (2005). *A Unifying View of Sparse Approximate Gaussian Process Regression*. Journal of Machine Learning Research, 6, 1939-1959. [JMLR](https://jmlr.org/papers/v6/quinonero-candela05a.html)
3. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes*. AISTATS (PMLR 5). [Proceedings](https://proceedings.mlr.press/v5/titsias09a.html)
4. Hensman, J., Fusi, N., & Lawrence, N. D. (2013). *Gaussian Processes for Big Data*. UAI. [Paper](https://arxiv.org/abs/1309.6835)
5. Bauer, M., van der Wilk, M., & Rasmussen, C. E. (2016). *Understanding Probabilistic Sparse Gaussian Process Approximations*. NeurIPS. [Proceedings](https://papers.nips.cc/paper/2016/hash/7250eb93b3c18cc9daa29cf58af7a004-Abstract.html)

## Related docs

- [Gaussian Process Tutorial Guide](../tutorials/gp.md)
- [Gaussian Process API](../api/models/gaussian_process.md)
