# Deep Ensembles

Deep ensembles quantify predictive uncertainty by training multiple independently
initialized neural networks and averaging their predictions at inference time.
Deep-UQ now exposes five ensemble variants so the same core idea can be used for
plain regression, regression with input-dependent noise, classification,
multi-output regression, and multi-output regression with predicted noise.

## Why Use Deep Ensembles

Deep ensembles remain one of the strongest practical UQ baselines for neural
networks because they do not require a variational posterior, Hessian
approximation, or sampler over weights. The uncertainty signal comes from the
spread across independently trained models.

Two ideas underpin the method family:

1. Independent models started from different random initializations settle into
   different local solutions.
2. Aggregating those solutions approximates model averaging and exposes
   epistemic uncertainty through prediction disagreement.

The classic ensemble argument goes back to Hansen and Salamon (1990), while the
modern deep-learning formulation used here follows Lakshminarayanan, Pritzel,
and Blundell (2017).

## Shared Setup

Let $\mathcal{D}=\{(x_i, y_i)\}_{i=1}^N$ and let $M$ denote the number of
ensemble members. Each member has parameters $	heta^{(m)}$ and predictive
mapping $f_{	heta^{(m)}}(x)$.

Each model is trained independently:

$$
	heta^{(m)} = rg\min_	heta \mathcal{L}^{(m)}(	heta; \mathcal{D}), \qquad m=1, \ldots, M.
$$

At prediction time, the ensemble combines member outputs instead of trusting any
single fitted network.

## 1. DeepEnsembleRegressor

`DeepEnsembleRegressor` is the plain regression ensemble. Each member predicts a
single deterministic output $f_{	heta^{(m)}}(x)$ and is trained with mean
squared error:

$$
\mathcal{L}_{\mathrm{MSE}}^{(m)} = rac{1}{N} \sum_{i=1}^N \lVert y_i - f_{	heta^{(m)}}(x_i) 
Vert_2^2.
$$

The ensemble predictive mean is

$$
\mu(x) = rac{1}{M} \sum_{m=1}^M f_{	heta^{(m)}}(x),
$$

and the epistemic variance used in Deep-UQ is the member-wise sample variance
(with `unbiased=False` in the implementation):

$$
\sigma^2_{\mathrm{epi}}(x) = rac{1}{M} \sum_{m=1}^M \left(f_{	heta^{(m)}}(x) - \mu(x)
ight)^2.
$$

This method is appropriate when the observation noise is negligible, already
accounted for in the dataset, or not the quantity of interest. In that case the
ensemble variance is interpreted as model uncertainty only.

**Deep-UQ interface:** `DeepEnsembleRegressor.predict_uq(x)` returns
`mean`, `epistemic_var`, and `total_var=epistemic_var`.

**Scientific notebook:**
[`DeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb`](../tutorials/sciml-deep-ensemble-adr1d.md)

**Primary references:** Hansen and Salamon (1990); Lakshminarayanan et al.
(2017).

## 2. HeteroscedasticDeepEnsembleRegressor

`HeteroscedasticDeepEnsembleRegressor` augments each member so that it predicts a
mean and an input-dependent variance:

$$
f_{	heta^{(m)}}(x) = \left(\mu_{	heta^{(m)}}(x), \log \sigma^2_{	heta^{(m)}}(x)
ight).
$$

Deep-UQ implements a Gaussian negative log-likelihood with a diagonal variance
head. Writing $s_{	heta^{(m)}}(x)=\log \sigma^2_{	heta^{(m)}}(x)$, each member
minimizes

$$
\mathcal{L}_{\mathrm{het}}^{(m)} = rac{1}{2N} \sum_{i=1}^N \left[ e^{-s_{	heta^{(m)}}(x_i)} \left(y_i - \mu_{	heta^{(m)}}(x_i)
ight)^2 + s_{	heta^{(m)}}(x_i) 
ight].
$$

The ensemble predictive mean remains

$$
\mu(x) = rac{1}{M} \sum_{m=1}^M \mu_{	heta^{(m)}}(x),
$$

while the total predictive variance is decomposed into epistemic and aleatoric
terms:

$$
\sigma^2_{\mathrm{epi}}(x) = rac{1}{M} \sum_{m=1}^M \left(\mu_{	heta^{(m)}}(x) - \mu(x)
ight)^2,
$$

$$
\sigma^2_{\mathrm{alea}}(x) = rac{1}{M} \sum_{m=1}^M \sigma^2_{	heta^{(m)}}(x),
$$

$$
\sigma^2_{\mathrm{tot}}(x) = \sigma^2_{\mathrm{epi}}(x) + \sigma^2_{\mathrm{alea}}(x).
$$

This is the right variant when the data contain input-dependent measurement
noise. In Deep-UQ, the predicted variance is clamped below by `min_variance`
for numerical stability.

**Deep-UQ interface:** `predict_uq(x)` returns `mean`, `epistemic_var`,
`aleatoric_var`, and `total_var`.

**Scientific notebook:**
[`HeteroscedasticDeepEnsemble_AdvectionDiffusionReaction1D_Tutorial.ipynb`](../tutorials/sciml-heteroscedastic-deep-ensemble-adr1d.md)

**Primary references:** Nix and Weigend (1994) for Gaussian mean/variance
prediction; Kendall and Gal (2017) for the epistemic/aleatoric decomposition;
Lakshminarayanan et al. (2017) for the ensemble construction.

## 3. DeepEnsembleClassifier

`DeepEnsembleClassifier` targets classification. Each member predicts logits
$z_{	heta^{(m)}}(x) \in \mathbb{R}^C$ and is trained with cross-entropy:

$$
\mathcal{L}_{\mathrm{CE}}^{(m)} = -rac{1}{N} \sum_{i=1}^N \log p_{	heta^{(m)}}(y_i \mid x_i),
$$

where

$$
p_{	heta^{(m)}}(y \mid x) = \mathrm{softmax}\left(z_{	heta^{(m)}}(x)
ight).
$$

The ensemble predictive probability vector is the average of member
probabilities:

$$
ar p(y \mid x) = rac{1}{M} \sum_{m=1}^M p_{	heta^{(m)}}(y \mid x).
$$

Deep-UQ also reports a per-class probability variance,

$$
\sigma^2_{p,c}(x) = rac{1}{M} \sum_{m=1}^M \left(p^{(m)}_c(x) - ar p_c(x)
ight)^2,
$$

which acts as a direct disagreement proxy. This is especially useful near
scientific safety/failure boundaries where member classifiers disagree about the
class region.

**Deep-UQ interface:** `predict_uq(x)` returns `probs`, `probs_var`, and sets
`mean=probs` for consistency with the package-wide `UQResult` contract.

**Scientific notebook:**
[`DeepEnsemble_Elasticity2D_Classification_Tutorial.ipynb`](../tutorials/sciml-deep-ensemble-elasticity-classification.md)

**Primary references:** Hansen and Salamon (1990); Lakshminarayanan et al.
(2017).

## 4. MultiOutputDeepEnsembleRegressor

`MultiOutputDeepEnsembleRegressor` generalizes the plain regressor to vector
outputs. Each member predicts

$$
f_{	heta^{(m)}}(x) \in \mathbb{R}^D,
$$

for example displacement and stress, or any other coupled set of scientific
outputs.

Training still uses MSE,

$$
\mathcal{L}_{\mathrm{MSE}}^{(m)} = rac{1}{N} \sum_{i=1}^N \lVert y_i - f_{	heta^{(m)}}(x_i) 
Vert_2^2,
$$

and Deep-UQ aggregates the mean componentwise:

$$
\mu(x) = rac{1}{M} \sum_{m=1}^M f_{	heta^{(m)}}(x).
$$

The package currently reports a diagonal epistemic covariance approximation, not
a full output covariance matrix:

$$
\Sigma_{\mathrm{epi}}(x) pprox \mathrm{diag}\!\left( rac{1}{M} \sum_{m=1}^M \left(f_{	heta^{(m)}}(x)-\mu(x)
ight) \odot \left(f_{	heta^{(m)}}(x)-\mu(x)
ight) 
ight).
$$

That choice keeps the interface simple and aligns with the existing `UQResult`
contract. It is appropriate when the main question is uncertainty magnitude per
output channel rather than the full cross-output covariance.

**Deep-UQ interface:** `predict_uq(x)` returns `mean`, `epistemic_var`, and
`total_var=epistemic_var`, each shaped like the vector output.

**Scientific notebook:**
[`MultiOutputDeepEnsemble_ElasticBar1D_Tutorial.ipynb`](../tutorials/sciml-multioutput-deep-ensemble-elastic-bar.md)

**Primary references:** Lakshminarayanan et al. (2017). The multi-output form
implemented in Deep-UQ is a direct vector-valued extension of the same ensemble
averaging principle.

## 5. HeteroscedasticMultiOutputDeepEnsembleRegressor

`HeteroscedasticMultiOutputDeepEnsembleRegressor` combines the previous two
extensions: each member predicts a vector mean together with a diagonal vector
variance:

$$
f_{	heta^{(m)}}(x) = \left(\mu_{	heta^{(m)}}(x), \log \sigma^2_{	heta^{(m)}}(x)
ight), \qquad \mu_{	heta^{(m)}}(x), \sigma^2_{	heta^{(m)}}(x) \in \mathbb{R}^D.
$$

The member loss is the diagonal multivariate Gaussian negative log-likelihood,
implemented channelwise in Deep-UQ:

$$
\mathcal{L}_{\mathrm{multi	ext{-}het}}^{(m)} = rac{1}{2N} \sum_{i=1}^N \sum_{d=1}^D \left[ rac{\left(y_{id}-\mu_{	heta^{(m)},d}(x_i)
ight)^2}{\sigma^2_{	heta^{(m)},d}(x_i)} + \log \sigma^2_{	heta^{(m)},d}(x_i) 
ight].
$$

Deep-UQ again returns a diagonal covariance decomposition:

$$
\Sigma_{\mathrm{epi}}(x) pprox \mathrm{diag}\!\left( rac{1}{M} \sum_{m=1}^M \left(\mu_{	heta^{(m)}}(x)-\mu(x)
ight) \odot \left(\mu_{	heta^{(m)}}(x)-\mu(x)
ight) 
ight),
$$

$$
\Sigma_{\mathrm{alea}}(x) pprox \mathrm{diag}\!\left( rac{1}{M} \sum_{m=1}^M \sigma^2_{	heta^{(m)}}(x) 
ight),
$$

$$
\Sigma_{\mathrm{tot}}(x) pprox \Sigma_{\mathrm{epi}}(x) + \Sigma_{\mathrm{alea}}(x).
$$

This variant is the most complete regression-side ensemble in the current
package: it captures member disagreement and input-dependent per-output noise at
the same time.

**Deep-UQ interface:** `predict_uq(x)` returns `mean`, `epistemic_var`,
`aleatoric_var`, and `total_var`, each matching the output shape.

**Scientific notebook:**
[`HeteroscedasticMultiOutputDeepEnsemble_Transport2D_Tutorial.ipynb`](../tutorials/sciml-heteroscedastic-multioutput-deep-ensemble-transport2d.md)

**Primary references:** Nix and Weigend (1994); Kendall and Gal (2017);
Lakshminarayanan et al. (2017). As with the plain multi-output regressor, this
is a package-level extension of the same ensemble principle with diagonal
Gaussian heads.

## Deep-UQ Interfaces

Available classes:

- `DeepEnsembleRegressor`
- `HeteroscedasticDeepEnsembleRegressor`
- `DeepEnsembleClassifier`
- `MultiOutputDeepEnsembleRegressor`
- `HeteroscedasticMultiOutputDeepEnsembleRegressor`
- `DeepEnsembleWrapper` (backward-compatible alias of `DeepEnsembleRegressor`)

## Recommended Scientific Examples

| Method | Scientific example | Tutorial |
|---|---|---|
| `DeepEnsembleRegressor` | 1D advection-diffusion-reaction | [Notebook guide](../tutorials/sciml-deep-ensemble-adr1d.md) |
| `HeteroscedasticDeepEnsembleRegressor` | 1D advection-diffusion-reaction with spatially varying noise | [Notebook guide](../tutorials/sciml-heteroscedastic-deep-ensemble-adr1d.md) |
| `DeepEnsembleClassifier` | elasticity-inspired failure map | [Notebook guide](../tutorials/sciml-deep-ensemble-elasticity-classification.md) |
| `MultiOutputDeepEnsembleRegressor` | 1D elastic bar (displacement + stress) | [Notebook guide](../tutorials/sciml-multioutput-deep-ensemble-elastic-bar.md) |
| `HeteroscedasticMultiOutputDeepEnsembleRegressor` | 2D advection-diffusion transport (concentration + flux) | [Notebook guide](../tutorials/sciml-heteroscedastic-multioutput-deep-ensemble-transport2d.md) |

## References

1. Hansen, L. K., & Salamon, P. (1990). *Neural Network Ensembles*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(10), 993-1001. DOI: [10.1109/34.58871](https://doi.org/10.1109/34.58871)
2. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. Advances in Neural Information Processing Systems 30. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38f21-Abstract.html)
3. Nix, D. A., & Weigend, A. S. (1994). *Estimating the Mean and Variance of the Target Probability Distribution*. Proceedings of the 1994 IEEE International Conference on Neural Networks. DOI: [10.1109/ICNN.1994.374138](https://doi.org/10.1109/ICNN.1994.374138)
4. Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* Advances in Neural Information Processing Systems 30. [NeurIPS proceedings](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need)
