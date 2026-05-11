# Conformal Prediction

## Overview

Conformal prediction provides **distribution-free, finite-sample coverage guarantees** for prediction intervals and sets. Unlike other UQ methods that rely on model assumptions (Gaussian posteriors, calibrated ensembles), conformal methods guarantee that the true value lies within the predicted interval with probability at least $1 - \alpha$, regardless of the model or data distribution.

The only assumption is **exchangeability** of the calibration and test data (weaker than i.i.d.).

## Key Concepts

### Nonconformity Scores

A nonconformity score $s(x, y)$ measures how "unusual" a data point is relative to the model's predictions. Common choices:

- **Absolute residual**: $s = |y - \hat{y}|$
- **Normalized residual**: $s = |y - \hat{y}| / \hat{\sigma}$ (uses model uncertainty)
- **Quantile score** (CQR): $s = \max(\hat{q}_\text{lo} - y, \; y - \hat{q}_\text{hi})$

### Coverage Guarantee

Given calibration scores $s_1, \ldots, s_n$ and a new test point, conformal prediction computes:

$$\hat{q} = \text{Quantile}\left(s_1, \ldots, s_n; \; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

The prediction interval $[\hat{y} - \hat{q}, \; \hat{y} + \hat{q}]$ satisfies:

$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This holds for **any** model, **any** data distribution, with **finite samples**.

## Methods in Deep-UQ

### Split Conformal Regression

The simplest and most widely used variant. Requires only a held-out calibration set.

```python
from deepuq.methods import SplitConformalRegressor

cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate(cal_loader)  # or (x_cal, y_cal) tuple
result = cp.predict_uq(x_test)

# result.metadata["conformal_lower"], result.metadata["conformal_upper"]
```

**Pros**: Works with any model, no retraining, fast.
**Cons**: Intervals are constant width (not adaptive to local uncertainty).

### Conformalized Quantile Regression (CQR)

Combines quantile regression with conformal calibration for **adaptive intervals** that are tighter where the model is confident and wider where it's uncertain.

```python
from deepuq.methods import CQRPredictor

# model outputs (q_lo, q_hi) for each input
cqr = CQRPredictor(model, alpha=0.1)
cqr.calibrate(cal_loader)
result = cqr.predict_uq(x_test)
```

**Pros**: Adaptive interval widths, still has coverage guarantee.
**Cons**: Requires training a quantile regression model.

### Conformal Classification (APS / RAPS)

Produces **prediction sets** — subsets of classes guaranteed to contain the true class.

```python
from deepuq.methods import ConformalClassifier

cc = ConformalClassifier(model, alpha=0.1, method="raps")
cc.calibrate(cal_loader)
result = cc.predict_uq(x_test)

# result.metadata["prediction_sets"] — boolean tensor of included classes
# result.metadata["set_sizes"] — number of classes in each set
```

- **APS** (Adaptive Prediction Sets): Orders classes by softmax probability, includes until cumulative probability exceeds threshold.
- **RAPS** (Regularized APS): Adds a penalty for large sets, producing smaller average set sizes.

### Conformal UQ Wrapper

Wraps **any existing UQ method** (Laplace, Ensembles, MC Dropout, etc.) to guarantee coverage:

```python
from deepuq.methods import ConformalUQWrapper, LaplaceWrapper

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="full")
la.fit(train_loader)

# Conformalize the Laplace intervals
conf_la = ConformalUQWrapper(la, alpha=0.1)
conf_la.calibrate(cal_loader)
result = conf_la.predict_uq(x_test)
```

This is the recommended approach when you want both the interpretability of a Bayesian method and the coverage guarantee of conformal prediction.

## When to Use Conformal Prediction

| Scenario | Recommendation |
|----------|---------------|
| Need guaranteed coverage | Split Conformal or CQR |
| Have a trained model, want quick UQ | Split Conformal |
| Want adaptive intervals | CQR |
| Classification with set-valued predictions | APS/RAPS |
| Already using Laplace/Ensembles, want calibration | ConformalUQWrapper |

## Comparison with Other Methods

| Property | Conformal | Laplace | Ensembles | MC Dropout |
|----------|-----------|---------|-----------|------------|
| Coverage guarantee | Yes (finite-sample) | No (asymptotic) | No | No |
| Requires retraining | No | No | Yes | No |
| Adaptive intervals | CQR only | Yes | Yes | Yes |
| Assumption | Exchangeability | Gaussian posterior | Independence | Dropout approx. |
| Computational cost | Very low | Medium | High | Low |

## References

- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*
- Romano, Patterson, Candes (2019). *Conformalized Quantile Regression*
- Angelopoulos, Bates (2021). *A Gentle Introduction to Conformal Prediction*
- Angelopoulos et al. (2021). *Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS)*
