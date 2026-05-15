---
title: Development Roadmap
description: Deep-UQ implementation roadmap — from foundation to research frontier.
---

# Deep-UQ Development Roadmap

This document defines the complete implementation plan for making Deep-UQ the definitive uncertainty quantification toolkit for deep learning. Each phase builds on the previous, with clear deliverables, implementation guidance, and acceptance criteria.

---

## Current State (v0.1.x)

**Methods:** Deep Ensembles, VI (BBB + Last-Layer), Laplace (6 Hessian structures), SGLD, MC Dropout, GPs (exact/sparse/multitask/deep kernel/spectral mixture/heteroscedastic/classification), Conformal Prediction (split, CQR, classification)

**Models:** MLP, PINN, CNN, ResNet, UNet, DeepONet (1D/2D), FNO (2D/3D), GNO, Diffusion

**API:** Unified `predict_uq()` → `UQResult(mean, epistemic_var, aleatoric_var, total_var)`

---

## Phase 1: Foundation & Calibration (v0.2.x)

**Timeline:** 6–8 weeks  
**Goal:** Fill the most-requested gaps and add proper evaluation infrastructure.

### 1.1 SWAG (Stochastic Weight Averaging — Gaussian)

**What:** Collect first and second moments of SGD trajectory after convergence; use them as a Gaussian posterior approximation.

**Why critical:** Single-training-run Bayesian approximation. Cheaper than ensembles, better than MC Dropout. The most commonly requested missing method in UQ toolkits.

**Implementation plan:**

```
src/deepuq/methods/swag.py
├── class SWAGCollector
│   ├── __init__(model, max_rank=20, collection_freq=1)
│   ├── collect(model)          # call after each epoch in SWA phase
│   ├── finalize()              # compute low-rank + diagonal covariance
│   └── state_dict / load_state_dict
├── class SWAGWrapper
│   ├── __init__(base_model, swag_collector, ...)
│   ├── sample_parameters(scale=1.0, diag_noise=True)
│   ├── predict_uq(x, n_samples=30) → UQResult
│   └── log_prob(x, y)         # marginal likelihood estimate
└── class MultiSWAG
    ├── __init__(swag_wrappers: list)
    └── predict_uq(x, n_samples_per_model=10) → UQResult
```

**Key design decisions:**

- Store running mean, running squared mean (diagonal), and a low-rank deviation matrix (columns = `θ_i - θ_mean`)
- Low-rank matrix capped at `max_rank` columns (default 20); older deviations discarded FIFO
- Sampling: `θ ~ N(θ_mean, 0.5 * Σ_diag + 0.5 * Σ_lowrank)` per Maddox et al. 2019
- `MultiSWAG`: run SWAG collection from K different initializations → combine predictions

**How a scientist implements this:**

1. Train your model normally to convergence
2. Switch learning rate to a constant (SWA learning rate)
3. Call `collector.collect(model)` after every epoch for N additional epochs
4. Call `collector.finalize()`
5. Wrap: `swag = SWAGWrapper(model, collector)`
6. Call `swag.predict_uq(x_test)`

**Acceptance criteria:**

- [ ] Matches published SWAG results on UCI regression (within 5% NLL)
- [ ] Works with all model architectures (MLP, FNO, DeepONet, etc.)
- [ ] Memory overhead < 2x model size for rank-20
- [ ] `predict_uq()` returns valid `UQResult`

**References:** Maddox et al., "A Simple Baseline for Bayesian Inference in Deep Learning" (NeurIPS 2019)

---

### 1.2 Post-Hoc Calibration Methods

**What:** Methods that adjust a trained model's confidence outputs to be calibrated, without retraining.

**Why critical:** Nearly every deployed model needs post-hoc calibration. Temperature scaling is the minimum viable UQ for production classification.

**Implementation plan:**

```
src/deepuq/methods/calibration/
├── __init__.py
├── _temperature.py
│   ├── class TemperatureScaling
│   │   ├── __init__(model)
│   │   ├── fit(val_loader)              # optimize T on validation NLL
│   │   ├── predict_calibrated(x) → probs
│   │   └── predict_uq(x) → UQResult
│   └── class VectorScaling
│       └── (per-class temperature + bias)
├── _isotonic.py
│   ├── class IsotonicCalibration
│   │   ├── fit(val_loader)              # fit isotonic regression per class
│   │   └── predict_calibrated(x) → probs
│   └── class BetaCalibration
├── _histogram.py
│   └── class HistogramBinning
└── _focal.py
    └── class FocalLossCalibration       # focal loss as implicit calibration
```

**How a scientist implements this:**

```python
from deepuq.methods.calibration import TemperatureScaling

# After training
ts = TemperatureScaling(trained_model)
ts.fit(val_loader)  # learns optimal temperature on held-out data

result = ts.predict_uq(x_test)
# result.mean = calibrated probabilities
```

**Acceptance criteria:**

- [ ] Temperature scaling reduces ECE by >50% on CIFAR-10 ResNet
- [ ] Isotonic calibration produces perfectly calibrated histograms on validation
- [ ] All methods work for both binary and multi-class classification
- [ ] Integrates with existing `UQResult` (calibrated probabilities as mean, entropy as variance)

---

### 1.3 Calibration & Evaluation Metrics Module

**What:** Comprehensive metrics for evaluating UQ quality.

**Why critical:** You can't improve what you can't measure. Every paper needs these metrics.

**Implementation plan:**

```
src/deepuq/metrics/
├── __init__.py
├── calibration.py
│   ├── expected_calibration_error(probs, labels, n_bins=15) → float
│   ├── maximum_calibration_error(probs, labels, n_bins=15) → float
│   ├── adaptive_calibration_error(probs, labels, n_bins=15) → float
│   ├── reliability_diagram(probs, labels) → (bin_confs, bin_accs, bin_counts)
│   ├── calibration_curve_regression(predicted_var, residuals, quantiles) → coverage
│   └── prediction_interval_coverage(lower, upper, y_true) → float
├── scoring.py
│   ├── negative_log_likelihood(mean, var, y_true) → float
│   ├── continuous_ranked_probability_score(mean, var, y_true) → float
│   ├── brier_score(probs, labels) → float
│   ├── log_score(probs, labels) → float
│   ├── interval_score(lower, upper, y_true, alpha) → float
│   └── energy_score(samples, y_true) → float
├── sharpness.py
│   ├── mean_prediction_interval_width(lower, upper) → float
│   ├── coefficient_of_variation(mean, var) → float
│   └── sharpness_calibration_tradeoff(mean, var, y_true) → dict
├── ood.py
│   ├── auroc_ood(in_scores, out_scores) → float
│   ├── auprc_ood(in_scores, out_scores) → float
│   ├── fpr_at_tpr(in_scores, out_scores, tpr=0.95) → float
│   └── detection_error(in_scores, out_scores) → float
├── selective.py
│   ├── risk_coverage_curve(uncertainties, errors) → (coverage, risk)
│   ├── aurc(uncertainties, errors) → float          # area under risk-coverage
│   ├── eaurc(uncertainties, errors) → float         # excess AURC
│   └── selective_accuracy(uncertainties, errors, coverage=0.8) → float
└── visualization.py
    ├── plot_reliability_diagram(probs, labels, ax=None) → matplotlib.Axes
    ├── plot_calibration_regression(predicted_var, residuals, ax=None)
    ├── plot_risk_coverage(uncertainties, errors, ax=None)
    └── plot_uncertainty_histogram(epistemic, aleatoric, ax=None)
```

**How a scientist uses this:**

```python
from deepuq.metrics import (
    expected_calibration_error,
    continuous_ranked_probability_score,
    auroc_ood,
    plot_reliability_diagram,
)

# After getting predictions
result = model.predict_uq(x_test)

# Regression metrics
crps = continuous_ranked_probability_score(result.mean, result.total_var, y_test)

# Classification calibration
ece = expected_calibration_error(result.mean, y_test, n_bins=15)
plot_reliability_diagram(result.mean, y_test)

# OOD detection
in_uncertainty = model.predict_uq(x_in).epistemic_var.mean(dim=-1)
out_uncertainty = model.predict_uq(x_ood).epistemic_var.mean(dim=-1)
auroc = auroc_ood(in_uncertainty, out_uncertainty)
```

**Acceptance criteria:**

- [ ] ECE implementation matches published values on standard benchmarks
- [ ] CRPS computed correctly for Gaussian predictive distributions
- [ ] All metrics accept both `UQResult` and raw tensors
- [ ] Visualizations produce publication-quality figures
- [ ] 100% test coverage on edge cases (zero variance, single sample, etc.)

---

### 1.4 Selective Prediction Module

**What:** Reject predictions where uncertainty is too high; report accuracy only on accepted predictions.

```
src/deepuq/methods/selective.py
├── class SelectivePredictor
│   ├── __init__(model, criterion="epistemic_var")
│   ├── predict_with_rejection(x, threshold=None, coverage=0.8)
│   │   → (predictions, mask_accepted, uncertainties)
│   ├── find_threshold(val_loader, target_coverage=0.8) → float
│   └── evaluate(test_loader) → SelectiveMetrics
└── class SelectiveMetrics
    ├── coverage: float
    ├── selective_accuracy: float
    ├── rejection_rate: float
    └── risk_coverage_auc: float
```

---

## Phase 2: Scalable Production Methods (v0.3.x)

**Timeline:** 6–8 weeks  
**Goal:** Methods that scale to real models (ResNets, Transformers, foundation models).

### 2.1 SNGP (Spectral-Normalized Neural Gaussian Process)

**What:** Replace the last layer with a random-feature GP, apply spectral normalization to hidden layers to preserve distance awareness.

**Why critical:** Google's recommended UQ method for production. Single forward pass. No ensembles needed. Works on ImageNet-scale models.

**Implementation plan:**

```
src/deepuq/methods/sngp.py
├── class SpectralNormalization
│   └── Wrapper that applies spectral norm to all Linear/Conv layers
├── class RandomFeatureGPLayer
│   ├── __init__(in_features, num_classes, num_random_features=1024)
│   ├── reset_covariance()      # call at start of each epoch
│   ├── update_covariance(x)    # accumulate precision matrix
│   ├── forward(x) → (logits, covariance)
│   └── predict_uq(x) → UQResult
└── class SNGPWrapper
    ├── __init__(base_model, last_layer_name, num_random_features=1024, ...)
    ├── apply_spectral_norm(norm_bound=6.0)
    ├── fit(train_loader)       # accumulate covariance during one pass
    └── predict_uq(x) → UQResult
```

**Key design decisions:**

- Spectral normalization with configurable bound (default 6.0) on all hidden layers
- Random Fourier Features (RFF) for the GP approximation (not inducing points)
- Mean-field approximation for the posterior covariance
- During training: accumulate precision matrix in a streaming fashion
- During inference: single forward pass → logits + variance from Laplace approx on last layer

**How a scientist implements this:**

```python
from deepuq.methods import SNGPWrapper

# Wrap any pre-trained or new model
sngp = SNGPWrapper(model, last_layer_name="fc3", num_random_features=1024)
sngp.apply_spectral_norm(norm_bound=6.0)

# Train normally (SNGP modifies forward pass)
for epoch in range(n_epochs):
    sngp.reset_covariance()
    for x, y in train_loader:
        logits, cov = sngp(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        sngp.update_covariance(x)

# Inference
result = sngp.predict_uq(x_test)
```

**Acceptance criteria:**

- [ ] Matches published SNGP AUROC on CIFAR-10 vs SVHN OOD benchmark
- [ ] Single forward pass inference (no sampling)
- [ ] Compatible with CNN, ResNet, MLP architectures
- [ ] <10% latency overhead vs vanilla model

**References:** Liu et al., "Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness" (NeurIPS 2020)

---

### 2.2 Batch Ensemble

**What:** Share most parameters across ensemble members; each member has a rank-1 multiplicative perturbation (r_i, s_i) applied to each weight matrix.

**Why critical:** N ensemble members in ~1x memory. Trains in a single forward/backward pass per batch by assigning different batch elements to different members.

**Implementation plan:**

```
src/deepuq/methods/batch_ensemble.py
├── class BatchEnsembleLinear(nn.Module)
│   ├── __init__(in_features, out_features, ensemble_size)
│   ├── Parameters: weight (shared), r_i, s_i (per member), bias_i
│   └── forward(x) → (batch * ensemble_size, out_features)
├── class BatchEnsembleConv2d(nn.Module)
│   └── Same pattern for convolutions
├── class BatchEnsembleWrapper
│   ├── __init__(base_model, ensemble_size=4)
│   ├── convert_to_batch_ensemble()   # replace Linear/Conv with BE versions
│   ├── forward(x) → stacked predictions
│   └── predict_uq(x) → UQResult
└── Utility: replicate_batch(x, ensemble_size) → x repeated for each member
```

**How a scientist implements this:**

```python
from deepuq.methods import BatchEnsembleWrapper

be = BatchEnsembleWrapper(model, ensemble_size=4)
be.convert_to_batch_ensemble()

# Train: batch is automatically split across ensemble members
for x, y in train_loader:
    preds = be(x)  # shape: (batch * 4, output_dim)
    loss = compute_loss(preds, y.repeat(4))
    loss.backward()
    optimizer.step()

result = be.predict_uq(x_test)
```

**Acceptance criteria:**

- [ ] Memory overhead < 5% vs single model for ensemble_size=4
- [ ] Training throughput within 2x of single model
- [ ] Diversity between members (disagreement > random)
- [ ] Matches published BatchEnsemble accuracy on CIFAR-10

**References:** Wen et al., "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning" (ICLR 2020)

---

### 2.3 Packed Ensembles

**What:** Partition network channels into subgroups; each subgroup is an independent "sub-network". All sub-networks share the same architecture but have independent weights within their channel partition.

**Implementation plan:**

```
src/deepuq/methods/packed_ensemble.py
├── class PackedLinear(nn.Module)
│   ├── __init__(in_features, out_features, num_packs, ...)
│   └── forward(x) → grouped computation
├── class PackedConv2d(nn.Module)
│   └── Uses grouped convolution (groups=num_packs)
└── class PackedEnsembleWrapper
    ├── __init__(base_model, num_packs=4, alpha=2)
    ├── convert_to_packed()
    └── predict_uq(x) → UQResult
```

**References:** Laurent et al., "Packed-Ensembles for Efficient Uncertainty Estimation" (ICLR 2023)

---

### 2.4 Improved MCMC

**What:** SGHMC (momentum-based SGLD) and Cyclical SGMCMC (exploration/exploitation cycles).

```
src/deepuq/methods/mcmc.py  (extend existing)
├── class SGHMCOptimizer(Optimizer)
│   ├── __init__(params, lr, momentum_decay, noise_scale)
│   └── step()   # includes friction + noise
├── class CyclicalSGMCMC
│   ├── __init__(model, optimizer_cls, cycle_length, n_cycles, ...)
│   ├── run(train_loader, n_epochs) → list[state_dict]  # collected samples
│   └── Internal: cosine schedule within each cycle, collect at cycle end
└── class PosteriorPredictive
    ├── __init__(base_model, samples: list[state_dict])
    └── predict_uq(x, n_samples=None) → UQResult
```

**References:** Chen et al., "Stochastic Gradient Hamiltonian Monte Carlo" (ICML 2014); Zhang et al., "Cyclical Stochastic Gradient MCMC" (ICLR 2020)

---

### 2.5 Flipout for VI

**What:** Per-example weight perturbations that are decorrelated across the batch. Better gradient variance than standard reparameterization (BBB).

```
src/deepuq/methods/vi.py  (extend existing)
├── class FlipoutLinear(nn.Module)
│   ├── __init__(in_features, out_features)
│   ├── Parameters: weight_mu, weight_sigma (log-scale)
│   └── forward(x) → applies random sign flips per sample
└── class FlipoutMLP
    ├── __init__(input_dim, hidden_dims, output_dim)
    └── predict_uq(x, n_samples=30) → UQResult
```

**References:** Wen et al., "Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches" (ICLR 2018)

---

### 2.6 Linearized Laplace Predictive

**What:** Instead of sampling from the Laplace posterior, use the GLM (Generalized Linear Model) predictive. Linearize the network around the MAP and compute the exact predictive distribution analytically.

```
src/deepuq/methods/laplace/_wrapper.py  (extend existing)
├── LaplaceWrapper.predict_uq(..., method="glm")  # new option
│   # Computes: f(x) ≈ f_MAP(x) + J(x)(θ - θ_MAP)
│   # Predictive: N(f_MAP(x), J(x) @ Σ_post @ J(x)^T)
│   # No sampling needed; exact Gaussian predictive
```

**Why:** Better OOD detection than sample-based Laplace. Faster inference. Theoretically grounded.

**References:** Immer et al., "Improving predictions of Bayesian neural nets via local linearization" (AISTATS 2021)

---

## Phase 3: Deployment & Integration (v0.4.x)

**Timeline:** 6–8 weeks  
**Goal:** Make Deep-UQ usable in production pipelines, not just research notebooks.

### 3.1 Active Learning Module

**What:** Use uncertainty to select the most informative data points for labeling.

```
src/deepuq/active/
├── __init__.py
├── strategies.py
│   ├── class UncertaintySampling
│   │   ├── __init__(model, criterion="epistemic_var")
│   │   └── select(pool_loader, n_samples) → indices
│   ├── class BALDSampling                # Bayesian Active Learning by Disagreement
│   │   └── select(pool_loader, n_samples) → indices
│   ├── class BatchBALD
│   │   └── select(pool_loader, n_samples) → indices
│   ├── class CoreSet
│   │   └── select(pool_loader, n_samples) → indices
│   └── class ExpectedModelChange
│       └── select(pool_loader, n_samples) → indices
├── loop.py
│   └── class ActiveLearningLoop
│       ├── __init__(model, strategy, train_fn, pool_dataset, ...)
│       ├── step() → (selected_indices, current_metrics)
│       └── run(n_iterations, n_samples_per_iter) → history
└── visualization.py
    └── plot_learning_curve(history)
```

**How a scientist uses this:**

```python
from deepuq.active import ActiveLearningLoop, BALDSampling

strategy = BALDSampling(model, n_mc_samples=20)
loop = ActiveLearningLoop(
    model=model,
    strategy=strategy,
    train_fn=my_train_function,
    pool_dataset=unlabeled_data,
    val_dataset=val_data,
)

history = loop.run(n_iterations=20, n_samples_per_iter=50)
```

---

### 3.2 Bayesian Optimization Module

**What:** Use GPs (from existing `deepuq.models.gp`) for sequential optimization of expensive black-box functions.

```
src/deepuq/optim/
├── __init__.py
├── acquisition.py
│   ├── expected_improvement(model, X_candidate, best_y) → scores
│   ├── upper_confidence_bound(model, X_candidate, beta=2.0) → scores
│   ├── probability_of_improvement(model, X_candidate, best_y) → scores
│   └── thompson_sampling(model, X_candidate) → scores
├── bo.py
│   └── class BayesianOptimizer
│       ├── __init__(bounds, kernel, acquisition="ei", ...)
│       ├── suggest(n_suggestions=1) → X_next
│       ├── observe(X, y)
│       ├── run(objective_fn, n_iterations) → OptResult
│       └── get_model() → GaussianProcessRegressor
└── visualization.py
    ├── plot_acquisition(optimizer, ax=None)
    └── plot_convergence(optimizer, ax=None)
```

---

### 3.3 Model Export with UQ

**What:** Export UQ-wrapped models for deployment in non-Python environments.

```
src/deepuq/export/
├── __init__.py
├── torchscript.py
│   ├── export_ensemble(ensemble_wrapper, sample_input) → ScriptModule
│   ├── export_mc_dropout(mc_wrapper, sample_input, n_forward=30) → ScriptModule
│   └── export_laplace_linearized(laplace_wrapper, sample_input) → ScriptModule
├── onnx.py
│   ├── export_to_onnx(wrapper, sample_input, path, ...)
│   └── Handles: mean output + variance output as multi-output graph
└── utils.py
    └── validate_export(original, exported, sample_input, atol=1e-5)
```

---

### 3.4 Distributed Ensemble Training

**What:** Train ensemble members across multiple GPUs/nodes efficiently.

```
src/deepuq/distributed/
├── __init__.py
├── parallel_ensemble.py
│   └── class DistributedEnsembleTrainer
│       ├── __init__(model_fn, ensemble_size, ...)
│       ├── train(train_loader, n_epochs, ...) → list[model]
│       └── Internal: each GPU trains one member, sync at predict time
└── utils.py
    └── gather_predictions(local_preds, world_size) → combined
```

---

### 3.5 Evidential Deep Learning

**What:** Train a network to output parameters of a Dirichlet (classification) or Normal-Inverse-Gamma (regression) distribution. Single forward pass gives both prediction and uncertainty.

```
src/deepuq/methods/evidential.py
├── class EvidentialRegression
│   ├── __init__(base_model)   # model outputs (γ, ν, α, β) per output
│   ├── loss(x, y) → NIG negative log-likelihood + regularizer
│   ├── predict_uq(x) → UQResult
│   │   # epistemic_var = β / (ν * (α - 1))
│   │   # aleatoric_var = β / (α - 1)
│   └── uncertainty_type: "evidential"
└── class EvidentialClassification
    ├── __init__(base_model, num_classes)  # model outputs Dirichlet concentrations
    ├── loss(x, y) → Dirichlet likelihood + KL regularizer
    └── predict_uq(x) → UQResult
        # epistemic_var from Dirichlet uncertainty (K / sum_alpha)
```

**References:** Amini et al., "Deep Evidential Regression" (NeurIPS 2020); Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018)

---

## Phase 4: Scientific ML Depth (v0.5.x)

**Timeline:** 8–10 weeks  
**Goal:** Become the reference toolkit for UQ in computational science.

### 4.1 Multi-Fidelity UQ

**What:** Combine data from cheap (low-fidelity) and expensive (high-fidelity) simulations. The GP learns a correlation structure between fidelities.

```
src/deepuq/models/gp/multifidelity.py
├── class MultiFidelityGP
│   ├── __init__(kernel_lo, kernel_hi, rho_prior=None)
│   ├── fit(X_lo, y_lo, X_hi, y_hi)
│   ├── predict_uq(X_new, fidelity="high") → UQResult
│   └── Information gain / value of information computation
└── class DeepMultiFidelityGP
    ├── __init__(feature_extractor, ...)
    └── Nonlinear fidelity correlation via neural network
```

**Use case:** You have 10,000 coarse mesh CFD runs and 50 fine mesh runs. Train on both, predict with high-fidelity uncertainty.

---

### 4.2 Physics-Constrained Uncertainty

**What:** Ensure that prediction intervals respect known physical constraints (conservation laws, monotonicity, positivity, symmetry).

```
src/deepuq/constraints/
├── __init__.py
├── hard.py
│   ├── class PositivityConstraint      # clip lower bound at 0
│   ├── class ConservationConstraint     # adjust intervals to respect ∫u dx = const
│   ├── class MonotonicityConstraint     # enforce ordered quantiles
│   └── class BoundConstraint           # enforce known min/max
├── soft.py
│   ├── class PhysicsRegularizedUQ      # penalize intervals that violate PDE residual
│   └── class ConstraintAwareLoss       # augmented loss with constraint terms
└── wrappers.py
    └── class ConstrainedUQResult(UQResult)
        └── Applies constraints post-hoc to any UQResult
```

**How a scientist uses this:**

```python
from deepuq.constraints import ConservationConstraint, ConstrainedUQResult

# Mass must be conserved: total integral = 1.0
constraint = ConservationConstraint(
    integration_weights=dx,  # quadrature weights
    conserved_quantity=1.0,
)

raw_result = model.predict_uq(x_test)
constrained = ConstrainedUQResult(raw_result, constraints=[constraint])
# constrained.mean integrates to 1.0
# constrained.total_var adjusted to respect feasible region
```

---

### 4.3 Spatiotemporal Uncertainty Propagation

**What:** When rolling out a time-dependent PDE solver autoregressively, uncertainty grows at each step. Track and propagate this correctly.

```
src/deepuq/propagation/
├── __init__.py
├── rollout.py
│   └── class UncertaintyRollout
│       ├── __init__(model, n_steps, propagation="moment_matching")
│       ├── predict_trajectory(x0, n_steps) → list[UQResult]
│       │   # Options: "moment_matching", "sampling", "unscented"
│       └── uncertainty_growth_rate(trajectory) → float
├── moment_matching.py
│   └── Propagate mean + covariance through the model using linearization
├── unscented.py
│   └── Sigma-point propagation (no Jacobians needed)
└── sampling.py
    └── Particle-based propagation (ensemble of trajectories)
```

**Use case:** FNO trained on Navier-Stokes predicts 10 timesteps ahead. At step 1, epistemic variance is small. By step 10, it's grown significantly. This module tracks that growth faithfully.

---

### 4.4 Neural ODE/SDE with UQ

**What:** Continuous-depth models where the dynamics themselves have uncertainty.

```
src/deepuq/models/neural_ode.py
├── class NeuralODE
│   ├── __init__(dynamics_net, solver="dopri5")
│   └── forward(x0, t_span) → trajectory
├── class BayesianNeuralODE
│   ├── __init__(dynamics_net, uq_method="swag")
│   └── predict_uq(x0, t_span, n_samples=30) → list[UQResult]
└── class NeuralSDE
    ├── __init__(drift_net, diffusion_net, solver="euler_maruyama")
    ├── forward(x0, t_span) → trajectory (stochastic)
    └── predict_uq(x0, t_span, n_paths=100) → UQResult
```

---

### 4.5 Functional Priors for Neural Operators

**What:** Instead of weight-space priors (which are hard to interpret), define priors in function space: "I believe the operator output is smooth" or "output should look like a GP with Matern kernel."

```
src/deepuq/priors/
├── __init__.py
├── functional.py
│   ├── class GPFunctionalPrior
│   │   ├── __init__(kernel, input_points)
│   │   └── log_prob(f_samples) → float
│   ├── class SmoothnesssPrior
│   │   ├── __init__(smoothness_order=2)
│   │   └── log_prob(f_samples) → penalizes high-frequency content
│   └── class PhysicsPrior
│       ├── __init__(pde_residual_fn)
│       └── log_prob(f_samples) → penalizes PDE residual
└── integration.py
    └── Utilities to combine functional priors with SWAG/Laplace/VI
```

---

### 4.6 Uncertainty for Sequence Models

**What:** UQ methods designed for Transformers, RNNs, and other sequence architectures.

```
src/deepuq/models/sequence.py
├── class UncertainTransformer
│   ├── __init__(d_model, nhead, num_layers, ...)
│   ├── Stochastic attention (dropout + ensemble heads)
│   └── predict_uq(x_seq) → UQResult (per-token uncertainty)
├── class RecurrentEnsemble
│   ├── __init__(cell_type="lstm", hidden_size, ensemble_size)
│   └── predict_uq(x_seq) → UQResult
└── class BayesianTransformerLayer
    └── Last-layer VI or Laplace on transformer output projection
```

---

## Phase 5: Research Frontier (v0.6.x)

**Timeline:** 8–10 weeks  
**Goal:** Implement cutting-edge methods before other toolkits.

### 5.1 Epistemic Neural Networks (ENN)

**What:** DeepMind's framework. An "epinet" — a small auxiliary network — is trained to predict epistemic uncertainty given the base model's features + a random seed.

```
src/deepuq/methods/enn.py
├── class EpiNet
│   ├── __init__(feature_dim, hidden_dims, output_dim, n_basis=50)
│   ├── forward(features, z_index) → epistemic_perturbation
│   └── Parameters: small MLP + learnable basis vectors
└── class ENNWrapper
    ├── __init__(base_model, feature_layer, epinet_hidden=[64, 64])
    ├── fit(train_loader, n_epochs)    # train epinet with randomized prior loss
    └── predict_uq(x, n_index=100) → UQResult
        # Sample different z indices, measure spread
```

**References:** Osband et al., "Epistemic Neural Networks" (NeurIPS 2023)

---

### 5.2 Conformal Prediction Under Distribution Shift

**What:** Standard conformal assumes exchangeability. Under shift, coverage guarantees break. Weighted and adaptive methods restore guarantees.

```
src/deepuq/methods/conformal/  (extend existing)
├── _weighted.py
│   └── class WeightedConformalPredictor
│       ├── __init__(model, weight_fn)   # importance weights for shift correction
│       └── calibrate(cal_loader, weights)
├── _adaptive.py
│   └── class AdaptiveConformalPredictor
│       ├── __init__(model, target_coverage=0.9)
│       ├── update(x_new, y_new)         # online update of threshold
│       └── predict_set(x) → intervals that adapt over time
└── _mondrian.py
    └── class MondrianConformalPredictor
        └── Group-conditional coverage (per-class or per-region)
```

---

### 5.3 Stein Variational Gradient Descent (SVGD)

**What:** Particle-based inference. Maintain K "particles" (model copies) and update them with a repulsive kernel + gradient to approximate the posterior.

```
src/deepuq/methods/svgd.py
├── class SVGDOptimizer
│   ├── __init__(particles: list[nn.Module], kernel="rbf", bandwidth="median")
│   ├── step(loss_fn, x, y)   # SVGD update on all particles
│   └── kernel_matrix(params) → (K, grad_K)
└── class SVGDWrapper
    ├── __init__(model_fn, n_particles=10, ...)
    ├── fit(train_loader, n_epochs)
    └── predict_uq(x) → UQResult (from particle disagreement)
```

**References:** Liu & Wang, "Stein Variational Gradient Descent" (NeurIPS 2016)

---

### 5.4 PAC-Bayes Bounds

**What:** Compute theoretical generalization guarantees. Given a posterior over weights, compute a certificate: "with probability 1-δ, the true risk is at most X."

```
src/deepuq/bounds/
├── __init__.py
├── pac_bayes.py
│   ├── mcallester_bound(kl_divergence, n_train, delta=0.05) → float
│   ├── catoni_bound(empirical_risk, kl_divergence, n_train, delta=0.05) → float
│   └── data_dependent_bound(model, train_loader, prior, delta=0.05) → float
└── compute.py
    └── class PACBayesCertifier
        ├── __init__(model, prior, posterior)
        ├── compute_kl() → float
        ├── compute_bound(train_loader, delta=0.05) → float
        └── optimize_bound(train_loader) → optimal_lambda
```

---

### 5.5 Posterior Networks

**What:** Use a normalizing flow to model the posterior predictive distribution. Instead of predicting parameters of a fixed distribution, predict an arbitrary density.

```
src/deepuq/methods/posterior_networks.py
├── class PosteriorNetwork
│   ├── __init__(encoder, flow, n_classes_or_output_dim)
│   ├── forward(x) → distribution parameters (concentration for Dirichlet)
│   ├── loss(x, y) → UCE loss (uncertainty cross-entropy)
│   └── predict_uq(x) → UQResult
│       # Epistemic: entropy of the predicted Dirichlet
│       # Aleatoric: expected entropy under the Dirichlet
└── class NormalizingFlowUQ
    ├── __init__(base_model, flow_layers=8)
    ├── forward(x) → samples from learned predictive distribution
    └── predict_uq(x, n_samples=200) → UQResult
```

**References:** Charpentier et al., "Posterior Network: Uncertainty Estimation without OOD Samples" (NeurIPS 2020)

---

### 5.6 Test-Time Augmentation UQ

**What:** Apply random augmentations at test time, measure prediction variance across augmented versions.

```
src/deepuq/methods/tta.py
└── class TTAWrapper
    ├── __init__(model, augmentations, n_augmentations=30)
    └── predict_uq(x) → UQResult
        # Apply each augmentation, forward pass, measure spread
```

---

## Implementation Guidelines for Contributors

### Architecture Principles

1. **Every method must return `UQResult`** — no exceptions. This is the API contract.
2. **Every method must accept any `nn.Module`** as base model (except architecture-specific methods like SNGP which need to modify layers).
3. **No external UQ dependencies** — implement everything in pure PyTorch.
4. **Lazy computation** — don't compute aleatoric/epistemic split if the method doesn't support it; leave as `None`.
5. **Device-agnostic** — all methods must work on CPU and CUDA without code changes.

### File Organization

```
src/deepuq/
├── methods/           # UQ algorithms (wrappers around models)
│   ├── calibration/   # post-hoc calibration
│   ├── conformal/     # conformal prediction variants
│   ├── laplace/       # Laplace approximation
│   └── *.py           # one file per method family
├── models/            # neural network architectures
│   └── gp/            # Gaussian process models
├── metrics/           # evaluation metrics
├── active/            # active learning
├── optim/             # Bayesian optimization
├── constraints/       # physics constraints for UQ
├── propagation/       # uncertainty propagation over time
├── bounds/            # PAC-Bayes and theoretical bounds
├── export/            # model export (ONNX, TorchScript)
├── distributed/       # multi-GPU training
├── priors/            # functional priors
├── types.py           # UQResult and shared types
└── utils.py           # shared utilities
```

### Testing Requirements

Every new method needs:

1. **Unit test**: Does it run? Does it return valid `UQResult`?
2. **Shape test**: Various input shapes, batch sizes, output dimensions.
3. **Correctness test**: Compare against a known reference implementation or published numbers on a toy problem.
4. **Integration test**: Works with at least MLP + one scientific ML model (FNO or DeepONet).
5. **Calibration test**: On a simple problem, is the uncertainty actually calibrated?

```python
# Template for method tests
def test_swag_basic():
    model = MLP(input_dim=2, hidden_dims=[32], output_dim=1)
    # ... train ...
    collector = SWAGCollector(model, max_rank=5)
    for _ in range(10):
        # ... train one epoch ...
        collector.collect(model)
    collector.finalize()
    
    swag = SWAGWrapper(model, collector)
    result = swag.predict_uq(torch.randn(16, 2))
    
    assert isinstance(result, UQResult)
    assert result.mean.shape == (16, 1)
    assert result.epistemic_var.shape == (16, 1)
    assert (result.epistemic_var > 0).all()
```

### Documentation Requirements

Every new method needs:

1. **API reference page**: Auto-generated from docstrings via mkdocstrings
2. **Method guide page** in `docs/methods/`: Mathematical background, when to use, comparison with alternatives
3. **Tutorial**: End-to-end notebook showing the method on a real problem
4. **Benchmarks entry**: Added to the benchmark suite with published reference numbers

### Contribution Workflow

```
1. Open issue describing the method + acceptance criteria
2. Create branch: feature/<method-name>
3. Implement in src/deepuq/methods/<name>.py
4. Add tests in tests/test_<name>.py
5. Add docs page + tutorial
6. Run benchmark comparison
7. PR with: implementation + tests + docs + benchmark results
```

---

## Success Metrics

### v0.2 (Phase 1 complete)
- 10+ UQ methods available
- Comprehensive metrics module with visualization
- All methods benchmarked on at least 2 standard datasets
- ECE/CRPS numbers published in docs

### v0.3 (Phase 2 complete)
- Methods that scale to ImageNet-class models (SNGP, BatchEnsemble)
- <2x overhead vs vanilla training for most methods
- Single-pass UQ options available (SNGP, Evidential)

### v0.4 (Phase 3 complete)
- Active learning loop running on real problems
- Models exportable to ONNX
- Multi-GPU ensemble training working

### v0.5 (Phase 4 complete)
- Multi-fidelity workflows demonstrated on engineering problems
- Spatiotemporal rollout UQ working with FNO/DeepONet
- Physics-constrained intervals published

### v0.6 (Phase 5 complete)
- 25+ UQ methods (most comprehensive toolkit available)
- Theoretical bounds computable
- Research-frontier methods available within 6 months of publication

---

## Competitive Target

By v0.6, Deep-UQ should be the **only toolkit** where a researcher or engineer can:

1. Pick any PyTorch model architecture
2. Choose from 25+ UQ methods (from cheap post-hoc to full Bayesian)
3. Evaluate with proper metrics (calibration, sharpness, OOD, selective)
4. Deploy with export tools
5. Scale with distributed training
6. Apply physical constraints to uncertainty
7. Use in active learning / Bayesian optimization loops
8. Get theoretical guarantees (PAC-Bayes)

All with **zero external UQ dependencies** and a **single unified API**.
