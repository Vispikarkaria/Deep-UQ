---
hide:
  - title
  - navigation
---

<div class="hero-panel reveal">
  <h1>Deep-UQ</h1>
  <p class="hero-tagline">Uncertainty quantification for deep learning. One package, 20+ methods, any PyTorch model.</p>
  <p>
    <a href="getting-started/installation/" class="md-button md-button--primary">Get Started</a>
    <a href="tutorials/" class="md-button">Tutorials</a>
    <a href="api/" class="md-button">API Reference</a>
  </p>
  <p style="margin-top: 1rem; position: relative;">
    <a href="https://pypi.org/project/uqdeepnn/"><img src="https://img.shields.io/pypi/v/uqdeepnn?style=flat-square&color=0b84f3" alt="PyPI"></a>
    <a href="https://github.com/Vispikarkaria/Deep-UQ/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Vispikarkaria/Deep-UQ/tests.yml?style=flat-square&label=tests" alt="Tests"></a>
    <a href="https://github.com/Vispikarkaria/Deep-UQ/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
    <a href="https://github.com/Vispikarkaria/Deep-UQ"><img src="https://img.shields.io/github/stars/Vispikarkaria/Deep-UQ?style=flat-square" alt="GitHub Stars"></a>
  </p>
</div>

## What is Deep-UQ?

Deep-UQ is a PyTorch toolkit that adds **calibrated uncertainty estimates** to neural network predictions. Train your model normally, then wrap it with any of six UQ methods to get confidence intervals that grow when the model is unsure.

```python
from deepuq.models import MLP
from deepuq.methods import LaplaceWrapper

model = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1)
# ... train as usual ...

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="full")
la.fit(train_loader)
la.optimize_prior_precision()

result = la.predict_uq(x_test)
# result.mean, result.epistemic_var, result.total_var
```

---

<div class="card-grid">
  <div class="card reveal">
    <h3>6 Method Families</h3>
    <p>Deep Ensembles, Variational Inference, Laplace, SGLD, MC Dropout, and Gaussian Processes — all behind a unified <code>predict_uq()</code> API.</p>
  </div>
  <div class="card reveal">
    <h3>Scientific ML Ready</h3>
    <p>First-class support for FNO, DeepONet, Graph Neural Operators, PINNs, and convolutional surrogates with 36 executable tutorials.</p>
  </div>
  <div class="card reveal">
    <h3>Zero External UQ Dependencies</h3>
    <p>All methods implemented natively in PyTorch. No GPyTorch, no laplace-torch — just <code>pip install uqdeepnn</code>.</p>
  </div>
</div>

---

## Choose Your Method

| If you need... | Use | Effort |
|---|---|---|
| Quick baseline, no retraining | **MC Dropout** or **Test-Time Augmentation** | Minimal |
| Post-hoc uncertainty on a trained model | **Laplace** or **Temperature Scaling** | Low |
| Single forward pass, distance-aware | **SNGP** or **Evidential DL** | Low |
| Single-run Bayesian approximation | **SWAG / MultiSWAG** | Low |
| Calibrated multi-model uncertainty | **Deep Ensembles** | Medium |
| Memory-efficient ensembles | **Batch Ensemble** or **Packed Ensemble** | Medium |
| Full Bayesian weight posteriors | **Variational Inference** or **Flipout** | Medium |
| Posterior samples via MCMC | **SGLD / SGHMC / Cyclical SGMCMC** | Medium |
| Particle-based inference | **SVGD** | Medium |
| Nonparametric with kernel priors | **Gaussian Processes** | Varies |
| Distribution-free coverage | **Conformal Prediction** (split, weighted, adaptive) | Low |
| Reject uncertain predictions | **Selective Prediction** | Low |

<p><a href="methods/deep-ensembles/">Detailed method guides →</a></p>

---

## Laplace Backends (all native)

| Structure | Speed | Quality | Best for |
|-----------|-------|---------|----------|
| `diag` | Fastest | Good | Large models, quick estimates |
| `lowrank_diag` | Fast | Better | Strong OOD detection |
| `block_diag` | Medium | High | Good accuracy/speed balance |
| `kron` | Medium | High | Multi-output layers |
| `full` | Slow | Reference | Small models, maximum quality |

---

## Model Architectures

| Family | Models | UQ Methods |
|--------|--------|-----------|
| **Dense** | MLP, PINN | All 6 methods |
| **Spatial** | CNN, ResNet, U-Net | Ensembles, MC Dropout, Laplace |
| **Operators** | DeepONet, FNO 2D/3D | Laplace, Ensembles |
| **Graph** | Graph Neural Operator | Ensembles, Laplace |
| **GPs** | Exact, Sparse, Deep Kernel, Multi-task | Native Bayesian |

<p><a href="models/architectures/">Full architecture inventory →</a></p>

---

## Install

```bash
pip install uqdeepnn
```

Or from source:

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git
cd Deep-UQ && pip install -e ".[dev,tests]"
```

---

## Tutorials

**36 executable notebooks** covering every method and architecture:

- [Core UQ methods](tutorials/index.md) — MC Dropout, SGLD, Laplace, VI, Ensembles
- [Scientific ML](tutorials/index.md) — DeepONet, FNO, PINNs, CNNs, U-Nets, Graph Operators
- [Gaussian Processes](tutorials/gp.md) — Exact, Sparse, Classification, Multi-task, Deep Kernel

---

## Unified Output

Every method returns a `UQResult` with consistent fields:

```python
result = method.predict_uq(x)
result.mean           # predictive mean
result.epistemic_var  # model uncertainty
result.aleatoric_var  # data noise (when modeled)
result.total_var      # combined uncertainty
```

---

## Links

| | |
|---|---|
| [Getting Started](getting-started/installation.md) | Installation and quickstart |
| [Method Guides](methods/deep-ensembles.md) | Mathematical details for each method |
| [API Reference](api/index.md) | Full class and function documentation |
| [Tutorials](tutorials/index.md) | 36 executable notebook guides |
| [Contributing](contributing.md) | How to contribute |
| [Changelog](changelog.md) | Release history |
