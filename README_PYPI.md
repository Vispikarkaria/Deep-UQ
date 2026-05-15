Deep-UQ is the most comprehensive uncertainty quantification toolkit for deep learning in PyTorch — **20+ UQ methods**, scientific ML architectures, evaluation metrics, active learning, and physics constraints, all behind a unified `predict_uq()` API.

## Install

```bash
pip install uqdeepnn
```

## Quick Example

```python
from deepuq.models import MLP
from deepuq.methods import LaplaceWrapper

model = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1)
# ... train as usual ...

la = LaplaceWrapper(model, likelihood="regression", hessian_structure="kron")
la.fit(train_loader)
result = la.predict_uq(x_test)
# result.mean, result.epistemic_var, result.total_var
```

## 20+ UQ Methods

**Bayesian & Ensemble:** Deep Ensembles, Batch Ensemble, Packed Ensemble, SWAG, MultiSWAG, Variational Inference (BBB, Last-Layer), SVGD

**Post-hoc & Single-pass:** Laplace (6 Hessian structures + GLM predictive), SNGP, Evidential DL, MC Dropout, Test-Time Augmentation, Temperature/Isotonic Calibration

**MCMC:** SGLD, SGHMC, Cyclical SGMCMC

**Gaussian Processes:** Exact, Sparse, Multi-Fidelity, Deep Kernel, Multi-task, Heteroscedastic, Spectral Mixture, Classification

**Distribution-free:** Conformal (split, CQR, weighted, adaptive), Selective Prediction

## Full Toolkit

- `deepuq.methods` — All 20+ UQ method wrappers
- `deepuq.models` — Neural architectures (MLP, FNO, DeepONet, GNO, PINN, CNN, UNet, GPs)
- `deepuq.metrics` — ECE, CRPS, AUROC, PICP, Brier, interval score, AURC
- `deepuq.active` — Active learning (Uncertainty Sampling, BALD, loop orchestration)
- `deepuq.constraints` — Physics constraints (positivity, bounds, conservation, monotonicity)
- `deepuq.propagation` — Uncertainty propagation for autoregressive rollouts

## Scientific ML Backbones

DeepONet (1D/2D), FNO (2D/3D), Graph Neural Operators, CNN/ResNet/U-Net surrogates, PINNs, Diffusion models — all with first-class UQ support.

## 50+ Tutorials

Every method has a self-contained executable notebook with mathematical explanations, training, prediction, and visualization.

## Key Features

- **Zero external UQ dependencies** — everything in pure PyTorch
- **Unified API** — every method returns `UQResult(mean, epistemic_var, aleatoric_var, total_var)`
- **Any architecture** — works with any `nn.Module`
- **Evaluation built-in** — calibration, scoring rules, OOD detection metrics
- **Production-ready** — selective prediction, post-hoc calibration, single-pass methods

## Documentation

- Docs: https://vispikarkaria.github.io/Deep-UQ/
- Tutorials: https://vispikarkaria.github.io/Deep-UQ/tutorials/
- API: https://vispikarkaria.github.io/Deep-UQ/api/
- GitHub: https://github.com/Vispikarkaria/Deep-UQ

## Package Names

- PyPI: `uqdeepnn`
- Import: `deepuq`
