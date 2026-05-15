Deep-UQ is a PyTorch toolkit for uncertainty-aware machine learning.

It provides 20+ uncertainty quantification methods, Gaussian-process models,
scientific machine learning backbones, and evaluation metrics — all behind
a unified `predict_uq()` API returning calibrated uncertainty estimates.

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

## UQ Methods

- **Post-hoc (no retraining):** MC Dropout, Laplace, SWAG, Temperature Scaling, Test-Time Augmentation
- **Single forward pass:** SNGP, Evidential Deep Learning
- **Ensembles:** Deep Ensembles, Batch Ensemble, Packed Ensemble
- **Bayesian:** Variational Inference (BBB, Flipout, Last-Layer), SVGD
- **MCMC:** SGLD, SGHMC, Cyclical SGMCMC
- **Gaussian Processes:** Exact, Sparse, Multi-task, Deep Kernel, Multi-Fidelity
- **Conformal:** Split, CQR, Weighted, Adaptive
- **Calibration:** Temperature Scaling, Vector Scaling, Isotonic Regression
- **Decision:** Selective Prediction with AURC evaluation

## Modules

- `deepuq.methods` — All UQ method wrappers
- `deepuq.models` — Neural architectures (MLP, FNO, DeepONet, GNO, PINN, CNN, UNet, GPs)
- `deepuq.metrics` — ECE, CRPS, AUROC, PICP, Brier score, interval score, AURC
- `deepuq.active` — Active learning (Uncertainty Sampling, BALD)
- `deepuq.constraints` — Physics constraints (positivity, bounds, conservation, monotonicity)
- `deepuq.propagation` — Uncertainty propagation for autoregressive rollouts

## Scientific ML Backbones

DeepONet, FNO (2D/3D), Graph Neural Operators, CNN/ResNet/U-Net surrogates, PINNs, Diffusion models — all with first-class UQ support.

## Documentation

- Docs: https://vispikarkaria.github.io/Deep-UQ/
- Tutorials: https://vispikarkaria.github.io/Deep-UQ/tutorials/
- API reference: https://vispikarkaria.github.io/Deep-UQ/api/
- GitHub: https://github.com/Vispikarkaria/Deep-UQ

## Package names

- PyPI package: `uqdeepnn`
- Python import: `deepuq`
