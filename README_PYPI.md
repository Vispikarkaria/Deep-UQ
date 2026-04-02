Deep-UQ is a PyTorch toolkit for uncertainty-aware machine learning.

It collects practical uncertainty quantification methods, Gaussian-process models,
and scientific machine learning backbones in one package, with a common focus on
predictive uncertainty for regression, classification, and field-to-field surrogate modeling.

## Install

```bash
pip install uqdeepnn
```

For the legacy `kron` and `full` Laplace backends used in older Deep-UQ
tutorials, install the optional Laplace extra:

```bash
pip install "uqdeepnn[laplace]"
```

## Import

```python
import deepuq
```

## Included methods

- Deep Ensembles
- Variational Inference (Bayes by Backprop, heteroscedastic VI, multi-output VI, and last-layer VI)
- Laplace Approximation
- MCMC via SGLD
- MC Dropout
- Gaussian Processes

## Scientific machine learning backbones

- DeepONet
- Fourier Neural Operator (FNO)
- Graph Neural Operators
- CNN / ResNet spatial surrogates
- U-Net backbones
- Physics-Informed Neural Networks (PINNs)

## Included data utilities and examples

- The Well Gray-Scott loader for graph-operator tutorials
- Scientific notebooks for operators, graph models, ensembles, VI, PINNs, diffusion, and Laplace UQ

## Documentation

- Docs: https://vispikarkaria.github.io/Deep-UQ/
- Tutorials: https://vispikarkaria.github.io/Deep-UQ/tutorials/
- API reference: https://vispikarkaria.github.io/Deep-UQ/api/
- Variational Inference guide: https://vispikarkaria.github.io/Deep-UQ/methods/variational-inference/
- GitHub: https://github.com/Vispikarkaria/Deep-UQ

## Package names

- PyPI package: `uqdeepnn`
- Python import: `deepuq`
- Project / docs name: `Deep-UQ`
