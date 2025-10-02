# deepuq

Unified deep learning uncertainty quantification (UQ) toolkit in PyTorch.

Implements **four** widely used methods:

1. **Variational Inference (VI)** — Bayes by Backprop with BayesianLinear layers.
2. **Laplace Approximation** — via `laplace-torch` with diagonal/kronecker/full Hessians.
3. **MCMC (SGLD)** — Stochastic Gradient Langevin Dynamics sampler for NN posteriors.
4. **MC Dropout** — Keep dropout active at test-time and aggregate Monte Carlo predictions.

<p align="center">
<img src="https://raw.githubusercontent.com/placeholder/uq_table.png" alt="UQ Table" width="600"/>
</p>

## Install (local)

```bash
git clone https://github.com/yourusername/deepuq.git
cd deepuq
pip install -e .
```

> Coming from PyPI? See the section **Publish to PyPI** below.

## Quickstart

```python
import torch
from deepuq.models.simple import MLP
from deepuq.methods.mc_dropout import MCDropoutWrapper

model = MLP(input_dim=784, hidden_dims=[256,128], output_dim=10, p_drop=0.2)
uq = MCDropoutWrapper(model, n_mc=50)
mean, var = uq.predict(torch.randn(32, 784))
print(mean.shape, var.shape)
```

See the **examples/** folder for end‑to‑end training scripts on MNIST/FashionMNIST.

## Methods

- **VI**: Place Gaussian posteriors over weights with reparameterization trick and KL regularization.
- **Laplace**: Fit a Gaussian around a MAP solution using the Hessian; calibrate with a prior precision.
- **MCMC (SGLD)**: Inject Gaussian noise into SGD steps to sample from the posterior.
- **MC Dropout**: Use dropout at inference; Monte Carlo average for mean and variance.

## Documentation

- API docs are in each module and the README sections below.
- Run `pydoc deepuq.methods.vi` etc., or open the examples.

## Contributing

PRs welcome. Please add tests under `tests/` and run `pytest`.

## License

MIT
