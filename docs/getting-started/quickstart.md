# Quickstart

This page shows minimal workflows for regression and uncertainty prediction.

## 1) MC Dropout in 10 lines

```python
import torch
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

x = torch.linspace(-3, 3, 200).unsqueeze(-1)
model = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1, p_drop=0.1)
uq = MCDropoutWrapper(model, n_mc=200, apply_softmax=False)
mean, var = uq.predict(x)
```

## 2) Bayes by Backprop ELBO step

```python
from deepuq.methods.vi import BayesByBackpropMLP, vi_elbo_step

model = BayesByBackpropMLP(input_dim=1, hidden_dims=[64, 64], output_dim=1, prior_sigma=0.1)
loss, nll, kl = vi_elbo_step(
    model,
    x_batch,
    y_batch,
    num_batches=len(train_loader),
    criterion=torch.nn.MSELoss(reduction="mean"),
    kl_weight=0.01,
    mc_samples=8,
)
```

## 3) Laplace around a trained MAP model

```python
from deepuq.methods import LaplaceWrapper

la = LaplaceWrapper(
    model=trained_map_model,
    likelihood="regression",
    hessian_structure="diag",
    subset_of_weights="last_layer",
)
la.fit(train_loader, prior_precision=30.0)
mean, var = la.predict(x_test, n_samples=200)
```

## 4) Exact GP baseline

```python
from deepuq.models import GaussianProcessRegressor, RBFKernel

gp = GaussianProcessRegressor(kernel=RBFKernel(lengthscale=0.5, outputscale=1.0), noise=0.02)
gp.fit(x_train, y_train)
mean, var = gp.predict(x_test)
```

## Next Steps

- Explore [method docs](../methods/variational-inference.md)
- Run full scripts in [examples](../examples/index.md)
- Walk through [tutorial guides](../tutorials/index.md)
