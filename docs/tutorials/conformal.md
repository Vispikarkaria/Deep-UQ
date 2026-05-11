# Conformal Prediction Tutorial

This tutorial demonstrates all conformal prediction methods in Deep-UQ using synthetic regression and classification problems.

## Notebook

The full executable notebook is at [`notebooks/conformal/Conformal_Prediction_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/conformal/Conformal_Prediction_Tutorial.ipynb).

## Quick Example: Split Conformal Regression

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from deepuq.models import MLP
from deepuq.methods import SplitConformalRegressor

# Generate data
torch.manual_seed(42)
x = torch.linspace(0, 6, 300).unsqueeze(-1)
y = torch.sin(x) + 0.2 * torch.randn_like(x)

# Split: train / calibration / test
x_train, y_train = x[:150], y[:150]
x_cal, y_cal = x[150:225], y[150:225]
x_test, y_test = x[225:], y[225:]

# Train model
model = MLP(input_dim=1, hidden_dims=[64, 64], output_dim=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(500):
    opt.zero_grad()
    torch.nn.functional.mse_loss(model(x_train), y_train).backward()
    opt.step()
model.eval()

# Conformal calibration (alpha=0.1 → 90% coverage target)
cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate((x_cal, y_cal))

# Predict with guaranteed intervals
result = cp.predict_uq(x_test)
lower = result.metadata["conformal_lower"]
upper = result.metadata["conformal_upper"]

# Check coverage
in_interval = ((y_test >= lower) & (y_test <= upper)).float().mean()
print(f"Empirical coverage: {in_interval:.1%}")  # >= 90%
```

## Quick Example: Conformalized Laplace

```python
from deepuq.methods import LaplaceWrapper, ConformalUQWrapper

# Fit Laplace
la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
la.fit(DataLoader(TensorDataset(x_train, y_train), batch_size=50))

# Conformalize it
conf_la = ConformalUQWrapper(la, alpha=0.1)
conf_la.calibrate((x_cal, y_cal))
result = conf_la.predict_uq(x_test)
```

## Key Takeaways

1. **Split conformal** gives constant-width intervals — simple but effective
2. **CQR** gives adaptive intervals — tighter where model is confident
3. **ConformalUQWrapper** adds coverage guarantees to any existing method
4. Coverage is guaranteed at $\geq 1 - \alpha$ regardless of model quality
5. Better models → narrower intervals (same coverage, less width)
