# Laplace regression example (Euler-Bernoulli)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from deepuq.models import MLP
from deepuq.methods import LaplaceWrapper

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L = 2.0
w_load = 4000.0
E = 210e9
I = 1e-6


def beam_deflection_mm(x: torch.Tensor) -> torch.Tensor:
    y_m = (w_load * x * (L**3 - 2 * L * x**2 + x**3)) / (24 * E * I)
    return 1e3 * y_m


def make_dataset(n_samples: int, noise_std: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 1, generator=gen) * L
    y_true = beam_deflection_mm(x)
    noise = noise_std * torch.randn(n_samples, 1, generator=gen)
    y = y_true + noise
    return x, y, y_true


x_train, y_train, _ = make_dataset(512, noise_std=0.05, seed=11)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)

x_grid = torch.linspace(0, L, 200).unsqueeze(-1)
y_grid_true = beam_deflection_mm(x_grid)

model = MLP(1, [128, 128], 1, p_drop=0.0).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x_batch)
        loss = nn.MSELoss()(preds, y_batch)
        loss.backward()
        optimizer.step()

la = LaplaceWrapper(model, likelihood='regression', hessian_structure='diag')
la.fit(train_loader, prior_precision=1.0)
mean, var = la.predict(x_grid.to(DEVICE), n_samples=200)
mean = mean.cpu()
std = var.sqrt().cpu()

lower = mean - 1.96 * std
upper = mean + 1.96 * std

plt.figure(figsize=(7, 4))
plt.scatter(x_train.numpy(), y_train.numpy(), s=14, alpha=0.5, label='Noisy samples')
plt.plot(x_grid.numpy(), y_grid_true.numpy(), color='black', linewidth=2, label='True deflection')
plt.plot(x_grid.numpy(), mean.numpy(), color='tab:blue', label='Laplace mean')
plt.fill_between(x_grid.squeeze().numpy(), lower.squeeze().numpy(), upper.squeeze().numpy(), alpha=0.2, color='tab:blue', label='95% interval')
plt.xlabel('Position x (m)')
plt.ylabel('Deflection y (mm)')
plt.title('Laplace confidence bounds')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
