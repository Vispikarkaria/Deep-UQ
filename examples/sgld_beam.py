# SGLD regression example (Euler-Bernoulli)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from deepuq.models import MLP
from deepuq.methods import collect_posterior_samples, predict_with_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


x_train, y_train, _ = make_dataset(256, noise_std=0.05, seed=3)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)

x_grid = torch.linspace(0, L, 200).unsqueeze(-1)
y_grid_true = beam_deflection_mm(x_grid)

model = MLP(1, [64, 64], 1, p_drop=0.0).to(DEVICE)
loss_fn = nn.MSELoss(reduction="mean")

samples = collect_posterior_samples(
    model,
    train_loader,
    n_steps=120,
    lr=5e-4,
    weight_decay=1e-4,
    burn_in=0.2,
    loss_fn=loss_fn,
    device=DEVICE,
)

mean, var = predict_with_samples(
    model, samples, x_grid, apply_softmax=False, device=DEVICE
)
mean = mean.cpu()
std = var.sqrt().cpu()

lower = mean - 1.96 * std
upper = mean + 1.96 * std

plt.figure(figsize=(7, 4))
plt.scatter(x_train.numpy(), y_train.numpy(), s=14, alpha=0.5, label="Noisy samples")
plt.plot(
    x_grid.numpy(),
    y_grid_true.numpy(),
    color="black",
    linewidth=2,
    label="True deflection",
)
plt.plot(x_grid.numpy(), mean.numpy(), color="tab:blue", label="SGLD mean")
plt.fill_between(
    x_grid.squeeze().numpy(),
    lower.squeeze().numpy(),
    upper.squeeze().numpy(),
    alpha=0.2,
    color="tab:blue",
    label="95% interval",
)
plt.xlabel("Position x (m)")
plt.ylabel("Deflection y (mm)")
plt.title("SGLD confidence bounds")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
