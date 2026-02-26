# Beam deflection regression example (Euler-Bernoulli)
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from deepuq.methods import BayesByBackpropMLP, vi_elbo_step

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Beam parameters (SI units)
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


def ema(values, alpha=0.2):
    smoothed = []
    for value in values:
        if not smoothed:
            smoothed.append(float(value))
        else:
            smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def evaluate_elbo(model, loader, criterion, kl_weight, num_batches, mc_samples=8):
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_items = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            loss, nll, kl = vi_elbo_step(
                model,
                x_batch,
                y_batch,
                num_batches=num_batches,
                criterion=criterion,
                kl_weight=kl_weight,
                mc_samples=mc_samples,
            )
            items = y_batch.numel()
            total_loss += loss.item() * items
            total_nll += nll.item() * items
            total_kl += kl.item() * items
            total_items += items
    return (
        total_loss / total_items,
        total_nll / total_items,
        total_kl / total_items,
    )


torch.manual_seed(13)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(13)

x_train, y_train, _ = make_dataset(512, noise_std=0.05, seed=13)
x_val, y_val, _ = make_dataset(192, noise_std=0.05, seed=31)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=128, shuffle=False)

x_grid = torch.linspace(0, L, 200).unsqueeze(-1)
y_grid_true = beam_deflection_mm(x_grid)

model = BayesByBackpropMLP(1, [128, 128], 1, prior_sigma=0.2).to(DEVICE)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

n_epochs = 180
num_batches = len(train_loader)
kl_weight = 0.01

history = {
    'epoch': [],
    'train_elbo': [],
    'val_elbo': [],
    'train_nll': [],
    'val_nll': [],
    'train_kl': [],
    'val_kl': [],
}

for epoch in range(n_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = vi_elbo_step(
            model,
            x_batch,
            y_batch,
            num_batches=num_batches,
            criterion=criterion,
            kl_weight=kl_weight,
            mc_samples=1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_elbo, train_nll, train_kl = evaluate_elbo(
        model,
        train_loader,
        criterion=criterion,
        kl_weight=kl_weight,
        num_batches=num_batches,
        mc_samples=8,
    )
    val_elbo, val_nll, val_kl = evaluate_elbo(
        model,
        val_loader,
        criterion=criterion,
        kl_weight=kl_weight,
        num_batches=num_batches,
        mc_samples=8,
    )

    history['epoch'].append(epoch + 1)
    history['train_elbo'].append(train_elbo)
    history['val_elbo'].append(val_elbo)
    history['train_nll'].append(train_nll)
    history['val_nll'].append(val_nll)
    history['train_kl'].append(train_kl)
    history['val_kl'].append(val_kl)

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"train_elbo={train_elbo:.4f} val_elbo={val_elbo:.4f} | "
            f"train_nll={train_nll:.4f} train_kl={train_kl:.4f}"
        )

# Posterior predictive (unchanged behavior: sample multiple weight draws)
model.eval()
with torch.no_grad():
    samples = []
    for _ in range(200):
        samples.append(model(x_grid.to(DEVICE), sample=True).cpu())
    samples = torch.stack(samples)
    mean = samples.mean(dim=0)
    std = samples.var(dim=0, unbiased=False).sqrt()

lower = mean - 1.96 * std
upper = mean + 1.96 * std

plt.figure(figsize=(7, 4))
plt.scatter(x_train.numpy(), y_train.numpy(), s=14, alpha=0.5, label='Noisy train samples')
plt.plot(x_grid.numpy(), y_grid_true.numpy(), color='black', linewidth=2, label='True deflection')
plt.plot(x_grid.numpy(), mean.numpy(), color='tab:blue', label='Posterior mean')
plt.fill_between(
    x_grid.squeeze().numpy(),
    lower.squeeze().numpy(),
    upper.squeeze().numpy(),
    alpha=0.2,
    color='tab:blue',
    label='95% interval',
)
plt.xlabel('Position x (m)')
plt.ylabel('Deflection y (mm)')
plt.title('Bayes-by-Backprop predictive interval')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
train_ema = ema(history['train_elbo'], alpha=0.2)
val_ema = ema(history['val_elbo'], alpha=0.2)
plt.plot(history['epoch'], history['train_elbo'], color='tab:blue', alpha=0.25, label='Train ELBO (raw)')
plt.plot(history['epoch'], history['val_elbo'], color='tab:orange', alpha=0.25, label='Val ELBO (raw)')
plt.plot(history['epoch'], train_ema, color='tab:blue', label='Train ELBO (EMA)')
plt.plot(history['epoch'], val_ema, color='tab:orange', label='Val ELBO (EMA)')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.title('ELBO trend (fixed objective)')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
