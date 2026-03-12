import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import LaplaceWrapper
from deepuq.models import FNO2D


def _make_darcy_like_dataset(
    n_samples: int = 12,
    resolution: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.arange(resolution, dtype=torch.float32) / max(resolution - 1, 1)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    generator = torch.Generator().manual_seed(17)
    fields = []
    targets = []
    for _ in range(n_samples):
        log_k = 0.15 * torch.sin(2.0 * math.pi * xx) + 0.1 * torch.cos(
            2.0 * math.pi * yy
        )
        log_k = log_k + 0.05 * torch.randn(
            (resolution, resolution), generator=generator
        )
        k = torch.exp(log_k).clamp(0.6, 2.0)
        q = 0.2 * torch.sin(2.0 * math.pi * (xx + yy))
        q = q + 0.05 * torch.randn((resolution, resolution), generator=generator)
        g = torch.zeros_like(q)
        g[0, :] = 0.05 * torch.sin(2.0 * math.pi * yy[0, :])
        g[-1, :] = 0.05 * torch.cos(2.0 * math.pi * yy[-1, :])
        g[:, 0] = g[:, 0] + 0.05 * torch.sin(2.0 * math.pi * xx[:, 0])
        g[:, -1] = g[:, -1] + 0.05 * torch.cos(2.0 * math.pi * xx[:, -1])
        target = 0.6 * q + 0.25 * torch.log(k) + 0.5 * g
        target = target + 0.1 * (
            torch.roll(q, 1, 0)
            + torch.roll(q, -1, 0)
            + torch.roll(q, 1, 1)
            + torch.roll(q, -1, 1)
        )
        inputs = torch.stack((k, q, g), dim=-1)
        fields.append(inputs)
        targets.append(target)
    return torch.stack(fields), torch.stack(targets)


def test_fno2d_last_layer_block_diag_laplace():
    x, y = _make_darcy_like_dataset(n_samples=12, resolution=8)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)

    model = FNO2D(
        in_channels=3,
        width=4,
        modes=(3, 3),
        n_blocks=1,
        head_hidden_dim=8,
        use_coordinate_features=False,
        use_nonlinearity=False,
        use_block_norm=False,
    )
    opt = torch.optim.RMSprop(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(4):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="block_diag",
        subset_of_weights="last_layer",
    )
    wrapper.fit(loader, prior_precision=50.0)

    x_eval = x[:3]
    mean, var = wrapper.predict(x_eval, n_samples=4)
    uq = wrapper.predict_uq(x_eval, n_samples=4)

    assert mean.shape == (3, 8, 8)
    assert var is not None and var.shape == (3, 8, 8)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert uq.mean.shape == (3, 8, 8)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (3, 8, 8)
