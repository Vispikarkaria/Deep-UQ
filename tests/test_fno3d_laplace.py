import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import LaplaceWrapper
from deepuq.models import FNO3D


def _make_heat_dataset(
    n_samples: int = 24,
    resolution: int = 8,
    kappa: float = 0.02,
    final_time: float = 0.05,
):
    grid = torch.arange(resolution, dtype=torch.float32) / resolution
    xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing="ij")
    freqs = torch.fft.fftfreq(resolution, d=1.0 / resolution)
    kx, ky, kz = torch.meshgrid(freqs, freqs, freqs, indexing="ij")
    decay = torch.exp(
        -4.0 * math.pi * math.pi * kappa * final_time * (kx**2 + ky**2 + kz**2)
    )
    generator = torch.Generator().manual_seed(11)
    fields = []
    targets = []
    for _ in range(n_samples):
        field = torch.zeros_like(xx)
        for fx in range(1, 3):
            for fy in range(1, 3):
                for fz in range(1, 3):
                    amp = torch.randn((), generator=generator) / (fx * fy * fz) ** 1.5
                    phase = 2.0 * math.pi * torch.rand((), generator=generator)
                    field = field + amp * torch.sin(
                        2.0 * math.pi * (fx * xx + fy * yy + fz * zz) + phase
                    )
        field_ft = torch.fft.fftn(field)
        target = torch.fft.ifftn(field_ft * decay).real
        fields.append(field.unsqueeze(-1))
        targets.append(target)
    return torch.stack(fields), torch.stack(targets)


def test_fno3d_last_layer_block_diag_laplace():
    x, y = _make_heat_dataset(n_samples=18, resolution=8)
    loader = DataLoader(TensorDataset(x, y), batch_size=6, shuffle=True)

    model = FNO3D(
        in_channels=1,
        width=4,
        modes=(3, 3, 3),
        n_blocks=1,
        head_hidden_dim=8,
        use_coordinate_features=False,
        use_nonlinearity=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(8):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="block_diag",
        subset_of_weights="last_layer",
    )
    wrapper.fit(loader, prior_precision=5.0)

    x_eval = x[:3]
    mean, var = wrapper.predict(x_eval, n_samples=4)
    uq = wrapper.predict_uq(x_eval, n_samples=4)

    assert mean.shape == (3, 8, 8, 8)
    assert var is not None and var.shape == (3, 8, 8, 8)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert uq.mean.shape == (3, 8, 8, 8)
    assert uq.total_var is not None and uq.total_var.shape == (3, 8, 8, 8)
