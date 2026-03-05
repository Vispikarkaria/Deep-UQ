import torch

from deepuq.models import DeepKernelGaussianProcessRegressor


def test_deep_kernel_gp_predict_uq() -> None:
    torch.manual_seed(0)
    z = torch.linspace(-1.2, 1.2, 36).unsqueeze(-1)
    x = torch.cat([z, z**2, torch.sin(2 * z)], dim=1)
    y = torch.sin(1.4 * z) + 0.05 * torch.randn_like(z)

    model = DeepKernelGaussianProcessRegressor(
        feature_dim=8,
        hidden_dims=(16, 16),
        epochs=30,
        lr=1e-3,
    )
    model.fit(x, y)

    z_test = torch.linspace(-1.8, 1.8, 20).unsqueeze(-1)
    x_test = torch.cat([z_test, z_test**2, torch.sin(2 * z_test)], dim=1)
    uq = model.predict_uq(x_test)

    assert uq.mean.shape == (20,)
    assert uq.total_var is not None and uq.total_var.shape == (20,)
    assert torch.isfinite(uq.total_var).all()
