import torch

from deepuq.models import SpectralMixtureGaussianProcessRegressor


def test_spectral_mixture_gp_regression_uq() -> None:
    torch.manual_seed(0)
    x = torch.linspace(-1.5, 1.5, 48).unsqueeze(-1)
    y = torch.sin(2.0 * x) + 0.25 * torch.sin(5.0 * x)
    y = y + 0.06 * torch.randn_like(x)

    model = SpectralMixtureGaussianProcessRegressor(
        num_mixtures=3, opt_steps=40, lr=0.03
    )
    model.fit(x, y)

    x_test = torch.linspace(-2.0, 2.0, 32).unsqueeze(-1)
    uq = model.predict_uq(x_test)

    assert uq.mean.shape == (32,)
    assert uq.total_var is not None and uq.total_var.shape == (32,)
    assert torch.isfinite(uq.total_var).all()
