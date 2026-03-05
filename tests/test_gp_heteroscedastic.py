import torch

from deepuq.models import HeteroscedasticGaussianProcessRegressor


def test_heteroscedastic_gp_predict_uq_fields() -> None:
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 64).unsqueeze(-1)
    y_clean = torch.sin(1.2 * x)
    noise = 0.04 + 0.03 * torch.abs(x)
    y = y_clean + noise * torch.randn_like(x)

    model = HeteroscedasticGaussianProcessRegressor(num_alternations=3)
    model.fit(x, y)

    x_test = torch.linspace(-3.0, 3.0, 40).unsqueeze(-1)
    uq = model.predict_uq(x_test)

    assert uq.mean.shape == (40,)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (40,)
    assert uq.aleatoric_var is not None and uq.aleatoric_var.shape == (40,)
    assert uq.total_var is not None and uq.total_var.shape == (40,)
    assert torch.all(uq.total_var >= uq.epistemic_var)
