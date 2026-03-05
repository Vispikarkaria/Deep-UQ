import torch

from deepuq.models import MultiTaskGaussianProcessRegressor


def test_multitask_gp_shapes_and_variance() -> None:
    torch.manual_seed(0)
    x = torch.linspace(-1.5, 1.5, 24).unsqueeze(-1)
    y1 = torch.sin(1.1 * x)
    y2 = 0.4 * torch.cos(0.8 * x)
    y = torch.cat([y1, y2], dim=1)

    model = MultiTaskGaussianProcessRegressor(num_tasks=2, opt_steps=30, lr=0.05)
    model.fit(x, y)

    x_test = torch.linspace(-2.0, 2.0, 18).unsqueeze(-1)
    uq = model.predict_uq(x_test)

    assert uq.mean.shape == (18, 2)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (18, 2)
    assert uq.total_var is not None and uq.total_var.shape == (18, 2)
    assert torch.isfinite(uq.total_var).all()
