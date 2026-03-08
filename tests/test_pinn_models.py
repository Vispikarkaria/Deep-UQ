import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import LaplaceWrapper
from deepuq.models import PINN1D, PINN2D


def test_pinn1d_shape_and_gradients():
    model = PINN1D(hidden_dims=(16, 16))
    x = torch.linspace(0.0, 1.0, 8, requires_grad=True).unsqueeze(-1)
    y = model(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    assert y.shape == (8, 1)
    assert grad.shape == x.shape


def test_pinn2d_shape_and_gradients():
    model = PINN2D(hidden_dims=(16, 16))
    coords = torch.rand(10, 2, requires_grad=True)
    y = model(coords)
    grad = torch.autograd.grad(y.sum(), coords, create_graph=True)[0]
    assert y.shape == (10, 1)
    assert grad.shape == coords.shape


def test_laplace_last_layer_smoke_on_pinns():
    coords_1d = torch.linspace(0.0, 1.0, 24).unsqueeze(-1)
    target_1d = torch.sin(torch.pi * coords_1d)
    loader_1d = DataLoader(TensorDataset(coords_1d, target_1d), batch_size=8, shuffle=True)

    model_1d = PINN1D(hidden_dims=(16, 16))
    opt_1d = torch.optim.Adam(model_1d.parameters(), lr=1e-2)
    for _ in range(20):
        for xb, yb in loader_1d:
            opt_1d.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model_1d(xb), yb)
            loss.backward()
            opt_1d.step()

    la_1d = LaplaceWrapper(
        model_1d,
        likelihood="regression",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    la_1d.fit(loader_1d, prior_precision=1.0)
    uq_1d = la_1d.predict_uq(coords_1d[:5], n_samples=8)
    assert uq_1d.mean.shape == (5, 1)
    assert uq_1d.total_var is not None and uq_1d.total_var.shape == (5, 1)

    coords_2d = torch.rand(32, 2)
    target_2d = torch.sin(torch.pi * coords_2d[:, :1]) * torch.cos(torch.pi * coords_2d[:, 1:2])
    loader_2d = DataLoader(TensorDataset(coords_2d, target_2d), batch_size=8, shuffle=True)

    model_2d = PINN2D(hidden_dims=(16, 16))
    opt_2d = torch.optim.Adam(model_2d.parameters(), lr=1e-2)
    for _ in range(20):
        for xb, yb in loader_2d:
            opt_2d.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model_2d(xb), yb)
            loss.backward()
            opt_2d.step()

    la_2d = LaplaceWrapper(
        model_2d,
        likelihood="regression",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    la_2d.fit(loader_2d, prior_precision=1.0)
    uq_2d = la_2d.predict_uq(coords_2d[:6], n_samples=8)
    assert uq_2d.mean.shape == (6, 1)
    assert uq_2d.total_var is not None and uq_2d.total_var.shape == (6, 1)
