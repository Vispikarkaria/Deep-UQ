import pytest
import torch

from deepuq.models import DeepONet1D, DeepONet2D


def test_deeponet1d_forward_uses_fixed_query_grid():
    grid = torch.linspace(0.0, 1.0, 17).unsqueeze(-1)
    model = DeepONet1D(
        branch_input_dim=12,
        latent_dim=10,
        hidden_dim=20,
        depth=3,
        query_grid=grid,
    )

    branch_inputs = torch.randn(4, 12)
    outputs = model(branch_inputs)

    assert outputs.shape == (4, 17)
    assert torch.isfinite(outputs).all()


def test_deeponet1d_predict_on_coords_shape():
    model = DeepONet1D(
        branch_input_dim=8,
        latent_dim=9,
        hidden_dim=16,
        depth=4,
    )

    branch_inputs = torch.randn(3, 8)
    coords = torch.rand(13, 1)
    outputs = model.predict_on_coords(branch_inputs, coords)

    assert outputs.shape == (3, 13)
    assert torch.isfinite(outputs).all()


def test_deeponet_forward_uses_fixed_query_grid():
    grid = torch.rand(20, 2)
    model = DeepONet2D(
        branch_input_dim=16,
        latent_dim=12,
        hidden_dim=24,
        depth=3,
        query_grid=grid,
    )

    branch_inputs = torch.randn(5, 16)
    outputs = model(branch_inputs)

    assert outputs.shape == (5, 20)
    assert torch.isfinite(outputs).all()


def test_deeponet_predict_on_coords_shape():
    model = DeepONet2D(
        branch_input_dim=9,
        latent_dim=10,
        hidden_dim=18,
        depth=4,
    )

    branch_inputs = torch.randn(3, 9)
    coords = torch.rand(11, 2)
    outputs = model.predict_on_coords(branch_inputs, coords)

    assert outputs.shape == (3, 11)
    assert torch.isfinite(outputs).all()


def test_deeponet_requires_query_grid_for_default_forward():
    model = DeepONet2D(branch_input_dim=4, latent_dim=8, hidden_dim=16, depth=3)

    with pytest.raises(RuntimeError, match="query_grid is not set"):
        _ = model(torch.randn(2, 4))
