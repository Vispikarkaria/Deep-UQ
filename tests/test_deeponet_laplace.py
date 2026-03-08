import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import LaplaceWrapper
from deepuq.models import DeepONet2D


def _make_operator_dataset(
    n_samples: int = 48,
    branch_input_dim: int = 16,
    n_query: int = 9,
):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(n_samples, branch_input_dim, generator=gen)
    weight = torch.randn(branch_input_dim, n_query, generator=gen)
    y = torch.tanh(x @ weight)
    return x, y


def test_last_layer_laplace_fits_deeponet_with_fixed_grid():
    query_grid = torch.rand(9, 2)
    model = DeepONet2D(
        branch_input_dim=16,
        latent_dim=10,
        hidden_dim=20,
        depth=3,
        query_grid=query_grid,
    )

    x, y = _make_operator_dataset(n_samples=48, branch_input_dim=16, n_query=9)
    loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    wrapper.fit(loader, prior_precision=1.0)

    mean, var = wrapper.predict(torch.randn(4, 16), n_samples=6)
    uq = wrapper.predict_uq(torch.randn(4, 16), n_samples=6)

    assert mean.shape == (4, 9)
    assert var is not None and var.shape == (4, 9)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert uq.mean.shape == (4, 9)
    assert uq.total_var is not None and uq.total_var.shape == (4, 9)
