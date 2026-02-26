import torch

from deepuq.models import SparseGaussianProcessRegressor


def test_sparse_gp_shapes_and_history():
    x = torch.linspace(-1, 1, 40).unsqueeze(-1)
    y = torch.cos(2 * torch.pi * x)

    gp = SparseGaussianProcessRegressor(
        num_inducing=8,
        learning_rate=1e-1,
        num_iterations=10,
        verbose=False,
    )
    gp.fit(x, y)

    mean, var = gp.predict(x[:5])

    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert len(gp.elbo_history) == gp.num_iterations
