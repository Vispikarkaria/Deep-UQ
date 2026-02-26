import torch

from deepuq.models import GaussianProcessRegressor


def test_gp_predict_shapes():
    x = torch.linspace(0, 1, 8).unsqueeze(-1)
    y = torch.sin(2 * torch.pi * x)

    gp = GaussianProcessRegressor(noise=1e-5)
    gp.fit(x, y)

    mean, var = gp.predict(x)

    assert mean.shape == (x.shape[0],)
    assert var.shape == (x.shape[0],)

    samples = gp.posterior_samples(x, n_samples=4)
    assert samples.shape == (4, x.shape[0])
