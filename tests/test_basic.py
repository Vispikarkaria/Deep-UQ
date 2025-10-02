import torch
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper, BayesByBackpropMLP, vi_elbo_step

def test_mc_dropout_shapes():
    model = MLP(16, [8], 4, p_drop=0.5)
    uq = MCDropoutWrapper(model, n_mc=5)
    x = torch.randn(2,16)
    mean, var = uq.predict(x)
    assert mean.shape == (2,4) and var.shape == (2,4)

def test_vi_step_runs():
    model = BayesByBackpropMLP(8, [8], 3)
    x = torch.randn(5,8); y = torch.tensor([0,1,2,1,0])
    loss, nll, kl = vi_elbo_step(model, x, y, n_batches=1)
    assert torch.isfinite(loss)
