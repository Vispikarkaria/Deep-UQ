import torch

from deepuq.models import ConditionalUNet2D, SinusoidalTimeEmbedding


def test_sinusoidal_time_embedding_shape():
    module = SinusoidalTimeEmbedding(embedding_dim=32)
    timesteps = torch.tensor([0, 1, 7, 12])
    embedding = module(timesteps)
    assert embedding.shape == (4, 32)
    assert torch.isfinite(embedding).all()


def test_conditional_unet2d_forward_shape():
    model = ConditionalUNet2D(
        x_channels=1,
        cond_channels=3,
        base_channels=8,
        time_dim=32,
    )
    x_t = torch.randn(2, 1, 16, 16)
    timesteps = torch.tensor([3, 9])
    condition = torch.randn(2, 3, 16, 16)
    output = model(x_t, timesteps, condition)
    assert output.shape == (2, 1, 16, 16)
    assert torch.isfinite(output).all()


def test_conditional_unet2d_uses_condition_channels():
    model = ConditionalUNet2D(
        x_channels=1,
        cond_channels=2,
        base_channels=8,
        time_dim=32,
    )
    x_t = torch.zeros(1, 1, 16, 16)
    timesteps = torch.tensor([5])
    condition_a = torch.zeros(1, 2, 16, 16)
    condition_b = torch.ones(1, 2, 16, 16)
    out_a = model(x_t, timesteps, condition_a)
    out_b = model(x_t, timesteps, condition_b)
    assert not torch.allclose(out_a, out_b)
