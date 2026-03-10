import torch

from deepuq.models import FNO2D, SpectralConv2D


def test_spectral_conv2d_preserves_spatial_shape():
    layer = SpectralConv2D(in_channels=3, out_channels=5, modes=(3, 3))
    x = torch.randn(2, 3, 8, 8)
    y = layer(x)
    assert y.shape == (2, 5, 8, 8)
    assert torch.isfinite(y).all()


def test_fno2d_forward_shape():
    model = FNO2D(in_channels=3, width=8, modes=(3, 3), n_blocks=2)
    x = torch.randn(2, 12, 12, 3)
    y = model(x)
    assert y.shape == (2, 12, 12)
    assert torch.isfinite(y).all()


def test_fno2d_coordinate_features_break_symmetry():
    model = FNO2D(
        in_channels=3,
        width=8,
        modes=(2, 2),
        n_blocks=1,
        head_hidden_dim=8,
        use_coordinate_features=True,
    )
    zeros = torch.zeros(1, 10, 10, 3)
    out = model(zeros)
    assert out.std() > 0.0
