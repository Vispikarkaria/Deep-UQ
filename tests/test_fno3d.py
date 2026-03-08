import torch

from deepuq.models import FNO3D, SpectralConv3D


def test_spectral_conv3d_preserves_spatial_shape():
    layer = SpectralConv3D(in_channels=4, out_channels=6, modes=(3, 3, 3))
    x = torch.randn(2, 4, 8, 8, 8)
    y = layer(x)
    assert y.shape == (2, 6, 8, 8, 8)
    assert torch.isfinite(y).all()


def test_fno3d_forward_shape():
    model = FNO3D(in_channels=1, width=12, modes=(3, 3, 3), n_blocks=3)
    x = torch.randn(2, 8, 8, 8, 1)
    y = model(x)
    assert y.shape == (2, 8, 8, 8)
    assert torch.isfinite(y).all()


def test_fno3d_is_coordinate_sensitive():
    model = FNO3D(in_channels=1, width=8, modes=(2, 2, 2), n_blocks=2)
    zeros = torch.zeros(1, 6, 6, 6, 1)
    out = model(zeros)
    assert out.std() > 0.0
