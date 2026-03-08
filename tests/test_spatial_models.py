import torch

from deepuq.methods import MCDropoutWrapper
from deepuq.models import CNNRegressor2D, ResNetRegressor2D, UNet2D, UNet3D


def test_cnn_regressor2d_shape():
    model = CNNRegressor2D(in_channels=2, out_channels=1, hidden_channels=(8, 12))
    x = torch.randn(3, 2, 16, 16)
    y = model(x)
    assert y.shape == (3, 1, 16, 16)


def test_resnet_regressor2d_shape():
    model = ResNetRegressor2D(in_channels=1, out_channels=2, width=12, n_blocks=3)
    x = torch.randn(2, 1, 20, 20)
    y = model(x)
    assert y.shape == (2, 2, 20, 20)


def test_unet2d_shape():
    model = UNet2D(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)


def test_unet3d_shape():
    model = UNet3D(in_channels=1, out_channels=2, base_channels=4)
    x = torch.randn(1, 1, 16, 16, 16)
    y = model(x)
    assert y.shape == (1, 2, 16, 16, 16)


def test_mc_dropout_smoke_on_cnn_and_unet():
    cnn = CNNRegressor2D(
        in_channels=1, out_channels=1, hidden_channels=(8, 8), dropout_p=0.1
    )
    unet = UNet2D(in_channels=1, out_channels=1, base_channels=8, dropout_p=0.1)
    x = torch.randn(2, 1, 16, 16)

    cnn_uq = MCDropoutWrapper(cnn, n_mc=4, apply_softmax=False).predict_uq(x)
    unet_uq = MCDropoutWrapper(unet, n_mc=4, apply_softmax=False).predict_uq(x)

    assert cnn_uq.mean.shape == (2, 1, 16, 16)
    assert cnn_uq.total_var is not None and cnn_uq.total_var.shape == (2, 1, 16, 16)
    assert unet_uq.mean.shape == (2, 1, 16, 16)
    assert unet_uq.total_var is not None and unet_uq.total_var.shape == (2, 1, 16, 16)
