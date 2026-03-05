import torch

from deepuq.models import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RationalQuadraticKernel,
    SpectralMixtureKernel,
)


def test_kernel_shapes_and_finiteness() -> None:
    x = torch.randn(8, 3)
    y = torch.randn(5, 3)

    kernels = [
        RBFKernel(lengthscale=1.0, outputscale=1.2),
        RBFKernel(lengthscale=torch.tensor([1.0, 0.7, 1.4]), outputscale=0.9),
        MaternKernel(lengthscale=0.8, outputscale=1.1, nu=1.5),
        RationalQuadraticKernel(lengthscale=0.9, outputscale=1.0, alpha=0.8),
        PeriodicKernel(lengthscale=1.1, outputscale=0.7, period=2.0),
        LinearKernel(variance=0.5, bias=0.1),
        SpectralMixtureKernel(
            weights=torch.tensor([0.6, 0.4]),
            means=torch.tensor([[0.2, 0.3, 0.1], [0.4, 0.1, 0.2]]),
            scales=torch.tensor([[0.8, 0.9, 0.6], [0.3, 0.2, 0.4]]),
        ),
    ]

    for kernel in kernels:
        cov = kernel(x, y)
        assert cov.shape == (8, 5)
        assert torch.isfinite(cov).all()


def test_kernel_composition_shapes() -> None:
    x = torch.randn(6, 2)
    k1 = RBFKernel(lengthscale=1.0, outputscale=1.0)
    k2 = PeriodicKernel(lengthscale=0.9, outputscale=0.8, period=2.0)
    k3 = LinearKernel(variance=0.2)

    cov_sum = (k1 + k2)(x, x)
    cov_prod = (k2 * k3)(x, x)
    assert cov_sum.shape == (6, 6)
    assert cov_prod.shape == (6, 6)
    assert torch.isfinite(cov_sum).all()
    assert torch.isfinite(cov_prod).all()
