"""Public Gaussian process APIs for Deep-UQ.

This module preserves historical import paths while delegating implementations
into ``deepuq.models.gp``.
"""

from .gp import (
    DeepKernelGaussianProcessRegressor,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    HeteroscedasticGaussianProcessRegressor,
    Kernel,
    LinearKernel,
    MaternKernel,
    MultiTaskGaussianProcessRegressor,
    OneVsRestGaussianProcessClassifier,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    RBFKernel,
    SparseGaussianProcessRegressor,
    SpectralMixtureGaussianProcessRegressor,
    SpectralMixtureKernel,
    SumKernel,
)

__all__ = [
    "Kernel",
    "RBFKernel",
    "MaternKernel",
    "RationalQuadraticKernel",
    "PeriodicKernel",
    "LinearKernel",
    "SpectralMixtureKernel",
    "SumKernel",
    "ProductKernel",
    "GaussianProcessRegressor",
    "SparseGaussianProcessRegressor",
    "GaussianProcessClassifier",
    "OneVsRestGaussianProcessClassifier",
    "HeteroscedasticGaussianProcessRegressor",
    "MultiTaskGaussianProcessRegressor",
    "SpectralMixtureGaussianProcessRegressor",
    "DeepKernelGaussianProcessRegressor",
]
