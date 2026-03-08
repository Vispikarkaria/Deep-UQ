"""Gaussian process model families and kernels."""

from .classification import (
    GaussianProcessClassifier,
    OneVsRestGaussianProcessClassifier,
)
from .deep_kernel import DeepKernelGaussianProcessRegressor
from .exact_regression import GaussianProcessRegressor
from .heteroscedastic import HeteroscedasticGaussianProcessRegressor
from .kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    RBFKernel,
    SpectralMixtureKernel,
    SumKernel,
)
from .multitask_icm import MultiTaskGaussianProcessRegressor
from .sparse_regression import SparseGaussianProcessRegressor
from .spectral_mixture import SpectralMixtureGaussianProcessRegressor

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
