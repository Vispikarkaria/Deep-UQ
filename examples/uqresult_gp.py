"""Example: standardized UQ output from Gaussian Process models."""

import torch

from deepuq.models import GaussianProcessRegressor, SparseGaussianProcessRegressor


def main() -> None:
    x = torch.linspace(-1.0, 1.0, 96).unsqueeze(-1)
    y = torch.sin(2.5 * x) + 0.05 * torch.randn_like(x)
    x_star = torch.linspace(-1.5, 1.5, 120).unsqueeze(-1)

    exact = GaussianProcessRegressor(noise=1e-4)
    exact.fit(x, y)
    exact_uq = exact.predict_uq(x_star)
    print(
        "exact_gp:",
        exact_uq.mean.shape,
        exact_uq.total_var.shape if exact_uq.total_var is not None else None,
    )

    sparse = SparseGaussianProcessRegressor(num_inducing=24, num_iterations=200)
    sparse.fit(x, y)
    sparse_uq = sparse.predict_uq(x_star)
    print(
        "sparse_gp:",
        sparse_uq.mean.shape,
        sparse_uq.total_var.shape if sparse_uq.total_var is not None else None,
    )


if __name__ == "__main__":
    main()
