"""Example: standardized UQ output from Laplace."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import LaplaceWrapper
from deepuq.models import MLP


def main() -> None:
    x = torch.linspace(-2.0, 2.0, 128).unsqueeze(-1)
    y = 0.4 * x + torch.sin(1.2 * x) + 0.05 * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = MLP(1, [64, 64], 1, p_drop=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

    la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")
    la.fit(loader, prior_precision=1.0)
    uq = la.predict_uq(x, n_samples=80)
    print("mean:", uq.mean.shape, "total_var:", uq.total_var.shape if uq.total_var is not None else None)


if __name__ == "__main__":
    main()
