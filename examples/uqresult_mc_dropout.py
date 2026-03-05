"""Example: standardized UQ output from MC Dropout."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import MCDropoutWrapper
from deepuq.models import MLP


def main() -> None:
    x = torch.linspace(-2.0, 2.0, 128).unsqueeze(-1)
    y = torch.sin(2.0 * x) + 0.08 * torch.randn_like(x)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = MLP(1, [64, 64], 1, p_drop=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

    uq_model = MCDropoutWrapper(model, n_mc=100, apply_softmax=False)
    uq = uq_model.predict_uq(x)
    print(
        "mean:",
        uq.mean.shape,
        "epistemic:",
        uq.epistemic_var.shape if uq.epistemic_var is not None else None,
    )


if __name__ == "__main__":
    main()
