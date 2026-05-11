import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import DeepEnsembleWrapper, LaplaceWrapper
from deepuq.models import GraphNeuralOperator2D


def _make_graph_field_dataset(
    n_samples: int = 20,
    resolution: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(0.0, 1.0, resolution)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    generator = torch.Generator().manual_seed(11)
    fields = []
    targets = []
    for _ in range(n_samples):
        a = 0.6 + 0.3 * torch.rand(1, generator=generator)
        b = 0.04 + 0.02 * torch.rand(1, generator=generator)
        field_a = 1.0 - 0.2 * torch.exp(
            -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.02 + b)
        )
        field_b = 0.25 * torch.exp(-((xx - 0.35) ** 2 + (yy - 0.65) ** 2) / (0.01 + b))
        field_b = field_b + 0.02 * torch.randn(
            (resolution, resolution), generator=generator
        )
        current = torch.stack((field_a, field_b.clamp_min(0.0)), dim=-1)
        lap_a = (
            torch.roll(current[..., 0], 1, 0)
            + torch.roll(current[..., 0], -1, 0)
            + torch.roll(current[..., 0], 1, 1)
            + torch.roll(current[..., 0], -1, 1)
            - 4.0 * current[..., 0]
        )
        lap_b = (
            torch.roll(current[..., 1], 1, 0)
            + torch.roll(current[..., 1], -1, 0)
            + torch.roll(current[..., 1], 1, 1)
            + torch.roll(current[..., 1], -1, 1)
            - 4.0 * current[..., 1]
        )
        next_a = (
            current[..., 0] + b * lap_a - a * current[..., 0] * current[..., 1] ** 2
        )
        next_b = (
            current[..., 1]
            + 0.5 * b * lap_b
            + a * current[..., 0] * current[..., 1] ** 2
        )
        target = torch.stack((next_a, next_b), dim=-1)
        fields.append(current)
        targets.append(target)
    return torch.stack(fields), torch.stack(targets)


def test_graph_operator_deep_ensemble_predict_uq() -> None:
    x, y = _make_graph_field_dataset(n_samples=12, resolution=8)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    models = [
        GraphNeuralOperator2D(
            in_channels=2, hidden_dim=10, message_dim=8, n_message_passing_steps=2
        )
        for _ in range(2)
    ]
    ensemble = DeepEnsembleWrapper(models)
    ensemble.fit(loader, epochs=3, lr=1e-3)
    uq = ensemble.predict_uq(x[:3])
    assert uq.mean.shape == (3, 8, 8, 2)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (3, 8, 8, 2)
    assert torch.isfinite(uq.mean).all()
    assert torch.isfinite(uq.epistemic_var).all()


def test_graph_operator_last_layer_laplace() -> None:
    x, y = _make_graph_field_dataset(n_samples=16, resolution=8)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    model = GraphNeuralOperator2D(
        in_channels=2,
        hidden_dim=8,
        message_dim=8,
        n_message_passing_steps=1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(4):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="block_diag",
        subset_of_weights="last_layer",
    )
    wrapper.fit(loader, prior_precision=50.0)
    uq = wrapper.predict_uq(x[:2], n_samples=4)
    assert uq.mean.shape == (2, 8, 8, 2)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (2, 8, 8, 2)
    assert torch.isfinite(uq.mean).all()
    assert torch.isfinite(uq.epistemic_var).all()
