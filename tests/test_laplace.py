import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import deepuq.methods.laplace as laplace_module
from deepuq.methods import LaplaceWrapper
from deepuq.models import MLP


def _make_regression_loader(input_dim: int, n_samples: int = 64, batch_size: int = 16):
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(n_samples, input_dim, generator=gen)
    w = torch.linspace(-0.5, 0.5, input_dim).unsqueeze(-1)
    y = x @ w + 0.1 * torch.randn(n_samples, 1, generator=gen)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _make_classification_loader(
    input_dim: int, n_classes: int = 3, n_samples: int = 72, batch_size: int = 18
):
    gen = torch.Generator().manual_seed(321)
    x = torch.randn(n_samples, input_dim, generator=gen)
    logits = torch.randn(input_dim, n_classes, generator=gen)
    y = torch.argmax(x @ logits, dim=1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _fit_simple_regression_model(input_dim: int = 4):
    model = MLP(input_dim=input_dim, hidden_dims=[12], output_dim=1, p_drop=0.0)
    loader = _make_regression_loader(input_dim=input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    return model, loader


def _fit_simple_classification_model(input_dim: int = 5, n_classes: int = 3):
    model = MLP(input_dim=input_dim, hidden_dims=[10], output_dim=n_classes, p_drop=0.0)
    loader = _make_classification_loader(input_dim=input_dim, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(15):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model, loader


def test_supported_hessian_structures_and_validation():
    expected = ("diag", "fisher_diag", "lowrank_diag", "block_diag", "kron", "full")
    assert LaplaceWrapper.supported_hessian_structures() == expected

    model = MLP(4, [8], 1)
    with pytest.raises(ValueError, match="Unsupported hessian_structure"):
        LaplaceWrapper(
            model, likelihood="regression", hessian_structure="not_a_structure"
        )


@pytest.mark.parametrize("structure", LaplaceWrapper.supported_hessian_structures())
def test_regression_predictive_outputs_are_finite(structure):
    model, train_loader = _fit_simple_regression_model(input_dim=4)
    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure=structure,
        subset_of_weights="last_layer",
    )

    wrapper.fit(train_loader, prior_precision=1.0)
    x_query = torch.randn(9, 4)
    mean, var = wrapper.predict(x_query, n_samples=10)

    assert mean.shape == (9, 1)
    assert var is not None
    assert var.shape == (9, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()
    assert (var >= 0).all()


@pytest.mark.parametrize("structure", LaplaceWrapper.supported_hessian_structures())
def test_classification_predictive_contract(structure):
    model, train_loader = _fit_simple_classification_model(input_dim=5, n_classes=3)
    wrapper = LaplaceWrapper(
        model,
        likelihood="classification",
        hessian_structure=structure,
        subset_of_weights="last_layer",
    )

    wrapper.fit(train_loader, prior_precision=1.0)
    x_query = torch.randn(11, 5)
    mean_probs, var = wrapper.predict(x_query, n_samples=8)

    assert var is None
    assert mean_probs.shape == (11, 3)
    assert torch.isfinite(mean_probs).all()
    assert ((mean_probs >= 0.0) & (mean_probs <= 1.0)).all()
    row_sums = mean_probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_subset_combinations_work_for_native_structures():
    model, train_loader = _fit_simple_regression_model(input_dim=6)
    for subset in ("last_layer", "all"):
        for structure in (
            "diag",
            "fisher_diag",
            "lowrank_diag",
            "block_diag",
            "kron",
            "full",
        ):
            wrapper = LaplaceWrapper(
                model,
                likelihood="regression",
                hessian_structure=structure,
                subset_of_weights=subset,
            )
            wrapper.fit(train_loader, prior_precision=1.0)
            mean, var = wrapper.predict(torch.randn(5, 6), n_samples=6)
            assert mean.shape == (5, 1)
            assert var is not None
            assert var.shape == (5, 1)


def test_full_all_parameter_guardrail_triggers_before_backend():
    model = MLP(input_dim=256, hidden_dims=[256, 256], output_dim=1, p_drop=0.0)
    train_loader = _make_regression_loader(input_dim=256, n_samples=32, batch_size=8)

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="full",
        subset_of_weights="all",
        full_max_params=1000,
    )
    with pytest.raises(ValueError, match="full_max_params"):
        wrapper.fit(train_loader, prior_precision=1.0)


def test_lowrank_rank_clipping_and_fallback_path():
    model, train_loader = _fit_simple_regression_model(input_dim=3)

    wrapper = LaplaceWrapper(
        model,
        likelihood="regression",
        hessian_structure="lowrank_diag",
        subset_of_weights="last_layer",
        lowrank_rank=999,
    )
    wrapper.fit(train_loader, prior_precision=1.0)
    mean, var = wrapper.predict(torch.randn(4, 3), n_samples=7)

    assert mean.shape == (4, 1)
    assert var is not None
    assert torch.isfinite(mean).all()
    assert torch.isfinite(var).all()


def test_backward_compatible_diag_path():
    model, train_loader = _fit_simple_regression_model(input_dim=4)
    wrapper = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag")

    wrapper.fit(train_loader, prior_precision=1.0)
    mean, var = wrapper.predict(torch.randn(6, 4), n_samples=12)

    assert mean.shape == (6, 1)
    assert var is not None
    assert var.shape == (6, 1)


def test_full_backend_uses_native():
    model = MLP(4, [8], 1)
    wrapper = LaplaceWrapper(model, likelihood="regression", hessian_structure="full")
    backend = wrapper._build_backend()
    assert isinstance(backend, laplace_module._FullLaplace)


def test_kron_backend_uses_native():
    # Single-output last-layer falls back to block_diag (kron degenerates)
    model = MLP(4, [8], 1)
    wrapper = LaplaceWrapper(model, likelihood="regression", hessian_structure="kron")
    backend = wrapper._build_backend()
    assert isinstance(backend, laplace_module._BlockDiagonalLaplace)

    # Multi-output uses true KronLaplace
    model_multi = MLP(4, [8], 3)
    wrapper_multi = LaplaceWrapper(
        model_multi, likelihood="classification", hessian_structure="kron"
    )
    backend_multi = wrapper_multi._build_backend()
    assert isinstance(backend_multi, laplace_module._KronLaplace)
