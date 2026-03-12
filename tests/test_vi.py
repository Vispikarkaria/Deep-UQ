import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.vi import (
    BayesByBackpropMLP,
    HeteroscedasticBayesByBackpropRegressor,
    HeteroscedasticMultiOutputBayesByBackpropRegressor,
    LastLayerVariationalInference,
    MultiOutputBayesByBackpropRegressor,
    predict_vi_uq,
    vi_elbo_step,
)


def test_vi_elbo_finite_for_classification():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(8, [16], 3)
    x = torch.randn(7, 8)
    y = torch.tensor([0, 1, 2, 1, 0, 2, 1])

    loss, nll, kl = vi_elbo_step(model, x, y, num_batches=4)

    assert loss.ndim == 0
    assert nll.ndim == 0
    assert kl.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_last_layer_classification_vi_elbo_uses_cross_entropy():
    torch.manual_seed(0)
    feature_extractor = _FlatFeatureExtractor(4, 6)
    model = LastLayerVariationalInference(
        feature_extractor,
        feature_dim=6,
        output_dim=3,
        task="classification",
    )
    x = torch.randn(9, 4)
    y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 1])

    loss, nll, kl = vi_elbo_step(model, x, y, num_batches=3)

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_vi_elbo_finite_for_regression():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(4, [12], 1)
    x = torch.randn(10, 4)
    y = torch.randn(10, 1)

    loss, nll, kl = vi_elbo_step(
        model,
        x,
        y,
        num_batches=5,
        criterion=nn.MSELoss(reduction="mean"),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_heteroscedastic_vi_elbo_uses_model_nll():
    torch.manual_seed(0)
    model = HeteroscedasticBayesByBackpropRegressor(3, hidden_dims=(10, 10))
    x = torch.randn(9, 3)
    y = torch.randn(9, 1)

    loss, nll, kl = vi_elbo_step(model, x, y, num_batches=3)

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_multioutput_vi_forward_shape():
    model = MultiOutputBayesByBackpropRegressor(5, output_dim=3, hidden_dims=(12,))
    x = torch.randn(6, 5)
    pred = model(x, sample=True)
    assert pred.shape == (6, 3)


def test_heteroscedastic_multioutput_split_shape():
    model = HeteroscedasticMultiOutputBayesByBackpropRegressor(
        4, output_dim=2, hidden_dims=(8,)
    )
    x = torch.randn(7, 4)
    pred = model(x, sample=True)
    mean, raw_var = model.split_prediction(pred)
    assert mean.shape == (7, 2)
    assert raw_var.shape == (7, 2)
    assert torch.all(model.predictive_variance(raw_var) > 0)


class _FlatFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        return feats.permute(0, 2, 3, 1).contiguous()


def test_last_layer_vi_regression_forward_shape():
    feature_extractor = _FlatFeatureExtractor(3, 6)
    model = LastLayerVariationalInference(
        feature_extractor,
        feature_dim=6,
        output_dim=2,
        task="regression",
    )
    x = torch.randn(5, 3)
    pred = model(x, sample=True)
    assert pred.shape == (5, 2)
    assert torch.isfinite(model.kl())


def test_last_layer_vi_classification_spatial_shape():
    feature_extractor = _SpatialFeatureExtractor(1, 4)
    model = LastLayerVariationalInference(
        feature_extractor,
        feature_dim=4,
        output_dim=3,
        task="classification",
    )
    x = torch.randn(2, 1, 8, 8)
    logits = model(x, sample=True)
    assert logits.shape == (2, 8, 8, 3)


def test_kl_scaling_respects_num_batches():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(6, [10], 2)
    x = torch.randn(9, 6)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])

    _, _, kl_small = vi_elbo_step(model, x, y, num_batches=1)
    _, _, kl_large = vi_elbo_step(model, x, y, num_batches=10)

    assert kl_small > kl_large
    assert torch.isclose(kl_small / kl_large, torch.tensor(10.0), rtol=1e-4)


def test_n_batches_backward_compat_emits_warning():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(5, [10], 2)
    x = torch.randn(6, 5)
    y = torch.tensor([0, 1, 1, 0, 1, 0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss, nll, kl = vi_elbo_step(model, x, y, n_batches=3)

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)
    assert any("deprecated" in str(w.message).lower() for w in caught)


def test_mc_samples_path_returns_scalars():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(3, [8], 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    loss, nll, kl = vi_elbo_step(
        model,
        x,
        y,
        num_batches=4,
        criterion=nn.MSELoss(reduction="mean"),
        kl_weight=0.01,
        mc_samples=4,
    )

    assert loss.ndim == 0
    assert nll.ndim == 0
    assert kl.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_vi_predict_uq_heteroscedastic_fields():
    torch.manual_seed(0)
    model = HeteroscedasticBayesByBackpropRegressor(2, hidden_dims=(8,))
    x = torch.randn(6, 2)
    uq = predict_vi_uq(model, x, n_samples=5)
    assert uq.mean.shape == (6, 1)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (6, 1)
    assert uq.aleatoric_var is not None and uq.aleatoric_var.shape == (6, 1)
    assert uq.total_var is not None and torch.all(uq.total_var >= uq.epistemic_var)


def test_vi_predict_uq_multioutput_fields():
    torch.manual_seed(0)
    model = HeteroscedasticMultiOutputBayesByBackpropRegressor(
        2, output_dim=3, hidden_dims=(8,)
    )
    x = torch.randn(4, 2)
    uq = predict_vi_uq(model, x, n_samples=4)
    assert uq.mean.shape == (4, 3)
    assert uq.aleatoric_var is not None and uq.aleatoric_var.shape == (4, 3)


def test_vi_predict_uq_last_layer_classification_probs():
    torch.manual_seed(0)
    feature_extractor = _FlatFeatureExtractor(3, 5)
    model = LastLayerVariationalInference(
        feature_extractor,
        feature_dim=5,
        output_dim=4,
        task="classification",
    )
    x = torch.randn(7, 3)
    uq = predict_vi_uq(model, x, n_samples=6)
    assert uq.probs is not None
    assert uq.probs_var is not None
    assert uq.probs.shape == (7, 4)
    row_sums = uq.probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_vi_elbo_rejects_invalid_args():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(2, [4], 2)
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])

    try:
        vi_elbo_step(model, x, y, num_batches=0)
        raise AssertionError("Expected ValueError for num_batches=0")
    except ValueError:
        pass

    try:
        vi_elbo_step(model, x, y, num_batches=2, mc_samples=0)
        raise AssertionError("Expected ValueError for mc_samples=0")
    except ValueError:
        pass


def _ema(values, alpha=0.2):
    smoothed = []
    for value in values:
        if not smoothed:
            smoothed.append(float(value))
        else:
            smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def test_short_training_smoothed_elbo_decreases():
    torch.manual_seed(7)
    x = torch.linspace(-2.0, 2.0, 128).unsqueeze(-1)
    y = 0.5 * x + torch.sin(1.3 * x)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    model = BayesByBackpropMLP(1, [24, 24], 1, prior_sigma=0.5)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    kl_weight = 0.01
    num_batches = len(train_loader)

    def evaluate_train_elbo(mc_samples=6):
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                loss, _, _ = vi_elbo_step(
                    model,
                    x_batch,
                    y_batch,
                    num_batches=num_batches,
                    criterion=criterion,
                    kl_weight=kl_weight,
                    mc_samples=mc_samples,
                )
                items = y_batch.numel()
                total += loss.item() * items
                count += items
        return total / count

    history = []
    for _ in range(60):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = vi_elbo_step(
                model,
                x_batch,
                y_batch,
                num_batches=num_batches,
                criterion=criterion,
                kl_weight=kl_weight,
                mc_samples=1,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        history.append(evaluate_train_elbo())

    ema_elbo = _ema(history, alpha=0.2)
    delta = ema_elbo[-1] - ema_elbo[0]
    decreasing_ratio = float((np.diff(np.array(ema_elbo)) <= 0.0).mean())

    assert delta < 0.0
    assert decreasing_ratio >= 0.65
