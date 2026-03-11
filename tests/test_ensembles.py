import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import (
    DeepEnsembleClassifier,
    DeepEnsembleWrapper,
    HeteroscedasticDeepEnsembleRegressor,
    HeteroscedasticMultiOutputDeepEnsembleRegressor,
    MultiOutputDeepEnsembleRegressor,
)
from deepuq.models import MLP, CNNRegressor2D
from deepuq.types import UQResult


def test_deep_ensemble_predict_shapes():
    models = [MLP(3, [8], 1, p_drop=0.0) for _ in range(3)]
    ensemble = DeepEnsembleWrapper(models)
    x = torch.randn(5, 3)

    mean, var = ensemble.predict(x)
    uq = ensemble.predict_uq(x)

    assert mean.shape == (5, 1)
    assert var.shape == (5, 1)
    assert isinstance(uq, UQResult)
    assert uq.metadata["method"] == "deep_ensemble"
    assert uq.total_var is not None and uq.total_var.shape == (5, 1)


def test_deep_ensemble_fit_smoke():
    x = torch.randn(48, 2)
    y = (x[:, :1] ** 2) + 0.1 * x[:, 1:2]
    loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)

    ensemble = DeepEnsembleWrapper([MLP(2, [16], 1, p_drop=0.0) for _ in range(2)])
    ensemble.fit(
        loader,
        epochs=10,
        loss_fn=torch.nn.functional.mse_loss,
        lr=1e-2,
        seed=123,
    )
    uq = ensemble.predict_uq(x[:4])
    assert torch.isfinite(uq.mean).all()
    assert uq.total_var is not None and torch.isfinite(uq.total_var).all()


def test_heteroscedastic_deep_ensemble_components():
    models = [MLP(2, [12], 2, p_drop=0.0) for _ in range(3)]
    ensemble = HeteroscedasticDeepEnsembleRegressor(models)
    x = torch.randn(6, 2)

    uq = ensemble.predict_uq(x)
    assert uq.mean.shape == (6, 1)
    assert uq.epistemic_var is not None and uq.epistemic_var.shape == (6, 1)
    assert uq.aleatoric_var is not None and uq.aleatoric_var.shape == (6, 1)
    assert uq.total_var is not None and torch.all(
        uq.total_var >= uq.epistemic_var
    )


def test_deep_ensemble_classifier_predict_uq():
    models = [MLP(2, [10], 3, p_drop=0.0) for _ in range(4)]
    ensemble = DeepEnsembleClassifier(models)
    x = torch.randn(7, 2)

    uq = ensemble.predict_uq(x)
    assert isinstance(uq, UQResult)
    assert uq.probs is not None and uq.probs.shape == (7, 3)
    assert uq.probs_var is not None and uq.probs_var.shape == (7, 3)
    row_sums = uq.probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_multioutput_regression_variants():
    x = torch.randn(5, 3)

    reg = MultiOutputDeepEnsembleRegressor([MLP(3, [8], 2, p_drop=0.0) for _ in range(2)])
    reg_uq = reg.predict_uq(x)
    assert reg_uq.mean.shape == (5, 2)
    assert reg_uq.total_var is not None and reg_uq.total_var.shape == (5, 2)

    hetero = HeteroscedasticMultiOutputDeepEnsembleRegressor(
        [MLP(3, [8], 4, p_drop=0.0) for _ in range(2)]
    )
    hetero_uq = hetero.predict_uq(x)
    assert hetero_uq.mean.shape == (5, 2)
    assert hetero_uq.aleatoric_var is not None
    assert hetero_uq.aleatoric_var.shape == (5, 2)


def test_classifier_fit_smoke():
    x = torch.randn(64, 2)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    classifier = DeepEnsembleClassifier([MLP(2, [16], 2, p_drop=0.0) for _ in range(2)])
    classifier.fit(loader, epochs=5, lr=5e-3, seed=10)
    uq = classifier.predict_uq(x[:6])
    assert uq.probs is not None
    assert torch.isfinite(uq.probs).all()


def test_heteroscedastic_fit_smoke():
    x = torch.randn(48, 2)
    y = x[:, :1].sin() + 0.1 * x[:, 1:2]
    loader = DataLoader(TensorDataset(x, y), batch_size=12, shuffle=True)

    ensemble = HeteroscedasticDeepEnsembleRegressor(
        [MLP(2, [16], 2, p_drop=0.0) for _ in range(2)]
    )
    ensemble.fit(loader, epochs=5, lr=5e-3, seed=7)
    uq = ensemble.predict_uq(x[:4])
    assert uq.total_var is not None
    assert torch.isfinite(uq.total_var).all()


def test_heteroscedastic_field_output_split():
    models = [CNNRegressor2D(in_channels=1, out_channels=4, hidden_channels=(8, 8)) for _ in range(2)]
    ensemble = HeteroscedasticMultiOutputDeepEnsembleRegressor(models)
    x = torch.randn(3, 1, 12, 12)
    uq = ensemble.predict_uq(x)
    assert uq.mean.shape == (3, 2, 12, 12)
    assert uq.aleatoric_var is not None and uq.aleatoric_var.shape == (3, 2, 12, 12)
