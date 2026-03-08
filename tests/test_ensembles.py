import torch
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods import DeepEnsembleWrapper
from deepuq.models import MLP
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
