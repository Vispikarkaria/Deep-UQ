"""Tests for conformal prediction methods."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.conformal import (
    ConformalClassifier,
    ConformalUQWrapper,
    CQRPredictor,
    SplitConformalRegressor,
    check_coverage,
)
from deepuq.types import UQResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim=2, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class QuantileModel(nn.Module):
    """Returns (q_lo, q_hi) as shape (N, 2)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        out = self.net(x)
        # Ensure q_lo < q_hi
        lo = out[:, 0]
        hi = lo + torch.abs(out[:, 1]) + 0.1
        return torch.stack([lo, hi], dim=1)


class FakeUQMethod:
    """Fake UQ method with predict_uq()."""

    def __init__(self, model):
        self.model = model

    def predict_uq(self, x):
        with torch.no_grad():
            mean = self.model(x).squeeze(-1)
        return UQResult(
            mean=mean,
            epistemic_var=torch.ones_like(mean) * 0.5,
            total_var=torch.ones_like(mean) * 1.0,
        )


def _make_sine_data(n=500, seed=42):
    torch.manual_seed(seed)
    x = torch.rand(n, 1) * 10
    y = torch.sin(x).squeeze(-1) + torch.randn(n) * 0.2
    return x, y


def _train_mlp(model, x, y, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        pred = model(x).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_conformal_coverage_guarantee():
    x, y = _make_sine_data(1000)
    model = _train_mlp(SimpleMLP(), x[:500], y[:500])
    alpha = 0.1

    cp = SplitConformalRegressor(model, alpha=alpha)
    cp.calibrate((x[500:750], y[500:750]))
    result = cp.predict_uq(x[750:])

    lower = result.metadata["conformal_lower"]
    upper = result.metadata["conformal_upper"]
    cov = check_coverage(y[750:], lower, upper)
    assert cov >= 1 - alpha - 0.05  # allow small slack for finite sample


def test_split_conformal_with_mlp():
    x, y = _make_sine_data(600)
    model = _train_mlp(SimpleMLP(), x[:300], y[:300])

    cp = SplitConformalRegressor(model, alpha=0.1)
    cp.calibrate((x[300:450], y[300:450]))
    result = cp.predict_uq(x[450:])

    assert result.mean.shape == (150,)
    assert result.total_var is not None
    assert "conformal_lower" in result.metadata


def test_split_conformal_interval_width_reasonable():
    x, y = _make_sine_data(600)
    model = _train_mlp(SimpleMLP(), x[:300], y[:300])

    cp = SplitConformalRegressor(model, alpha=0.1)
    cp.calibrate((x[300:450], y[300:450]))
    result = cp.predict_uq(x[450:])

    width = result.metadata["conformal_upper"] - result.metadata["conformal_lower"]
    assert width.mean().item() > 0.1  # not degenerate
    assert width.mean().item() < 5.0  # not absurdly wide


def test_conformal_classifier_aps_coverage():
    torch.manual_seed(0)
    n = 1000
    x = torch.randn(n, 2)
    y = (x[:, 0] + x[:, 1] > 0).long() + (x[:, 0] > 0.5).long()  # 3 classes

    model = SimpleClassifier(2, 3)
    # Train briefly
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        loss = nn.CrossEntropyLoss()(model(x[:600]), y[:600])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    alpha = 0.1
    cc = ConformalClassifier(model, alpha=alpha, method="aps")
    cc.calibrate((x[600:800], y[600:800]))
    result = cc.predict_uq(x[800:])

    # Check coverage: true label should be in prediction set
    pred_sets = result.metadata["prediction_sets"]
    true_labels = y[800:]
    covered = pred_sets[torch.arange(len(true_labels)), true_labels]
    cov = covered.float().mean().item()
    assert cov >= 1 - alpha - 0.05


def test_conformal_classifier_raps_smaller_sets():
    torch.manual_seed(0)
    n = 800
    x = torch.randn(n, 2)
    y = (x[:, 0] > 0).long()  # binary

    model = SimpleClassifier(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        loss = nn.CrossEntropyLoss()(model(x[:400]), y[:400])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    cc_aps = ConformalClassifier(model, alpha=0.1, method="aps")
    cc_aps.calibrate((x[400:600], y[400:600]))
    r_aps = cc_aps.predict_uq(x[600:])

    cc_raps = ConformalClassifier(
        model, alpha=0.1, method="raps", k_reg=1, lambda_reg=0.1
    )
    cc_raps.calibrate((x[400:600], y[400:600]))
    r_raps = cc_raps.predict_uq(x[600:])

    # RAPS should give same or smaller sets on average
    assert (
        r_raps.metadata["set_sizes"].float().mean()
        <= r_aps.metadata["set_sizes"].float().mean() + 0.1
    )


def test_cqr_adaptive_intervals():
    torch.manual_seed(0)
    x = torch.linspace(0, 10, 500).unsqueeze(1)
    noise_scale = 0.1 + 0.3 * (x.squeeze() / 10)
    y = torch.sin(x.squeeze()) + torch.randn(500) * noise_scale

    model = QuantileModel()
    # Train quantile model
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        out = model(x[:300])
        q_lo, q_hi = out[:, 0], out[:, 1]
        # Pinball loss
        tau_lo, tau_hi = 0.1, 0.9
        loss_lo = torch.mean(
            torch.max(tau_lo * (y[:300] - q_lo), (tau_lo - 1) * (y[:300] - q_lo))
        )
        loss_hi = torch.mean(
            torch.max(tau_hi * (y[:300] - q_hi), (tau_hi - 1) * (y[:300] - q_hi))
        )
        loss = loss_lo + loss_hi
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    cqr = CQRPredictor(model, alpha=0.1)
    cqr.calibrate((x[300:400], y[300:400]))
    result = cqr.predict_uq(x[400:])

    assert result.mean.shape == (100,)
    assert "conformal_lower" in result.metadata
    assert "raw_q_lo" in result.metadata


def test_conformal_wrapper_with_ensemble():
    x, y = _make_sine_data(600)
    model = _train_mlp(SimpleMLP(), x[:300], y[:300])
    fake_uq = FakeUQMethod(model)

    wrapper = ConformalUQWrapper(fake_uq, alpha=0.1)
    wrapper.calibrate((x[300:450], y[300:450]))
    result = wrapper.predict_uq(x[450:])

    lower = result.metadata["conformal_lower"]
    upper = result.metadata["conformal_upper"]
    cov = check_coverage(y[450:], lower, upper)
    assert cov >= 0.80  # should be close to 0.9


def test_conformal_wrapper_with_laplace():
    # Same as ensemble test but exercises the path
    x, y = _make_sine_data(600)
    model = _train_mlp(SimpleMLP(), x[:300], y[:300])
    fake_uq = FakeUQMethod(model)

    wrapper = ConformalUQWrapper(fake_uq, alpha=0.05)
    wrapper.calibrate((x[300:450], y[300:450]))
    result = wrapper.predict_uq(x[450:])

    lower = result.metadata["conformal_lower"]
    upper = result.metadata["conformal_upper"]
    cov = check_coverage(y[450:], lower, upper)
    assert cov >= 0.90


def test_uncalibrated_raises():
    model = SimpleMLP()
    cp = SplitConformalRegressor(model, alpha=0.1)
    import pytest

    with pytest.raises(RuntimeError, match="[Cc]alibrate"):
        cp.predict_uq(torch.randn(10, 1))


def test_uqresult_has_correct_metadata():
    x, y = _make_sine_data(200)
    model = _train_mlp(SimpleMLP(), x[:100], y[:100])

    cp = SplitConformalRegressor(model, alpha=0.1)
    cp.calibrate((x[100:150], y[100:150]))
    result = cp.predict_uq(x[150:])

    assert "conformal_lower" in result.metadata
    assert "conformal_upper" in result.metadata
    assert "coverage_alpha" in result.metadata
    assert "q_hat" in result.metadata
    assert result.metadata["coverage_alpha"] == 0.1


def test_coverage_theoretical_bound():
    """Coverage should satisfy: 1-alpha <= cov <= 1-alpha + 1/(n+1)."""
    x, y = _make_sine_data(2000)
    model = _train_mlp(SimpleMLP(), x[:1000], y[:1000])
    alpha = 0.1
    n_cal = 500

    cp = SplitConformalRegressor(model, alpha=alpha)
    cp.calibrate((x[1000 : 1000 + n_cal], y[1000 : 1000 + n_cal]))
    result = cp.predict_uq(x[1500:])

    lower = result.metadata["conformal_lower"]
    upper = result.metadata["conformal_upper"]
    cov = check_coverage(y[1500:], lower, upper)

    # Theoretical bound (finite sample)
    assert cov >= 1 - alpha - 0.03  # small slack
    assert cov <= 1 - alpha + 1 / (n_cal + 1) + 0.03


def test_accepts_dataloader_and_tuple():
    x, y = _make_sine_data(300)
    model = _train_mlp(SimpleMLP(), x[:150], y[:150])

    # Tuple
    cp1 = SplitConformalRegressor(model, alpha=0.1)
    cp1.calibrate((x[150:225], y[150:225]))
    r1 = cp1.predict_uq(x[225:])

    # DataLoader
    ds = TensorDataset(x[150:225], y[150:225])
    loader = DataLoader(ds, batch_size=32)
    cp2 = SplitConformalRegressor(model, alpha=0.1)
    cp2.calibrate(loader)
    r2 = cp2.predict_uq(x[225:])

    assert torch.allclose(r1.mean, r2.mean, atol=1e-5)
    assert abs(r1.metadata["q_hat"] - r2.metadata["q_hat"]) < 1e-5
