import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.vi import BayesByBackpropMLP, vi_elbo_step


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
        criterion=nn.MSELoss(reduction='mean'),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


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
        warnings.simplefilter('always')
        loss, nll, kl = vi_elbo_step(model, x, y, n_batches=3)

    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)
    assert any('deprecated' in str(w.message).lower() for w in caught)


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
        criterion=nn.MSELoss(reduction='mean'),
        kl_weight=0.01,
        mc_samples=4,
    )

    assert loss.ndim == 0
    assert nll.ndim == 0
    assert kl.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)


def test_vi_elbo_rejects_invalid_args():
    torch.manual_seed(0)
    model = BayesByBackpropMLP(2, [4], 2)
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])

    try:
        vi_elbo_step(model, x, y, num_batches=0)
        raise AssertionError('Expected ValueError for num_batches=0')
    except ValueError:
        pass

    try:
        vi_elbo_step(model, x, y, num_batches=2, mc_samples=0)
        raise AssertionError('Expected ValueError for mc_samples=0')
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
    criterion = nn.MSELoss(reduction='mean')
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
