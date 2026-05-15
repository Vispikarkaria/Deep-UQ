"""Tests for SGHMCOptimizer and CyclicalSGMCMC."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepuq.methods.mcmc import CyclicalSGMCMC, SGHMCOptimizer


def _simple_model():
    return nn.Linear(4, 2)


def _make_loader(n=32, in_dim=4, n_classes=2, batch_size=8):
    x = torch.randn(n, in_dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TestSGHMCOptimizer:
    def test_parameters_change(self):
        model = _simple_model()
        opt = SGHMCOptimizer(model.parameters(), lr=1e-2)
        x = torch.randn(4, 4)
        initial_params = {k: v.clone() for k, v in model.named_parameters()}

        out = model(x)
        out.sum().backward()
        opt.step()

        for name, p in model.named_parameters():
            assert not torch.equal(p.data, initial_params[name]), f"{name} did not change"

    def test_velocity_maintained(self):
        model = _simple_model()
        opt = SGHMCOptimizer(model.parameters(), lr=1e-2)

        x = torch.randn(4, 4)
        out = model(x)
        out.sum().backward()
        opt.step()

        # Check velocity exists and is non-zero
        for p in model.parameters():
            state = opt.state[p]
            assert "velocity" in state
            assert not torch.all(state["velocity"] == 0)

        # Second step: velocity should still be present
        opt.zero_grad()
        out = model(x)
        out.sum().backward()
        opt.step()

        for p in model.parameters():
            assert "velocity" in opt.state[p]


class TestCyclicalSGMCMC:
    def test_correct_number_of_samples(self):
        model = _simple_model()
        loader = _make_loader()
        n_cycles = 3
        samples_per_cycle = 2
        runner = CyclicalSGMCMC(
            model,
            SGHMCOptimizer,
            cycle_length=10,
            n_cycles=n_cycles,
            samples_per_cycle=samples_per_cycle,
        )
        samples = runner.run(loader, nn.CrossEntropyLoss())
        assert len(samples) == n_cycles * samples_per_cycle

    def test_samples_differ(self):
        model = _simple_model()
        loader = _make_loader()
        runner = CyclicalSGMCMC(
            model,
            SGHMCOptimizer,
            cycle_length=10,
            n_cycles=4,
            samples_per_cycle=3,
        )
        samples = runner.run(loader, nn.CrossEntropyLoss())

        # At least some samples should differ
        first = samples[0]
        any_different = False
        for s in samples[1:]:
            for k in first:
                if not torch.equal(first[k], s[k]):
                    any_different = True
                    break
            if any_different:
                break
        assert any_different, "All collected samples are identical"
