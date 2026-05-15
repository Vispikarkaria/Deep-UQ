"""Tests for SWAG implementation."""

import copy

import torch
from torch import nn

from deepuq.methods.swag import MultiSWAG, SWAGCollector, SWAGWrapper
from deepuq.types import UQResult


def _make_model():
    """Simple 2-layer net for testing."""
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))


class TestSWAGCollector:
    def test_basic_collection_and_finalization(self):
        model = _make_model()
        collector = SWAGCollector(model, max_rank=5)

        # Simulate training epochs with param changes
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(10):
            x = torch.randn(8, 4)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            collector.collect(model)

        assert collector.n_collected == 10
        collector.finalize()
        assert collector.mean.shape[0] > 0
        assert collector.diag_var.shape == collector.mean.shape
        # max_rank=5 so deviation matrix has 5 columns
        assert collector.deviation_matrix.shape[1] == 5

    def test_collection_freq(self):
        model = _make_model()
        collector = SWAGCollector(model, max_rank=20, collection_freq=3)

        for _ in range(9):
            collector.collect(model)

        assert collector.n_collected == 3

    def test_state_dict_roundtrip(self):
        model = _make_model()
        collector = SWAGCollector(model, max_rank=5)
        for _ in range(5):
            collector.collect(model)
        collector.finalize()

        state = collector.state_dict()
        new_collector = SWAGCollector(model, max_rank=5)
        new_collector.load_state_dict(state)

        assert torch.allclose(new_collector.mean, collector.mean)
        assert torch.allclose(new_collector.diag_var, collector.diag_var)


class TestSWAGWrapper:
    def _build_wrapper(self):
        model = _make_model()
        collector = SWAGCollector(model, max_rank=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(10):
            x = torch.randn(8, 4)
            loss = model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            collector.collect(model)
        collector.finalize()
        return SWAGWrapper(model, collector)

    def test_sampling_produces_different_params(self):
        wrapper = self._build_wrapper()
        wrapper.sample_parameters()
        p1 = torch.cat([p.reshape(-1) for p in wrapper.base_model.parameters()]).clone()
        wrapper.sample_parameters()
        p2 = torch.cat([p.reshape(-1) for p in wrapper.base_model.parameters()])
        assert not torch.allclose(p1, p2)

    def test_predict_uq_returns_valid_result(self):
        wrapper = self._build_wrapper()
        x = torch.randn(16, 4)
        result = wrapper.predict_uq(x, n_samples=10)

        assert isinstance(result, UQResult)
        assert result.mean.shape == (16, 1)
        assert result.epistemic_var.shape == (16, 1)
        assert result.total_var.shape == (16, 1)
        assert result.metadata["method"] == "swag"
        assert result.metadata["n_samples"] == 10

    def test_epistemic_var_positive(self):
        wrapper = self._build_wrapper()
        x = torch.randn(16, 4)
        result = wrapper.predict_uq(x, n_samples=30)
        assert (result.epistemic_var > 0).all()


class TestMultiSWAG:
    def test_multi_swag_works(self):
        wrappers = []
        for _ in range(3):
            model = _make_model()
            collector = SWAGCollector(model, max_rank=5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(10):
                x = torch.randn(8, 4)
                loss = model(x).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                collector.collect(model)
            collector.finalize()
            wrappers.append(SWAGWrapper(model, collector))

        multi = MultiSWAG(wrappers)
        x = torch.randn(16, 4)
        result = multi.predict_uq(x, n_samples_per_model=5)

        assert isinstance(result, UQResult)
        assert result.mean.shape == (16, 1)
        assert result.epistemic_var.shape == (16, 1)
        assert (result.epistemic_var > 0).all()
        assert result.metadata["method"] == "multi_swag"
        assert result.metadata["n_models"] == 3
        assert result.metadata["total_samples"] == 15
