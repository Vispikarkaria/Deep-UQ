"""SWAG (Stochastic Weight Averaging - Gaussian) for approximate Bayesian inference.

Implements weight-space posterior approximation via a low-rank plus diagonal
Gaussian fitted during SGD training. ``predict_uq`` returns a
:class:`deepuq.types.UQResult` with epistemic uncertainty from posterior samples.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from deepuq.types import UQResult


def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Return a 1-D tensor of all model parameters."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _set_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Load a flat parameter vector into the model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset : offset + numel].reshape(p.shape))
        offset += numel


class SWAGCollector:
    """Collects running statistics for SWAG during training.

    Parameters
    ----------
    model:
        The model whose parameters are being tracked.
    max_rank:
        Maximum number of deviation columns to store (FIFO).
    collection_freq:
        Collect every N calls to :meth:`collect`. Default 1 (every call).
    """

    def __init__(self, model: nn.Module, max_rank: int = 20, collection_freq: int = 1):
        self.max_rank = max_rank
        self.collection_freq = collection_freq
        self._call_count = 0
        self._n_collected = 0

        params = _flatten_params(model)
        self._num_params = params.numel()

        self._mean = torch.zeros_like(params)
        self._sq_mean = torch.zeros_like(params)
        self._deviations: list[torch.Tensor] = []

    @torch.no_grad()
    def collect(self, model: nn.Module) -> None:
        """Record current model parameters into running statistics."""
        self._call_count += 1
        if self._call_count % self.collection_freq != 0:
            return

        params = _flatten_params(model)
        self._n_collected += 1
        n = self._n_collected

        # Online mean and squared mean
        self._mean = self._mean + (params - self._mean) / n
        self._sq_mean = self._sq_mean + (params**2 - self._sq_mean) / n

        # Store deviation from current running mean
        deviation = params - self._mean
        self._deviations.append(deviation)
        if len(self._deviations) > self.max_rank:
            self._deviations.pop(0)

    def finalize(self) -> None:
        """Compute final covariance components. Call after training."""
        if self._n_collected < 2:
            raise RuntimeError("Need at least 2 collected samples to finalize.")
        # Diagonal variance
        self._diag_var = (self._sq_mean - self._mean**2).clamp_min(1e-30)
        # Deviation matrix: columns are stored deviations
        self._deviation_matrix = torch.stack(self._deviations, dim=1)

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def diag_var(self) -> torch.Tensor:
        return self._diag_var

    @property
    def deviation_matrix(self) -> torch.Tensor:
        return self._deviation_matrix

    @property
    def n_collected(self) -> int:
        return self._n_collected

    def state_dict(self) -> dict:
        """Serialize collector state."""
        return {
            "mean": self._mean,
            "sq_mean": self._sq_mean,
            "deviations": self._deviations,
            "n_collected": self._n_collected,
            "call_count": self._call_count,
            "max_rank": self.max_rank,
            "collection_freq": self.collection_freq,
            "num_params": self._num_params,
            "diag_var": getattr(self, "_diag_var", None),
            "deviation_matrix": getattr(self, "_deviation_matrix", None),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore collector state."""
        self._mean = state["mean"]
        self._sq_mean = state["sq_mean"]
        self._deviations = state["deviations"]
        self._n_collected = state["n_collected"]
        self._call_count = state["call_count"]
        self.max_rank = state["max_rank"]
        self.collection_freq = state["collection_freq"]
        self._num_params = state["num_params"]
        if state.get("diag_var") is not None:
            self._diag_var = state["diag_var"]
        if state.get("deviation_matrix") is not None:
            self._deviation_matrix = state["deviation_matrix"]


class SWAGWrapper(nn.Module):
    """Wraps a base model with a SWAG collector for Bayesian prediction.

    Parameters
    ----------
    base_model:
        The trained model architecture (used as a template for forward passes).
    swag_collector:
        A finalized :class:`SWAGCollector` with computed covariance components.
    """

    method_name = "swag"

    def __init__(self, base_model: nn.Module, swag_collector: SWAGCollector):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.collector = swag_collector

    @torch.no_grad()
    def sample_parameters(self, scale: float = 1.0, diag_noise: bool = True) -> None:
        """Sample parameters from the SWAG posterior and load into base_model.

        Sampling formula:
            θ ~ θ_mean + (1/√2) * diag_noise + (1/√(2(K-1))) * D @ z
        """
        mean = self.collector.mean
        dev_mat = self.collector.deviation_matrix
        K = dev_mat.shape[1]

        # Low-rank component
        z = torch.randn(K, device=mean.device)
        low_rank = dev_mat @ z / (2.0 * (K - 1)) ** 0.5

        # Diagonal component
        if diag_noise:
            diag_std = self.collector.diag_var.sqrt()
            diag = diag_std * torch.randn_like(mean) / (2.0**0.5)
        else:
            diag = torch.zeros_like(mean)

        sampled = mean + scale * (diag + low_rank)
        _set_params(self.base_model, sampled)

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor, n_samples: int = 30) -> UQResult:
        """Run multiple forward passes with sampled weights and aggregate.

        Parameters
        ----------
        x:
            Input tensor.
        n_samples:
            Number of posterior weight samples.

        Returns
        -------
        UQResult with predictive mean and epistemic variance.
        """
        preds = []
        self.base_model.eval()
        for _ in range(n_samples):
            self.sample_parameters()
            preds.append(self.base_model(x).unsqueeze(0))

        preds_t = torch.cat(preds, dim=0)
        mean = preds_t.mean(dim=0)
        var = preds_t.var(dim=0, unbiased=False)

        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "n_samples": n_samples,
                "max_rank": self.collector.max_rank,
            },
        )


class MultiSWAG(nn.Module):
    """Combines multiple SWAG models for improved uncertainty estimation.

    Parameters
    ----------
    swag_wrappers:
        List of :class:`SWAGWrapper` instances (e.g., trained from different
        initializations).
    """

    method_name = "multi_swag"

    def __init__(self, swag_wrappers: Sequence[SWAGWrapper]):
        super().__init__()
        if len(swag_wrappers) == 0:
            raise ValueError("MultiSWAG requires at least one SWAGWrapper.")
        self.wrappers = nn.ModuleList(swag_wrappers)

    @torch.no_grad()
    def predict_uq(self, x: torch.Tensor, n_samples_per_model: int = 10) -> UQResult:
        """Aggregate predictions across all SWAG models.

        Parameters
        ----------
        x:
            Input tensor.
        n_samples_per_model:
            Number of posterior samples per SWAG model.

        Returns
        -------
        UQResult with predictive mean and epistemic variance.
        """
        all_preds = []
        for wrapper in self.wrappers:
            wrapper.base_model.eval()
            for _ in range(n_samples_per_model):
                wrapper.sample_parameters()
                all_preds.append(wrapper.base_model(x).unsqueeze(0))

        preds_t = torch.cat(all_preds, dim=0)
        mean = preds_t.mean(dim=0)
        var = preds_t.var(dim=0, unbiased=False)

        return UQResult(
            mean=mean,
            epistemic_var=var,
            aleatoric_var=None,
            total_var=var,
            probs=None,
            probs_var=None,
            metadata={
                "method": self.method_name,
                "n_models": len(self.wrappers),
                "n_samples_per_model": n_samples_per_model,
                "total_samples": len(self.wrappers) * n_samples_per_model,
            },
        )
