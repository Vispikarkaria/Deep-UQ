"""Temperature and vector scaling calibration methods."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepuq.types import UQResult


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for classification models.

    Learns a single scalar T that divides logits before softmax, minimizing
    NLL on a held-out validation set.

    Parameters
    ----------
    model:
        A PyTorch model that outputs raw logits.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(1.5))

    def fit(
        self,
        val_loader: torch.utils.data.DataLoader,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> TemperatureScaling:
        """Optimize temperature on a validation set using LBFGS.

        Parameters
        ----------
        val_loader:
            DataLoader yielding (inputs, targets) batches.
        max_iter:
            Maximum LBFGS iterations.
        lr:
            Learning rate for the optimizer.

        Returns
        -------
        self
        """
        self.model.eval()
        # Collect all logits and labels
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits_list.append(self.model(inputs))
                labels_list.append(targets)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_calibrated(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for input x.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Calibrated class probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            return F.softmax(logits / self.temperature, dim=-1)

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return a UQResult with calibrated probabilities and entropy-based uncertainty.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        UQResult with mean=calibrated probs, epistemic_var=predictive entropy.
        """
        probs = self.predict_calibrated(x)
        # Predictive entropy as epistemic uncertainty proxy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return UQResult(
            mean=probs,
            epistemic_var=entropy,
            probs=probs,
            metadata={
                "method": "temperature_scaling",
                "temperature": self.temperature.item(),
            },
        )


class VectorScaling(nn.Module):
    """Per-class temperature and bias calibration.

    Learns a vector of temperatures W and biases b such that
    calibrated logits = logits * W + b.

    Parameters
    ----------
    model:
        A PyTorch model that outputs raw logits.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._w: nn.Parameter | None = None
        self._b: nn.Parameter | None = None

    def _init_params(self, num_classes: int) -> None:
        if self._w is None:
            self._w = nn.Parameter(torch.ones(num_classes) * 1.5)
            self._b = nn.Parameter(torch.zeros(num_classes))

    def fit(
        self,
        val_loader: torch.utils.data.DataLoader,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> VectorScaling:
        """Optimize per-class parameters on a validation set using LBFGS.

        Parameters
        ----------
        val_loader:
            DataLoader yielding (inputs, targets) batches.
        max_iter:
            Maximum LBFGS iterations.
        lr:
            Learning rate for the optimizer.

        Returns
        -------
        self
        """
        self.model.eval()
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits_list.append(self.model(inputs))
                labels_list.append(targets)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        num_classes = logits.shape[-1]
        self._init_params(num_classes)

        optimizer = torch.optim.LBFGS([self._w, self._b], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self._w + self._b
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def predict_calibrated(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities for input x."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            if self._w is None:
                raise RuntimeError("Must call fit() before predict_calibrated().")
            scaled = logits / self._w + self._b
            return F.softmax(scaled, dim=-1)

    def predict_uq(self, x: torch.Tensor) -> UQResult:
        """Return a UQResult with calibrated probabilities and entropy-based uncertainty."""
        probs = self.predict_calibrated(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return UQResult(
            mean=probs,
            epistemic_var=entropy,
            probs=probs,
            metadata={"method": "vector_scaling"},
        )
