"""Test-Time Augmentation (TTA) wrapper for uncertainty quantification."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from deepuq.types import UQResult


class TTAWrapper:
    """Wraps any nn.Module to produce UQ estimates via test-time augmentation.

    Parameters
    ----------
    model:
        Any ``nn.Module``. Does not need a ``predict_uq`` method.
    augmentations:
        List of callables ``(x -> augmented_x)``. If None, default noise
        augmentations are generated.
    n_augmentations:
        Number of default noise augmentations to create when ``augmentations``
        is None.
    """

    def __init__(
        self,
        model: nn.Module,
        augmentations: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        n_augmentations: int = 30,
    ) -> None:
        self.model = model
        if augmentations is not None:
            self.augmentations = augmentations
        else:
            self.augmentations = [
                self._make_noise_aug(i) for i in range(n_augmentations)
            ]

    @staticmethod
    def _make_noise_aug(i: int) -> Callable[[torch.Tensor], torch.Tensor]:
        """Create a noise augmentation with scale proportional to i."""

        def aug(x: torch.Tensor) -> torch.Tensor:
            return x + torch.randn_like(x) * 0.01 * i

        return aug

    def predict_uq(
        self, x: torch.Tensor, n_augmentations: int | None = None
    ) -> UQResult:
        """Run TTA and return uncertainty estimates.

        Parameters
        ----------
        x:
            Input tensor.
        n_augmentations:
            If provided, use only the first ``n_augmentations`` augmentations.

        Returns
        -------
        UQResult with mean, epistemic_var, and total_var.
        """
        augs = self.augmentations
        if n_augmentations is not None:
            augs = augs[:n_augmentations]

        self.model.eval()
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for aug_fn in augs:
                x_aug = aug_fn(x)
                out = self.model(x_aug)
                outputs.append(out)

        stacked = torch.stack(outputs, dim=0)  # (n_aug, batch, ...)
        mean = stacked.mean(dim=0)
        epistemic_var = stacked.var(dim=0)

        return UQResult(
            mean=mean,
            epistemic_var=epistemic_var,
            total_var=epistemic_var,
            metadata={"method": "tta", "n_augmentations": len(augs)},
        )
