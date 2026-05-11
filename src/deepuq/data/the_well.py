"""Utilities for working with local The Well Gray-Scott subsets.

The helpers in this module intentionally target a lightweight, local archive
format so the package does not depend on a heavier The Well reader stack. The
expected archive is a ``.npz`` file containing either full trajectories or
pre-built next-step pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class GrayScottPairs:
    """One-step Gray-Scott state pairs with regime metadata.

    Attributes
    ----------
    current_states:
        Tensor with shape ``[N, H, W, 2]`` storing the input state at time
        ``t``.
    next_states:
        Tensor with shape ``[N, H, W, 2]`` storing the target state at time
        ``t + Δt``.
    feed_rates:
        Tensor with shape ``[N]`` storing the Gray-Scott feed rate ``f`` for
        each pair.
    kill_rates:
        Tensor with shape ``[N]`` storing the Gray-Scott kill rate ``k`` for
        each pair.
    """

    current_states: torch.Tensor
    next_states: torch.Tensor
    feed_rates: torch.Tensor
    kill_rates: torch.Tensor

    def subset(self, indices: torch.Tensor) -> GrayScottPairs:
        return GrayScottPairs(
            current_states=self.current_states[indices],
            next_states=self.next_states[indices],
            feed_rates=self.feed_rates[indices],
            kill_rates=self.kill_rates[indices],
        )


def _find_archive(root: str | Path, file_name: str | None = None) -> Path:
    root_path = Path(root).expanduser().resolve()
    if file_name is not None:
        archive = root_path / file_name
        if not archive.exists():
            raise FileNotFoundError(f"Could not find Gray-Scott archive: {archive}")
        return archive
    candidates = [
        root_path / "gray_scott_subset.npz",
        root_path / "gray_scott.npz",
        root_path / "the_well_gray_scott.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a local Gray-Scott .npz archive under the provided root. "
        "Expected one of: gray_scott_subset.npz, gray_scott.npz, "
        "the_well_gray_scott.npz."
    )


def _pairs_from_states(
    states: torch.Tensor,
    feed_rates: torch.Tensor,
    kill_rates: torch.Tensor,
    step: int,
) -> GrayScottPairs:
    if states.dim() != 5 or states.size(-1) != 2:
        raise ValueError(
            "Expected states with shape [n_trajectories, time, H, W, 2] for Gray-Scott trajectories."
        )
    if step < 1 or step >= states.size(1):
        raise ValueError("step must be at least 1 and smaller than the time dimension.")
    current = states[:, :-step].reshape(-1, *states.shape[2:])
    nxt = states[:, step:].reshape(-1, *states.shape[2:])
    repeats = states.size(1) - step
    feed = feed_rates.repeat_interleave(repeats)
    kill = kill_rates.repeat_interleave(repeats)
    return GrayScottPairs(
        current_states=current, next_states=nxt, feed_rates=feed, kill_rates=kill
    )


def load_gray_scott_pairs(
    root: str | Path,
    *,
    file_name: str | None = None,
    step: int = 1,
) -> GrayScottPairs:
    """Load Gray-Scott next-step pairs from a local `.npz` archive.

    Supported archive layouts
    -------------------------
    1. Trajectory layout:
       - ``states``: ``[n_trajectories, time, H, W, 2]``
       - ``feed_rates`` or ``feed``: ``[n_trajectories]``
       - ``kill_rates`` or ``kill``: ``[n_trajectories]``

    2. Pair layout:
       - ``current_states``: ``[N, H, W, 2]``
       - ``next_states``: ``[N, H, W, 2]``
       - ``feed_rates`` or ``feed``: ``[N]``
       - ``kill_rates`` or ``kill``: ``[N]``
    """
    archive = _find_archive(root, file_name=file_name)
    with np.load(archive, allow_pickle=False) as data:
        if "current_states" in data and "next_states" in data:
            current = torch.from_numpy(data["current_states"]).float()
            nxt = torch.from_numpy(data["next_states"]).float()
            feed = torch.from_numpy(data.get("feed_rates", data.get("feed"))).float()
            kill = torch.from_numpy(data.get("kill_rates", data.get("kill"))).float()
            return GrayScottPairs(
                current_states=current,
                next_states=nxt,
                feed_rates=feed,
                kill_rates=kill,
            )

        if "states" not in data:
            raise ValueError(
                "Gray-Scott archive must contain either (current_states, next_states) or a states trajectory array."
            )
        states = torch.from_numpy(data["states"]).float()
        feed_arr = data.get("feed_rates", data.get("feed"))
        kill_arr = data.get("kill_rates", data.get("kill"))
        if feed_arr is None or kill_arr is None:
            raise ValueError(
                "Gray-Scott archive must include feed_rates/feed and kill_rates/kill metadata."
            )
        feed = torch.from_numpy(feed_arr).float()
        kill = torch.from_numpy(kill_arr).float()
        return _pairs_from_states(states, feed, kill, step=step)


def split_gray_scott_pairs_by_regime(
    pairs: GrayScottPairs,
    *,
    train_regimes: list[tuple[float, float]],
    ood_regimes: list[tuple[float, float]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    atol: float = 1e-6,
) -> dict[str, GrayScottPairs]:
    """Split Gray-Scott pairs into train/val/test/OOD by ``(feed, kill)`` regime."""
    if (
        val_fraction <= 0.0
        or test_fraction <= 0.0
        or val_fraction + test_fraction >= 1.0
    ):
        raise ValueError(
            "val_fraction and test_fraction must be positive and sum to less than 1."
        )

    def regime_mask(regimes: list[tuple[float, float]]) -> torch.Tensor:
        mask = torch.zeros_like(pairs.feed_rates, dtype=torch.bool)
        for feed, kill in regimes:
            mask |= torch.isclose(
                pairs.feed_rates, torch.tensor(feed), atol=atol
            ) & torch.isclose(pairs.kill_rates, torch.tensor(kill), atol=atol)
        return mask

    train_mask = regime_mask(train_regimes)
    ood_mask = regime_mask(ood_regimes)
    if not torch.any(train_mask):
        raise ValueError("No samples matched the requested train_regimes.")
    if not torch.any(ood_mask):
        raise ValueError("No samples matched the requested ood_regimes.")

    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    train_indices = train_indices[
        torch.randperm(train_indices.numel(), generator=generator)
    ]
    n_train_total = train_indices.numel()
    n_val = max(int(round(val_fraction * n_train_total)), 1)
    n_test = max(int(round(test_fraction * n_train_total)), 1)
    n_core = max(n_train_total - n_val - n_test, 1)
    core = train_indices[:n_core]
    val = train_indices[n_core : n_core + n_val]
    test = train_indices[n_core + n_val : n_core + n_val + n_test]
    if test.numel() == 0:
        test = val[:1]
    return {
        "train": pairs.subset(core),
        "val": pairs.subset(val),
        "test": pairs.subset(test),
        "ood": pairs.subset(torch.nonzero(ood_mask, as_tuple=False).flatten()),
    }
