from __future__ import annotations

import numpy as np
import torch

from deepuq.data import load_gray_scott_pairs, split_gray_scott_pairs_by_regime


def test_load_gray_scott_pairs_from_pair_archive(tmp_path) -> None:
    archive = tmp_path / "gray_scott_subset.npz"
    current = np.random.rand(6, 8, 8, 2).astype(np.float32)
    nxt = np.random.rand(6, 8, 8, 2).astype(np.float32)
    feed = np.array([0.02, 0.02, 0.022, 0.022, 0.04, 0.04], dtype=np.float32)
    kill = np.array([0.05, 0.05, 0.051, 0.051, 0.065, 0.065], dtype=np.float32)
    np.savez(
        archive,
        current_states=current,
        next_states=nxt,
        feed_rates=feed,
        kill_rates=kill,
    )

    pairs = load_gray_scott_pairs(tmp_path)
    assert pairs.current_states.shape == (6, 8, 8, 2)
    assert pairs.next_states.shape == (6, 8, 8, 2)
    assert torch.allclose(pairs.feed_rates, torch.tensor(feed))
    assert torch.allclose(pairs.kill_rates, torch.tensor(kill))


def test_split_gray_scott_pairs_by_regime(tmp_path) -> None:
    archive = tmp_path / "gray_scott_subset.npz"
    current = np.random.rand(10, 8, 8, 2).astype(np.float32)
    nxt = np.random.rand(10, 8, 8, 2).astype(np.float32)
    feed = np.array([0.02] * 4 + [0.022] * 3 + [0.04] * 3, dtype=np.float32)
    kill = np.array([0.05] * 4 + [0.051] * 3 + [0.065] * 3, dtype=np.float32)
    np.savez(
        archive,
        current_states=current,
        next_states=nxt,
        feed_rates=feed,
        kill_rates=kill,
    )

    pairs = load_gray_scott_pairs(tmp_path)
    split = split_gray_scott_pairs_by_regime(
        pairs,
        train_regimes=[(0.02, 0.05), (0.022, 0.051)],
        ood_regimes=[(0.04, 0.065)],
        val_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )
    assert split["train"].current_states.numel() > 0
    assert split["val"].current_states.numel() > 0
    assert split["test"].current_states.numel() > 0
    assert split["ood"].current_states.shape[0] == 3
