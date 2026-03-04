from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    seed: int = 42
    batch_size: int = 64
    hidden_dims: tuple[int, int] = (64, 64)
    train_epochs: int = 120
    vi_epochs: int = 120
    num_samples: int = 60
    lr: float = 1e-3
    max_train_points: int = 5000
    max_exact_gp_train: int = 600
    max_sparse_gp_train: int = 3000


def preset(name: str) -> BenchmarkConfig:
    if name == "quick":
        return BenchmarkConfig(
            train_epochs=30,
            vi_epochs=30,
            num_samples=20,
            max_train_points=2000,
            max_exact_gp_train=300,
            max_sparse_gp_train=1200,
        )
    if name == "full":
        return BenchmarkConfig()
    raise ValueError(f"Unknown benchmark preset {name!r}. Use 'quick' or 'full'.")
