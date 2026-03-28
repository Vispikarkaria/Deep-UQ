"""Dataset helpers for external scientific benchmarks."""

from .the_well import (
    GrayScottPairs,
    load_gray_scott_pairs,
    split_gray_scott_pairs_by_regime,
)

__all__ = [
    "GrayScottPairs",
    "load_gray_scott_pairs",
    "split_gray_scott_pairs_by_regime",
]
