from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionDataset:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def _to_regression_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> RegressionDataset:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    return RegressionDataset(
        name=name,
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1),
    )


def load_regression_datasets(seed: int = 42) -> List[RegressionDataset]:
    datasets: List[RegressionDataset] = []
    cache_dir = Path("benchmarks/cache/sklearn_data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    x_diabetes, y_diabetes = load_diabetes(return_X_y=True)
    datasets.append(_to_regression_dataset("diabetes", x_diabetes, y_diabetes, seed))

    try:
        x_cal, y_cal = fetch_california_housing(
            return_X_y=True,
            data_home=str(cache_dir),
        )
        datasets.append(
            _to_regression_dataset("california_housing", x_cal, y_cal, seed)
        )
    except Exception:
        pass

    # OpenML can be unavailable in offline environments. We skip gracefully.
    try:
        energy = fetch_openml(
            name="energy-efficiency",
            version=1,
            as_frame=True,
            data_home=str(cache_dir),
        )
        frame = energy.frame
        if frame is not None and "Y1" in frame.columns:
            x_energy = frame.drop(columns=["Y1", "Y2"], errors="ignore").to_numpy(
                dtype=np.float32
            )
            y_energy = frame["Y1"].to_numpy(dtype=np.float32)
            datasets.append(
                _to_regression_dataset("energy_efficiency", x_energy, y_energy, seed)
            )
    except Exception:
        pass

    return datasets
