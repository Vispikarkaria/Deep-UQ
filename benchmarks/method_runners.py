from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.config import BenchmarkConfig
from benchmarks.datasets import RegressionDataset
from deepuq.methods import (
    BayesByBackpropMLP,
    LaplaceWrapper,
    MCDropoutWrapper,
    predict_vi_uq,
    vi_elbo_step,
)
from deepuq.models import GaussianProcessRegressor, MLP, SparseGaussianProcessRegressor


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _subset_train_data(
    x: torch.Tensor,
    y: torch.Tensor,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= max_points:
        return x, y
    idx = torch.randperm(x.shape[0])[:max_points]
    return x[idx], y[idx]


def _train_mlp_regression(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()


def run_mc_dropout(dataset: RegressionDataset, cfg: BenchmarkConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    x_train, y_train = _subset_train_data(dataset.x_train, dataset.y_train, cfg.max_train_points)
    train_loader = _loader(x_train, y_train, cfg.batch_size)
    model = MLP(dataset.x_train.shape[1], cfg.hidden_dims, 1, p_drop=0.1)

    t0 = time.perf_counter()
    _train_mlp_regression(model, train_loader, cfg.train_epochs, cfg.lr)
    train_time = time.perf_counter() - t0

    wrapper = MCDropoutWrapper(model, n_mc=cfg.num_samples, apply_softmax=False)
    t1 = time.perf_counter()
    uq = wrapper.predict_uq(dataset.x_test)
    infer_time = time.perf_counter() - t1

    return {
        "method": "mc_dropout",
        "mean": uq.mean.detach().cpu(),
        "var": uq.total_var.detach().cpu() if uq.total_var is not None else None,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def run_laplace(dataset: RegressionDataset, cfg: BenchmarkConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    x_train, y_train = _subset_train_data(dataset.x_train, dataset.y_train, cfg.max_train_points)
    train_loader = _loader(x_train, y_train, cfg.batch_size)
    model = MLP(dataset.x_train.shape[1], cfg.hidden_dims, 1, p_drop=0.0)

    t0 = time.perf_counter()
    _train_mlp_regression(model, train_loader, cfg.train_epochs, cfg.lr)
    la = LaplaceWrapper(model, likelihood="regression", hessian_structure="diag", subset_of_weights="last_layer")
    la.fit(train_loader, prior_precision=1.0)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    uq = la.predict_uq(dataset.x_test, n_samples=cfg.num_samples)
    infer_time = time.perf_counter() - t1
    return {
        "method": "laplace_diag",
        "mean": uq.mean.detach().cpu(),
        "var": uq.total_var.detach().cpu() if uq.total_var is not None else None,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def run_vi(dataset: RegressionDataset, cfg: BenchmarkConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    x_train, y_train = _subset_train_data(dataset.x_train, dataset.y_train, cfg.max_train_points)
    train_loader = _loader(x_train, y_train, cfg.batch_size)
    model = BayesByBackpropMLP(dataset.x_train.shape[1], cfg.hidden_dims, 1, prior_sigma=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss(reduction="mean")
    num_batches = len(train_loader)

    t0 = time.perf_counter()
    model.train()
    for _ in range(cfg.vi_epochs):
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, _, _ = vi_elbo_step(
                model,
                xb,
                yb,
                num_batches=num_batches,
                criterion=criterion,
                kl_weight=0.01,
                mc_samples=1,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    uq = predict_vi_uq(model, dataset.x_test, n_samples=cfg.num_samples, apply_softmax=False)
    infer_time = time.perf_counter() - t1
    return {
        "method": "vi_bayes_by_backprop",
        "mean": uq.mean.detach().cpu(),
        "var": uq.total_var.detach().cpu() if uq.total_var is not None else None,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def run_exact_gp(dataset: RegressionDataset, cfg: BenchmarkConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    x_train, y_train = _subset_train_data(dataset.x_train, dataset.y_train, cfg.max_train_points)
    if x_train.shape[0] > cfg.max_exact_gp_train:
        idx = torch.randperm(x_train.shape[0])[: cfg.max_exact_gp_train]
        x_train = x_train[idx]
        y_train = y_train[idx]

    model = GaussianProcessRegressor(noise=1e-4)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    uq = model.predict_uq(dataset.x_test)
    infer_time = time.perf_counter() - t1
    return {
        "method": "exact_gp",
        "mean": uq.mean.detach().cpu().unsqueeze(-1),
        "var": uq.total_var.detach().cpu().unsqueeze(-1) if uq.total_var is not None else None,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def run_sparse_gp(dataset: RegressionDataset, cfg: BenchmarkConfig) -> Dict[str, object]:
    _set_seed(cfg.seed)
    x_train, y_train = _subset_train_data(dataset.x_train, dataset.y_train, cfg.max_train_points)
    if x_train.shape[0] > cfg.max_sparse_gp_train:
        idx = torch.randperm(x_train.shape[0])[: cfg.max_sparse_gp_train]
        x_train = x_train[idx]
        y_train = y_train[idx]

    model = SparseGaussianProcessRegressor(
        num_inducing=min(64, x_train.shape[0]),
        learning_rate=0.05,
        num_iterations=300 if cfg.train_epochs >= 100 else 120,
        verbose=False,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    uq = model.predict_uq(dataset.x_test)
    infer_time = time.perf_counter() - t1
    return {
        "method": "sparse_gp",
        "mean": uq.mean.detach().cpu().unsqueeze(-1),
        "var": uq.total_var.detach().cpu().unsqueeze(-1) if uq.total_var is not None else None,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def run_all_methods(dataset: RegressionDataset, cfg: BenchmarkConfig) -> List[Dict[str, object]]:
    runners = [run_mc_dropout, run_laplace, run_vi, run_exact_gp, run_sparse_gp]
    outputs: List[Dict[str, object]] = []
    for run in runners:
        try:
            outputs.append(run(dataset, cfg))
        except Exception as exc:
            outputs.append(
                {
                    "method": run.__name__.replace("run_", ""),
                    "mean": None,
                    "var": None,
                    "train_time_sec": float("nan"),
                    "infer_time_sec": float("nan"),
                    "error": str(exc),
                }
            )
    return outputs
