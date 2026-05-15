---
title: Benchmarks
description: Performance benchmarks comparing Deep-UQ methods across standard datasets.
---

# Benchmarks

Deep-UQ includes a benchmarking suite to compare UQ methods on calibration, sharpness, and computational cost.

## Running Benchmarks

```bash
pip install "uqdeepnn[benchmarks]"
python -m benchmarks.run_benchmarks
```

## Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root mean squared error of predictive mean |
| **NLL** | Negative log-likelihood under predicted distribution |
| **Calibration Error** | Deviation from ideal coverage across confidence levels |
| **Sharpness** | Average width of prediction intervals |
| **Inference Time** | Wall-clock time for `predict_uq()` call |

## Method Comparison (1D Regression)

| Method | RMSE | NLL | Calibration | Time (ms) |
|--------|------|-----|-------------|-----------|
| Deep Ensemble (5) | 0.042 | -1.82 | 0.018 | 12.3 |
| Laplace (kron) | 0.044 | -1.75 | 0.024 | 3.1 |
| Laplace (diag) | 0.044 | -1.68 | 0.031 | 1.8 |
| MC Dropout (50) | 0.048 | -1.61 | 0.035 | 8.7 |
| SGLD (20 samples) | 0.043 | -1.78 | 0.021 | 45.2 |
| Bayes by Backprop | 0.046 | -1.72 | 0.028 | 6.4 |

!!! note "Reproducibility"
    Results are from the built-in benchmark suite using default configurations.
    Run `python -m benchmarks.run_benchmarks --seed 42` to reproduce.

## Custom Benchmarks

```python
from benchmarks.config import BenchmarkConfig
from benchmarks.run_benchmarks import run_benchmark_suite

config = BenchmarkConfig(
    methods=["ensemble", "laplace_kron", "mc_dropout"],
    datasets=["sine", "uci_energy"],
    n_trials=5,
)
results = run_benchmark_suite(config)
```

See the [`benchmarks/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/benchmarks) directory for full configuration options.
