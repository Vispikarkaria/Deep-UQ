# Benchmark Suite

This folder contains a manual/local benchmark pipeline for comparing Deep-UQ
methods on multiple regression datasets.

## Datasets (v1)

- Diabetes (`sklearn.datasets.load_diabetes`)
- California Housing (`sklearn.datasets.fetch_california_housing`)
- Energy Efficiency (`OpenML`, optional and skipped if unavailable)

## Run

Quick run:

```bash
python benchmarks/run_benchmarks.py --preset quick
```

Full run:

```bash
python benchmarks/run_benchmarks.py --preset full
```

## Outputs

- `benchmarks/results/results.csv`
- `benchmarks/results/summary.md`

Downloaded sklearn/OpenML datasets are cached under `benchmarks/cache/sklearn_data/`.
