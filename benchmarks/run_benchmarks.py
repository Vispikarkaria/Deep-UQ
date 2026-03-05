#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.config import BenchmarkConfig, preset
from benchmarks.datasets import load_regression_datasets
from benchmarks.metrics import regression_metrics
from benchmarks.method_runners import run_all_methods
from benchmarks.report import write_csv, write_markdown_summary


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def run(config: BenchmarkConfig, output_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for dataset in load_regression_datasets(seed=config.seed):
        for output in run_all_methods(dataset, config):
            row = {
                "dataset": dataset.name,
                "method": output["method"],
            }
            if output.get("mean") is None:
                row.update(
                    {
                        "rmse": "nan",
                        "mae": "nan",
                        "nll": "nan",
                        "coverage95": "nan",
                        "interval_width95": "nan",
                        "train_time_sec": _fmt(
                            float(output.get("train_time_sec", float("nan")))
                        ),
                        "infer_time_sec": _fmt(
                            float(output.get("infer_time_sec", float("nan")))
                        ),
                        "status": f"error: {output.get('error', 'unknown')}",
                    }
                )
            else:
                metrics = regression_metrics(
                    dataset.y_test, output["mean"], output.get("var")
                )
                row.update(
                    {
                        "rmse": _fmt(metrics["rmse"]),
                        "mae": _fmt(metrics["mae"]),
                        "nll": _fmt(metrics["nll"]),
                        "coverage95": _fmt(metrics["coverage95"]),
                        "interval_width95": _fmt(metrics["interval_width95"]),
                        "train_time_sec": _fmt(float(output["train_time_sec"])),
                        "infer_time_sec": _fmt(float(output["infer_time_sec"])),
                        "status": "ok",
                    }
                )
            rows.append(row)

    write_csv(rows, output_dir / "results.csv")
    write_markdown_summary(rows, output_dir / "summary.md")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Deep-UQ regression benchmarks.")
    parser.add_argument("--preset", default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    cfg = preset(args.preset)
    if args.seed is not None:
        cfg.seed = args.seed
    output_dir = Path(args.output_dir)
    rows = run(cfg, output_dir)
    print(f"Wrote {len(rows)} benchmark rows to {output_dir}")


if __name__ == "__main__":
    main()
