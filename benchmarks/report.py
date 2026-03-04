from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("# Benchmark Summary\n\nNo rows generated.\n")
        return

    headers = [
        "dataset",
        "method",
        "rmse",
        "mae",
        "nll",
        "coverage95",
        "interval_width95",
        "train_time_sec",
        "infer_time_sec",
        "status",
    ]
    lines = [
        "# Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
