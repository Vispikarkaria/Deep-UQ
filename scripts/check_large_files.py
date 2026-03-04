#!/usr/bin/env python3
"""Guard against newly committed oversized files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(p) for p in result.stdout.splitlines() if p]


def _format_size(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{mib:.2f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-threshold-mb", type=float, default=20.0)
    parser.add_argument("--fail-threshold-mb", type=float, default=50.0)
    args = parser.parse_args()

    warn_bytes = int(args.warn_threshold_mb * 1024 * 1024)
    fail_bytes = int(args.fail_threshold_mb * 1024 * 1024)
    if warn_bytes > fail_bytes:
        parser.error("warn threshold must be <= fail threshold")

    warnings: list[tuple[Path, int]] = []
    failures: list[tuple[Path, int]] = []
    for path in _tracked_files():
        if not path.exists() or not path.is_file():
            continue
        size = path.stat().st_size
        if size >= fail_bytes:
            failures.append((path, size))
        elif size >= warn_bytes:
            warnings.append((path, size))

    for path, size in sorted(warnings, key=lambda item: item[1], reverse=True):
        print(f"WARNING: {path} is large ({_format_size(size)})")

    for path, size in sorted(failures, key=lambda item: item[1], reverse=True):
        print(f"ERROR: {path} exceeds fail threshold ({_format_size(size)})")

    if failures:
        print(
            f"Found {len(failures)} files above {args.fail_threshold_mb:.1f} MiB. "
            "Either shrink files or raise threshold intentionally."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
