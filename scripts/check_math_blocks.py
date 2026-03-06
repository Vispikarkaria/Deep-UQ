#!/usr/bin/env python3
"""Fail if docs markdown display-math blocks contain markdown-sensitive lines.

This prevents GitHub markdown mis-parsing in $$...$$ blocks, including:
- operator-only lines such as '=', '-', '+', '*'
- list-like lines such as '- <expr>' or '+ <expr>' that Markdown can treat as bullets
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


OPERATOR_ONLY = {"=", "-", "+", "*"}
LIST_PREFIX_RE = re.compile(r"^[+\-*]\s+")
ORDERED_LIST_PREFIX_RE = re.compile(r"^\d+\.\s+")
DEFAULT_GLOB = "docs/methods/*.md"


@dataclass
class Violation:
    path: Path
    block_index: int
    line_number: int
    token: str
    reason: str


def iter_display_blocks(text: str):
    """Yield (block_index, start_line, block_content) for each $$...$$ block."""
    parts = text.split("$$")
    for idx, block in enumerate(parts[1::2], start=1):
        # Count lines before this block starts.
        prefix = "$$".join(parts[: 2 * idx - 1])
        start_line = prefix.count("\n") + 1
        yield idx, start_line, block


def check_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8")
    if text.count("$$") % 2 != 0:
        # Keep this script scoped to operator-only guard as requested.
        # Unbalanced delimiters are reported as a top-level violation-like message.
        print(f"[math-guard] WARNING: odd number of '$$' delimiters in {path}")

    violations: list[Violation] = []
    for block_idx, start_line, block in iter_display_blocks(text):
        for rel_line, line in enumerate(block.splitlines(), start=1):
            token = line.strip()
            if token in OPERATOR_ONLY:
                violations.append(
                    Violation(
                        path=path,
                        block_index=block_idx,
                        line_number=start_line + rel_line - 1,
                        token=token,
                        reason="operator-only line",
                    )
                )
                continue

            if LIST_PREFIX_RE.match(token) or ORDERED_LIST_PREFIX_RE.match(token):
                violations.append(
                    Violation(
                        path=path,
                        block_index=block_idx,
                        line_number=start_line + rel_line - 1,
                        token=token,
                        reason="markdown-list-like line",
                    )
                )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check markdown display-math blocks for markdown-sensitive lines."
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=f"Path glob to scan (default: {DEFAULT_GLOB})",
    )
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        print(f"[math-guard] No files matched glob: {args.glob}")
        return 0

    all_violations: list[Violation] = []
    for path in paths:
        all_violations.extend(check_file(path))

    if not all_violations:
        print(
            f"[math-guard] OK: {len(paths)} files checked; no markdown-sensitive lines found in display math blocks."
        )
        return 0

    print("[math-guard] Found markdown-sensitive lines inside $$...$$ blocks:")
    for violation in all_violations:
        print(
            f"  - {violation.path}:{violation.line_number} "
            f"(block #{violation.block_index}) {violation.reason}: '{violation.token}'"
        )

    print("[math-guard] Fix by keeping operators/list markers away from start-of-line inside $$...$$ blocks.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
