#!/usr/bin/env python3
"""Verify public `deepuq.methods` exports are covered by the API docs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import deepuq.methods as methods  # noqa: E402

DOCS_DIR = REPO_ROOT / "docs" / "api" / "methods"
MKDOCS_YML = (REPO_ROOT / "mkdocs.yml").read_text()
API_INDEX = (REPO_ROOT / "docs" / "api" / "index.md").read_text()


def main() -> int:
    failures: list[str] = []
    for export in methods.__all__:
        obj = getattr(methods, export)
        module_name = obj.__module__.split(".")[-1]
        page = DOCS_DIR / f"{module_name}.md"
        if not page.exists():
            failures.append(
                f"{export}: missing API page docs/api/methods/{module_name}.md"
            )
            continue

        page_text = page.read_text()
        directive = f"::: deepuq.methods.{module_name}"
        if directive not in page_text:
            failures.append(
                f"{export}: page docs/api/methods/{module_name}.md is missing '{directive}'"
            )

        rel_path = f"api/methods/{module_name}.md"
        if rel_path not in MKDOCS_YML:
            failures.append(f"{export}: {rel_path} missing from mkdocs.yml nav")
        if f"methods/{module_name}.md" not in API_INDEX:
            failures.append(
                f"{export}: methods/{module_name}.md missing from docs/api/index.md"
            )

    if failures:
        print("API documentation coverage check failed:\n")
        for item in failures:
            print(f"- {item}")
        return 1

    print("API documentation coverage check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
