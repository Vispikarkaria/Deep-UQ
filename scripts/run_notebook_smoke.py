from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"


def iter_notebooks() -> list[Path]:
    return sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))


def compile_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    for idx, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        compile(cell.source, f"{path}#cell-{idx}", "exec")


def execute_notebook(path: Path) -> None:
    os.chdir(path.parent)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    nb = nbformat.read(path, as_version=4)
    ns = {"__name__": "__main__", "__file__": str(path)}
    for idx, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        exec(compile(cell.source, f"{path}#cell-{idx}", "exec"), ns, ns)


def run_one(path: Path, mode: str) -> int:
    try:
        if mode == "compile":
            compile_notebook(path)
        else:
            execute_notebook(path)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {path}: {exc}", file=sys.stderr)
        return 1
    print(f"OK {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check Deep-UQ notebooks.")
    parser.add_argument("--mode", choices=["compile", "exec"], default="compile")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--notebook", action="append", default=[])
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        return run_one(Path(args.run_one).resolve(), args.mode)

    notebooks = (
        [Path(nb).resolve() for nb in args.notebook]
        if args.notebook
        else iter_notebooks()
    )
    failures: list[Path] = []
    for path in notebooks:
        cmd = [sys.executable, __file__, "--mode", args.mode, "--run-one", str(path)]
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        if args.mode == "exec":
            env.setdefault("DEEPUQ_NOTEBOOK_QUICK", "1")
        result = subprocess.run(cmd, cwd=REPO_ROOT, timeout=args.timeout, env=env)
        if result.returncode != 0:
            failures.append(path)
    if failures:
        print("\nNotebook smoke-check failures:", file=sys.stderr)
        for path in failures:
            print(f"- {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    print(f"Validated {len(notebooks)} notebooks in {args.mode} mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
