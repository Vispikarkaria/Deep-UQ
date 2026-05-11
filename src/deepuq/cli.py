"""Deep-UQ command-line interface."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepuq",
        description="Deep-UQ: Unified deep learning uncertainty quantification toolkit",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit"
    )
    subparsers = parser.add_subparsers(dest="command")

    bench_parser = subparsers.add_parser(
        "benchmark", help="Run the benchmark suite"
    )
    bench_parser.add_argument(
        "--preset",
        default="quick",
        choices=["quick", "standard", "full"],
        help="Benchmark preset (default: quick)",
    )
    bench_parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for results",
    )

    info_parser = subparsers.add_parser("info", help="Show package information")

    args = parser.parse_args()

    if args.version:
        from deepuq._version import __version__

        print(f"deepuq {__version__}")
        sys.exit(0)

    if args.command == "benchmark":
        from benchmarks.run_benchmarks import main as run_bench

        sys.argv = ["deepuq", "--preset", args.preset]
        run_bench()
    elif args.command == "info":
        from deepuq._version import __version__

        print(f"Deep-UQ v{__version__}")
        print(f"PyPI: uqdeepnn")
        print(f"Docs: https://vispikarkaria.github.io/Deep-UQ/")
        print(f"Repo: https://github.com/Vispikarkaria/Deep-UQ")
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
