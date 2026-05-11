# Contributing to Deep-UQ

Thank you for your interest in contributing to Deep-UQ! This guide will help you get started.

## Code of Conduct

Be respectful, constructive, and collaborative. We welcome contributors of all experience levels.

## Development Setup

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git
cd Deep-UQ
pip install -e ".[dev,tests,docs]"
pre-commit install
```

## Development Workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes with tests.
3. Run the quality checks locally before pushing.
4. Open a pull request against `master`.

## Quality Checks

```bash
# Formatting
black --check .

# Linting
ruff check .

# Type checking
mypy

# Tests
pytest -q

# Docs (optional)
mkdocs build --strict
```

All checks run in CI via GitHub Actions. PRs must pass before merging.

## Pull Request Guidelines

- **One concern per PR** — keep changes focused and reviewable.
- **Add tests** for new functionality or bug fixes.
- **Update docs** if you change public API behavior.
- **Keep notebooks executable** — code cells must run without error.
- **Follow existing style** — Black formatting, Ruff linting, clear naming.

## Commit Messages

Use clear, imperative-mood commit messages:

- `Add sparse GP predict_uq method`
- `Fix Laplace kron backend numerical stability`
- `Update FNO tutorial with new data loader`

## Adding a New UQ Method

1. Implement in `src/deepuq/methods/` following existing patterns.
2. Return `UQResult` from `predict_uq()`.
3. Add tests in `tests/`.
4. Add a tutorial notebook in `notebooks/`.
5. Update `docs/` and the method inventory in the README.

## Adding a New Model Architecture

1. Implement in `src/deepuq/models/` following existing patterns.
2. Ensure compatibility with at least one UQ method.
3. Add tests in `tests/`.
4. Update the architecture inventory in docs.

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Python/PyTorch versions
- Minimal reproducible example
- Expected vs actual behavior

## Requesting Features

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and describe:

- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
