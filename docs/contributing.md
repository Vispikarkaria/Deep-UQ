# Contributing

Thanks for contributing to Deep-UQ.

## Development Setup

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git
cd Deep-UQ
pip install -e .
```

## Run Tests

```bash
pytest -q
```

## Optional Docs Build

```bash
pip install -r docs/requirements.txt
mkdocs build --strict
```

## Pull Request Checklist

- Add or update tests for code changes
- Keep APIs backward compatible unless clearly versioned
- Update tutorials/docs if behavior changes
- Keep notebook code cells executable and syntactically valid

## Coding Style

- Use clear names and simple control flow
- Add comments only when code is not self-evident
- Preserve stable return contracts in public wrappers
