# Releasing `uqdeepnn`

This project uses tag-driven GitHub Actions release automation with PyPI
trusted publishing.

## One-time setup

1. In PyPI, configure a Trusted Publisher for this repo:
   - Repository: `Vispikarkaria/Deep-UQ`
   - Workflow: `.github/workflows/release.yml`
   - Environment: `pypi`
2. In GitHub, ensure environment `pypi` exists (Settings -> Environments).

## Release flow

1. Bump version in `pyproject.toml`.
2. Update changelog/docs as needed.
3. Commit and push to `master`.
4. Create and push a version tag:

```bash
git tag v0.1.8
git push origin v0.1.8
```

5. Workflow `release.yml` will:
   - build wheel + sdist,
   - run `twine check`,
   - publish to PyPI using trusted publishing.

## Notes

- Version tags should match `pyproject.toml` (`vX.Y.Z`).
- Keep `master` green (`tests.yml`, `lint.yml`, `docs.yml`) before tagging.
