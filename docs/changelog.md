# Changelog

## Unreleased

Highlights:

- Added CI workflows for tests (`tests.yml`) and quality checks (`lint.yml`).
- Added tag-driven PyPI release workflow (`release.yml`) with trusted publishing.
- Introduced standardized uncertainty container `deepuq.UQResult`.
- Added non-breaking `predict_uq` APIs across methods and GP models.
- Added manual multi-dataset benchmark suite under `benchmarks/`.
- Added tracked-data policy docs and large-file guard script.
- Added packaging extras in `pyproject.toml` (`dev`, `tests`, `docs`, `benchmarks`, `notebooks`).
- Updated docs links and usage/examples for the unified uncertainty API.

## 0.1.4 — 2026-03-03

Highlights:

- Fixed homepage rendering behavior for GitHub Pages deployment.
- Added cache-busted docs assets (`extra-v2.css`, `extra-v2.js`) to avoid stale browser rendering.
- Removed stale `laplace-torch` language from Laplace notebooks/examples.

## 0.1.3 — 2026-03-03

Highlights:

- Removed external `laplace-torch` dependency from package requirements.
- Added native `kron` and `full` Laplace backends under `LaplaceWrapper`.
- Updated Laplace docs/usage notes to reflect native support for all Hessian structures.
- Refreshed Laplace Hessian comparison tutorial updates.

## 0.1.2 — 2026-03-03

Highlights:

- Expanded Laplace support through `LaplaceWrapper`:
  - native `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`
  - native `kron` and `full` implementations (no `laplace-torch` dependency)
- Added Laplace notebooks under `notebooks/laplace/`:
  - full-Hessian tutorial
  - multi-structure comparison tutorial
- Improved VI/Laplace tutorial consistency and diagnostics
- Refined README method table and package documentation links

## 0.1.1

- Packaging and release updates for PyPI publication

## 0.1.0

- Initial public package release
