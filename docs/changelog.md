# Changelog

## 0.1.10 — 2026-03-08

Highlights:

- Added reusable 3D Fourier Neural Operator components: `SpectralConv3D`, `FNOBlock3D`, and `FNO3D`.
- Added `notebooks/sciml/FNO3D_Heat_Laplace_Tutorial.ipynb` covering a 3D periodic heat-equation surrogate with slice-based uncertainty visualization.
- Added tests for 3D FNO forward behavior and last-layer Laplace compatibility.
- Added tutorial docs and site navigation for the new 3D SciML notebook.

## 0.1.9 — 2026-03-08

Highlights:

- Added `DeepONet1D` for fixed-grid 1D operator-learning workflows that remain compatible with `LaplaceWrapper`.
- Added `notebooks/sciml/DeepONet_Poisson1D_Laplace_Tutorial.ipynb` covering a 1D Poisson problem with sparse forcing sensors, residual DeepONet training, and last-layer Laplace uncertainty bands.
- Extended DeepONet tests to cover the 1D model and last-layer Laplace compatibility.
- Expanded the 2D Burgers SciML notebook explanations and comments for easier first-time reading.
- Added tutorial docs and site navigation for the new 1D SciML notebook.

## 0.1.8 — 2026-03-07

Highlights:

- Added CI workflows for tests (`tests.yml`) and quality checks (`lint.yml`).
- Added tag-driven PyPI release workflow (`release.yml`) with trusted publishing.
- Introduced standardized uncertainty container `deepuq.UQResult`.
- Added non-breaking `predict_uq` APIs across methods and GP models.
- Added manual multi-dataset benchmark suite under `benchmarks/`.
- Added tracked-data policy docs and large-file guard script.
- Added packaging extras in `pyproject.toml` (`dev`, `tests`, `docs`, `benchmarks`, `notebooks`).
- Updated docs links and usage/examples for the unified uncertainty API.
- Added `DeepONet2D` for operator-learning experiments in scientific machine learning.
- Added a new `notebooks/sciml/DeepONet_Burgers_Laplace_Tutorial.ipynb` tutorial covering 2D viscous Burgers operator learning with Laplace uncertainty.
- Added tests for DeepONet forward behavior and Laplace last-layer compatibility.
- Added tutorial docs and navigation for the new SciML notebook section.

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
