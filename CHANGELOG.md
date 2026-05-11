# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.19] - 2026-04-01

### Fixed
- Restored automatic `laplace-torch` integration for `hessian_structure="kron"` and `"full"` when the optional dependency is installed.

### Added
- `laplace` extra (`uqdeepnn[laplace]`) for PyPI users needing the `laplace-torch` backend.

## [0.1.18] - 2026-03-28

### Added
- `GraphNeuralOperator2D` as a grid-as-graph message-passing neural operator with optional last-layer Laplace compatibility.
- Local The Well Gray-Scott loading utilities.
- Graph-neural-operator ensemble tutorial under `notebooks/graphs/`.

## [0.1.17] - 2026-03-12

### Added
- Heteroscedastic VI regression, multi-output VI regression, and last-layer VI wrappers.
- Five executable VI tutorials under `notebooks/vi/`.
- Notebook smoke validation support.

### Changed
- Strengthened VI API docs with workflow conventions and tensor-shape guidance.

## [0.1.16] - 2026-03-10

### Added
- Five scientific deep-ensemble tutorials (ADR, heteroscedastic, classification, multi-output, transport).
- Regression, classification, multi-output, and heteroscedastic ensemble test coverage.

### Changed
- Expanded deep-ensemble method guide with equations, uncertainty decompositions, and citations.

## [0.1.15] - 2026-03-10

### Added
- `ConditionalUNet2D` and `SinusoidalTimeEmbedding` in `deepuq.models.diffusion`.
- Conditional diffusion heat-field reconstruction tutorial.
- Diffusion-model unit tests.

### Fixed
- CPU/CUDA device mismatch in notebook diffusion schedule.

## [0.1.14] - 2026-03-10

### Added
- 2D Fourier Neural Operator: `SpectralConv2D`, `FNOBlock2D`, `FNO2D`.
- FNO2D Darcy-flow Laplace tutorial with GIF export.

## [0.1.13] - 2026-03-10

### Changed
- Refined PINN Poisson tutorial with `Adam -> LBFGS` schedule and higher-prior last-layer Laplace.
- Compared `diag` and `block_diag` last-layer Laplace backends in PINN notebook.

## [0.1.12] - 2026-03-08

### Changed
- Switched PyPI long-description to dedicated `README_PYPI.md`.

## [0.1.11] - 2026-03-08

### Added
- `DeepEnsembleWrapper` for regression-first multi-model UQ.
- `CNNRegressor2D`, `ResNetRegressor2D`, `UNet2D`, `UNet3D`, `PINN1D`, `PINN2D`.
- Model architecture inventory page and SciML tutorial notebooks.

## [0.1.10] - 2026-03-08

### Added
- 3D Fourier Neural Operator: `SpectralConv3D`, `FNOBlock3D`, `FNO3D`.
- FNO3D heat-equation tutorial with slice-based uncertainty visualization.

## [0.1.9] - 2026-03-08

### Added
- `DeepONet1D` for 1D operator-learning with Laplace compatibility.
- 1D Poisson DeepONet tutorial.

## [0.1.8] - 2026-03-07

### Added
- CI workflows for tests, lint, and tag-driven PyPI release.
- Standardized `UQResult` uncertainty container.
- `predict_uq` APIs across all methods and GP models.
- Multi-dataset benchmark suite.
- `DeepONet2D` and Burgers operator-learning tutorial.

## [0.1.4] - 2026-03-03

### Fixed
- Homepage rendering for GitHub Pages deployment.
- Stale browser cache issues with docs assets.

## [0.1.3] - 2026-03-03

### Changed
- Removed external `laplace-torch` dependency; added native `kron` and `full` backends.

## [0.1.2] - 2026-03-03

### Added
- Native Laplace backends: `diag`, `fisher_diag`, `lowrank_diag`, `block_diag`, `kron`, `full`.
- Laplace tutorial notebooks.

## [0.1.1] - 2026-03-01

### Changed
- Packaging updates for PyPI publication.

## [0.1.0] - 2026-03-01

### Added
- Initial public release.
