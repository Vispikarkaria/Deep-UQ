# Changelog

## 0.1.15 — 2026-03-10

Highlights:

- Added `ConditionalUNet2D` and `SinusoidalTimeEmbedding` in `deepuq.models.diffusion` for conditional diffusion tutorials on scientific fields.
- Added `notebooks/sciml/ConditionalDiffusion_Heat2D_Tutorial.ipynb` for sparse-sensor 2D heat-field reconstruction with sample-based uncertainty.
- Added diffusion-model docs coverage via `docs/tutorials/sciml-conditional-diffusion-heat2d.md` and `docs/api/models/diffusion.md`.
- Updated the model-architecture inventory and tutorial index to include generative / diffusion models.
- Added diffusion-model unit tests and fixed the notebook diffusion schedule to avoid CPU/CUDA device mismatch during execution.

## 0.1.14 — 2026-03-10

Highlights:

- Added reusable 2D Fourier Neural Operator components: `SpectralConv2D`, `FNOBlock2D`, and `FNO2D`.
- Added `notebooks/sciml/FNO2D_Darcy_Laplace_Tutorial.ipynb` covering a 2D Darcy-flow operator surrogate with three input fields and last-layer Laplace uncertainty maps.
- Added tests for 2D FNO forward behavior and last-layer Laplace compatibility.
- Added tutorial docs and architecture-table coverage for the new Darcy operator-learning example.
- Added GIF export in the Darcy notebook to compare input fields, true pressure, predictive mean, error, and epistemic uncertainty across samples.

## 0.1.13 — 2026-03-10

Highlights:

- Refined the SciML PINN Poisson tutorial with richer interior anchors, an `Adam -> LBFGS` training schedule, and a higher-prior last-layer Laplace fit.
- Updated the PINN notebook to compare `diag` and `block_diag` last-layer Laplace backends and keep the more stable posterior view.
- Clarified the deep-ensemble Poisson tutorial to distinguish parameter-space OOD uncertainty from physical-space data sparsity.
- Refreshed the affected notebooks and tutorial pages so the GitHub repo and docs site stay aligned with the improved notebook behavior.

## 0.1.12 — 2026-03-08

Highlights:

- Switched PyPI long-description rendering to a dedicated `README_PYPI.md`.
- Removed GitHub workflow badges and repo-specific overview content from the PyPI package description.
- Aligned naming in the package presentation around `Deep-UQ` (docs/project), `uqdeepnn` (PyPI), and `deepuq` (Python import).

## 0.1.11 — 2026-03-08

Highlights:

- Added `DeepEnsembleWrapper` as a regression-first multi-model UQ baseline.
- Added new predictive backbones: `CNNRegressor2D`, `ResNetRegressor2D`, `UNet2D`, `UNet3D`, `PINN1D`, and `PINN2D`.
- Added the new model architecture inventory page plus API docs for ensembles, spatial models, and PINNs.
- Added new SciML tutorial notebooks for deep ensembles, CNN / ResNet heat surrogates, U-Net diffusion surrogates, and PINN Poisson problems.

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
