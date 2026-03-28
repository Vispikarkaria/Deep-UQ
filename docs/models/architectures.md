# Model Architectures

Deep-UQ separates **predictive architectures** from **uncertainty methods**.
This page is the canonical inventory of model families available in the
package, while the UQ method pages describe how posterior or predictive
uncertainty is computed.

For convolutional backbones, the primary UQ paths are **MC Dropout** and
**Deep Ensembles**. For coordinate-input PINNs, the primary UQ path is
**last-layer Laplace** because the current Laplace implementation integrates
naturally with final `nn.Linear` heads.

Legend: <span class="family-mark family-mark--yes">✓</span> supported,
<span class="family-mark family-mark--no">✗</span> not the primary path.

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Dense / Parametric Models

These models operate on vectors or coordinate-augmented point inputs and are a
natural fit for classical regression baselines, ensembles, and PINN-style
training.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| `MLP` | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | Parametric regression / classification | [Deep Ensembles](../tutorials/sciml-deep-ensemble-poisson1d.md) |
| `PINN1D` | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 1D physics-informed PDE surrogate | [PINN + Poisson](../tutorials/sciml-pinn-poisson.md) |
| `PINN2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 2D physics-informed PDE surrogate | [PINN + Poisson](../tutorials/sciml-pinn-poisson.md) |
</div>

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Spatial Convolutional Models

These are field-to-field backbones for regular scientific grids. Their main
UQ story in Deep-UQ is stochastic inference through dropout or population-level
uncertainty through ensembles.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| `CNNRegressor2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | Compact 2D PDE surrogate baseline | [CNN / ResNet + Heat2D](../tutorials/sciml-cnn-resnet-heat2d.md) |
| `ResNetRegressor2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | Stronger residual 2D surrogate baseline | [CNN / ResNet + Heat2D](../tutorials/sciml-cnn-resnet-heat2d.md) |
| `UNet2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | Multi-scale 2D field-to-field surrogate | [UNet + Diffusion2D](../tutorials/sciml-unet-diffusion2d.md) |
| `UNet3D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | Volumetric 3D field surrogate backbone | [UNet + Diffusion2D](../tutorials/sciml-unet-diffusion2d.md) |
</div>

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Operator-Learning Models

These models learn mappings between whole fields. DeepONet uses separate branch
and trunk networks, while FNO performs global spectral mixing on regular grids.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| `DeepONet1D` | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 1D field-to-field operator learning | [DeepONet + 1D Poisson](../tutorials/sciml-deeponet-poisson1d.md) |
| `DeepONet2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 2D field-to-field operator learning | [DeepONet + Burgers](../tutorials/sciml-deeponet-burgers.md) |
| `FNO2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 2D Darcy / regular-grid PDE surrogate | [FNO2D + Darcy](../tutorials/sciml-fno2d-darcy.md) |
| `FNO3D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | 3D periodic PDE surrogate | [FNO3D + 3D Heat](../tutorials/sciml-fno3d-heat.md) |
</div>

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Graph Operator Models

These models treat a regular grid as a graph and use local message passing
instead of global spectral mixing. They are a good bridge between grid-based
operator learning and future unstructured-mesh models.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| `GraphNeuralOperator2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | Grid-as-graph operator learning for reaction-diffusion and mesh-style surrogates | [Graph Operator + Gray-Scott](../tutorials/graph-operator-grayscott-ensemble.md) |
</div>

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Generative / Diffusion Models

These models represent uncertainty through the spread of generated conditional
samples rather than by wrapping a deterministic predictor with Laplace,
dropout, or ensembles.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| `ConditionalUNet2D` | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | Conditional field reconstruction with sample-based UQ | [Conditional Diffusion + Heat2D](../tutorials/sciml-conditional-diffusion-heat2d.md) |
</div>

<div class="family-table-block architecture-matrix reveal" markdown="1">
### Gaussian Process Models

The GP family is function-space Bayesian from the start, so the uncertainty
story is native rather than layered on after training.

| Model | 1D | 2D | 3D | Laplace | MC Dropout | Deep Ensemble | Primary Use | Notebook |
|---|---|---|---|---|---|---|---|---|
| GP regressors / classifiers | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--yes">✓</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | <span class="family-mark family-mark--no">✗</span> | Kernel-based Bayesian modeling | [Gaussian Processes](../tutorials/gp.md) |
</div>

## Deep Ensemble Method Note

`DeepEnsembleWrapper` is a **UQ method**, not a predictive model class. It can
wrap any compatible deterministic backbone and is the primary ensemble-based
uncertainty path for `MLP`, `CNNRegressor2D`, `ResNetRegressor2D`, `UNet2D`, and
`UNet3D`.
