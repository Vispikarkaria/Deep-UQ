# Tutorials

These guides summarize each notebook and link directly to source notebooks in GitHub.

## Tutorial Families

### Deep Ensembles

- [Deep Ensembles + Parametric Poisson1D](sciml-deep-ensemble-poisson1d.md)
- Full method docs: [`/methods/deep-ensembles/`](../methods/deep-ensembles.md)
- Notebook: [`notebooks/sciml/DeepEnsemble_ParametricPoisson1D_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/DeepEnsemble_ParametricPoisson1D_Tutorial.ipynb)

### Variational Inference

- [Bayes by Backprop guide](bayes-by-backprop.md)
- Full method docs: [`/methods/variational-inference/`](../methods/variational-inference.md)
- Notebook: [`notebooks/BayesByBackprop_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/BayesByBackprop_Tutorial.ipynb)

### Laplace Approximation

- [Laplace comparison guide](laplace-comparison.md)
- Full method docs: [`/methods/laplace/`](../methods/laplace.md)
- Notebook directory: [`notebooks/laplace/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks/laplace)

### MCMC / SGLD

- [SGLD guide](sgld.md)
- Full method docs: [`/methods/mcmc-sgld/`](../methods/mcmc-sgld.md)
- Notebook: [`notebooks/SGLD_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/SGLD_Tutorial.ipynb)

### MC Dropout

- [MC Dropout guide](mc-dropout.md)
- Full method docs: [`/methods/mc-dropout/`](../methods/mc-dropout.md)
- Notebook: [`notebooks/MC_Dropout_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/MC_Dropout_Tutorial.ipynb)

### Gaussian Processes

- [Gaussian Processes guide](gp.md)
- Full method docs: [`/methods/gaussian-processes/`](../methods/gaussian-processes.md)
- Notebook directory: [`notebooks/gp/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks/gp)

### Scientific ML / Operator Learning

- [DeepONet + Burgers + Laplace guide](sciml-deeponet-burgers.md)
- Uses the reusable `DeepONet2D` model together with the existing Laplace backends
- Notebook: [`notebooks/sciml/DeepONet_Burgers_Laplace_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/DeepONet_Burgers_Laplace_Tutorial.ipynb)
- [DeepONet + 1D Poisson + Laplace guide](sciml-deeponet-poisson1d.md)
- Uses the reusable `DeepONet1D` model for a field-to-field 1D operator-learning problem with shaded UQ bands
- Notebook: [`notebooks/sciml/DeepONet_Poisson1D_Laplace_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/DeepONet_Poisson1D_Laplace_Tutorial.ipynb)
- [FNO3D + 3D Heat + Laplace guide](sciml-fno3d-heat.md)
- Uses the reusable `FNO3D` model for a 3D field-to-field heat-diffusion surrogate with slice-based uncertainty maps
- Notebook: [`notebooks/sciml/FNO3D_Heat_Laplace_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/FNO3D_Heat_Laplace_Tutorial.ipynb)
- [FNO2D + Darcy Flow + Laplace guide](sciml-fno2d-darcy.md)
- Uses the reusable `FNO2D` model for a 2D Darcy operator surrogate with three input fields and Laplace uncertainty maps
- Notebook: [`notebooks/sciml/FNO2D_Darcy_Laplace_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/FNO2D_Darcy_Laplace_Tutorial.ipynb)
- [CNN / ResNet + Heat2D UQ guide](sciml-cnn-resnet-heat2d.md)
- Uses `CNNRegressor2D` and `ResNetRegressor2D` on a 2D heat source-to-solution map with MC Dropout and optional ensembles
- Notebook: [`notebooks/sciml/CNN_ResNet_Heat2D_UQ_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/CNN_ResNet_Heat2D_UQ_Tutorial.ipynb)
- [UNet + Diffusion2D UQ guide](sciml-unet-diffusion2d.md)
- Uses `UNet2D` for 2D field-to-field diffusion prediction with stochastic uncertainty bands/maps
- Notebook: [`notebooks/sciml/UNet_Diffusion2D_UQ_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/UNet_Diffusion2D_UQ_Tutorial.ipynb)
- [PINN + Poisson + Laplace guide](sciml-pinn-poisson.md)
- Uses `PINN1D` and `PINN2D` for physics-informed Poisson problems with last-layer Laplace uncertainty
- Notebook: [`notebooks/sciml/PINN_Poisson_Laplace_Tutorial.ipynb`](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/PINN_Poisson_Laplace_Tutorial.ipynb)

## Notebook Source Directory

- Main notebooks: [`notebooks/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks)
- Laplace notebooks: [`notebooks/laplace/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks/laplace)
- Gaussian Process notebooks: [`notebooks/gp/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks/gp)
- Scientific ML notebooks: [`notebooks/sciml/`](https://github.com/Vispikarkaria/Deep-UQ/tree/master/notebooks/sciml)
