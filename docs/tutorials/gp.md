# Tutorial: Gaussian Processes

The GP tutorial suite lives under `notebooks/gp/` and covers exact, sparse, classification, heteroscedastic, multitask, spectral, and deep-kernel models.

## Notebook Index

- [GP_Exact_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Exact_Tutorial.ipynb)
- [GP_Sparse_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Sparse_Tutorial.ipynb)
- [GP_Kernel_Zoo_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Kernel_Zoo_Tutorial.ipynb)
- [GP_Classification_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Classification_Tutorial.ipynb)
- [GP_Heteroscedastic_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Heteroscedastic_Tutorial.ipynb)
- [GP_MultiTask_ICM_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_MultiTask_ICM_Tutorial.ipynb)
- [GP_SpectralMixture_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_SpectralMixture_Tutorial.ipynb)
- [GP_DeepKernel_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_DeepKernel_Tutorial.ipynb)
- [GP_Model_Comparison.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/gp/GP_Model_Comparison.ipynb)

## Legacy Paths

The original root notebook paths are kept as lightweight compatibility stubs:

- `notebooks/GaussianProcess_Tutorial.ipynb`
- `notebooks/SparseGaussianProcess_Tutorial.ipynb`

## Common Evaluation Protocol

Across notebooks, we emphasize calibration-first metrics:

- RMSE
- Gaussian NLL
- 95% interval coverage
- Mean 95% interval width

Each notebook uses deterministic seeds and quick default configs for CPU-friendly runtime.
