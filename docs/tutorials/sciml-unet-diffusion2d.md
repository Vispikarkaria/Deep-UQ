# U-Net + Diffusion2D UQ

Notebook: [UNet_Diffusion2D_UQ_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/spatial/UNet_Diffusion2D_UQ_Tutorial.ipynb)

This tutorial uses `UNet2D` for a 2D diffusion field-to-field mapping. The main
UQ path is MC Dropout, while the tutorial also discusses how the same backbone
can be paired with ensemble uncertainty for stronger multi-model diversity.

Primary references:

- Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
