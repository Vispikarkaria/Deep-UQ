# Scientific Conditional Diffusion for Heat Fields

Notebook: [ConditionalDiffusion_Heat2D_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/sciml/generative/ConditionalDiffusion_Heat2D_Tutorial.ipynb)

This tutorial builds a **conditional diffusion model** for 2D heat fields.
The task is a scientific field-reconstruction problem rather than a deterministic
operator map: sparse sensor measurements of a smooth heat field are used to
construct a conditional distribution over the full field.

Key ideas:

- the physical field comes from the 2D heat equation,
- conditioning is intentionally incomplete: sparse sensors plus a mask and a
  smooth inpainting baseline,
- predictive uncertainty is computed from the **spread of generated samples**,
- uncertainty should be largest away from sensors and larger on OOD fields than
  on in-domain fields.

The notebook shows:

- heat-field generation from an exact spectral heat solver,
- a reusable `ConditionalUNet2D` denoiser,
- DDPM-style training on residual fields,
- conditional sampling with data-consistency clamping at sensor points,
- sample mean, predictive standard deviation, and in-domain vs OOD comparisons.

Primary references:

- Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*
- Song et al. (2021), *Score-Based Generative Modeling through Stochastic Differential Equations*
- Ronneberger, Fischer, Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*
