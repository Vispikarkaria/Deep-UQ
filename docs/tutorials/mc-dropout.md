# Tutorial: MC Dropout

Notebook: [MC_Dropout_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/MC_Dropout_Tutorial.ipynb)

## Purpose

Demonstrate predictive uncertainty from dropout-enabled neural networks via Monte Carlo inference.

## Data Setup

- Beam-like nonlinear regression case
- Structured deflection behavior
- Regions with sparse supervision to stress uncertainty estimates

## Core Logic

- Train MLP with dropout
- Enable dropout during evaluation
- Aggregate many stochastic passes

## Expected Outputs

- predictive mean curve
- uncertainty bands that reflect data support and complexity
