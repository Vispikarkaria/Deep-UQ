# Graph Operator + Gray-Scott + Deep Ensembles

Notebook: [GraphOperator_GrayScott_Ensemble_Tutorial.ipynb](https://github.com/Vispikarkaria/Deep-UQ/blob/master/notebooks/graphs/GraphOperator_GrayScott_Ensemble_Tutorial.ipynb)

This tutorial introduces a graph-based neural operator for scientific machine
learning using a Gray-Scott reaction-diffusion system. The main model is
`GraphNeuralOperator2D`, which treats a regular Cartesian grid as a graph and
applies local message passing over node embeddings.

Primary task:

- input: two-species state `[A(x,y,t), B(x,y,t)]`
- output: one-step-ahead state `[A(x,y,t+\Delta t), B(x,y,t+\Delta t)]`

Primary UQ method:

- `DeepEnsembleWrapper` over multiple independently trained graph operators

Optional comparison:

- last-layer Laplace with `LaplaceWrapper(..., subset_of_weights="last_layer", hessian_structure="block_diag")`

Why this tutorial matters:

- it is the package's first graph-based neural-operator example,
- it shows how to use a graph model even when the underlying data lives on a
  regular grid,
- it compares in-domain and held-out Gray-Scott regimes,
- it visualizes field-wise predictive mean and epistemic standard deviation for
  both species.

Dataset note:

- if a local The Well Gray-Scott archive is available, the notebook loads it
  through `deepuq.data.the_well`,
- otherwise quick mode falls back to a small synthetic Gray-Scott generator so
  the notebook remains executable in a self-contained way.

Primary references cited in the notebook:

- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*
- Battaglia et al., *Relational inductive biases, deep learning, and graph networks*
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
- MacKay (1992), Ritter et al. (2018), and Daxberger et al. (2021) for the optional Laplace comparison
