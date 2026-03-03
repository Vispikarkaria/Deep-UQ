# Variational Inference (Bayes by Backprop)

Deep-UQ implements mean-field Bayes by Backprop with Gaussian posteriors over weights.

## Core Components

- `GaussianPosterior`: parameterized by `mu` and `rho`; `sigma = softplus(rho)`.
- `BayesianLinear`: sampled linear layer with analytic KL against Gaussian prior.
- `BayesByBackpropMLP`: convenience MLP composed of Bayesian layers.
- `vi_elbo_step`: ELBO helper with KL scaling by `num_batches` and optional MC averaging.

## Objective

For a mini-batch:

\[
\mathcal{L}_{ELBO} = \mathbb{E}_{q(w)}[-\log p(y|x,w)] + eta \cdot rac{1}{N_b} KL(q(w)\|p(w))
\]

- `N_b`: number of optimizer steps per epoch (`len(train_loader)`)
- `beta` (`kl_weight` in code): KL weighting factor

## Practical Guidance

- Use constant `kl_weight` for comparable epoch-to-epoch ELBO traces.
- Use `mc_samples > 1` (for example `8`) for lower-variance logging.
- Track ELBO, NLL, and KL separately.

## References

- [Bayes by Backprop Tutorial](../tutorials/bayes-by-backprop.md)
- [VI API](../api/methods/vi.md)
