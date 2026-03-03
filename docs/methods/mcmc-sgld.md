# MCMC (SGLD)

Deep-UQ includes Stochastic Gradient Langevin Dynamics for posterior sampling in neural networks.

## Components

- `SGLDOptimizer`: SGD update plus calibrated Gaussian noise.
- `collect_posterior_samples`: collects model snapshots after burn-in.
- `predict_with_samples`: Monte Carlo prediction from collected parameter states.

## Update Rule

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) + \sqrt{2\eta}\,\epsilon_t,
\quad \epsilon_t \sim \mathcal{N}(0, I)
$$

## Practical Guidance

- Start with small learning rates.
- Use sufficient burn-in before collecting snapshots.
- Evaluate predictive mean and variance from many retained samples.

## References

- [SGLD Tutorial Guide](../tutorials/sgld.md)
- [MCMC API](../api/methods/mcmc.md)
