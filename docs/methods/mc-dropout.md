# MC Dropout

MC Dropout treats dropout masks as approximate posterior samples at inference.

## Component

- `MCDropoutWrapper`:
  - keeps dropout active during prediction
  - performs `n_mc` stochastic forward passes
  - returns predictive mean and variance

## Practical Guidance

- Increase `n_mc` for smoother uncertainty estimates.
- Use `apply_softmax=True` for classification logits.
- Use dropout in training architecture (`p_drop > 0`) if MC Dropout is required at inference.

## References

- [MC Dropout Tutorial Guide](../tutorials/mc-dropout.md)
- [MC Dropout API](../api/methods/mc_dropout.md)
