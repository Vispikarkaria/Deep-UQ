# Examples

The `examples/` folder contains runnable scripts for each method family.

## Run Commands

```bash
python examples/vi_beam.py
python examples/laplace_beam.py
python examples/sgld_beam.py
python examples/mc_dropout_beam.py
python examples/uqresult_laplace.py
python examples/uqresult_mc_dropout.py
python examples/uqresult_gp.py
```

## Example Map

- `examples/vi_beam.py`: Bayes by Backprop ELBO training and uncertainty prediction
- `examples/laplace_beam.py`: MAP + Laplace approximation workflow
- `examples/sgld_beam.py`: posterior sampling via SGLD
- `examples/mc_dropout_beam.py`: MC Dropout inference
- `examples/uqresult_laplace.py`: standardized `UQResult` output for Laplace
- `examples/uqresult_mc_dropout.py`: standardized `UQResult` output for MC Dropout
- `examples/uqresult_gp.py`: standardized `UQResult` output for exact/sparse GP

## Benchmarks

Manual/local benchmark suite:

```bash
python benchmarks/run_benchmarks.py --preset quick
```

Source: [examples directory](https://github.com/Vispikarkaria/Deep-UQ/tree/master/examples)
