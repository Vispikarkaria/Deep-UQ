# Notebook Data Folder Policy

This folder stores notebook-local datasets to keep tutorials runnable out of the box.

## Current contents

- `MNIST/raw/`: MNIST files used by notebook workflows.

## Intended use

- Educational/tutorial execution inside this repository.
- Not intended as a canonical dataset mirror.

## Cache and generated artifacts

- Keep temporary benchmark caches outside tracked folders.
- Put generated outputs under ignored paths such as:
  - `benchmarks/cache/`
  - `tmp/`
