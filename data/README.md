# Data Folder Policy

This repository keeps selected tutorial datasets tracked in Git for reproducibility.

## Current contents

- `cifar-10-batches-py/`: CIFAR-10 python batch files from the official dataset release.

## Provenance and licensing

- CIFAR-10 is distributed by the University of Toronto. See
  `data/cifar-10-batches-py/readme.html` for original details and attribution.
- Before redistributing any dataset, verify upstream license terms.

## Usage notes

- These files are intended for examples/tutorials, not production pipelines.
- For large-scale experiments, prefer external storage and local caching.

## Size policy

- CI runs `scripts/check_large_files.py` to flag oversized tracked files.
- Default fail threshold is 50 MiB per file in CI lint checks.
