# Installation

## Install from PyPI

```bash
pip install uqdeepnn
```

Optional dependency for Laplace `kron` and `full` structures:

```bash
pip install "laplace-torch>=0.1.7"
```

## Install from Source

```bash
git clone https://github.com/Vispikarkaria/Deep-UQ.git
cd Deep-UQ
pip install -e .
```

## Verify Installation

```bash
python -c "import deepuq; print(deepuq.__version__)"
```

## Common Environments

### Conda

```bash
conda create -n deepuq python=3.11 -y
conda activate deepuq
pip install uqdeepnn
```

### GPU-enabled PyTorch
Install a CUDA-compatible PyTorch build first, then install `uqdeepnn`.
