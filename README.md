# Welcome to Odisseo (Optimized Differentiable Integrator for Stellar Systems Evolution of Orbits)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/odisseo/badge/?version=latest)](https://odisseo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14992689.svg)](https://doi.org/10.5281/zenodo.14992689)


`odisseo` differentiable direct Nbody written in `JAX`.



## Installation

`odisseo` is installed with [`uv`](https://docs.astral.sh/uv/), which sets up the
virtual environment in `.venv`:

```bash
git clone https://github.com/vepe99/Odisseo.git
cd Odisseo
uv sync
```

This installs the CPU build of `JAX`. On a machine with NVIDIA GPUs, sync the
extra matching the CUDA major version of the driver (Linux only):

```bash
uv sync --extra cuda12
uv sync --extra cuda13
```

Commands then run inside that environment with `uv run`, for example
`uv run python script.py` or `uv run --with jupyter jupyter lab`.


## Notebooks for Getting Started

- Self gravitating system
    - [2 body problem](notebooks/2body.ipynb)
    - [Self gravitating Plummer sphere](notebooks/Plummer.ipynb)

- External Potentials
    - [Plummer sphere in NFW potential](notebooks/Plummer_in_NFWpotential.ipynb)

- Gradient
    - [Plummer sphere in NFW with gradient](notebooks/gradient_test/grad_NFW_Potential.ipynb)
