Odisseo documentation
=====================

# Introduction and Installation

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

self
first_simulation.md
conservation.md
units.md
parallelism.md
```

`odisseo` differentiable direct Nbody written in `JAX`.

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} $\nabla$ Differentiable N-body code

Written in `JAX`, `Odisseo` is fully differentiable - a simulation can be differentiated with respect to any input parameter - and just-in-time compiled for fast execution on CPU, GPU, or TPU. 

+++
[Learn more »](./first_simulation.md)
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Well-considered Numerical Methods

Particular importance is put in the choice of the [simulation units](./units.md) and what [checks](./conservation.md) can be run to see if the simulation underwent numerical errors as a post-processing strategy. 

+++
[Learn more »](./units.md)
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Easily parallelized 

Distribution strategies are easily implemented to take full advantage of multi-device machines.

+++
[Learn more »](./parallelism.md)
:::

::::

# Installation

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



# Notebooks for Getting Started

```{toctree}
:caption: Self gravitating system
:maxdepth: 1

notebooks/2body.ipynb
notebooks/Plummer.ipynb
```

```{toctree}
:caption: External Potentials
:maxdepth: 1

notebooks/Plummer_in_NFWpotential.ipynb
```

```{toctree}
:caption: Gradient
:maxdepth: 1

notebooks/gradient_test/grad_NFW_Potential.ipynb
```

```{toctree}
:caption: GD1 stream
:maxdepth: 1

notebooks/GD1.ipynb
```

```{toctree}
:maxdepth: 2
:caption: API 

apidocs/odisseo/odisseo

```

# Roadmap

- [x] Implement simple `initial_conditions` (two body, self gravitating Plummer sphere )
- [x] Implement units conversion
- [x] Implement gradient through the `time_integration` 
- [x] Implement diffrax backend for integrators
- [ ] Implement sphere sky projection
- [x] Implement key `external_potential` for disrupted dwarf galaxies scenarios (Navarro-Frank-White halo, Miyamoto-Nagai disk)
- [x] Multi gpu parallelization
- [ ] Implement adaptive time stepping
