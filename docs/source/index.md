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


## Installation

`odisseo` can be installed via by cloning the repo and then via `pip`

```bash
git clone https://github.com/vepe99/Odisseo.git
cd Odisseo
pip install .
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).


## Fondamentals
The core of Odisseo's pipeline and how to run a simulation is described in detail in [Running a simulation](./first_simulation.md). Particular importance is put in the choise of the [simulation units](./units.md) and what [checks](./conservation.md) can be run to see if the simulation underwent numerical errors as a post-process strategy. Lastly, some distribution strategy are shown in [Distributed simulations](./parallelism.md) to take full advantage of multi-devices machines.

## Notebooks for Getting Started

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

## Roadmap

- [x] Implement simple `initial_conditions` (two body, self gravitating Plummer sphere )
- [x] Implement units conversion
- [x] Implement gradient trought the `time_integration` 
- [x] Implement diffrax backend for integrators
- [ ] Implement sphere sky projection
- [x] Implement key `external_potential` for distrupted dwarf galaxies scenarios (Navarro-Frank-White halo, Myamoto-Nagai disk)
- [x] Multi gpu parallalization
- [ ] Implement adaptive time stepping
