Odisseo documentation
=====================

# Introduction and Installation

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

self

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

:::{tip} Get started with this [simple example]().
:::

## Showcase

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

## Roadmap

- [x] Implement simple `initial_conditions` (two body, self gravitating Plummer sphere )
- [x] Implement units conversion
- [x] Implement gradient trought the `time_integration` 
- [ ] Implement higher order integrators (IAS15, WHFast512)
- [ ] Implement sphere sky projection
- [x] Implement key `external_potential` for distrupted dwarf galaxies scenarios (Navarro-Frank-White halo, Myamoto-Nagai disk)
