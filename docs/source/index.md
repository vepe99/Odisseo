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


# Contribute to the software

Contributions are welcome, from bug fixes and documentation improvements to new
integrators, potentials and initial conditions. The project is developed in the
open at [github.com/vepe99/Odisseo](https://github.com/vepe99/Odisseo) and is
released under the MIT license.

The usual workflow is:

1. Open an issue first describing what you would like to change, so the design
   can be discussed before any code is written. For small fixes (typos, broken
   links, docstrings) a pull request on its own is fine.
2. Fork the repository and create a branch.
3. Set up the development environment, including the test dependencies:

   ```bash
   git clone https://github.com/<your-username>/Odisseo.git
   cd Odisseo
   uv sync --extra tests
   ```

4. Make the change. New public functions should carry a docstring in the same
   style as the surrounding code, since the API documentation is generated
   automatically from the source with `sphinx-autodoc2`.
5. Add a test under `tests/` covering the new behaviour, and run the suite:

   ```bash
   uv run pytest
   ```

   A physics-oriented change should also preserve the conservation diagnostics
   described in [Conservation](./conservation.md): energy and angular momentum
   errors must not degrade for the example problems.
6. Open a pull request, describing what the change does and why,
   and linking the issue it addresses.

If you are looking for a place to start, the unchecked entries in the
[Roadmap](#roadmap) above (sky projection and adaptive time stepping) are open,
and so is the extension of the gallery of example notebooks.

# Report issues or problems with the software

Please report bugs through the
[GitHub issue tracker](https://github.com/vepe99/Odisseo/issues), after checking
whether the problem is already covered by an open issue.

A report is most useful when it contains:

- a short description of what you expected and what happened instead;
- a minimal script or notebook cell that reproduces the problem, together with
  the simulation configuration used (number of particles, integrator, time step,
  external potential);
- the full traceback, if an error is raised;
- the output of `uv run python -c "import jax; print(jax.__version__); print(jax.devices())"`,
  so that the `JAX` version and the backend (CPU, GPU or TPU) are known;
- the operating system and, on GPU, the CUDA major version used in
  `uv sync --extra cuda12` or `uv sync --extra cuda13`.

Silent numerical problems deserve a report as well: if a simulation runs to
completion but the energy or angular momentum drift reported by the
[conservation checks](./conservation.md) is unexpectedly large, that is a bug
worth opening an issue for, not a user error.

# Seek support

For questions about using `odisseo` — how to set up initial conditions, choose
units, pick an integrator, or differentiate through a simulation — the first
places to look are the [first simulation](./first_simulation.md),
[units](./units.md) and [parallelism](./parallelism.md) pages, the example
notebooks listed above, and the API reference.

If the answer is not there, open an issue on the
[tracker](https://github.com/vepe99/Odisseo/issues) with the `question` label.
Questions are answered in the open so that other users can find them later.

For matters that do not belong in a public issue, such as collaborations or
scientific use of the code, contact the maintainers directly at
`giuseppe.viterbo@iwr.uni-heidelberg.de`.
