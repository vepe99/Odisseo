# Welcome to Odisseo (Optimized Differentiable Integrator for Stellar Systems Evolution of Orbits)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/odisseo/badge/?version=latest)](https://odisseo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14992689.svg)](https://doi.org/10.5281/zenodo.14992689)


`odisseo` differentiable direct Nbody written in `JAX`.



## Installation

`odisseo` can be installed via by cloning the repo and then via `pip`

```bash
git clone https://github.com/vepe99/Odisseo.git
cd Odisseo
pip install .
```


## Notebooks for Getting Started

- Self gravitating system
    - [2 body problem](notebooks/2body.ipynb)
    - [Self gravitating Plummer sphere](notebooks/Plummer.ipynb)

- External Potentials
    - [Plummer sphere in NFW potential](notebooks/Plummer_in_NFWpotential.ipynb)

- Gradient
    - [Plummer sphere in NFW with gradient](notebooks/gradient_test/grad_NFW_Potential.ipynb)


### Unified Integration API

Use `odisseo.integrate(...)` as the main entrypoint. Backend selection is done via `SimulationConfig.acceleration_scheme`:

- direct schemes (`DIRECT_ACC`, `DIRECT_ACC_LAXMAP`, `DIRECT_ACC_MATRIX`, ...)
- `FMM_ACC` for the Jaccpot-FMM coupler workflow

Key FMM tuning fields in `SimulationConfig`:
- `fmm_refresh_every`, `fmm_leaf_size`, `fmm_max_order`
- `fmm_preset`, `fmm_basis`, `fmm_theta`, `fmm_mac_type`
- `fmm_farfield_mode`, `fmm_nearfield_mode`, `fmm_nearfield_edge_chunk_size`, `fmm_tree_leaf_target`

Example:

```python
from odisseo.integration_api import integrate
from odisseo.option_classes import SimulationConfig, SimulationParams, FMM_ACC

cfg = SimulationConfig(
    N_particles=128,
    acceleration_scheme=FMM_ACC,
    num_timesteps=200,
    fixed_timestep=True,
    fmm_refresh_every=4,
)
params = SimulationParams(G=1.0, t_end=1.0)
state_out = integrate(state0, masses, cfg, params)
```
