# Welcome to Odisseo (Optimized Differentiable Integrator for Stellar Systems Evolution of Orbits)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/odisseo/badge/?version=latest)](https://odisseo.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14992689.svg)](https://doi.org/10.5281/zenodo.14992689)


`odisseo` differentiable direct Nbody written in `JAX`.

## Highlight: 200k-Particle Galaxy Disk (FMM)

Generated with the new `jaccpot` large-`N` radix integration path in ODISSEO:

![Galaxy disk evolution (200k particles)](notebooks/scalability/galaxy_disk_gpu9.gif)



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
- `fmm_preset`, `fmm_basis`, `fmm_theta`, `fmm_runtime_path`, `fmm_mac_type`
- `fmm_farfield_mode`, `fmm_nearfield_mode`, `fmm_nearfield_edge_chunk_size`, `fmm_tree_leaf_target`
- `fmm_auto_large_n_profile`, `fmm_large_n_min_particles`, `fmm_large_n_force_fp32`

Large-`N` GPU runs can now auto-switch from `fmm_preset="fast"` to jaccpot's
radix fast lane (`"large_n_gpu"` preset + `"large_n"` runtime path) when
`fmm_auto_large_n_profile=True` and particle count exceeds
`fmm_large_n_min_particles`.

> **Multi-step accuracy note (device-resident treecode walk).** The optional
> device-resident treecode far/near builder
> (`JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK=1`) must use bounding-**sphere**
> MAC extents (`JACCPOT_STATIC_STRICT_FUSED_TREECODE_MAC=dual`, the default) for
> time integration. The cheaper axis-aligned box (`bh`) extents are only
> statically accurate: they under-bound the source multipole radius and, in the
> frozen-topology fast lane, inject a non-conservative per-step force error that
> heats the system and blows the run up over ~tens of steps. Use `bh` only for
> single-shot force evaluations. See
> [`docs/STATIC_RADIX_FUSED_STATUS.md`](docs/STATIC_RADIX_FUSED_STATUS.md) and
> jaccpot's `docs/treecode_mac_stability.md`.

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

Galaxy-disk large-`N` example (auto-selects jaccpot radix fast lane on GPU):

```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py --n-particles 200000 --num-steps 200
```

Render snapshots live after the run (use `--mode render`):

```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py --n-particles 200000 --num-steps 200 --mode render --live --snapshot-stride 1 --snapshot-chunk-steps 20
```

Record a movie (`.gif` or `.mp4`; use `--mode render`):

```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py --n-particles 200000 --num-steps 200 --mode render --movie-path ./galaxy_disk.gif --movie-fps 24 --snapshot-stride 1 --snapshot-chunk-steps 20
```

Render multiple projections from one simulation run (no duplicate integration):

```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --n-particles 200000 --num-steps 200 \
  --mode render \
  --movie-path ./galaxy_disk.mp4 \
  --movie-projections xy,xz \
  --movie-fps 24 \
  --snapshot-stride 1 \
  --snapshot-chunk-steps 20
```

This produces:
- `./galaxy_disk_xy.mp4` (face-on)
- `./galaxy_disk_xz.mp4` (edge-on)

### AGAMA-Based IC Generation (Reusable Fixed IC Files)

Preferred generator (SCM-style, rotating disk target):

```bash
micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py \
  --output /export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz \
  --n-particles 200000 \
  --seed 7
```

Strict production behavior of this generator:
- SCM convergence is required by default (fail-fast on iteration failure).
- No fallback sampling is used unless `--allow-scm-fallback` is explicitly set.
- Rotation acceptance gate is enforced (`prograde_fraction` and `median_vphi` minima).

Run the galaxy simulation by loading that fixed IC file:

Set `ODISSEO_IC_ROOT` (optional) to control the default persistent IC cache root used by benchmarking helpers.

```bash
python notebooks/scalability/galaxy_disk_fmm_large_n.py \
  --mode render \
  --n-particles 200000 \
  --num-steps 40 \
  --fmm-preset large_n_gpu \
  --fmm-runtime-path large_n \
  --fmm-tree-build-mode static_radix \
  --fmm-leaf-size 256 \
  --fmm-refresh-every 1 \
  --no-fmm-large-n-environment-overrides \
  --ic-source load \
  --ic-input-path /export/home/tbuck/Odisseo/notebooks/scalability/ic_cache/odisseo_fixed_agama_ic_200k.npz \
  --no-ic-require-runtime-potential-match \
  --movie-path /tmp/galaxy_agama_scm.mp4 \
  --movie-projections xy,xz \
  --movie-fps 20 \
  --render-backend density \
  --render-resolution 768 \
  --snapshot-stride 1 \
  --snapshot-chunk-steps 1
```

Legacy note:
- `tools/agama_generate_equilibrium_ic.py` is deprecated and kept only for
  transition/debug compatibility. New production IC calibration should use
  `tools/agama_generate_scm_disk_ic.py`.
