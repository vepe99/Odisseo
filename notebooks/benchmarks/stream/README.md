# GD1 stream: Odisseo vs gala and galpy, direct N-body

The GD1 stream of `notebooks/GD1.ipynb`, evolved with direct-summation
self-gravity by Odisseo (GPU fp64/fp32 and 1 CPU core), gala and galpy (1 CPU
core each). Two scripts, `stream_benchmark.py` for timing and
`stream_conservation.py` for energy and angular momentum, write their results
to `results/`.

## Reproduce

```bash
cd notebooks/benchmarks/stream
PY=../../../.venv/bin/python        # Odisseo venv
CPY=.venv_cpu/bin/python            # gala + galpy venv
uv venv .venv_cpu --python 3.13
VIRTUAL_ENV=$PWD/.venv_cpu uv pip install gala galpy numpy scipy astropy

# timing -> results/stream_summary.csv, stream_timing.{pdf,png}
$PY  stream_benchmark.py setup --N 500 1000 2000 4000 8000
$PY  stream_benchmark.py run --code odisseo      --N 500 1000 2000 4000 8000   # GPU via autocvd
$PY  stream_benchmark.py run --code odisseo_fp32 --N 500 1000 2000 4000 8000
$PY  stream_benchmark.py run --code odisseo_cpu  --N 500 1000 2000 4000 8000 --core 63
$CPY stream_benchmark.py run --code gala         --N 500 1000 2000 4000 8000 --core 62
$CPY stream_benchmark.py run --code galpy        --N 500 1000 2000 4000 8000 --core 61
$PY  stream_benchmark.py analyze

# conservation -> results/conservation/conservation.{pdf,png}
$PY  stream_conservation.py run --code odisseo      --N 500 2000      # same for odisseo_fp32, odisseo_cpu
$CPY stream_conservation.py run --code gala         --N 500 2000      # same for galpy
$PY  stream_conservation.py analyze
```

## Problem

* **Progenitor.** Plummer sphere (M = 10^4.05 Msun, a = 100 pc) on the GD1
  orbit, integrated back 3 Gyr and then evolved forward 3 Gyr.
* **External potential.** NFW + Miyamoto–Nagai + power law with cutoff
  (MWPotential2014). gala's and galpy's accelerations match Odisseo's to
  < 1e-13; the run aborts if they don't.
* **Self-gravity.** Direct summation, Plummer softening 0.1 pc.
* **Integration.** Leapfrog, 1000 fixed steps (dt = 3 Myr), fp64 unless
  `odisseo_fp32`.

| | Odisseo | gala | galpy |
|---|---|---|---|
| N-body engine | `time_integration`, `DIRECT_ACC` | `DirectNBody` + `LeapfrogIntegrator` | no N-body integrator: galpy external forces + numpy direct sum, KDK |
| Force evaluations per step | 2 | 1 (particles updated sequentially) | 1 |

## Results (A100-PCIE-40GB; CPU runs on 1 core of the same host)

Wall-clock time to 3 Gyr in seconds, median over repeats:

| N | Odisseo GPU fp64 | Odisseo GPU fp32 | Odisseo 1 core | gala | galpy |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.56 | 0.30 | 45 | 19.8 | 14.9 |
| 1000 | 0.64 | 0.35 | 144 | 82.6 | 51.9 |
| 2000 | 0.90 | 0.42 | 475 | 344 | 179 |
| 4000 | 1.75 | 0.76 | 1676 | 1338 | 680 |
| 8000 | 6.19 | 2.09 | 6265 | 5174 | 2622 |

Conservation at N = 500 and 2000, from snapshots every 10 steps:

| | max \|ΔE/E\| | max \|ΔL_z\|/\|L\| |
|---|---|---|
| Odisseo fp64 (GPU, CPU), galpy | 2.2e-4 | 2.5e-15 |
| Odisseo fp32 (GPU) | 2.2e-4 | 1e-7 |
| gala | 3.2e-4 | 2.9e-4 |

Odisseo fp64 and galpy integrate the same leapfrog and their curves overlap.
gala updates particles one at a time within a step, which breaks the pairwise
force symmetry and costs it the L_z conservation.
