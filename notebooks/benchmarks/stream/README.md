# GD1 stream: Odisseo (GPU) vs gala and galpy (CPU), direct N-body

The stream problem from `notebooks/GD1.ipynb`, evolved with direct-summation
self-gravity in three codes at increasing particle number N.

Everything is in **`stream_benchmark.py`** (subcommands `setup`, `run`,
`analyze`). Results go to `results/`, initial conditions to `data/` (ignored
by git, re-creatable bit-for-bit with `setup`).

## Reproduce

```bash
cd notebooks/benchmarks/stream
PY=../../../.venv/bin/python        # Odisseo venv: setup, Odisseo runs, analyze
CPY=.venv_cpu/bin/python            # gala + galpy (kept out of the Odisseo venv)
uv venv .venv_cpu --python 3.13
VIRTUAL_ENV=$PWD/.venv_cpu uv pip install gala galpy numpy scipy astropy

$PY  stream_benchmark.py setup --N 500 1000 2000 4000 8000 16000 32000 64000
$PY  stream_benchmark.py run --code odisseo --N 500 1000 2000 4000 8000 16000 32000 64000   # GPU via autocvd
for i in 0 1 2 3 4; do N=$((500 << i));                                                     # 1 core per process
  $PY stream_benchmark.py run --code odisseo_cpu --N $N --core $((59 + i)) > results/odisseo_cpu_N$N.log 2>&1 &
done; wait
$CPY stream_benchmark.py run --code gala  --N 500 1000 2000 4000 8000
$CPY stream_benchmark.py run --code galpy --N 500 1000 2000 4000 8000
$PY  stream_benchmark.py analyze     # stream_summary.csv, stream_timing.pdf
```

`run ... --n-steps 10` is a quick test mode: nothing is written to the CSV and
the final state goes to `results/test/`. Before timing, the gala and galpy runs
check their external acceleration against Odisseo's (`data/potential_check.npz`)
and abort if it differs by more than 1e-10.

## Problem (identical for every code)

* **Units.** Odisseo code units: 10 kpc, 10^4 Msun, 3 Gyr (G = astropy G in
  these units, 4.0487e-4).
* **Progenitor.** A Plummer sphere with M = 10^4.05 Msun and a = 100 pc,
  sampled by `odisseo.initial_condition.Plummer_sphere` with key 500000000.
  It is placed on the centre-of-mass phase-space point obtained by integrating
  the present-day GD1 point, x = (11.8, 0.79, 6.4) kpc and
  v = (109.5, −254.5, −90.3) km/s, back 3 Gyr.
* **External potential.** NFW (M = 4.368e11 Msun, r_s = 16 kpc),
  Miyamoto–Nagai (M = 6.819e10 Msun, a = 3 kpc, b = 0.28 kpc), and a power law
  with exponential cutoff (M = 4.50e9 Msun, α = 1.8, r_c = 1.9 kpc). This is
  MWPotential2014.
  * gala: `NFWPotential`, `MiyamotoNagaiPotential`, `PowerLawCutoffPotential`.
  * galpy: `NFWPotential`, `MiyamotoNagaiPotential`,
    `PowerSphericalPotentialwCutoff` (amplitude converted from total mass).
  * **Checked:** at 2000 random points between 0.3 and 30 kpc, gala's and
    galpy's accelerations agree with Odisseo's to < 7e-14 relative. gala's
    potential agrees to 4e-15; galpy's bulge potential differs only by a
    constant zero point.
* **Self-gravity.** Direct summation with Plummer softening ε = 0.1 pc.
* **Integration.** 3 Gyr in 1000 fixed steps (dt = 3 Myr) of second-order
  leapfrog, fp64 throughout. The notebook's adaptive Tsit5 is replaced by
  leapfrog because gala's `DirectNBody` offers only leapfrog, Ruth4 and DOP853,
  and galpy has no N-body integrator.

## What each code actually runs

| | Odisseo | gala | galpy |
|---|---|---|---|
| N-body engine | `time_integration`, `DIRECT_ACC` (JAX/XLA) | `gala.dynamics.nbody.DirectNBody` (Cython/C) | **galpy has no N-body integrator.** External force from galpy's C potentials (`evaluateRforces` / `evaluatezforces`), self-gravity from a float64 numpy direct sum, KDK loop in Python |
| Integrator | `LEAPFROG` = velocity Verlet, **2 force evaluations per step** | `LeapfrogIntegrator`, 1 force evaluation per step, **sequential update** (see below) | KDK, 1 force evaluation per step |
| Hardware | 1 A100 (40 GB) and 1 host thread; `odisseo_cpu`: 1 CPU core | 1 CPU core | 1 CPU core (BLAS/OpenMP threads forced to 1) |
| Timed region | compiled call to `block_until_ready()`, after warm-up; compile time recorded separately | `integrate_orbit` only; building the N `PlummerPotential` objects is recorded as `setup_s` | the KDK loop |
| Repeats | 5 (3 for N ≥ 32000) | 3 for N ≤ 1000, else 1 | 3 for N ≤ 1000, else 1 |

* **Odisseo vs galpy.** After 10 steps from the N = 500 IC, Odisseo (fp64)
  and the galpy-based integrator agree to 7e-16 in position and 4e-15 in
  velocity, so they integrate the same discrete map.
* **gala's N-body leapfrog is not a synchronous leapfrog.**
  `gala/integrate/cyintegrators/leapfrog.pyx` (`leapfrog_integrate_nbody`)
  loops over particles and updates each one in place. Particle i drifts, then
  takes its force from particles 0…i−1, which have already moved this step,
  and from i+1…N−1, which have not. A Python re-implementation of this
  sequential scheme reproduces gala to 2e-16. Compared with synchronous KDK it
  differs at O(dt·a_self) from step 2 onwards (~1e-5 code length after 10
  steps at N = 500). The cost per step is still one O(N²) pass, so the timings
  are comparable.
* **Interaction counts.** Odisseo performs twice the pairwise interactions per
  step of gala and galpy (N² × 2 × 1000 vs N² × 1001). Wall-clock time to
  solution is the fair comparison; `stream_summary.csv` also lists the
  pair-interaction rate.

## Results (2026-11-18, A100-PCIE-40GB, single CPU core on the same host)

Wall-clock to 3 Gyr (1000 steps), median over repeats, from `results/stream_summary.csv`:

| N | Odisseo (1 GPU) | Odisseo (1 CPU core) | gala (1 core) | galpy-based (1 core) | gala / Odisseo GPU | galpy / Odisseo GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.56 s | 45 s | 19.8 s | 14.9 s | 35× | 27× |
| 1000 | 0.64 s | 144 s | 82.6 s | 51.9 s | 129× | 81× |
| 2000 | 0.90 s | 475 s | 344 s | 179 s | 383× | 200× |
| 4000 | 1.75 s | 1676 s | 1338 s | 680 s | 763× | 388× |
| 8000 | 6.19 s | 6265 s | 5174 s (1.4 h) | 2622 s | 835× | 423× |
| 16000 | 30.9 s | – | – | – | | |
| 32000 | 113 s | – | – | – | | |
| 64000 | 454 s ¹ | – | – | – | | |

¹ `DIRECT_ACC` ran out of memory at N = 64000 (a 64000² fp64 buffer is
30.5 GiB) and is recorded as an error row. The time shown is
`DIRECT_ACC_LAXMAP` (batch 10000).

* **Plot.** `stream_timing.{pdf,png}` shows time to solution against N on
  the common grid N = 500–8000 only. The GPU-only points (N ≥ 16000) are in
  `stream_summary.csv`.
* **Odisseo on one CPU core** (`run --code odisseo_cpu`):
  * Setup: XLA CPU backend, `--xla_cpu_multi_thread_eigen=false
    intra_op_parallelism_threads=1`, the process pinned to one core with
    `sched_setaffinity`, no warm-up run, and the same repeat policy as
    gala/galpy.
  * Each N ran as its own process on its own core (cores 59–63), at the same
    time as the others.
  * It is 2.2× slower than gala at N = 500 and 1.2× at N = 8000. Part of this
    is that Odisseo evaluates the forces twice per leapfrog step. Per
    interaction, Odisseo-CPU rises from 1.1e7 to 2.0e7/s with N, faster than
    gala's 1.2e7/s from N ≈ 2000.
  * The GPU gives about 1000× over Odisseo's own single-core run at N = 8000.
  * CPU and GPU Odisseo give the same streams. The median particle position
    differs by ≤ 2e-8 kpc; the largest single-particle difference (0.05 kpc
    at N = 8000) is chaotic growth of fp round-off differences between the
    XLA CPU and GPU kernels.
* **Throughput:**
  * Odisseo saturates at about 1.8–2.1e10 pair interactions/s from N ≈ 4000
    upwards. Below that, the time is dominated by per-step launch overhead
    (~0.5 ms per step).
  * gala's C loop does about 1.2e7 interactions/s and galpy+numpy about
    2.4e7. Both scale exactly as N², so CPU N = 16000 would take about 6 h
    (gala) and 3 h (galpy).
* **Energy error** (same float64 estimator for all codes: numpy direct-sum
  self-potential plus Odisseo's external potential): |dE/E| ≈ 1.2e-4 for Odisseo and galpy, and **2.4e-4 for gala**
  at every N.
* **gala's final state is systematically offset** from Odisseo's and galpy's
  (median φ1 1.5° lower, v_R 5 km/s higher at every N). Its external potential
  and test-particle leapfrog match Odisseo's to round-off, so the offset and
  the doubled energy error come from how self-gravity is integrated, most
  likely the sequential particle update described above.

Run conditions: the gala and galpy runs ran concurrently (one core each).
From about 11:33 they overlapped with PeTar's `run_reference.py`, which used
up to 16 OpenMP threads, and the host load average was 12–29 on 64 cores. Both
CPU codes are single threaded and compute-bound, and their per-N throughput
stayed flat (gala 1.17–1.26e7 int/s), so the effect looks small. For
publication-grade CPU numbers, rerun them on an idle machine.
