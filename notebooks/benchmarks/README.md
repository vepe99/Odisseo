# Odisseo vs. PeTar on a single NVIDIA A100

This benchmark compares Odisseo (JAX/XLA on CUDA) with PeTar, an established
C++/CUDA N-body code, on one A100. Both codes start from the same initial
conditions, use the same Plummer softening, run to the same physical end time,
and have their energy error measured by the same estimator. The scripts report
numbers only.

| File | Role |
|---|---|
| `ic.py` | Generates the initial conditions once per (N, seed) into `data/ic/`. Every runner reads these files. |
| `run_odisseo.py` | Odisseo throughput sweep, N = 2^10 … 2^17, fp64 and fp32, `DIRECT_ACC` and `DIRECT_ACC_LAXMAP` |
| `run_reference.py` | PeTar (GPU build), N = 2^12, 2^14, 2^16, with a wall-clock cap |
| `run_matched.py` | Odisseo runs whose dt is tuned until |dE/E| matches the PeTar runs (same N, IC and t_end) |
| `plot.py` | Writes `results/summary.csv` and `results/throughput.{pdf,png}` |
| `common.py` | CSV writer, float64 energy estimator, GPU and provenance helpers |

Results are written to `results/`:

* `odisseo_throughput.csv`, `reference.csv`, `odisseo_matched.csv`: tidy CSVs,
  one row per (code, N, precision, scheme, repeat). Rows are fsync'ed as they
  are written.
* `provenance_*.json`: versions, environment, `nvidia-smi -q`, build info,
  and the step tables.
* `petar_help.txt` and `petar_work/*.log` / `*.input.par`: PeTar's help text,
  its stdout, and the full parameter set it actually used.
* `summary.csv` and `throughput.{pdf,png}`.

## Reproduce

```bash
# 1. GPU JAX (from the repo root; the driver here supports CUDA 13.2)
uv sync --extra cuda13 --inexact

# 2. PeTar GPU build (see "PeTar build" below for the exact commands)

# 3. Run, in this order (each script also has --dry-run: smallest N only, results/dryrun/)
cd notebooks/benchmarks
../../.venv/bin/python run_odisseo.py      # throughput sweep, about 1.5 h
../../.venv/bin/python run_reference.py    # PeTar; reads the same IC files
../../.venv/bin/python run_matched.py      # Odisseo at PeTar's |dE/E|
../../.venv/bin/python plot.py
```

Run the scripts one after another, not at the same time. PeTar uses many CPU
cores, and the Odisseo sweep evaluates energies on 32 CPU threads, so
overlapping them would perturb both codes' timings.

## Test problem (identical for every code)

* **Initial conditions.** Odisseo's `Plummer_sphere` is sampled in float64 on
  the CPU, seed 42, with Plummer radius a = 3π/16. The sample is moved to its
  centre-of-mass frame and rescaled so that, *without softening*, W = −1/2 and
  T = 1/4 hold exactly for that finite sample. These are Hénon units:
  G = M = 1, E = −1/4, Q = 1/2, and every particle has mass 1/N.
  The file is generated once per N and stores a SHA-256 of the arrays, which
  every load checks. Runners convert the file; they never regenerate it.
  `run_reference.py` writes PeTar's ASCII input with `repr()` (round-trip
  exact). It then reads PeTar's own time-0 snapshot back and asserts that
  mass, position and velocity equal the IC file to round-off. The measured
  maximum relative difference is stored as `ic_ingest_max_rel_diff`; it was
  0.0 in the test runs.
* **Softening: Plummer, ε = 2^-7 = 0.0078125, for all codes** (PeTar:
  `--soft-eps 0.0078125`). Reasons for this value:
  1. It is a power of two, so it is exact in fp32 and fp64.
  2. It is close to the ε ≈ 0.01 commonly used for softened direct-summation
     benchmarks in Hénon units.
  3. It is smaller than the mean interparticle spacing at the half-mass radius
     (≈ 0.77 N^-1/3 ≈ 0.015) up to N = 2^17, so softening does not dominate the
     dynamics at any N in the sweep.

  PeTar applies ε² in all its force paths. The GPU tree kernel takes `eps2`
  (`src/force_gpu_cuda.cu`), and `HardManager::setEpsSq` passes it to both the
  Hermite and the AR interactions (`src/hard.hpp:41-44`,
  `src/ar_interaction.hpp:319`). Both codes therefore integrate the same
  Plummer-softened Hamiltonian.
* **Physical end time.** t_end = 1 Hénon time unit for every code, N and
  series.

## Energy error (the equal-accuracy rule)

|dE/E| = |E(t_end) − E(0)| / |E(0)|. **Both codes use the same estimator**,
`common.total_energy`: a float64 numpy direct sum on the CPU with the same
Plummer softening, skipping r = 0 pairs (Odisseo's convention).

* Odisseo: the estimator is applied to the final state of the warm-up run.
  This is identical across repeats, because the runs are deterministic.
* PeTar: the estimator is applied to its own time-0 snapshot and to its final
  snapshot. PeTar's self-reported error is kept separately, in
  `petar_dE_E_reported`. It is not used for matching: PeTar's energy comes
  from its fp32 tree potential and has a floor of about 4e-7 relative. Its
  initial energy differs from the float64 direct sum by that amount at
  N = 1024.

`run_matched.py` tunes Odisseo's dt to the reference accuracy:

* The target for each N is the median |dE/E| of the PeTar repeats.
* For each (precision, scheme), Odisseo walks a ladder of step counts,
  n_steps = round(8 · √2^k).
* It picks the smallest n_steps for which that rung **and the next one** both
  give |dE/E| ≤ target. The leapfrog energy error is not monotonic in dt at
  coarse dt; in the N = 4096 test, n = 16 gave 1.5e-3 and n = 23 gave 3.2e-3.
* That configuration is then timed. Each row stores `target_dE_E`,
  `accuracy_ratio` and `accuracy_matched` (ratio in [1/3, 3]), and the whole
  ladder.
* `plot.py` draws Odisseo points with `accuracy_matched = False` as grey
  crosses, never as part of a series, and annotates each PeTar point with its
  |dE/E|.

## Odisseo measurement (`run_odisseo.py`, `run_matched.py`)

* **GPU.**
  * `autocvd(num_gpus=1)` picks a free GPU (the repo convention on this shared
    machine). The GPU is selected after the CPU-only IC preparation.
  * The run fails loudly unless `nvidia-smi` reports an **A100**. It records
    whether the card is the 40 GB or 80 GB variant, from `memory.total`.
  * It aborts if another process is on that GPU. During timing, every row
    re-checks, and any foreign process changes the row's `status` to
    `shared_gpu`; `plot.py` excludes those rows.
* **API.**
  * `odisseo.time_integration.time_integration` with `SimulationConfig(
    fixed_timestep=True, return_snapshots=False, integrator=LEAPFROG,
    acceleration_scheme=DIRECT_ACC | DIRECT_ACC_LAXMAP, softening=2^-7,
    batch_size=10000, double_map=False)` and
    `SimulationParams(G=1, t_end=1)`.
  * `leapfrog` evaluates the acceleration twice per step, so
    `n_force_evals = 2 * n_steps`.
* **Compile time.** `time_integration.lower(...).compile()` is timed on its
  own (`compile_s`, a one-off cost per configuration) and excluded from all
  throughputs.
* **Timing.** One warm-up run, then at least 5 timed repeats. Each repeat runs
  from inputs already on the device until `out.block_until_ready()` returns,
  measured with `time.perf_counter()`. `plot.py` reports the median and the
  interquartile range.
* **Precision.**
  * fp64 (primary) runs with `jax_enable_x64=True`.
  * fp32 is a separate, labelled series run in its **own process with x64
    disabled**. Odisseo's `lax.cond` in `single_body_acc` cannot trace fp32
    inputs inside an x64 process.
  * Both precisions are checked: the lowered HLO must contain `f64` ops for
    fp64 and none for fp32, and the output dtype must match.
  * Energies are always evaluated in float64.
* **Throughput sweep steps.**
  * `n_steps` is calibrated **once per N**, on fp64 `DIRECT_ACC`, so that one
    integration to t_end = 1 takes about 30 s.
  * All other series at that N reuse the same value, so they integrate the
    same problem with the same dt.
  * Since throughput is nearly flat in N, N² · n_steps comes out roughly
    constant.
  * The step table is in `steps_<run_id>.json`.
* **Metric.**
  * interactions/s = N² · n_force_evals / wall_s. N² includes the i = j pair,
    which the kernels evaluate and zero. With shared fixed steps this count
    is exact.
  * GFLOP/s = interactions/s × 20 / 10⁹. The 20 flops per interaction follows
    the convention of Nyland, Harris & Prins (2007, GPU Gems 3, ch. 31). It
    counts the softened acceleration only, so it is a convention, not a
    hardware counter.

## PeTar (`run_reference.py`)

### Build (as done on the benchmark machine)

```bash
mkdir -p ~/petar_build && cd ~/petar_build
git clone https://github.com/lwang-astro/PeTar.git
git clone https://github.com/FDPS/FDPS.git && (cd FDPS && git checkout v7.0)   # README: v7.1 may not work
git clone https://github.com/lwang-astro/SDAR.git
cd PeTar
export PATH=/usr/local/cuda-12.2/bin:$PATH
./configure --prefix=$HOME/petar_build/install --enable-cuda \
            --with-cuda-prefix=/usr/local/cuda-12.2 --with-mpi=no
# two manual edits to the generated Makefile (recorded in the provenance JSON as a diff):
#   -ccbin g++-11 : nvcc 12.2 does not support the default host g++ 13
#   -arch=sm_80   : compile the CUDA kernel for the A100 instead of the nvcc default
sed -i 's|^NVCC = nvcc  -std=c++17|NVCC = nvcc -ccbin g++-11 -arch=sm_80 -std=c++17|' Makefile
make -j8 && make install      # -> install/bin/petar -> petar.omp.avx2.gpu
```

* **Resulting binary.** `petar.omp.avx2.gpu`: OpenMP, AVX2 SIMD on the host,
  CUDA tree kernel. The C++ parts are built with `-O3 -march=core-avx2`; the
  full compile lines are in the provenance file.
* **Not used.** MPI (single node, one GPU) and stellar evolution.
* **Recorded.** The git commits of PeTar, FDPS and SDAR go into
  `provenance_reference_*.json`.
* **Precision of the force kernels, checked in the source, not assumed.**
  * The GPU long-range (tree) kernel `src/force_gpu_cuda.cu` is
    **single precision**: `float3`/`float4` positions and accelerations and
    `rsqrtf`.
  * The short-range Hermite/AR integration runs on the CPU in double
    precision.
  * PeTar is therefore a mixed-precision code: the CSV labels it
    `precision = mixed`.

### Flags (verified against `petar -h` of the installed binary)

The help text is saved to `results/petar_help.txt`. Before running,
`run_reference.py` checks that every flag it passes appears in that text and
skips the run otherwise. Command used:

```
petar -i 1 -u 0 -G 1 -t 1.0 -o 1.0 --soft-eps 0.0078125 -b 0 -w 1 -a 0 -f data input.dat
```

| Flag | `petar -h` text (verbatim) | Why |
|---|---|---|
| `-i 1` | "Data read(r)/write(w) format BINARY(B)/ASCII(A): ... rw-A (1)" | ASCII input and output, so the input file is written at full precision with `repr()` and snapshots can be read back |
| `-u 0` | "Input data unit, 0: unknown, referring to G" | Hénon (N-body) units |
| `-G 1` | "Gravitational constant: 1" | G = 1 |
| `-t 1.0` | "End time of simulation" | same t_end as Odisseo |
| `-o 1.0` | "Output time interval for particle dataset snapshots" | snapshots at t = 0 and t = t_end only (the IC check and the final energy) |
| `--soft-eps 0.0078125` | "Softening epsilon: 0" | same Plummer ε as Odisseo |
| `-b 0` | "Number of primordial binaries for initialization" | no primordial binaries |
| `-w 1` | "File writing style: ... 1. write snapshots, status, and profile separately" | the snapshots are needed for the energy estimator |
| `-a 0` | "Data output style: 0 - create new output files and overwrite existing ones" | each run starts from a clean work directory |
| `-f data` | "Prefix of filenames for output data" | |

* **Defaults kept unchanged:**
  * `-T 0.3` (tree opening angle)
  * `--hermite-eta 0.1`
  * `--energy-err-hard 1e-4`
  * `--energy-err-ar 1e-8`
  * `-r 0` and `-s 0`: the code derives r_out and dt_soft from N and σ; for
    N = 4096 it chose r_out = 0.0125 and dt_soft = 2^-9.
  * quadrupole tree moments, enabled at configure time
* **Where the values used are recorded:** every run's `input.par`, which PeTar
  writes with all parameter values it used, is saved under
  `results/petar_work/`.
* **Environment:** `OMP_NUM_THREADS=16` (CLI `--omp-threads`) and
  `OMP_STACKSIZE=128M` (PeTar's configure recommends this). A single process
  is used, without MPI.

### CPU usage on the shared machine

* PeTar runs on CPU and GPU together. Before each run, `run_reference.py`
  waits until the 1-minute load average plus the OpenMP thread count fits
  within 80 % of the cores (`--max-cpu-fraction`).
* Each row records `loadavg_before`, `loadavg_after` and `cpu_cores_used`
  (child CPU time divided by wall time).
* `cpu_cores_used` is part of PeTar's cost: Odisseo uses one host thread,
  while PeTar uses up to `omp_threads` cores **plus** the GPU.

### Metric for PeTar

* PeTar is not an all-pairs code, so an N² interaction rate would not be
  defensible. Its rows have
  `metric_basis = wallclock_to_fixed_time`: `wall_s` is the external
  wall-clock of the whole PeTar process to t_end, including start-up and the
  two snapshot writes.
* PeTar's own per-step timer total is stored separately in
  `petar_internal_s`.
* Its step statistics are parsed from its log and recorded, but not converted
  into a rate:
  * `tree_steps`
  * `h4_steps_total` (Hermite steps)
  * `ar_steps_total`
  * `ep_ep_total` and `ep_sp_total` (particle–particle and
    particle–superparticle tree interactions)
* Repeats: 5 by default. Each run is capped at `--wallclock-cap`, 3600 s by
  default. A run that exceeds the cap is recorded as `status = timeout`, and
  the remaining repeats at that N are skipped.

## Where the algorithms differ

Configured as above, PeTar and Odisseo integrate **the same softened N-body
problem from the same initial state to the same time, and are judged by the
same energy estimator**. They do **not** use the same algorithm.

| | Odisseo | PeTar |
|---|---|---|
| Long-range force | all-pairs direct summation, O(N²) | Barnes–Hut tree (θ = 0.3, quadrupole), O(N log N), fp32 on the GPU, with a changeover cutoff |
| Short-range force | the same direct sum | direct Hermite 4th order within r_out, on the CPU in fp64 |
| Time steps | one shared, fixed dt (leapfrog) | shared tree step (dt_soft) plus block individual Hermite steps |
| Close encounters | softened only | slow-down algorithmic regularisation (SDAR) for bound groups |
| Hardware used | 1 GPU and 1 host thread | 1 GPU and `omp_threads` CPU cores |

A timing difference between the two codes therefore includes these
algorithmic differences (tree vs. direct summation, individual vs. shared
steps, regularisation, CPU cores used). **It is not a measure of a
language or runtime overhead (Python/JAX vs. C++/CUDA) alone.**

NBODY6++GPU, the other code suggested, uses a 4th-order Hermite scheme with an
Ahmad–Cohen neighbour scheme and KS/chain regularisation, and has no Plummer
softening. For that reason it was not used.
