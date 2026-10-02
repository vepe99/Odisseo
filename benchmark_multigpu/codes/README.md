# Code-vs-code comparison: jaccpot vs pkdgrav3

## Why pkdgrav3 and not Bonsai

Bonsai is a Barnes-Hut treecode with monopole+quadrupole moments and, on this
hardware, only a working *single*-GPU path (its MPI multi-GPU run wedged the
node's driver — see `../README.md`). Comparing jaccpot's FMM against it means
comparing two different algorithm classes at two different accuracy knobs.

pkdgrav3 removes both problems:

* **It is itself an FMM.** `gravity/moments.h` carries 4th-order reduced
  multipoles (`MOMR`) and 5th-order local expansions (`LOCR`).
* **It drives every visible GPU from one process.** `mdl2/cuda/mdlcuda.cu`
  builds one `Device` per `cudaGetDeviceCount` entry and dispatches each work
  packet to the least-busy one, so the device count is chosen purely by
  `CUDA_VISIBLE_DEVICES` — the same knob this harness uses for jaccpot, and no
  extra MPI ranks on a single node.
* **It is the credible baseline**: the trillion-particle Titan runs.

## Method: Pareto fronts, never points

pkdgrav3's expansion order is fixed at compile time, so its only runtime
accuracy knob is `dTheta`: it traces a **1-D** accuracy-vs-cost curve. jaccpot
has `(order, theta)`: a **2-D** family. A single "X is N times faster" number is
therefore not a result. Both codes are swept over their own knobs against the
*same* float64 direct sum on the *same* particles, and what gets compared is the
lower envelope of each cloud.

## The softening trap

jaccpot uses **Plummer** softening (`r^2 + eps^2`); pkdgrav3 uses a
**compact-support spline** (`gravity/pp.h::EvalPP` — a polynomial correction
inside `2h`, exactly Newtonian outside). At matched `eps` the two codes compute
*different physics*, so a force-error comparison would be measuring the
softening kernel rather than the algorithm.

Accordingly `write_tipsy_dark` **refuses** to write a nonzero softening unless
you pass `allow_softening=True`, and `compare_force.py` pins `eps = 0` on both
sides. Use a smooth IC (Plummer sphere, uniform box) so there are no
near-coincident pairs for an unsoftened kernel to blow up on.

## Environments (they are not interchangeable)

| side | interpreter | why |
| --- | --- | --- |
| jaccpot | `/export/home/tbuck/jaccpot/.venv/bin/python` (**jax 0.10.2**) | the `odisseo` env has jax **0.9.0**, whose `ragged_all_to_all` silently corrupts the distributed halo — multi-GPU forces come back wrong with healthy-looking diagnostics |
| pkdgrav3 | its own embedded python3.12 | `source /export/home/tbuck/pkdgrav3/env_pkdgrav3.sh` first (Boost/FFTW-MPI/MPICH/CUDA 12.9 all live in the `pkdgrav3-deps` micromamba env) |

## Running

```bash
source /export/home/tbuck/pkdgrav3/env_pkdgrav3.sh
export CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o -q)
JAX_ENABLE_X64=1 /export/home/tbuck/jaccpot/.venv/bin/python \
  benchmark_multigpu/codes/compare_force.py --n 200000 --ic plummer --ndevs 1 2
```

`--ndevs 1` uses jaccpot's **single-GPU** lane (`FastMultipoleMethod`), not the
distributed driver: the distributed path cannot run on one device at all,
because its LET stage builds a coarse tree out of the *remote* particle set,
which is empty with no remote devices. `--ndevs 2` uses the distributed
`make_force_evaluator` path.

## Things that will bite you

* **pkdgrav3 exits 0 even when its embedded interpreter raises.** Check for the
  output file, never the return code. `compare_force.py` does this.
* Field selectors are `PKDGRAV.PKD_FIELD.FIELD_ACCELERATION` (an enum member),
  not the module-level `FIELD_*` names that `modules/PKDGRAV.py` appears to use.
* `reorder()` must be called before `get_array()`, or the rows are in tree order
  and do not line up with the jaccpot arrays.
* `bMemAcceleration=True` is required or `get_array` returns zeros.
* Memory: JAX preallocates ~75% of VRAM, so `nvidia-smi` is meaningless for the
  jaccpot side. `compare_force.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  reads `memory_stats()['peak_bytes_in_use']`.
* pkdgrav3 leans on the host CPU hard (`-sz` cores) and jaccpot does not, so the
  core count is recorded with every timing and must be quoted with any result.

## Validation status

`aggL2_signflip` in every error dict is a convention tripwire — if it is the
*small* number, the comparison is wired backwards.

N=20k Plummer, 1x A100, eps=0: pkdgrav3 at theta=0.7 gives **aggL2 = 4.61e-4**
against the float64 direct sum (median 1.9e-4, p90 6.4e-4), reproducing its
documented "~0.1% RMS at 0.7"; signflip = 2.0; positions round-trip exactly
through Tipsy + `reorder()`. `Total tiles processed on the GPU: 100.00 %`.
