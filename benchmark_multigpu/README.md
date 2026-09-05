# Multi-GPU FMM & yggdrax-tree benchmarks

Reproducible, paper-grade benchmarks for the multi-GPU distributed FMM (jaccpot)
and the yggdrax tree (vs. **jztree**). Feeds the jaccpot and yggdrax papers.

## Layout

```
benchmark_multigpu/
  common/        shared helpers (no GPU state)
    ic.py          IC generators: uniform, plummer, separated clusters, + tipsy reader for the 200k disk IC
    reference.py   chunked O(N^2) direct sum (float64 accuracy ground truth)
    timing.py      timed_call: warmup + block_until_ready + min-of-repeats
    env.py         provenance capture (git SHAs of jaccpot/yggdrax/jztree + jax/GPU info)
    capacity.py    overflow-retry traversal-cap calibration
    bonsai.py      single-GPU Bonsai runner + tipsy writer + log parser
  fmm/
    accuracy.py    Phase 1: FMM error vs order p and theta, vs direct sum
    performance.py Phase 2: steady-state force-eval wall time vs #GPUs (+ Bonsai 1-GPU baseline)
    scaling.py     Phase 3: strong & weak scaling (1/2/3 GPUs)
  tree/
    tree_bench.py  Phase 4: yggdrax vs jztree (build / traversal / decomposition)
  notebooks/
    make_figures.py  render paper figures from artifacts/*.json (no GPU)
  artifacts/       produced .json/.npz + .pdf figures
  run_all.sh       orchestrate all phases (picks free GPUs via autocvd)
  build_bonsai_mpi.sh   DISABLED (MPI multi-GPU wedged the node; single-GPU only)
```

## Environment

- Python: `/export/home/tbuck/micromamba/envs/odisseo/bin/python` (JAX 0.9; editable jaccpot/yggdrax/jztree). Always set `JAX_ENABLE_X64=1`.
- GPUs: select free devices via `autocvd` — e.g. `CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o -q)`. **Never hard-code device 0.**
- The reusable driver lives in jaccpot: `jaccpot.distributed.distributed_fmm_accelerations` / `make_force_evaluator` / `partition_for_devices` / `DistributedFMMConfig`.

## Reference choice (why direct sum, not Bonsai)

Accuracy is measured **only** against the exact direct sum (`common/reference.py`).
Bonsai is itself an approximate Barnes-Hut treecode, so FMM-vs-Bonsai would compare
two approximations. Bonsai is a **performance** competitor (single-GPU baseline),
never an accuracy reference.

## Run

```bash
cd /export/home/tbuck/Odisseo-bench-multigpu
# one phase at a time (recommended; each picks its own GPUs):
CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o -q) JAX_ENABLE_X64=1 \
  ../micromamba/envs/odisseo/bin/python benchmark_multigpu/fmm/accuracy.py --n 20000
# ... performance.py / scaling.py / tree/tree_bench.py similarly
# then render figures (no GPU):
../micromamba/envs/odisseo/bin/python benchmark_multigpu/notebooks/make_figures.py --which all
# or everything:
bash benchmark_multigpu/run_all.sh
```

Every artifact embeds a provenance block (repo SHAs, JAX version, GPU model) so a
figure always traces back to the exact code + hardware.

## Status / caveats (2026-07-11)

- Driver extracted and py_compile-clean; it **traced and ran end-to-end on 2 GPUs**
  before the node was wedged (see below) — needs a confirming re-run of
  `jaccpot/tests/test_distributed_fmm_driver.py`.
- **GPU-node incident:** the Bonsai *MPI* multi-GPU smoke test hit CUDA error 700
  and wedged the node's driver (nvidia-smi hangs; D-state processes across users).
  It needs an admin `nvidia-smi -r` / driver reload / reboot. Do NOT self-reset.
  Consequently Bonsai is **single-GPU baseline only** and `build_bonsai_mpi.sh` is
  disabled.
- The FMM harness scripts (performance/scaling), the tree comparison, and the
  Bonsai runner/parser are written and syntax-checked but **not yet validated on a
  live GPU** — do that first once the node is healthy (start small: `--n 20000`,
  2 GPUs), watching for capacity `overflow=True` (grow caps via `--calibrate`).
- Only 3 GPUs are free on this box → scaling curves run 1-2-3 GPUs.
