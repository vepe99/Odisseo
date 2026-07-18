# RTX 2080 Ti benchmark — ODISSEO jaccpot FMM vs Bonsai

Updated on-node (`compgpu5`, RTX 2080 Ti / sm_75) benchmark of the current
ODISSEO jaccpot FMM (fused device-only fast-lane, the built-in default since
`70811e6`) against the Bonsai BH-tree, on the **same** 200k-particle AGAMA disk
IC, live self-gravity + static external NFW halo, `G=1`, 4000 steps / `t_end=2.0`.

This mirrors `benchmark_a100/` but for the local Turing card. The one code
difference: **Pallas is off** — the fused near-field / M2L Pallas kernels are
gated to compute capability ≥ 8.0 (Ampere); sm_75 falls back to pure-JAX anyway.

## ⚠️ Node blocker (as of 2026-07-10)

GPU2 (`0000:1C:00.0`) on `compgpu5` is **hardware-faulted** ("Unknown Error").
This poisons `cuInit(0)` for the whole node: every CUDA process (JAX *and*
Bonsai) fails with `CUDA_ERROR_UNKNOWN`, regardless of `CUDA_VISIBLE_DEVICES`.
It also breaks `nvidia-smi -L` (exit 255), which is why `autocvd` crashes.

**Nothing here can run until an admin resets GPU2** (`nvidia-smi -r -i 2`) or
reloads the driver / reboots the node. `pick_gpu.sh` works around the *autocvd*
crash (it uses `--query-gpu`, which tolerates the missing card) but it cannot fix
the driver-level `cuInit` fault.

## Files

| File | Purpose |
|---|---|
| `env_rtx2080.sh` | fused fast-lane env, Pallas off (sm_75) |
| `pick_gpu.sh` | autocvd-equivalent free-GPU picker that skips the faulted GPU2 |
| `run_odisseo_perf.sh` | ODISSEO FMM perf (200k, 200 steps, 1 warmup + 3 measured) |
| `run_bonsai_reference.sh` | Bonsai 4000-step run on the same IC (regenerates the tipsy IC if missing) |
| `run_all.sh` | preflight cuInit check → both → `compare.py` |
| `compare.py` | parses both timings → `summary.{json,md}` + printed table |

## Run (once the node is healthy)

```bash
cd /export/home/tbuck/Odisseo
bash benchmark_rtx2080/run_all.sh          # preflight + odisseo + bonsai + compare
# or individually:
bash benchmark_rtx2080/run_all.sh odisseo
bash benchmark_rtx2080/run_all.sh bonsai
bash benchmark_rtx2080/run_all.sh compare
```

The ODISSEO perf number (median of 3 runs after 1 warmup, compile excluded) is
directly comparable to the A100 perf (77.1 ms/step) and, ×4000, to the Bonsai
full-run loop time on the same card.

## Reference: current standing on the A100 (`benchmark_a100/`)

| Code | per-step | full 4000-step |
|---|---|---|
| Bonsai (BH-tree) | ~10.0 ms | loop 40.1 s |
| ODISSEO jaccpot FMM (fused, Pallas on) | 77.1 ms | showcase 412.8 s |

→ ODISSEO ≈ 7.7× slower than Bonsai per step on the A100. The 2080 Ti numbers
will be slower in absolute terms but the *ratio* is the interesting comparison
(and, per the H100 note, the FMM near-field launch-bound behaviour is very
hardware-dependent).
