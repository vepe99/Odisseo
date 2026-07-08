# Static-Radix Fused FMM — Canonical Status

Single source of truth for the jaccpot FMM ↔ Odisseo strict fused static-radix
work. Supersedes the dated handoff/status/plan notes now under
[`docs/archive/`](archive/) (kept for provenance).

_Last updated: 2026-07-08 — fused Pallas near-field P2P kernel implemented on
A100 (sm_80): 4.0x on the near-field force eval. See
[Pallas near-field P2P kernel — IMPLEMENTED](#pallas-near-field-p2p-kernel--implemented-2026-07-08-a100-sm_80)._

## Goal

Run a 200k-particle Agama galaxy disk with `jaccpot` FMM self-gravity **inside
Odisseo's time integration** under a strict static policy:

- one JIT compile at startup, **no per-step recompiles**,
- **constant-shape data structures** across refreshes,
- **no host round-trips** in the hot loop (device-resident),
- **no fallback** to slow paths,
- warm full step **< 0.5 s** on one GPU,
- physics unchanged, and ultimately fast enough to **beat jaxfmm / Dehnen-style FMM**.

## Architecture

Three sibling repos, dependency chain `Odisseo → jaccpot → yggdrax`, coupled at
runtime via lazy imports in `odisseo/jaccpot_coupling.py` (no packaged dependency):

- **yggdrax** — tree/traversal library: Morton codes, static-radix tree build +
  template refresh, dual-tree far/near interaction lists, compact far-pairs.
- **jaccpot** — FMM solver: upward (P2M/M2M), M2L, downward (L2L/L2P), near-field,
  and the strict fused runtime (`jaccpot/runtime/_fmm_impl.py`,
  `_large_n_pipeline.py`, `_large_n_nearfield.py`).
- **Odisseo** — integrator + coupling (`odisseo/jaccpot_coupling.py`,
  `integration_api.py`, `option_classes.py`) + benchmark tooling (`tools/`,
  `notebooks/scalability/galaxy_disk_fmm_large_n.py`).

The production path is the **strict fused static-radix lane**: refresh → upward →
M2L → downward → evaluate → velocity-Verlet update run inside one compiled scan,
device-resident. Entry point `strict_run_v2` (raw-tensor multi-step runner);
Odisseo drives it from the strict lane in `jaccpot_coupling.py`.

## Current state (working / met)

- Strict fused lane is the production default: `strict_fused_mode_active=True`,
  `fallback_count=0`, planner bypassed, device-resident refresh.
- **No per-step recompile**: one compile per fixed profile; overflow → controlled
  reprofile to the next fixed cap tier, not dynamic shapes.
- **Constant shapes**: static-radix template refresh + fixed-capacity padded
  buffers (interaction/neighbor caps, flat compact far-pairs).
- **Correctness fixes landed & regression-guarded**:
  - fresh compact far-pair rebuild is the default with active-count masking of
    padded slots (`yggdrax.CompactTaggedFarPairs.far_pair_count`), fixing stale
    far-pair reuse that corrupted forces under particle drift (unsafe reuse is
    now opt-in);
  - velocity-Verlet endpoint correction (one endpoint FMM eval per step, was
    reusing a single acceleration at both endpoints);
  - NFW potential finite-at-`r→0` small-radius series (`odisseo/potentials.py`).

### Performance

- Reference GPU (per archived 2026-06 handoffs): warm **200k/1 ≈ 0.31 s/step**,
  200k/20 ≈ 0.33 s/step — target met. Trajectory: ~8.8 s/step (May) →
  ~1.95 → 1.00 → 0.471 → 0.397 → **0.312 s/step** (flat compact far-pairs +
  static node-range reuse).
- This-machine validation (RTX 2080 Ti, 11 GB, jax 0.9.0): canonical 200k/2
  `tools/walltime_ab_compare.py --fixed-policy --fixed-neighbor-cap 1048576`
  → `strict_fused_mode_active=True`, `fallback_count=0`, `compile_count=1` /
  `execute_count=2` (no per-step recompile), 0 overflow/neighbor reprofiles,
  all-finite. ~0.78 s/step with `--profile-breakdown` diagnostics on (slower,
  weaker GPU; not comparable to the 0.31 s reference figure).

## Deferred test debt (pre-existing; tracked, not blocking)

Verified against committed HEAD — these fail independently of the consolidated
work and are cleanup follow-ups:

- **jaccpot (14)**: 9 stale policy assertions in `tests/unit/test_solver_api.py`
  and `test_large_n_fast_path_policy.py` (defaults evolved: `target_owned_block_size`
  8→32, gpu-preset caps, `process_block` 256); 5 in `tests/integration/test_fmm.py`
  — 2 small-N strict `prepare_state` tests that must set
  `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH=0` (the strict fail-fast is
  the intended contract; see the sibling `test_strict_fused_moved_endpoint_matches_fresh_prepare`),
  1 cache-key spy, 2 for the deliberately-abandoned adaptive/class_major preset.
- **Odisseo (3)**: environmental under jax 0.9.0 — `test_initial_condition.py`
  (jaxtyping), `test_integrators.py` (diffrax `saveat`); files untouched by this work.

## jaxfmm head-to-head (2026-07-07, RTX 2080 Ti, fp32)

jaxfmm 0.2.0 is installed. Compared via `jaccpot/bench/bench_fused_eval_vs_jaxfmm.py`
(times the strict *fused* eval-only path via the new
`FastMultipoleMethod.strict_fused_prepared_eval_fn`, not the slow non-fused
`evaluate_prepared_state`). falcON is out of scope (no build; Dehnen appears only
as the MAC).

Eval at 200k, matched accuracy (p=4, theta=0.6, leaf/N_max=256), warm min-of-20:

| quantity | jaxfmm | jaccpot fused eval-only |
| --- | --- | --- |
| potential | 0.051 s | ~0.12 s (far 0.006 + near-potential 0.116) |
| force (3-vector) | — | 0.279 s (far 0.006 + near-force 0.267) |

**Root cause of the gap (not what the aspiration assumed):**
- jaccpot's **FMM far-field is excellent** — 0.006 s; never the bottleneck.
- The near-field direct P2P is **~98% of the eval** and the whole gap.
- ~half the apparent 5x is **force-vs-potential**: jaxfmm returns a scalar
  potential; jaccpot returns 3-vector forces (~2.3x costlier). Potential-to-
  potential the gap is **~2.4x**, not 5x.
- The residual ~2x is **GPU utilization** in the near-field P2P. A pure-JAX
  dense-block P2P (jaxfmm-style, verified to parity 3e-8 vs the canonical
  near-field) does **not** close it — it matches the current ~0.24 s force
  near-field regardless of batch size (~7% of GPU peak). Only a tiled **Pallas
  P2P kernel** could plausibly close it; deferred as uncertain-payoff future work.

**Conclusion:** jaccpot does not currently beat jaxfmm on raw eval throughput.
Its defensible advantage is the in-integrator strict fused static-shape
zero-recompile full step, which jaxfmm's functional `eval_potential` does not
itself provide.

## GPU profiling + near-field optimization attempt (2026-07-07)

Profiled the production device-resident `strict_run_v2` scan on RTX 2080 Ti
(sm_75) via `jaccpot/bench/profile_fused_gpu_util.py` (nvidia-smi dmon) and nsys:

- **The GPU is NOT idle-bound.** During execution SM occupancy is ~97-98%
  (dmon); the fused lane keeps the GPU busy. There is a one-time ~29s compile per
  `num_steps` profile (GPU idle then; amortized in production).
- **The near-field is fragmented + inefficient.** nsys shows ~2,500 tiny
  (~1-2us) fused kernels per step (`input_dynamic_slice_reduce`,
  `loop_dynamic_slice`, `loop_add`) vs a handful of efficient GEMMs for the
  far-field. The near-field runs at ~7% of FLOP peak: memory-bound, because XLA
  materializes the WxW distance matrix to HBM.

**Pure-JAX near-field restructure: no robust win.** `bench_fused_eval_vs_jaxfmm.py`
and a formulation sweep (vectorized vs componentwise dense-block, batch 512-8192,
all parity ~2e-7 vs the current near-field) cap at ~0.24s -- the same as the
current ~0.267s within measurement error (the small apparent gain comes from
pre-gathering positions outside the timed region). Pure JAX cannot escape the
HBM-materialization ceiling; consistent with the archived "unroll knobs
exhausted" note.

**On sm_75 the Pallas lane was blocked** — a SRAM-tiling Pallas P2P kernel is the
only lever that reaches better near-field efficiency, and on the RTX 2080 Ti
cluster (compute capability **7.5 / sm_75**) even a minimal Pallas/Triton kernel
fails: `'nvvm.cp.async.bulk.wait_group' op is not supported on sm_75`. jaccpot's
`pallas_nearfield_*_supported()` guards `>= 8.0`. **This is a hardware property,
not a code limitation** — see the update below.

## Pallas near-field P2P kernel — IMPLEMENTED (2026-07-08, A100 sm_80)

**Hardware correction:** the sm_75 blocker above does *not* apply to every
machine. The development host used here has **8x NVIDIA A100-PCIE-40GB, compute
capability 8.0 (sm_80)**, jax/jaxlib 0.9.0. Pallas/Triton compiles and runs
here; `pallas_nearfield_fused_supported()` returns `True`.

A fused tiled Pallas near-field P2P kernel now exists and is wired into the
radix fast lane in jaccpot (branch `feat/pallas-nearfield-fused`):

- `jaccpot/pallas/nearfield_fused_leaf.py` provides two register-blocked kernels
  that keep the `W_t x W_s` distance products in registers (never HBM) and emit
  **acceleration + potential** leaf-major:
  - `nearfield_fused_leaf_*` for the materialized per-particle source payload;
  - `nearfield_leafpair_*` for the **compact prepacked source-leaf-id layout the
    production fused lane actually uses** — source leaves are gathered by id from
    the small `leaf_positions` table inside the kernel and invalid slots are
    skipped with `lax.cond`, avoiding the multi-GB dense source materialization
    (the materialized `(num_leaves, ~2048, W)` payload OOMs at leaf=256).
- Gated by `use_pallas` + `pallas_nearfield_fused_supported()` with the pure-JAX
  paths as fallback (unchanged default behavior; CPU/CI use interpret mode via
  `JACCPOT_NEARFIELD_PALLAS_INTERPRET=1`). `return_potential=True` on the fast
  lane is now implemented (was `NotImplementedError`).
- Tunables: `JACCPOT_NEARFIELD_PALLAS_{TARGET_SUBTILE,NUM_WARPS,NUM_STAGES}`;
  default target-subtile 32 (power-of-two, best A100 occupancy).

**Correctness:** the Pallas paths match the pure-JAX baselines and a brute-force
direct sum to **~1e-15 (fp64) / ~1e-6 (fp32)** for both acceleration and
potential — see `jaccpot/tests/unit/operators/test_pallas_nearfield_fused.py`
and the `*_pallas_*` tests in `tests/unit/core/test_near_field.py`.

**Performance (A100, same-process warm min-of-25, 200k, p=4, theta=0.6,
leaf=256, fp32, near-field-only force eval):**

| near-field force P2P | time | speedup |
| --- | --- | --- |
| pure-JAX radix fast lane | 0.219 s | 1.0x |
| fused Pallas leaf-pair (subtile 32) | 0.055 s | **4.0x** |

The near-field was ~98% of the fused eval, so this collapses the dominant cost
(jaxfmm potential eval on the same box is ~0.013 s; the residual gap is
force-vs-potential + the small far-field). Benchmark:
`jaccpot/bench/bench_fused_eval_vs_jaxfmm.py --use-pallas both --near-only`
(now uses `autocvd --gpu-select free` to pin an uncontended GPU — timing on a
shared/contended GPU is meaningless and produced the earlier noisy baselines).

## Next steps (deferred)

1. Enable `use_pallas` on the Odisseo→jaccpot production coupling path and
   re-baseline the full strict-fused step (this status recorded the isolated
   near-field eval; the end-to-end `strict_run_v2` step gain is pending a clean
   uncontended run).
2. Optional: a potential-only kernel variant (skips the 3-vector accumulation)
   for jaxfmm potential-to-potential parity; tune num_stages/subtile per leaf
   size; extend the leaf-pair kernel to the overflow/target-block payloads.
3. Retire the deferred test debt above. Note: 9 pre-existing
   `test_solver_api.py` / `test_large_n_fast_path_policy.py` failures are stale
   solver-policy default expectations (block_size 32 vs 8, traversal caps),
   unrelated to this work and present on the base branch.

## Reproduce

```bash
# canonical 200k/2 strict fused smoke (pin a free GPU, disable preallocation)
CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
micromamba run -n odisseo python tools/walltime_ab_compare.py \
  --out-root /tmp/smoke_200k_2 --n-particles 200000 --num-steps 2 \
  --no-autocvd --fixed-policy --fixed-neighbor-cap 1048576 \
  --perf-warmup-runs 1 --perf-measure-runs 1 --profile-breakdown
```
