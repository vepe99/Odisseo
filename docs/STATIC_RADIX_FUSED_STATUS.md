# Static-Radix Fused FMM — Canonical Status

Single source of truth for the jaccpot FMM ↔ Odisseo strict fused static-radix
work. Supersedes the dated handoff/status/plan notes now under
[`docs/archive/`](archive/) (kept for provenance).

_Last updated: 2026-07-07._

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

## Next steps (deferred)

1. **Pallas near-field P2P kernel** — the only remaining lever for the residual
   ~2x eval gap; guarded/flagged with a pure-JAX fallback, parity-gated
   (`tools/fused_payload_force_parity.py`), and kept only if it wins walltime
   while preserving `fallback_count=0` and recompile-free.
2. Retire the deferred test debt above.

## Reproduce

```bash
# canonical 200k/2 strict fused smoke (pin a free GPU, disable preallocation)
CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
micromamba run -n odisseo python tools/walltime_ab_compare.py \
  --out-root /tmp/smoke_200k_2 --n-particles 200000 --num-steps 2 \
  --no-autocvd --fixed-policy --fixed-neighbor-cap 1048576 \
  --perf-warmup-runs 1 --perf-measure-runs 1 --profile-breakdown
```
