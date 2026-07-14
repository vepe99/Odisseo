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

## Treecode device-resident walk — MAC choice & multi-step stability (2026-07-14)

The opt-in device-resident per-leaf **treecode walk**
(`JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK=1`) replaces the host-iterated yggdrax
dual-tree walk (kills its launch storm) and is the intended high-performance far/near
builder. Its acceptance criterion (MAC) can use two per-node extent recipes, selected by
`JACCPOT_STATIC_STRICT_FUSED_TREECODE_MAC`:

Note the fast lane recomputes **all node geometry every step** — particles are
re-Morton-sorted (fresh leaf membership) and node centers/bounding-sphere radii/box
extents/multipoles/interaction lists are rebuilt from the current positions; only the
tree *shape* (node index-ranges, leaf count, buffer capacities) is frozen (to avoid
recompilation). So the choice below is a bound-*tightness* issue, not stale geometry.

- **`bh`** — axis-aligned box `max_extent` (half-width). Cheaper (accepts fewer
  far/M2L pairs → faster) and *statically* as accurate as the sphere (t=0 force parity
  vs the dual walk ~0.03 %). **But dynamically UNSTABLE**: the box `max_extent`
  systematically *under-bounds* the true source multipole radius (the bounding sphere
  circumscribes the box, ≈√3× larger isotropic, more when anisotropic). Feeding the
  smaller extent makes the MAC accept far pairs at smaller `d` than the sphere would →
  **`bh` effectively runs at a coarser opening angle than the requested θ**, systematically
  under-resolving the far field. Instantaneously tiny (~0.03 %), but it is a *coherent,
  non-gradient* force bias, and velocity-Verlet does not conserve energy under a
  non-conservative force → it **accumulates into secular heating** and blows up
  (200k/order-4/real: max|v| 7 → 20 → 142 → >10³ over 300 steps; total energy diverges).
  Not an overflow effect (huge caps don't fix it).
- **`dual`** (**DEFAULT since 2026-07-14**) — reproduce the configured dual-tree MAC
  extents (dehnen bounding-**sphere** radius for the large-N preset). This is the correct
  multipole-radius bound, keeps every accepted pair inside the θ budget, and gives
  **accuracy-profile parity** with the validated dual walk. Stable: over 300 steps
  `max|v|` 7.34, `dKE/KE0` 5.6e-2, `|dLz/Lz0|` 1.9e-3 — matching the dual-walk baseline
  (7.33 / 5.6e-2 / 2.2e-3) in both the complex and real (Dehnen) bases.

**Cost of the fix:** dehnen accepts deeper → more M2L pairs → ~16 % slower per step
(200k/order-4/real, A100: ~58 ms/step `dual` vs ~50 ms/step `bh`), still launch-storm-free.

**Guidance:** use the default `dual` for any multi-step integration; `bh` is safe only
for single-shot/static force evaluations. Full write-up:
`jaccpot/docs/treecode_mac_stability.md`; code in
`jaccpot/runtime/_interaction_cache.py` (`_build_treecode_artifacts_strict_streamed`,
`_treecode_mac_extents`) and `jaccpot/experimental/treecode_walk.py` (`_mac_ok`).

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
and the `*_pallas_*` tests in `tests/unit/core/test_near_field.py`. At
production scale (200k, leaf=256) the full fused-eval acceleration matches the
pure-JAX path to **rel-L2 3.5e-6** (median 2.6e-6).

**Reproducibility note:** the Pallas kernel accumulates in a sequential register
loop, XLA uses a tree reduction — so results are **not bit-identical** (they
differ at the fp32 ~1e-6 level). In a live N-body integration that per-step
difference amplifies chaotically: a 10-step (2 Gyr) `strict_run_v2` A/B diverged
by rel-L2 ~0.19 in the final state. This is expected float32 chaos (both
trajectories are equally valid), not a correctness defect — confirmed by the
tight per-step accel parity above. Use fp64 if bitwise trajectory reproducibility
against the JAX path is required.

**Performance (A100, same-process warm min-of-25, 200k, p=4, theta=0.6,
leaf=256, fp32, near-field-only force eval):**

| force eval (200k, leaf=256) | pure-JAX | fused Pallas | speedup |
| --- | --- | --- | --- |
| near-field-only P2P | 0.219 s | 0.055 s | **4.0x** |
| full fused eval (near + far) | 0.217 s | 0.055 s | **3.9x** |

The near-field was ~98% of the fused eval, so the full-eval and near-only gains
track each other (far-field ~0.006 s). Both runs were same-process warm min-of-25
on an uncontended A100 (min ≈ mean). jaxfmm potential eval on the same box is
~0.013 s; the residual gap to it is force-vs-potential (jaccpot returns 3-vector
forces) plus the small far-field. Benchmark:
`jaccpot/bench/bench_fused_eval_vs_jaxfmm.py --use-pallas both [--near-only]`
(uses `autocvd --gpu-select {free,least-used}` to pin an uncontended GPU — timing
on a shared/contended GPU is meaningless and produced the earlier noisy baselines).

**Odisseo production wiring:** `odisseo/jaccpot_coupling.py::_build_fmm_solver`
threads `use_pallas` into the solver from `ODISSEO_FMM_USE_PALLAS` (default off;
set `1`/`true` to enable on Ampere+). The solver still auto-falls back to pure
JAX on unsupported hardware. Verified: the flag propagates to
`solver._impl.use_pallas`.

## Fused lane on concentrated galaxy ICs — static-block cap auto-size (2026-07-08)

Running the 200k **Agama disk** on the fast fused lane exposed a prepare-side
blocker independent of the Pallas kernel: the fused near-field packs each target
leaf's neighbour source leaves into a fixed-shape `(num_leaves,
max_blocks_per_leaf, block_size)` **static-target-block** payload, and the cap
was a fixed int (default 32, ladder 8..128) with no data-driven sizing. A
centrally-concentrated disk has dense inner leaves with huge near-neighbour
counts (measured: the central leaf is "near" **781 of 782** leaves at theta=0.6),
so prepare raised `static target-block cap exceeded` at every ladder value.
(Uniform-random points — as in the eval benchmark — have bounded degree and fit
cap 64, which is why the 3.9x eval held there.)

Fixes:
- **jaccpot** (`runtime/_large_n_pipeline.py`): the cap auto-sizes to the densest
  leaf at eager prepare (`ceil(max_leaf_degree/block_size)*headroom`, from an
  extended caps ladder), supports `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF=auto`,
  and caches the resolved cap so the traced strict refresh reuses the same fixed
  shape (zero-recompile). At leaf=256 the cap auto-sizes 32→256 (~0.8 GB payload).
  Unit test: `jaccpot/tests/unit/test_static_target_blocks_cap.py`.
- **Odisseo** (`jaccpot_coupling.py`): the coupling defaults the static-block cap
  to `auto`; and — the second half of the fix — wraps the `strict_run_v2` call in
  `_temporary_large_n_environment` so the large-N env (notably `TARGET_BLOCK_SIZE`)
  stays active while the device-resident scan is **traced/compiled**, not just
  during the eager prepare. Without this the traced refresh re-resolved
  `block_size` to its default (4→32) and the fused static-target-block preflight
  mismatched.

Result: the concentrated Agama IC now prepares and runs on the **fast fused
lane** (~0.4 s/step once compiled, vs the ~9 s/step non-static dynamic fallback);
pallas engages (final-state A/B differs). Verified 4–40 step runs.

**Neighbor-edge cap — auto-sized up front (2026-07-08).** A *second* fixed-shape
cap, the neighbor-edge profile cap, also underestimated concentrated ICs (its
N-based bootstrap gave 209648 vs the disk's ~800768 active edges). Key findings:
it cannot grow mid-scan (the eager prepare builds the bootstrap-sized initial
carry while the traced refresh builds the full list → growing the refresh breaks
the `lax.scan` carry), and it cannot be measured via a separate yggdrax
`build_leaf_neighbor_lists` (that counts leaf-neighbours, ~6x fewer than
jaccpot's dual-tree traversal edges). But the neighbor-edge list is just int
edge ids (~6 MB at 800768), so over-provisioning is cheap. Fix
(`jaccpot_coupling.py::_default_fused_neighbor_edge_cap`): the coupling sets
`JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP` **generously up front** (before
any prepare caches the env-config) — default 16 edges/particle (3.2M / ~26 MB at
200k), tunable via `ODISSEO_FMM_NEIGHBOR_EDGE_PER_PARTICLE_CAP`; extreme ICs can
still set the jaccpot env directly. Result: the 200k Agama disk now runs the
**fast fused lane fully automatically** (no manual caps).

## Next steps (deferred)

0. (done 2026-07-08) The **neighbor-edge** profile cap is now auto-sized up front
   for concentrated ICs (generous `edges/particle` default; see above). A future
   refinement could measure the exact count via jaccpot's own traversal instead
   of a generous heuristic, but the heuristic is cheap and covers realistic disks.
1. `use_pallas` is wired into the Odisseo coupling (`ODISSEO_FMM_USE_PALLAS`);
   the full fused *eval* is re-baselined above (3.9x). An end-to-end 10-step
   `strict_run_v2` A/B (via `tools/walltime_ab_compare.py`, `--variant-env
   JACCPOT_STATIC_STRICT_FUSED_MODE=on --variant-env ODISSEO_FMM_USE_PALLAS=1`)
   ran clean (final state all-finite, physical). It is compile/IO-dominated at
   10 steps (whole-process wall only ~1.08x), so it is *not* a per-step
   throughput measure — the isolated self-force eval (3.9x) is. A longer-horizon
   step-throughput A/B (many steps, or reading the simulator's measured median
   rather than process wall) remains a nice-to-have.
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
