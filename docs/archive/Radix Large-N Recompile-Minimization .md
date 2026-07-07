# ARCHIVED DOCUMENT

This document is retained for historical context and is not the current source of truth.

Current navigation:
- `docs/STATIC_RADIX_HANDOFF_INDEX_2026-05-19.md`

## Radix Large-N Recompile-Minimization Plan (Tree-Change Safe)

### Summary
Implement a **radix large-N–only** incremental runtime path with **static-shape tree + multipole data structures** for fixed `N`, using a **fixed-capacity padded state** and **public explicit APIs** for partial rebuild and multipole refresh.  
Primary KPI: **recompile count reduction** (cold-compile events), targeting compile-at-startup then reuse through end-of-run in the ideal case.  
Hard constraints: **no runtime regression** and **no accuracy/conservation regression** versus current validated baseline.

### Priority Update (2026-04-23, aligned with current ODISSEO session)
1. Top priority is fixed-`N` static-shape reuse:
- keep tree-structure and multipole containers shape-stable across refreshes
- update numeric payloads in-place where possible
2. Recompute policy:
- recompute topology and multipoles only when particle positions require it
- avoid introducing new dynamic tensor shapes during these updates
3. Compile policy:
- aim for one startup compile (or one bounded profile set), then stable reuse
- treat unexpected new compile events as regressions to investigate

### Implementation Changes
1. **Introduce a static-shape “Compiled Profile” for radix large-N prepared states**
- Add a profile object carried in prepared state with fixed capacities and static kernel knobs:
  - max nodes, max leaves, max far pairs, max nearfield blocks, max leaf particle slots.
  - fixed nearfield/farfield chunking and tile parameters already used by the large-N lane.
- Add profile fingerprint/hash to diagnostics so compile reuse can be measured per run.
- Capacity overflow policy: if runtime payload exceeds profile capacity, trigger controlled full rebuild/reprofile (not silent shape drift).
- Explicitly enforce shape-stability for tree + multipole containers under a fixed particle count contract.

2. **Add public explicit incremental APIs (selected scope only)**
- `refresh_prepared_state(...)`:
  - Reuses existing topology and interaction scaffolding when profile-compatible.
  - Rebuilds only dynamic particle-dependent numeric payloads (sorted arrays, upward/downward numerical content) under fixed shapes.
- `update_multipoles_only(...)`:
  - Updates upward (and dependent downward locals) from new masses/positions when topology mapping is still valid.
  - Does not rebuild tree topology or interaction structure.
- `rebuild_topology_in_place(...)`:
  - Recomputes topology but keeps same compiled profile if capacity permits (pads to capacity).
- All 3 methods constrained to radix + large_n runtime path; other modes raise clear `NotImplementedError` for now.

3. **Partial rebuild strategy (decision-complete)**
- Reuse tiers (checked in order):
  1. **Full reuse**: same topology key/signature -> refresh numeric payload only.
  2. **Topology-changed but capacity-compatible**: rebuild topology/interactions into existing padded profile.
  3. **Capacity overflow or invariant failure**: automatic full rebuild with new profile.
- Invariants for reuse:
  - tree type radix, execution lane large_n, basis solidfmm, fixed dtype/profile knobs.
  - profile capacities not exceeded.
- Existing fallback behavior remains automatic (no fail-fast default).
- For perf test mode, support optional fail-fast on shape drift/compile drift.

4. **Compile-awareness instrumentation**
- Add lightweight counters/timestamps to runtime state:
  - compile cache hits/misses for evaluate kernels.
  - profile transitions (old fingerprint -> new fingerprint).
  - number of fallback full rebuilds due to capacity overflow.
- Surface these through a small public diagnostics accessor (read-only dict/struct).

5. **Integration with existing caches**
- Keep current topology and interaction cache keys, but bind them to profile fingerprint.
- Prevent dynamic-shape artifacts from leaking into compiled path by always materializing padded buffers for large-N lane.
- Keep current behavior for non-targeted modes unchanged.

### Public API / Interface Additions
- New public methods on solver/runtime wrapper (radix large-N only):
  - `refresh_prepared_state(...)`
  - `update_multipoles_only(...)`
  - `rebuild_topology_in_place(...)`
- New read-only diagnostics accessor:
  - returns compile/rebuild/profile counters for benchmarking and regression tests.

### Test Plan
1. **No-recompile reuse tests**
- Same profile, changing particle layouts/masses across steps:
  - assert compile miss count does not increase after first warm compile.
  - assert profile fingerprint stable.

2. **Topology-change under fixed capacity**
- Force topology key changes while staying within capacity:
  - assert no new compile.
  - assert numerical parity against full prepare/evaluate baseline within existing tolerances.

3. **Capacity-overflow fallback**
- Exceed profile capacities intentionally:
  - assert automatic full rebuild path taken.
  - assert profile fingerprint changes and compile miss increments exactly once per new profile.

4. **Public API behavior tests**
- Each new method:
  - valid in radix large-N.
  - raises clear error outside supported scope.
  - preserves output parity with current full `prepare_state + evaluate_prepared_state`.

5. **Performance acceptance checks**
- Add benchmark gate comparing before/after:
  - reduced compile events across repeated changing-topology steps.
  - no unacceptable steady-state runtime regression for warm eval path.
  - no accuracy/conservation regression versus current baseline tolerance envelope.

6. **Accuracy + runtime regression gates (required for rollout)**
- Runtime gate:
  - reject changes if total runtime or steady-state prepare/evaluate regress beyond agreed threshold.
- Accuracy gate:
  - reject changes if conservation metrics (`|ΔE/E0|`, `|ΔL|/|L0|`, COM drift) regress beyond agreed threshold.
- Reporting gate:
  - always publish before/after runtime + conservation side-by-side for each profile variant.

### Assumptions / Defaults
- Scope is intentionally limited to **radix + large_n + solidfmm** in iteration 1.
- Default behavior on drift is **auto fallback rebuild** (not fail-fast).
- Prioritize **recompile-count reduction** over minimal memory use; padded capacities are accepted.
- Public APIs are introduced now, but marked as large-N/radix constrained until later expansion.
