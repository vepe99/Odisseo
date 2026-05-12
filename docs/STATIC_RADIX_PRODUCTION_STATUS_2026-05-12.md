# Static-Radix Production Status - 2026-05-12

## Scope

Cross-repo checkpoint after reading:

- `docs/STATIC_RADIX_TARGET_BLOCK_OVERFLOW_HANDOFF_2026-05-06.md`

and scanning current `Odisseo`, `jaccpot`, and `yggdrax` branches.

## Repository Snapshot

### Odisseo

- Branch: `feat/static-radix-memory-fit`
- Local modifications include:
  - `notebooks/scalability/galaxy_disk_fmm_large_n.py`
  - `odisseo/jaccpot_coupling.py`
  - `odisseo/option_classes.py`
- Recent additions already support:
  - disabling large-N env overrides,
  - explicit initial-acceleration diagnostics,
  - IC velocity potential controls and metadata capture.

### jaccpot

- Branch: `fix/radix-fast-lane-overflow`
- Latest committed fix: `48fa71c` (`Fix radix fast-lane overflow nearfield`)
- Current uncommitted work (validated today):
  - `jaccpot/runtime/_fmm_impl.py`
  - `jaccpot/runtime/_large_n_pipeline.py`
  - `tests/integration/test_fmm.py`

### yggdrax

- Branch: `feat/static-radix-memory-fit` (clean)
- Recent static-radix API and traversal updates are present on `main`.
- `build_tree` is exported in `yggdrax/__init__.py` and exists in `yggdrax/tree.py`.

## Confirmed Behavior Today

1. The fast-lane nearfield evaluator in `jaccpot/runtime/_large_n_nearfield.py` now includes overflow contributions when present, via:
   - radix overflow payload (`compute_leaf_p2p_accelerations_radix_payload_pairs_only`) and
   - generic target-block overflow fallback (`compute_leaf_p2p_accelerations_target_block_pairs_only`).

2. Focused integration tests pass in local environment (`micromamba run -n odisseo`):
   - `test_radix_fast_lane_includes_overflow_target_blocks`
   - `test_static_radix_refresh_rebuilds_current_large_n_payloads`

3. The uncommitted refresh changes in `jaccpot` are currently important:
   - They rebuild large-N prepared payloads from current positions during static-radix refresh instead of reusing stale nearfield payloads.
   - They disable static-radix interaction-cache reuse that can attach stale traversal/neighbor payloads.

## Current Risk Register

1. Performance risk under heavy overflow remains unresolved for production:
   - Correctness is restored, but overflow-heavy layouts (e.g. tiny target block size + tiny fast prefix) can still be too slow.

2. Runtime safety in Odisseo currently depends on configuration discipline:
   - block-size-4 overrides should remain disabled for production unless overflow is eliminated or made fast.

## Structured Commit Plan

1. `jaccpot`: commit validated static-radix refresh correctness patch + tests.
2. `Odisseo`: commit runtime safety default(s) and diagnostics docs updates.
3. `jaccpot`: implement/benchmark overflow-performance strategy (prefer no-overflow layout first, then optimized overflow path if needed).
4. `Odisseo`: rerun and record galaxy acceleration report + short movie validation using matched IC/runtime settings.

## Next Immediate Actions

1. Commit current validated `jaccpot` refresh fixes.
2. Add a short benchmark harness/checkpoint to compare:
   - block size 32 (overflow 0),
   - block size 4 + fast prefix 256 (overflow 0),
   - block size 4 + small prefix (overflow > 0).
3. Gate production profile to a safe no-overflow configuration until benchmarked overflow path is acceptable.
