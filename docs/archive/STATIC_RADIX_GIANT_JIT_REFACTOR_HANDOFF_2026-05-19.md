# Static-Radix Giant-JIT Refactor Handoff (2026-05-19)

## Goal

Maximize throughput and GPU utilization in the strict static-radix 200k lane by reducing host-side refresh orchestration, while keeping physics behavior unchanged.

## Frozen Production Comparison Lane

- `fmm_preset=large_n_gpu`
- `fmm_runtime_path=large_n`
- `fmm_tree_build_mode=static_radix`
- `fmm_leaf_size=256`
- `fmm_refresh_every=1`
- fixed Agama IC file (`--ic-source load --ic-input-path <same-file>`)
- strict fail-fast policy unchanged

## Single Perf Oracle

Use only `tools/walltime_ab_compare.py` for keep/revert decisions.

- Baseline env:
  - `JACCPOT_STATIC_STRICT_GPU_MODE=on`
  - `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH=0`
- Variant env: exactly one knob at a time (`--variant-env KEY=VALUE`)
- Summary output: `walltime_ab_summary.json` with only `frozen_baseline`, `baseline`, `variant`, `delta`

## Current Blocker

Main blocker is still host orchestration in strict refresh internals (tree/upward -> dual/downward -> nearfield handoff seams) in jaccpot runtime, not wrapper-level Odisseo jit toggles. Previous wrapper-side giant-jit experiments were neutral or slower at 200k.

## Status Log

- 2026-05-19: Added dedicated giant-jit handoff.
- 2026-05-19: Tested strict-lane scaffold routing toggle in Odisseo (`ODISSEO_FMM_STRICT_USE_CORE_SCAFFOLD=1`); 200k showed no meaningful gain.
- 2026-05-19: Added external-acceleration support in scaffold loop; 200k remained flat/slower.
- 2026-05-19: Tested jaccpot experimental compiled-step flag (`JACCPOT_STATIC_STRICT_EXPERIMENTAL_COMPILED_STEPS=1`); 200k slower.
- 2026-05-19: Added jaccpot fused prepare entry (`JACCPOT_LARGE_N_FUSE_TREE_DUAL_PREPARE=1`); tiny run stable/slightly faster, 200k not yet showing clear throughput win.
- 2026-05-19 cleanup: removed Odisseo strict-core scaffold experimental routing from active strict path.
- 2026-05-19 cleanup: normalized strict refresh reporting to use `refresh_total_seconds` fallback to component-sum when needed.
- 2026-05-19: Added deterministic A/B harness (`tools/strict_ab_compare.py`) for diagnostics.
- 2026-05-19: Added canonical wall-time-only A/B harness (`tools/walltime_ab_compare.py`) for throughput decisions.

## Keep/Revert Rule

- Keep a slice only if 200k wall time improves by a meaningful margin (target >3%).
- Revert any slice that regresses wall time before trying next slice.

## Next Slice

Move refactor inside jaccpot strict refresh internals only:
- reduce Python-stage seams between tree/upward, dual/downward, nearfield prep handoff,
- keep algorithm/physics identical,
- re-run 200k wall-time A/B against frozen baseline.
- 2026-05-20: jaccpot internal strict-path cleanup/refactor slice applied:
  - removed `JACCPOT_STATIC_STRICT_EXPERIMENTAL_COMPILED_STEPS` branch from `strict_run_v2` (single stable strict route only),
  - made fused tree+dual prepare the default internal candidate path with explicit opt-out `JACCPOT_LARGE_N_DISABLE_FUSED_TREE_DUAL_PREPARE=1`.
- 2026-05-20 correctness gate: strict tiny run passed under baseline lane env (`JACCPOT_STATIC_STRICT_GPU_MODE=on`, `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH=0`) with `n=256`, `num_steps=1`.
- 2026-05-20 throughput gate (wall-time-only A/B, fixed IC `/tmp/odisseo_fixed_agama_ic_200k.npz`):
  - Command: `micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_fused_default --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_LARGE_N_DISABLE_FUSED_TREE_DUAL_PREPARE=1`
  - Baseline (fused default): `178.6696 s`
  - Variant (fused disabled): `178.7808 s`
  - Delta variant-baseline: `+0.1112 s` (variant slower, but effectively neutral)
  - Decision: no meaningful throughput gain (>3% target not met); keep as simplification only, not as a confirmed performance win.
- 2026-05-20 refactor slice (deeper strict fusion): in `jaccpot/runtime/_fmm_impl.py`, strict streamed-direct branch now bypasses extra host orchestration stages (far-pair planning/autotune/select) and feeds compact COO far pairs directly into downward compute.
- 2026-05-20 final single timing run after full slice completion (wall-time-only, fixed IC `/tmp/odisseo_fixed_agama_ic_200k.npz`):
  - Command: `micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_final_fused_slice --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_LARGE_N_DISABLE_FUSED_TREE_DUAL_PREPARE=1`
  - Baseline (fused default): `176.2964 s`
  - Variant (fused disabled): `176.7022 s`
  - Delta variant-baseline: `+0.4058 s` (variant slower)
  - Decision: fused-default remains preferable, but gain is small and below the >3% meaningful-win threshold.
- 2026-05-20 implementation: added strict fused-lane contract/routing scaffolding in jaccpot runtime:
  - new env controls: `JACCPOT_STATIC_STRICT_FUSED_MODE={off,auto,on}` and `JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET` (default capacities `100000,200000,400000`),
  - static profile key matching over `(n, leaf=256, refresh=1, dtype=float32, tree=static_radix, preset=large_n_gpu)`,
  - strict `strict_run_v2` routes eligible requests to new `_strict_run_v2_fused_profile(...)` with hard fallback to existing strict path,
  - added observability counters/flags and fallback reason export (`strict_fused_*`).
- 2026-05-20 ODISSEO diagnostics mapping extended with `runtime_strict_fused_*` fields.
- 2026-05-20 validation:
  - tiny (`n=256`, `num_steps=1`, forced profile set `256`) shows fused routing attempted and then fallback (`runtime_strict_fused_last_fallback_reason=fused_runtime_error:TypeError`),
  - production-shape short check (`n=200000`, `num_steps=1`) shows same: fused route attempted, then fallback with `TypeError`.
- Current blocker to full fused execution: strict fused scan still cannot remain entirely in compiled execution due runtime/type constraints inside `strict_prepare_refresh_and_evaluate`/prepare pipeline under traced scan carry; fallback currently preserves correctness.
- 2026-05-20 fused carry-structure fix: updated `_strict_run_v2_fused_profile(...)` to warm-start when `prepared_state is None` and then run scan with stable non-None `PreparedStateLike` carry.
- 2026-05-20 fused activation validation:
  - tiny (`n=256`, forced fused profile set `256`): `runtime_strict_fused_mode_active=true`, `fallback_count=0`.
  - production-shape short (`n=200000`, `num_steps=1`): `runtime_strict_fused_mode_active=true`, `fallback_count=0`.
- 2026-05-20 throughput gate (wall-time-only, fixed IC `/tmp/odisseo_fixed_agama_ic_200k.npz`):
  - Command: `JACCPOT_STATIC_STRICT_FUSED_MODE=on JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=100000,200000,400000 micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_fused_on_vs_off --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=off`
  - Baseline (fused on): `178.0405 s`
  - Variant (fused off): `177.2095 s`
  - Delta variant-baseline: `-0.8310 s` (fused off faster)
  - Decision: current fused lane is functionally active but not yet throughput-positive; do not keep as default-on for performance lane yet.
- 2026-05-20 post-refactor throughput gate (fused-device-mode branch):
  - Command: `JACCPOT_STATIC_STRICT_FUSED_MODE=on JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET=100000,200000,400000 micromamba run -n odisseo python tools/walltime_ab_compare.py --ic-input-path /tmp/odisseo_fixed_agama_ic_200k.npz --out-root /tmp/odisseo_walltime_ab_fused_device_mode --n-particles 200000 --num-steps 20 --state-dtype float32 --leaf-size 256 --refresh-every 1 --variant-env JACCPOT_STATIC_STRICT_FUSED_MODE=off`
  - Baseline (fused on): `179.5595 s`
  - Variant (fused off): `175.5996 s`
  - Delta variant-baseline: `-3.9599 s` (fused off faster)
  - Interpretation: fused-mode control flow is active and stable, but current fused body still underutilizes GPU / incurs extra overhead relative to strict non-fused path.
  - Decision: do **not** enable fused mode as default for perf lane.

## Current Root-Cause Hypothesis (GPU utilization still low)

- Fused loop exists at control-flow level, but per-step refresh still executes a fragmented set of kernels (tree/upward, dual build, nearfield prep/eval) with relatively low arithmetic intensity and synchronization boundaries.
- Host orchestration was reduced (planner/cache bypass in fused-device mode), but not fully eliminated from refresh internals and payload construction.
- External acceleration + leapfrog update are cheap versus refresh cost; occupancy remains governed by refresh subgraph, not integration math.

## Next Focused Work Items

1. Move strict fused refresh onto a dedicated pre-specialized kernel configuration object (no in-loop env/config resolution, no dynamic branch selection in refresh body).
2. Collapse nearfield payload/precompute stages further so fused loop reuses static-capacity buffers without repeated rebuild logic each step.
3. Introduce optional refresh cadence >1 inside fused lane for controlled experiments (still static and fail-fast), to test whether refresh frequency dominates utilization wall.
4. Add one profile-enabled diagnostic run comparing fused on/off to isolate which refresh substage time grew in fused mode after latest changes.
