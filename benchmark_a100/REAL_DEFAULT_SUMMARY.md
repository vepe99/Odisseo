# Real-harmonics radix fast lane: verification + made the default (2026-07-18)

A100 (compgpu10), N=200k, order 4, θ=0.8, leaf 256, static_radix / large_n_gpu,
refresh_every=1, dehnen MAC, `env_fused.sh`, 100 steps × 3 measured runs (median),
IC `odisseo_agama_ic_200k_scm8_exp.npz`. All cells: fused lane active,
`fastlane_hits=1`, `fallback_count=0`.

## The question
Is the "radix fast lane + real harmonics + large-N" setup the best-performing FMM
inside ODISSEO's time integration, for BOTH the pure-JAX and Pallas backends?

## Answer: yes, on A100 — real is best-or-tied in both backends

4-cell A/B, {complex, real} × {pure-JAX, Pallas} (ms/step):

| cell             | before (main) | after (refactor) |
|------------------|--------------:|-----------------:|
| pure-JAX complex |        274.15 |           276.73 |
| pure-JAX real    |        272.36 |       **269.88** |
| Pallas complex   |        113.27 |           113.49 |
| **Pallas real**  |    **108.42** |       **108.65** |

- **Pallas real is the fastest cell** (~4.3% ahead of Pallas complex).
- **pure-JAX real ties/beats complex** on A100. The earlier RTX 2080 (sm_75, Turing)
  finding that pure-JAX real was +4.5–18% slower is **Turing-specific, not universal**;
  on the production A100 real wins or ties in both backends.
- The pure-JAX real cell **improved 272.4 → 269.9 ms/step** after the refactor removed
  the per-step complex→real conversion (native pure-real upward sweep). Pallas/complex
  cells are unchanged (within run-to-run noise).

## What was made the default
- **jaccpot**: default `basis` complex → **real**; the radix large-N fast lane now runs
  **pure-real end to end** (native real P2M/M2M upward; no `complex_to_dehnen_real_coeffs`
  conversion anywhere on the real path). complex/solidfmm retained for cross-checking.
- **jaccpot**: the fused production lane now **hard-errors instead of silently falling
  back** to a slower path (fast-lane blocked, or N not in the profile set).
- **ODISSEO** `SimulationConfig`: `fmm_basis="real"`, `fmm_tree_build_mode="static_radix"`,
  new `fmm_use_pallas=None` (auto: Ampere sm_80+ → Pallas on, pure-JAX on sm_75/CPU).
  Verified: with no `ODISSEO_FMM_USE_PALLAS` env, auto-detect enables Pallas → 107.9 ms/step.

## Energy / correctness
Native-real vs complex over 100 steps: identical to fp32 noise —
max|dE/E0| 1.630e-4 (real) vs 1.633e-4 (complex); max|dL/L0| 4.583e-4 vs 4.584e-4.
Runtime parity test `test_real_basis_tracks_complex_basis` green.

Raw reports: `benchmark_a100/phase0_baseline/` (before), `benchmark_a100/phase5_after/` (after).
