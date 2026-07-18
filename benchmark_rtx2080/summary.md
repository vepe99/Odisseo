# RTX 2080 Ti — ODISSEO jaccpot FMM vs Bonsai (accuracy-matched)

Updated 2026-07-13, on `compgpu5` (RTX 2080 Ti, sm_75, Pallas OFF). 200k-particle
AGAMA disk (`scm8_exp` IC), live self-gravity + static external NFW halo, `G=1`,
dt=5e-4, softening=0.002, same IC for both codes.

## Fairness basis
Bonsai is a **Barnes-Hut tree** (monopole+quadrupole, θ=0.5), not an FMM. We match
the **accuracy budget** using each code's force approximation:
- FMM force accuracy measured as **relative force error vs a direct-sum reference**
  (`--initial-accel-report`, 4096 sampled targets).
- Bonsai uses quadrupole (order-2) multipoles at θ=0.5. Its accuracy is not directly
  dumpable (no per-particle accel output; its `Etot` excludes the external halo so
  energy drift is not a usable metric), so we match by multipole truncation and note
  that FMM order-2 (with local expansions) is *at least as accurate* as BH order-2 —
  a conservative match that does not flatter the FMM.

## Bonsai (BH, θ=0.5, quadrupole), RTX 2080 Ti
**9.34 ms/step** — loop 37.4 s / total 40.5 s for 4000 steps. (≈ same as the A100's
40.1 s: at 200k Bonsai is launch/host-bound and barely uses the bigger card.)

## FMM accuracy ↔ cost curve (complex basis)
| θ | order | force err p50 | force err p90 | ms/step | vs Bonsai |
|---|---|---|---|---|---|
| 0.5 | 1 | 2.00% | 5.82% | 544 | 58× |
| 0.5 | 2 | 0.39% | 1.20% | 551 | 59× |
| 0.5 | 3 | 0.061% | 0.27% | 559 | 60× |
| 0.5 | 4 | 0.016% | 0.071% | 568 | 61× |
| 0.8 | 1 | 7.17% | 18.8% | 338 | 36× |
| 0.8 | 2 | 2.37% | 6.06% | 343 | 37× |
| 0.8 | 3 | 0.70% | 2.51% | 350 | 37× |
| 0.8 | 4 | 0.28% | 0.97% | 361 | 39× |

**Accuracy-matched result:** at Bonsai-level accuracy (~0.3–0.4% median force error)
the most efficient FMM point is **θ=0.8, order 4 (0.28%) = 361 ms/step ≈ 39× slower
than Bonsai**; the naive θ=0.5/order-2 match is 551 ms ≈ 59×. The ratio is robust
across the plausible accuracy band because FMM cost is nearly flat there.

## Real (Dehnen) basis on the fast lane — currently a regression
Same force error as complex (numerically correct parity), but slower, because the
fast lane routes real through the *slow per-pair* M2L, not the grouped/cached kernel:
| point | complex ms | real ms | Δ |
|---|---|---|---|
| order2/θ0.5 | 551 | 576 | +4.5% |
| order4/θ0.5 | 568 | 647 | +14% |
| order2/θ0.8 | 343 | 367 | +7.1% |
| order4/θ0.8 | 361 | 426 | +18% |

## Two structural findings
1. **Order barely affects FMM cost** (544→568 ms across orders 1→4 at θ=0.5). The
   far-field/multipole order is *not* the cost driver — restricting order to match
   accuracy buys almost nothing. θ (near-field pair count) matters more. Consistent
   with the launch/near-field-bound diagnosis (jaccpot
   `docs/h100_fastlane_launchbound_2026-07-10.md`).
2. **Real basis is slower than complex** on the fast lane (see table): the fast lane
   uses the slow per-pair real M2L (`_accumulate_real_m2l_chunked_scan`), not the
   grouped/cached `m2l_rot_scale_real_batch_cached_blocks`. Folding the grouped/cached
   real kernel into the streamed fast lane is the remaining work — but it only speeds
   the far field, which the curve shows is not the bottleneck.

## Bottom line
On this hardware, at equal accuracy, ODISSEO's FMM fast lane is **~38–60× slower than
Bonsai** for 200k. The gap is dominated by the near-field / kernel-launch path, not
the multipole basis or order. Priority for speedup: collapse the launch-bound
near-field (the ~1000 tiny per-step kernels), not the far-field basis.

## Repro
```
bash benchmark_rtx2080/run_bonsai_reference.sh          # Bonsai (needs sm_75 binary)
bash benchmark_rtx2080/run_accuracy_sweep.sh            # complex curve
ORDERS="2 4" THETAS="0.5 0.8" BASIS=real bash benchmark_rtx2080/run_accuracy_sweep.sh
micromamba run -n odisseo python benchmark_rtx2080/parse_accuracy_sweep.py
```
(Bonsai sm_75 binary: `Bonsai/runtime/CMakeLists.txt` GENCODE now includes
`compute_75,code=sm_75`; A100-only binary backed up as `bonsai2_slowdust.sm80.bak`.)
