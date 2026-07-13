# Radix fast-lane optimization investigation — RTX 2080 Ti (2026-07-13)

Goal: find a fixable performance win in the radix fast lane. Working hypothesis
(from the H100 note) was "launch-bound near-field." **That hypothesis does not
hold on this hardware** — findings below.

## Headline diagnosis: the 2080 Ti is COMPUTE-bound, not launch-bound
Steady-state SM utilization during a measured 200k step (order4/θ0.8):
**p50 = 97%, mean = 93%** (62/65 active samples in the 80–100% bucket). The GPU
is saturated with real arithmetic. This is the *opposite* of the A100/H100 fused
lane (SM ~1% busy, launch-latency-bound). The launch-bound problem is real but
**A100/H100-specific**; the slow 2080 Ti hides the launch gaps behind per-kernel
compute (exactly as the H100 note predicted for the "slow 2080").

## Where the ~358 ms/step goes (order4/θ0.8, 200k)
Via `JACCPOT_LARGE_N_EVAL_DIAG_MODE` differencing (near_zero / far_zero / zero):
- **per-step PREPARE / rebuild (tree + upward + M2L + near-field payload build): ~70%**
- near-field P2P eval: ~24% (~86 ms)
- far-field L2P eval: ~5% (~17 ms)
The prepare/rebuild runs **every step** and dominates.

## Every accessible knob is exhausted (no easy win)
| Lever | Result |
|---|---|
| **Expansion order** 1→4 | cost flat (544→568 ms θ0.5; 338→361 θ0.8) → far-field/M2L is NOT the bottleneck |
| **Near-field cap** 64→auto(32) | payload halved `[782,64,32]`→`[782,32,32]`, **~0 speed change** (padding slots were already ~free) — production already defaults to 32 (auto-grow), so no prod bug |
| **refresh_every** >1 | **hard-rejected**: "strict static-radix production requires refresh_every=1 for endpoint-correct velocity-Verlet". Rebuild every step is by design (same cadence as Bonsai) |
| **Leaf size** | 64/128 fail (fused-lane fixed-shape profile caps sized for 256); 256 = 357 ms (optimum); 512 = 485 ms (slower). Default 256 already best |

## Conclusion
On the RTX 2080 Ti the fast lane is at its compute-bound floor for these knobs;
production defaults (order 4, leaf 256, cap auto→32, θ per accuracy) are already
near-optimal. **No easy/safe speed fix exists on this hardware.** At equal accuracy
it stays ~38–60× behind Bonsai (see `summary.md`), because the per-step FMM rebuild
is genuinely expensive arithmetic while Bonsai (launch-bound at 200k) is cheap.

## Where a real win lives (structural, needs the right hardware to validate)
1. **Launch fusion for A100/H100** (the actual production targets): collapse the
   ~1000 tiny per-step kernels of the near-field / rebuild into few big kernels.
   This is the jaccpot Phase-5 near-field-Pallas direction. Pays off ONLY where the
   lane is launch-bound (A100/H100) — cannot be validated on the 2080 Ti (compute-bound).
2. **Cheaper per-step rebuild** (helps the 2080 Ti's 70% prepare): the tree/payload
   build is rebuilt every step; reducing that arithmetic is the only 2080-Ti lever,
   but it is a structural change, not a knob.
3. **Near-field pair symmetry** (Newton's 3rd law): near-field is ~24%; exploiting
   i↔j symmetry could ~halve it (~12% total) — algorithmically sound but a hard
   fixed-shape/scatter-add kernel change.

Recommendation: the impactful fix (launch fusion) belongs on A100/H100. There is no
low-risk knob-level fix to land for the 2080 Ti.

---

## UPDATE (clean decomposition + leaf sweep)

Corrected decomposition (clean `--profile-breakdown` medians; the earlier "70%
rebuild" was a compile-contaminated mis-measurement):
- **near-field P2P = 78% (278 ms)**, far-field M2L+L2L = 19% (70 ms), tree rebuild
  + upward = **2%** (6.5 ms). The bottleneck is the near-field force eval, NOT rebuild.
- Confirmed both ways: refresh-diag (upward/downward/full) and eval-diag
  (near_zero/far_zero/zero) agree.

Accuracy-preserving levers — ALL exhausted, default config already optimal:
- **Order** 1→4: cost flat → far-field not the driver.
- **Near-field cap** 64→auto(32): payload halved, ~0 speed change (padding was free).
- **Leaf size** sweep 96/128/160/192/256/512 = 517/431/410/446/**357**/485 ms/step:
  U-shaped, **minimum at the default 256**. Smaller leaves explode far-field M2L
  (far pairs 65k→208k→316k for 256→128→96) faster than near-field shrinks; larger
  grow near-field. Leaf<256 needs `FUSED_COMPACT_FAR_PAIR_CAP` raised from 131072.
- (θ retune shifts near→far and IS faster — θ1.0/o6 = 292 ms — but collapses accuracy
  0.28%→0.96%, so ruled out.)

Conclusion: production defaults (order 4, leaf 256, cap auto, θ 0.8) are already the
compute-optimal accuracy-preserving point on sm_75. The near-field pure-JAX P2P is at
its floor; the only remaining levers are structural: (1) Newton's-3rd-law pair symmetry
(~1.3-1.6x near-field realistically, but risky scatter-add rewrite of the static-shape
target-owned kernel, and it HURTS the launch-bound A100/H100), or (2) the sm_80 Pallas
register-tiled near-field kernel — which is really the A100/H100 fix, unavailable on sm_75.
