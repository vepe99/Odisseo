> **CORRECTION, 2026-08-29 (later the same day).** The conclusions in the sections below
> titled "THE RESULT THAT INVERTS THE USUAL TRADE" and "THE ACCURACY FLOOR ... (confirmed)"
> are **float32 artefacts**, and are kept here unaltered as the raw record only.
>
> A sixth grid point was measured afterwards: **fp64, theta 0.4, order 6 -> rel_l2 1.068e-05**.
> Order 3 -> 6 buys **1.04x in fp32 and 118x in fp64**, and in fp64 *tightening* theta is worth
> 53x. The float32 round-off floor was masking the truncation knob entirely. Do NOT act on
> "loosening the MAC is the right route to accuracy" -- it holds only below that floor.
>
> The round-off is in one place: the per-target accumulator in `_nearfield_leafpair_kernel`
> (`jaccpot/pallas/nearfield_fused_leaf.py:691`). Near-field chunking splits over *target
> leaves*, not sources, so widening the chunk carry recovers exactly zero.
>
> Also superseded below: "The stepping loop was aborted before any step timing" -- 20 971 520
> particles on 5 cards ran 125 steps at a median of 69.2 s/step, dL/L 2.6e-06 (`run21m.log`).

# Measured this session (2026-08-28), jaccpot main @21bcf65, yggdrax main @9812af8

## Reproduction check
ndev2 leaf256 131072/dev: self_near 357,946, rel_l2 4.113e-04, 0.190 s
  -> matches the record (357,946 / 4.1e-4 / 0.200 s) to the digit.

## The headline: 10^7 on 5 cards
ndev5 leaf512 2,097,152/dev = 10,485,760 total, order 3, theta 0.4, dehnen, x64
  every overflow flag CLEAR; peak 3.92 GiB of a 23.7 GiB budget per card
  self_near 28,110,976 leaf pairs
  cross_near [5.81, 5.87, 5.82, 8.48, 5.78] M  = 31.76 M total
  cross_far  [2.56, 2.60, 2.50, 3.88, 2.57] M
  self_far   ~1.59 M per device
  build 60.69 s, oracle 435 s
  rel_l2 5.007e-03 -> just over the harness's 5e-3 gate, TIMING WITHHELD

## Cross-domain near field overtakes the intra-domain one
  per-leaf near neighbours (leaf 512):
    ndev2, 8.39M total: self 1541 / leaf, cross  705 / leaf  (cross = 31 %)
    ndev5, 10.49M total: self 1373 / leaf, cross 1551 / leaf (cross = 53 %)
  at FIXED total N = 2.097M, leaf 256:
    ndev2: self 9.52M + cross 5.29M = 14.81M pairs, 4.08 s
    ndev4: self 7.85M + cross 9.33M = 17.18M pairs, 3.54 s   (1.15x for 2x devices)
  device 3 carried 46 % more cross_near than its peers -> RCB load imbalance.

## Cost scaling of the distributed lane (recorded, 2xA100, leaf 256, fp64, p3, th0.4)
  N total     s/force   self_near
  65,536       0.136       29,378
  262,144      0.200      357,946
  1,048,576    1.324    3,232,328
  2,097,152    4.081    9,522,178
  3,145,728    9.051   16,640,926
  4,194,304   14.534   25,347,010
  6,291,456   31.671   45,879,650
  8,388,608   44.770   69,574,540
  => time ~ N^1.7 ; near pairs ~ N^1.45  (leaf fixed)
  leaf 512: 8.39M -> 31.70 s ; 16.78M -> 93.24 s (rel_l2 4.5e-3)

## Near field is ~99 % of the single-card evaluation and is NOT gather-bound
  N=1e6 disc, leaf 256, p4, th0.6: 1.442e11 near particle-pairs = 137,500 sources
  per target = 13 % of a full direct sum. Kernel at 27.8 % of peak -> rewrite
  capped at 3.6x. "Fewer pairs" is the only lever.

## The untapped lever: order is free, theta is not
  N=1e6, disc, fp32, leaf 128 (jaccpot bench/results/near_field):
    th 0.5 p4 0.612 s rel_rms 1.354e-3 | p6 0.600 s 2.731e-4 | p8 0.627 s 1.202e-4
    th 0.6 p4 0.484 s rel_rms 3.304e-3
    th 0.7 p4 0.431 s rel_rms 7.666e-3
  The order axis was ONLY ever swept at theta=0.5. The (loose theta x high order)
  corner is unmeasured, and it is where the throughput is: near work ~ theta^-3.
  The distributed lane's defaults are order 3 / theta 0.4 -- the wrong corner.

## Not blockers (checked)
  * ICs: agama samples 1e7 disc particles in 131 s (1e6 in 20.1 s). SIGILL gone.
  * Memory: 3.92 GiB per card at 10^7 on 5 cards.
  * Env: one editable yggdrax; jaccpot.mutual.distributed imports outside pytest.
  * partitioner is a real DistributedFMMConfig field, default "rcb".

## Hardware
  8x A100-PCIE-40GB, NO NVLINK anywhere (PIX/NODE/SYS only), GPUs 0-3 on NUMA 0 and
  4-7 on NUMA 1 -> any 5-GPU set crosses the CPU interconnect.
  All 8 cards had other users' processes resident (6.5-15.5 GB each) during these runs.

## Lanes that do not exist yet
  * No distributed stepping driver: jaccpot/distributed/fmm.py has make_force_evaluator
    and distributed_fmm_accelerations only -- no leapfrog/verlet scan.
  * DistributedBlockStepFMM exists but has only ever run at N=128-256 on forced CPU
    devices. Never on a GPU.
  * ODISSEO has no multi-GPU FMM lane at all: resolve_lane -> direct | fmm_forward |
    fmm_differentiable | fmm_blockstep, all single-device. Its only shard_map is a
    direct-sum in dynamics.py.
  * render_callback streams from the single-GPU strict_run_v2 scan only.
  * Dehnen mass-dependent MAC still cannot reach the fast lane; the cheap route
    (mac_type="dehnen_theta", per-node effective theta) is measured and REFUTED
    (12-9300x worse error at 1.35-15x more work).

## MEASURED 2026-08-28: 10^7 on five A100s, with a timing
  ndev5 leaf512 2,097,152/dev = 10,485,760, theta 0.4, dehnen, x64 config, fp32 IC
    order 3: rel_l2 5.007e-03, build  60.7 s, peak 3.92 GiB  (timing withheld by gate)
    order 6: rel_l2 4.811e-03, build 106.2 s, peak 4.02 GiB, **37.70 s / force**
  Order 3 -> 6 doubles far-field work and moves the error 4 %.
  => THE ERROR AT 10^7 IS NOT TRUNCATION-LIMITED. Expansion order cannot buy it.
     Leading hypothesis: fp32 accumulation over ~7e5 near sources per target whose
     net force is a small residual of a much larger sum of |terms|.
     sqrt(7e5)*6e-8 = 5e-5 relative to sum|terms|; a ~100x cancellation factor in a
     near-uniform disc puts that at 5e-3, which is what is measured.
     Test: scratchpad/ladder_fp64.py (same point, float64 positions).

## THE RESULT THAT INVERTS THE USUAL TRADE (2026-08-28)
10,485,760 particles, 5 x A100, leaf 512, order 6, dehnen MAC, fp32 IC, x64 config:

  theta  self_near    cross_near   cross_far   self_far   rel_l2     s/force
  0.4    28,110,976   31,756,761   14,111,170  7,953,088  4.811e-03   37.70
  0.7    11,916,458    8,212,404    3,751,286  4,051,034  9.695e-04   23.25

LOOSENING theta 0.4 -> 0.7 is 1.62x FASTER and 5.0x MORE ACCURATE.
  self near pairs  2.36x fewer
  cross near pairs 3.87x fewer   (cross share 53% -> 41%)
  cross far pairs  3.76x fewer

That is backwards for an FMM unless the error is dominated by the NEAR field's
float32 accumulation rather than by multipole truncation -- which is also what the
order 3 -> 6 null result says (4 % for double the far work). Fewer near pairs =
less accumulated round-off = better answer.

Consequence: the distributed lane's shipped defaults (order 3, theta 0.4) are the
worst corner of the grid at this scale -- slower AND less accurate than theta 0.7.

Time fell only 1.62x while near work fell 2.4-3.9x => ~10 s of fixed overhead
(tree build, halo exchange, launches) at 10^7 on 5 PCIe cards. At theta 0.7 the lane
is much closer to communication/overhead-bound than to near-field-bound.

## THE ACCURACY FLOOR AT 10^7 IS FLOAT32 NEAR-FIELD ROUND-OFF (confirmed)
10,485,760 particles, 5 x A100, leaf 512, dehnen MAC. Four points, one variable each:

  precision  theta  order  rel_l2       s/force   peak GiB   self_near
  fp32       0.4    3      5.007e-03    withheld    3.92     28,110,976
  fp32       0.4    6      4.811e-03      37.70     4.02     28,110,976   order:  -4 %
  fp32       0.7    6      9.695e-04      23.25     4.02     11,916,458   theta:  -5.0x
  fp64       0.4    3      1.256e-03      90.31     4.46     28,111,504   precision: -4.0x

Two independent interventions -- shrink the near field, or widen the accumulator --
each buy ~4-5x. Raising the expansion order buys 4 %. That is the signature of a
round-off-dominated near-field sum, not of multipole truncation.

THE OPERATIONAL CONCLUSION: theta 0.7 in fp32 (9.7e-4 at 23.25 s) is slightly MORE
accurate than theta 0.4 in fp64 (1.26e-3 at 90.31 s) and 3.9x cheaper. Loosening the
MAC is the right route to accuracy at this scale; raising precision is not.
The shipped defaults (order 3, theta 0.4, fp32) are the worst corner on every axis.

Caveat: 1.256e-03 remains in fp64, so round-off is the dominant term at theta 0.4 but
not the only one. A theta sweep in fp64 would separate the remainder.

## 2026-08-29: EVERYTHING ABOVE WAS fp32. The precision grid.
VERIFIED directly: with jax_enable_x64=True, float32 in -> float32 accel out
(coeff_dtype = lp.dtype). The harness's `_disc` returns float32, so the ENTIRE
recorded distributed-ceiling table is fp32 compute under an x64 config.

RECORD DEFECT: bench/distributed_ceiling_ladder.py:57 says the ceiling table is
"order 3, theta 0.4, dehnen, fp64" and docs/distributed_per_device_ceiling.md:16
says "real basis + dehnen MAC, float64". Both inverted. Worse, the harness's
fp32-vs-fp64 MEMORY comparison (lines 88-98) is VACUOUS -- it contrasts the mesh
"in fp64" against the single-GPU lane's "order 4, theta 0.8, fp32", but both arms
were fp32; only order and theta differed. Its conclusion ("fp32 is not a memory
lever") is roughly right but nothing in the record established it.
=> my 2026-08-28 run was the FIRST fp64 execution of this lane.

10,485,760 particles, 5 x A100, leaf 512, dehnen, m2l_chunk 65536, nearfield_chunk 512:

  precision theta order  rel_l2      s/force  peak GiB  build s  self_near
  fp32      0.4   3      5.007e-03   (gated)    3.92      60.7   28,110,976
  fp32      0.4   6      4.811e-03    37.70     4.02     106.2   28,110,976
  fp32      0.7   6      9.695e-04    23.25     4.02      70.2   11,916,458
  fp64      0.4   3      1.256e-03    90.31     4.46     131.5   28,111,504
  fp64      0.7   6      5.640e-04    43.90     4.47     102.3   11,916,642

MATCHED PRECISION PAIR (theta 0.7, order 6, only dtype differs):
  error   1.72x BETTER in fp64
  time    1.89x slower
  memory  1.11x  <-- fp64 is nearly free on memory (peak is cap-sized INT buffers)
  build   1.46x slower

THE ROUND-OFF SIGNATURE, CONFIRMED: fp64's benefit SHRINKS as the near field shrinks.
  theta 0.4, 28.1M near leaf pairs -> fp64 gains 4.0x
  theta 0.7, 11.9M near leaf pairs -> fp64 gains 1.7x
Treating fp64 as ~pure truncation, the inferred fp32 round-off term falls
4.85e-03 -> 7.89e-04 (6.1x) for a 2.36x smaller near field. Fewer terms in the
near-field sum = less accumulated rounding = less for fp64 to fix. Exactly the
prediction.

OPERATIONAL: fp32 at theta 0.7 (9.70e-04, 23.25 s) is MORE accurate and 3.9x cheaper
than fp64 at theta 0.4 (1.26e-03, 90.31 s). Loosen the MAC before reaching for fp64.
If you need the best available accuracy, fp64 + theta 0.7 gives 5.64e-04 at 43.90 s.
