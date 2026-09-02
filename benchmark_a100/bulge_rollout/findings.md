# A realistic disc + bulge at 21 million particles: audit, ICs, and what broke

Session of 2026-09-01. Upstream pulled first, then audited, then the IC built, then the
rollout. Numbers below are measured on this box (8 x A100-PCIE-40GB, no NVLink, GPUs 0-3
and 4-7 on different NUMA nodes) unless marked otherwise.

## 1. What upstream had landed since the last mesh session

`jaccpot` +63 commits, `yggdrax` +4, `nornax` unchanged, `Odisseo` even with
`origin/jaccpot-integration`.

- **The Dehnen (2014) section 5 mass-dependent MAC now reaches the distributed lane, on
  BOTH walks** (jaccpot #274 + yggdrax #54). Measured upstream at N=1 048 576: **4.4-7.7x
  better p99.99 force error at 0.77-0.88x the near-field work**, rising to 8.8x (leaf 512)
  and 15.1x (leaf 1024) at four devices with total N held fixed. The self-only ablation is
  1.87x at 1.06x the work, so most of the win is in the CROSS walk -- which is the half
  that needed the yggdrax change.
- **The traversal caps were recalibrated for the criterion**: queues 2-4x SMALLER, far/near
  caps 2-4x LARGER. The second half is a correctness fix, not tuning -- two caps were
  under-provisioned and only `auto_scale_caps=True` was hiding it.
- **The float32 near-field round-off floor is gone** (`nearfield_accum="wide"`): 439x
  accuracy for 1.8 % time and 0 % memory.

### Audit findings

1. **`docs/dehnen_mass_mac_status_and_plan.md:113` is stale.** Its section 0 heading still
   reads "PLUMBING DONE, UNMEASURED, and self-only", which sections 0b and 0c directly
   below it contradict with GPU measurements of both walks.
2. **The record names its own missing measurement** and it is exactly this session's run:
   *"What is still missing before this is a paper number: N=10^7 and 5 devices ... and a
   matched WALL-CLOCK comparison rather than matched work."*
3. **ODISSEO's `MeshOptions` could not express the criterion** -- no `mac_type` or
   `adaptive_eps` field, so the new capability was unreachable from the lane. Fixed.
4. **`tools/mesh_galaxy_run.py` never passed `nearfield_accum`**, so it inherited jaccpot's
   own default of `"input"` -- meaning *every number that script has produced was taken at
   the float32 round-off floor*. That is also why the earlier record concluded that
   LOOSENING theta buys accuracy: an artefact of the floor. Fixed, and defaulted to `wide`.
5. **`--repartition-every` was declared but not implemented, in the script AND in the
   lane.** A dead field in both. Implemented in the script (see section 3).
6. **The script hardcoded the 11 diagnostic field names**, so it silently dropped
   `force_scale_min`/`force_scale_max` -- which are precisely trap 14's witness. Now
   imported from `jaccpot.distributed.fmm.DIAG_FIELDS`.

## 2. The initial conditions: a live bulge, and three ways it goes wrong

`tools/agama_generate_scm_disk_ic.py` gained a sampled, self-gravitating bulge (Dehnen
gamma=1 = Hernquist, in the total potential, QuasiSpherical DF, its own SCM component).
The NFW halo stays analytic, as before. Built:

    N = 21 012 480  =  disc 17 510 400 (M=6.0)  +  bulge 3 502 080 (M=1.2)
    Hernquist a = 0.08 (800 pc), half-mass radius 2.414a = 0.193
    particle mass 3.426535e-07, IDENTICAL across components (max/min = 1.000000)
    N divides ndev*leaf for ndev in {5,6} x leaf in {256,512,1024}

Three defects surfaced while building it. Each would have produced a plausible-looking
galaxy that was quietly not the requested one.

**The rotation acceptance gate must look at the DISC alone.** A pressure-supported bulge
sits at `prograde_frac ~ 0.5` by construction (measured 0.4999). Over the concatenated
population the gate read **0.916** against its own 0.90 floor -- it would have *passed*, by
0.016, and then failed as soon as anyone raised the bulge fraction. Worse, it can pass a
disc that has stopped rotating if the bulge fraction happens to compensate. The gate now
evaluates the disc slice, before the shuffle, and reports the bulge separately.

**The output must be SHUFFLED.** The rollout trims N to a multiple of `ndev * leaf_size` by
taking a PREFIX. On a disc-then-bulge concatenation that trim deletes bulge particles only,
silently changing the mass ratio. `--quantum` now emits an exactly-divisible N so no trim
happens at all, and the shuffle is the belt to that braces (bulge fraction in the first
10 % of rows: 0.1668 against 0.1667 expected).

**AGAMA's `sample(k)` spreads the DF's whole mass over `k`.** The rejection loop that
enforces `--rmax-code` takes a large first draw and a small top-up draw, so the two come
back with per-particle masses differing by the ratio of their sizes. Concatenating them and
rescaling the total produced a **two-population IC in the same component: 1.17e-04 against
6.09e-05**, a factor 1.9. Unequal masses make the heavier species sink by dynamical
friction -- a numerical secular effect that looks exactly like the bulge growth this run is
meant to measure. Fixed by assigning one uniform mass per component (which is what a single
draw returns anyway) plus a guard that raises if AGAMA ever stops doing so.

### The tail had to be clipped, and the reason is not aesthetic

A Hernquist profile has `M(>r) ~ 2a/r`, so the expected largest radius among N particles is
`~2aN` -- **560 000 code units** at a=0.08 and N=3.5e6. Measured on the first build:
**rmax = 3 656 740**. One particle, setting the tree's bounding box, collapsing Morton
resolution for all 21 million others. `--rmax-code 20.0` (10x the halo scale radius) clips
**0.798 %** of the bulge's mass -- against an analytic prediction of 0.795 % -- and brings
the box to +/-20 with the mass still inside r ~ 1.5.

### The IC is in equilibrium in the potential the rollout actually uses

Velocities are drawn in the SCM potential, whose halo is a live DF component; the rollout
adds the *analytic* NFW instead. That mismatch is real and pre-existing, and it is small:
the SCM halo *expands* slightly, so the runtime potential is **2-6 % deeper** in
`v_circ` inside R=0.5 and agrees to 1.000 beyond R=1. The disc contracts marginally rather
than flying apart. Quantified rather than assumed:

    r         halo M(r): NFW / SCM        v_run/v_scm
    0.05      0.0302 / 0.0228 = 0.754     1.0246
    0.20      0.4401 / 0.3407 = 0.774     1.0574
    0.50      2.3144 / 2.0928 = 0.904     1.0193
    1.00      7.2132 / 6.9822 = 0.968     1.0027
    2.00     19.3147 / 19.1401 = 0.991    1.0005

## 3. What the run script gained

- `--nearfield-accum` (default `wide`), `--mac-type`, `--adaptive-eps`,
  `--mac-cross-criterion` / `--no-mac-cross-criterion`.
- `--overflow-every` (default 25). Caps are static; pair counts are NOT -- the disc clusters
  and a live bulge concentrates faster than the disc does, so a capacity that cleared on the
  first force can overflow at step 400, **silently truncating the near list, which makes the
  run read FASTER while the force goes wrong**. Previously checked on the first force only.
- `--repartition-every`, now implemented. `cap` depends only on `(n, ndev, leaf_size)` and
  never on positions, so this is a host permutation plus a `device_put` with no recompile.
  The aligner was changed to take `rank_in` as an OPERAND (converging on
  `odisseo.mesh_coupling.make_aligner`) so a repartition does not retrace it, and the
  realignment is re-verified against `scatter_to_input_order` after every repartition.
  It matters more here than in any previous run: over a quarter orbit an inner particle
  completes ~5x more azimuth than an outer one, so a frozen RCB decomposition shears.
- `--probe` -- an fp64 direct-sum accuracy measurement against ALL sources, at t=0, on the
  self-gravity term only (adding the analytic halo to both sides would inflate the
  denominator and report an error several times smaller than the solver's own). Host numpy
  on purpose: doing it on device needs `jax_enable_x64`, a process-wide switch that would
  change the lane being measured. `rel_l2` from a subsampled reference is only comparable at
  the SAME `--probe`.
- A trap-14 guard: under `dehnen_error`, a force scale whose min equals its max is the
  `jnp.ones(...)` fallback, which runs a different and looser criterion, faster, and reports
  it nowhere. The run now refuses to start.

## 4. An intermittent 5-order-of-magnitude force defect, and its trigger

`tests/test_mesh_lane.py::test_a_short_mesh_rollout_conserves_angular_momentum`
(N=1024, ndev=2, leaf 64, theta 0.7, order 4, 6 steps) fails on GPU with
`dL/L ~ 3e-03` against its own 1e-4 bar -- and passes at `dL/L ~ 1e-08` on the same
hardware. **It is not my change**: it fails identically with every local edit stashed.

The trigger is `XLA_PYTHON_CLIENT_PREALLOCATE`:

| arm | dL/L | verdict |
|---|---|---|
| GPU, preallocate default (=true) | 3.118e-03, 3.118e-03, 2.794e-03, 3.284e-03, 3.118e-03 | **5/5 FAIL** |
| GPU, `preallocate=false`, 8 cards visible | ~1e-08 | pass |
| GPU, `preallocate=false`, 2 cards visible | ~1e-08 | pass |
| forced CPU, 2 devices (the documented way) | ~1e-08 | pass |

Three distinct wrong values across five runs, all in 2.8-3.3e-03, so it is not
round-off scatter and not a single deterministic bug either. Notably **no overflow flag
fires** in the failing runs, and the lane checks them every 2 steps.

Two things this is NOT, both measured rather than assumed:

- **Not the pallas near-field backend.** jaccpot's own config comment says pallas-vs-baseline
  equivalence "is NOT asserted by the test suite" because CI runs on CPU, so this was the
  first suspect. On two A100s, all four (backend x accumulator) arms land at <= 2e-08, and
  `wide` *improves* pallas by 5.7x:

      baseline/input 7.6618e-09   baseline/wide 7.6618e-09
      pallas/input   2.0130e-08   pallas/wide   3.5257e-09   <- best of the four

- **Not the caps, theta, or order.** At this size the far field is inert (112 self and 128
  cross near leaf pairs -- essentially a direct sum), and the force matches an exact fp64
  direct sum to `rel_l2 = 2.9e-07`, *identically* across theta 0.4/0.7, order 4/6 and caps
  x8. Whatever moves dL/L by five orders does not move the force at this size.

**And the force is FINE while it happens.** In the same failing process (dL/L =
2.794e-03), a direct comparison against an exact fp64 direct sum over all pairs gives
`rel_l2 = 7.06e-07`, with every overflow flag zero and the far field inert
(`self_far_pairs = [0, 0]`, `cross_far_pairs = [0, 0]`; near counts
`self [56, 56]`, `cross [64, 64]`, identical to the passing arm). The realignment check
against `scatter_to_input_order` also passes. So whatever this is, it is **downstream of
the FMM** -- in the rollout's state handling, not the solver. An error of 2.8e-03 in the
summed angular momentum with a correct force is about the size a few permuted or stale
rows out of 1024 would produce, not a uniformly degraded force.

That also bounds the blast radius: our production rollout runs
`tools/mesh_galaxy_run.py`, not `integrate_mesh_jaccpot`, and reports dL/L on a cadence.

**Mitigation adopted for the production run**: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`CUDA_VISIBLE_DEVICES` pinned to the six held cards (so the two cards other users are on
are never touched), and `--probe` measuring the force error directly at the production
configuration before any long run commits.

**Not root-caused.** A memory-allocator flag has no business changing a force by 3e-03, and
the mechanism is still unknown. It is filed here rather than guessed at.

## 5. The MAC pretest at 21 012 480 particles on 6 x A100

All arms: leaf 512, order 6, `nearfield_accum=wide`, fp32, dt 2.5e-4, softening 0.008278
(82.8 pc, the script's derived value), `--probe 256 --probe-seed 20260901`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `CUDA_VISIBLE_DEVICES=0,2,4,5,6,7`. `rel_l2` is
against an fp64 direct sum over ALL 21 M sources for 256 targets, on the self-gravity term
only, and is comparable ONLY at this same probe.

### Arm 1 -- geometric `dehnen`, theta 0.7 (the config the 21 M disc-only run used)

    median 63.74 s/step   (against 69.47 s at 21 M on FIVE cards, so ~1.09x for 1.2x cards)
    dL/L 7.71e-07, dP 6.48e-06, COM drift 2.71e-07 after 2 steps
    probe: rel_l2 1.5200e-02   median 1.2089e-03   p99 2.5390e-02   max 3.8257e-02
    peak ~31 GB of 40 GB per card
    978 steps (a quarter orbit) at this rate = 17.3 h

**The error is concentrated where the mass is.** Median relative error 1.2e-03 against an
`rel_l2` of 1.5e-02 means the L2 figure is carried by the few targets with the LARGEST
`|a|` -- the bulge cusp. That is the geometric MAC's characteristic failure: one opening
angle for a mass distribution spanning orders of magnitude in density. It is also exactly
what a mass-dependent criterion is built to fix, since eq (16a) equalises *absolute* force
error rather than subtended angle.

For scale, the record's theta-0.7 figure at 10 485 760 / 5 cards is 9.7e-04 -- 15x better
than measured here. The differences are the bulge (a Hernquist cusp where that run had a
disc with a central hole) and a softening 2.4x smaller in code units.

The probe's own softening convention is not in question: the identical Plummer form
`r^2 + eps^2` reproduces the lane to `rel_l2 = 2.9e-07` at N=1024, where the far field is
inert and the walk is effectively a direct sum.

Full arm-1 record, every overflow flag zero and `force_scale_min/max` correctly 0.0/0.0
(the criterion is off, and a nonzero pair here would mean it had silently engaged):

    first force incl. compile   249.7 s
    realignment verified         4.2 s
    probe                      134.7 s
    median step                 63.66 s   (330 095 particles/s)
    self_near   12 660 744      self_far   6 100 074
    cross_near  50 074 764      cross_far 15 588 198

**Cross-domain near-field work is 79.8 % of the near field here** (ratio 3.96 : 1). The
record has 31 % at two devices and 53 % at five, so this is the same qualitative story
further along -- but NOT a clean extension of that trend, because the record's device sweep
held total N fixed near 10^6 and this point is at 21 M. The device count and N are
confounded. What it does establish is that in THIS configuration the cross walk is where
four fifths of the near-field work is, which makes yggdrax #54's cross-walk pair policy the
dominant lever rather than a refinement.

**Every dL/L in round 1 is the DIAGNOSTIC's round-off, not the physics.** `invariants`
summed 21 million float32 cross products on device. A tree reduction over N=2.1e7 carries
`log2(N) * eps = 2.9e-06` relative error, and dL/L is the norm of a DIFFERENCE of two such
sums -- so the diagnostic's own floor sits at ~3e-06, which is the range every measurement
was in (1.1e-07 to 5.4e-06).

The tell was decisive: `geo` and `geo_prealloc` have forces agreeing to **seven significant
figures** (rel_l2 1.5199893e-02 against 1.5199920e-02) and yet reported dL/L of 2.6e-06 and
1.1e-07 at the same step -- a 23x "difference" that is entirely reduction order. Two runs
with the same force cannot have different angular-momentum drift.

So the earlier readings here -- a coherent linear drift extrapolating to 9.2e-04 at 978
steps, and before that a two-point power-law fit giving index 1.77 and 4.5e-02 -- were both
measuring noise, and neither survives. (The two-point fit was independently wrong anyway:
two points fit a power law exactly by construction.)

`invariants` now reduces in **float64 on the host**. Not `jnp.float64` on device: x64 is a
process-wide switch, and enabling it to fix a diagnostic would also let python floats
promote the lane's float32 arrays -- at ~31 GB of 40 GB per card, not something to do by
accident. It costs one 252 MB `device_get` per call, against a 60 s step.

This does NOT touch the other round-1 results: `rel_l2` is computed in float64 on the host
against an fp64 direct sum, and the pair counts, overflow flags and step times are exact or
integer. Nor does it touch section 4's finding, where L was computed in numpy float64 by the
test itself and the effect (2.8e-03) is four orders above that configuration's floor.

**The error is not a softening artefact.** Softening 0.008278 is 5.0-21.5x the mean
interparticle spacing everywhere in the bulge (r 0.02 to 0.2), so the near field is
pair-exact and generously smoothed. The 1.5e-02 is FMM approximation error.

### Arm 4 -- the preallocate anomaly does NOT reach the production configuration

Same arm as 1, only `XLA_PYTHON_CLIENT_PREALLOCATE` flipped to its default:

    geo           (preallocate=false)  rel_l2 1.5199893e-02   median 63.66 s/step
    geo_prealloc  (preallocate=true)   rel_l2 1.5199920e-02   median 63.13 s/step

**Agreement to seven significant figures.** Section 4's defect is specific to the tiny
configuration it was found in (N=1024, leaf 64, 8 leaves per device, far field inert); it
does not appear at 21 M / 6 devices / leaf 512. The production run still pins
`preallocate=false` and its own devices, but the risk it was guarding against is now bounded
rather than merely mitigated.

### Arms 2 and 3 -- the criterion does not FIT at leaf 512, and that is a real result

Both `dehnen_error` arms died identically, on all six cards, before the first force
completed:

    RESOURCE_EXHAUSTED: Out of memory while trying to allocate 63.78GiB
    hlo_rematerialization: can't reduce below 28.07GiB; only to 62.63GiB from 69.62GiB

The criterion's derived caps at 21 012 480 / 6 devices / leaf 512 are **2x the geometric
ones** on exactly the two axes that matter:

    cap                              geometric      dehnen_error
    cross_max_pair_queue             8 388 608      16 777 216
    cross_max_interactions_per_node     32 768          65 536

and 68 482 828 288 / 16 777 216 = **4082 bytes per wavefront entry**, consistent with order-6
expansion data (49 coefficients x complex128 = 784 B, several arrays deep). The geometric arm
already peaks at ~31 GB of 40 GB, so doubling that queue cannot fit and no amount of
rematerialisation rescues it.

This is not a failure of the criterion, it is the cap calibration meeting a scale it was not
calibrated at: jaccpot's own record says the coefficients were solved on 2 and 4 devices at
N=1 048 576, and notes that the linear `remote` factor on the cross caps "is the one a
five-device run leans on hardest". At six devices and twenty times the particles, it is the
one that breaks.

`tools/mesh_galaxy_run.py` gained `--pair-queue`, `--cross-queue`, `--cross-neighbors` and
`--cross-interactions` (named as in `bench/distributed_ceiling_ladder.py`) so the caps can be
pinned rather than derived, plus a `# caps:` line so a run's own log records what it used.
Round 2 pins the criterion's two oversized caps to the geometric values -- a footprint known
to fit -- and lets the OVERFLOW FLAGS answer whether the criterion actually needs more. That
distinction matters: an under-provisioned cap truncates the walk, which reads faster and
computes a wrong force.

### Round 2 -- pinning the cross caps is not enough, and the numbers say why

Criterion at leaf 512 with `cross_max_pair_queue` and `cross_max_interactions_per_node`
pinned to the geometric values. The ask drops from 63.78 GiB to **46.17 GiB** and still does
not fit. Pinning two caps recovered 17.6 GiB, so those two were not the whole story.

The full cap table at 21 012 480 / 6 devices makes it obvious -- the criterion is larger on
*four* axes, not two, and `max_interactions_per_node` is **4x**:

    config                        leaves     self_q  self_nbr  self_int     cross_q cross_nbr cross_int
    leaf 512 dehnen                6,840  2,097,152     8,192     4,096   8,388,608    65,536    32,768
    leaf 512 dehnen_error e1e-05   6,840  2,097,152    16,384    16,384  16,777,216    65,536    65,536
    leaf 1024 dehnen               3,420    524,288     4,096     2,048   2,097,152    32,768    16,384
    leaf 1024 dehnen_error e1e-05  3,420  1,048,576     8,192     8,192   4,194,304    32,768    32,768

A far-list memory proxy -- `interactions_per_node x num_leaves x 392 B` (order 6 is 49
coefficients at complex64) -- ranks them. It overestimates absolute memory because those
lists are chunked, but the ratios are the point:

    leaf 512  dehnen        self  10.2 GiB   cross  81.8 GiB   sum   92.1 GiB   <- FITS, 31 GB peak
    leaf 512  dehnen_error  self  40.9 GiB   cross 163.7 GiB   sum  204.6 GiB   <- OOM
    leaf 1024 dehnen        self   2.6 GiB   cross  20.5 GiB   sum   23.0 GiB
    leaf 1024 dehnen_error  self  10.2 GiB   cross  40.9 GiB   sum   51.1 GiB   <- about half of the config that fits

**Leaf 1024 shrinks every cap ~4x**, because num_leaves halves *and* each cap halves. That
is the principled route rather than pinning caps until something fits, and it is independently
where the criterion is strongest: jaccpot's record has its advantage growing 2.7x -> 7.0x ->
21.8x across leaf 256 -> 512 -> 1024 on one GPU, and 8.80x -> 15.06x at four devices.

Round 3 therefore runs BOTH arms at leaf 1024 with caps left DERIVED (nothing pinned, so the
overflow flags mean what they say), which separates the criterion's contribution from the leaf
size's. `geo@leaf512` (63.66 s/step, rel_l2 1.5200e-02) is the third leg of the comparison.

`--self-neighbors` / `--self-interactions` were deliberately NOT added. Pinning the
criterion's self caps down to geometric values would have been pinning caps the calibration
says the criterion genuinely needs -- jaccpot's own record records
`max_interactions_per_node` as having been UNDER-provisioned for the criterion, and that the
far caps peak at LOOSE eps rather than tight. Forcing them smaller buys a run that fits and
truncates.

### Round 3 -- the criterion RUNS at leaf 1024, and it is worth 3.77x

`mac_type="dehnen_error"`, `adaptive_eps=1e-5`, leaf 1024, caps left DERIVED (nothing
pinned), 21 012 480 particles on 6 x A100. **Every overflow flag zero.**

    force_scale range [0.008366, 179.2]  -- a 21 420x spread
    first force incl. compile   1003.8 s   (4x the geometric arm's 249.7 s)
    median step                  137.86 s
    peak                         40 105 MiB of 40 960 on GPU 0, stable, 31.4 GB elsewhere

The force scale is the first thing to read, not the last: trap 14's failure mode is
`build_adaptive_policy_state` substituting `jnp.ones(...)` for a missing force scale, which
runs eq (16a) against `eps * 1` -- a different, looser criterion that accepts more and runs
FASTER. Its signature is a CONSTANT scale. 21 420x is emphatically not constant, and the
spread is far above the 15.6x the record reports from its CPU plumbing test, because this
galaxy has a Hernquist cusp and that one did not.

| | geo @ leaf 512 | crit @ leaf 1024 | ratio |
|---|---|---|---|
| rel_l2 | 1.5200e-02 | **4.0344e-03** | **3.77x better** |
| median | 1.2089e-03 | 6.717e-04 | 1.80x |
| p99 | 2.5390e-02 | 7.481e-03 | 3.39x |
| max | 3.8257e-02 | 1.589e-02 | 2.41x |
| self_near | 12 660 744 | 12 548 308 | 1.01x |
| cross_near | 50 074 764 | **19 162 479** | **2.61x fewer** |
| dL/L after 2 steps | (fp32 noise) | **1.013e-09** | -- |
| median s/step | 63.66 | 137.86 | **2.17x SLOWER** |

**The tail moves more than the median, which is the criterion's signature.** p99 improves
3.39x against the median's 1.80x: eq (16a) equalises ABSOLUTE force error, so it refines
hardest exactly where the geometric MAC is worst -- the high-|a| particles in the cusp that
were carrying `rel_l2`.

**But leaf pairs are not the cost; particle pairs are.** Near work goes as
`near_leaf_pairs x leaf^2`, and leaf 1024 doubles the second factor:

    geo  @ leaf 512   62 735 508 leaf pairs  x 512^2  = 1.645e13 particle pairs
    crit @ leaf 1024  31 710 787 leaf pairs  x 1024^2 = 3.325e13 particle pairs   (2.02x MORE)

Predicted 2.02x slower from that alone; measured 2.17x. So the criterion halves the leaf-pair
work and the leaf size more than gives it back. At 137.86 s/step a quarter orbit (978 steps)
is **37.4 h**, against 17.3 h for the geometric arm -- over the budget.

`--m2l-chunk`/`--nearfield-chunk` are the untried lever, and the distinction matters:
chunking makes a walk FIT, shrinking a cap makes it TRUNCATE. Round 4 asks whether the
criterion fits at leaf 512 -- where the cheap particle-pair work is -- with the far and near
loops chunked instead.

### Round 4 -- chunking does NOT reach the buffer that OOMs. A clean negative.

Criterion at leaf 512, cross caps pinned as in round 2, plus `--m2l-chunk 8192`
(8x smaller than the 65536 default) and `--nearfield-chunk 128` (4x smaller than 512):

    RESOURCE_EXHAUSTED: Out of memory while trying to allocate 46.17GiB

**The same 46.17 GiB, to the byte, as round 2 without any chunking.** An identical byte
count under an 8x smaller far chunk and a 4x smaller near chunk is proof rather than
inference: the offending allocation is not inside either chunked loop. `--m2l-chunk` and
`--nearfield-chunk` cannot reach it, so the only remaining levers on it are the caps
themselves -- which the calibration says the criterion genuinely needs -- or the leaf size.

**Conclusion: at 21 012 480 particles on 6 x A100-40GB, `mac_type="dehnen_error"` runs at
leaf 1024 and nowhere else.** Not because the criterion is expensive in flops, but because
its cap derivation, solved on 2 and 4 devices at N=1 048 576, produces buffers that do not
fit six devices at twenty times the particle count. That is a concrete, reproducible
scaling limit worth reporting upstream, and it is separable from the criterion's physics --
which, where it does fit, delivers 3.77x.

## 6. The production run

Decision, with all four candidates measured rather than argued:

    option                     steps   hours   rel_l2      coverage in a 23 h budget
    geo  @512  dt 2.5e-4         978    17.3   1.52e-02    100 % (quarter orbit)
    crit @1024 dt 5e-4           489    18.5   4.03e-03    100 % (quarter orbit)   <- CHOSEN
    crit @1024 dt 2.5e-4         978    37.0   4.03e-03     62 % (608 steps)
    geo  @512  dt 5e-4           489     8.6   1.52e-02    100 % (quarter orbit)

`crit @1024, dt 5e-4` buys **3.77x better force accuracy for the full quarter orbit at the
same wall clock**, and the cost is a timestep 2x coarser than originally chosen:
`dt/t_dyn = 0.068` at the softening length in the bulge core, against 0.034. That is inside
normal practice (`dt < 0.05-0.1 t_dyn`) and is the timestep the earlier 21 M disc-only
rollout ran at. The trade is a systematic time-integration error against a 3.77x smaller
random force error, which is why it was put to the user rather than assumed.

Launched configuration:

    N 21 012 480 = 17 510 400 disc + 3 502 080 bulge, 6 x A100, RCB
    leaf 1024, theta 0.7, order 6, fp32 state, nearfield_accum=wide
    mac_type=dehnen_error, adaptive_eps=1e-5, cross criterion ON, caps DERIVED
    dt 5e-4, 489 steps = 0.2445 code = 36.4 Myr = a quarter orbit at the half-mass radius
    softening 0.008278 (82.8 pc)
    probe 256 (fp64 direct sum), repartition every 100, checkpoint every 100,
    overflow flags re-read every 10, diagnostics every 10 (float64, on the host)
    movies: xy AND xz at 800^2 -- edge-on is the only view that shows a bulge
    max-hours 23, stops cleanly on budget with a full snapshot

### The launch was aborted, and why -- not a failure of the configuration

The run was launched at 14:41 on cards 0,2,4,5,6,7 and killed four minutes later, during
compilation. A collaborator's `athenaPK` job (user `lstorcks`, one job across three GPUs)
had taken **GPUs 4, 5 and 6** -- three of the six reserved cards -- using 10.9 / 2.3 / 11.5
GB at 99-100 % utilisation. `crit@1024` needs 31 450 MiB on every worker card, and cards 4
and 6 had 30 029 and 29 489 MiB free. It would have OOM'd at the first force, ~15 minutes in.

**The reservation did not work, and it is worth recording why.** The holder took a
deliberately small allocation (~434 MiB per card) so that killing it would free everything
instantly and leave the full 40 GB for the real run. That makes it invisible as a claim: a
colleague reading `nvidia-smi` sees a card at 1 % with 434 MB used and correctly concludes it
is free. A reservation that is polite enough not to waste memory is also too quiet to
reserve anything. On a shared box without a scheduler there is no mechanism that fixes this,
which is the actual reason to move to a queued system.

After the abort exactly one 6-card set still fitted in memory -- coordinator 0 plus workers
1, 2, 3, 5, 7 -- but it necessarily included card 5 (athenaPK, 99 %) and cards 1 and 3
(another user, ~35 %). A `shard_map` mesh is synchronous, so it runs at the pace of its
slowest card; time-slicing one card against a compute-saturated neighbour would have put the
489 steps well past the 23 h budget, while also slowing the collaborator's job. The
coordinator would also have had 352 MiB of headroom (40 457 free against 40 105 needed),
so any growth on athenaPK's side would have killed a run 20 h deep.

Decision (the user's): stop competing for cards, write the setup out, and move to HoreKa.
The GPU reservation was released at that point.

**Everything needed to run it elsewhere is measured, and none of it is invalidated by the
abort**: the IC, the configuration, the accuracy, the step time, the memory footprint and the
cap behaviour. See `SETUP.md` in this directory.

## 7. The 4-card run: a cap cliff, and 17,825,792 particles

Three of the six cards were taken, four came free later, so the question became what fits on
four. The answer is set by a **discrete cap threshold**, not a gradual memory limit.

`cross_max_interactions_per_node` doubles once leaves-per-device passes ~4,608. Scanning N at
ndev 4, leaf 1024, criterion, eps 1e-5 (estimate = cap proxy scaled to the measured ndev 6 /
leaf 1024 point of 40,105 MiB):

    N            leaves/dev   self_int   cross_int   est. MiB   % of card
    15,728,640        3,840      8,192      16,384     27,490      67.1 %
    16,777,216        4,096      8,192      16,384     29,226      71.4 %
    17,825,792        4,352      8,192      16,384     30,963      75.6 %   <- ceiling
    18,874,368        4,608      8,192    **32,768**   53,534     130.7 %   <- OOM
    20,971,520        5,120      8,192      32,768     60,768     148.4 %

So **17,825,792 is the largest N that runs on four 40 GB cards at leaf 1024**. One rung
further and a single cap doubles, taking the footprint from three quarters of a card to a
third more than one. This is the same power-of-two rounding jaccpot's own cap record warns
compounds a coefficient; here it produces a cliff between adjacent particle counts.

Counter-intuitively **four cards are LIGHTER per card than six at these sizes**, because the
cross caps carry a `remote` factor in (ndev - 1): at ndev 4 that factor is 3 against 5, which
halves `cross_max_interactions_per_node` and outweighs the extra leaves per device.

### Measured, 17,825,792 on 4 x A100-40GB (probe, 4 steps)

    peak            27,692 MiB coordinator / 17,540 MiB worker  (68 % / 43 % of a card)
                    -- the proxy predicted 30,963, so it over-estimates by 12 %
    first force     1025.9 s incl. compile
    median step     154.43 s   (115,429 particles/s)
    force_scale     [0.009708, 169.7] -- 17,478x spread, so not the eps*1 fallback
    overflow flags  all zero
    probe           rel_l2 2.8765e-03  median 6.660e-04  p99 5.044e-03  max 7.792e-03
    self_near 11,779,046   cross_near 14,018,631  (cross share 54.3 %)

**Fewer cards is MORE accurate here**, which follows from the error being cross-domain
limited. Against the 6-card / 21.0 M run of the same criterion and leaf:

    metric        6 cards / 21.0M   4 cards / 17.8M   ratio
    rel_l2            4.0344e-03        2.8765e-03    1.40x better
    p99               7.481e-03         5.044e-03     1.48x better
    max               1.589e-02         7.792e-03     2.04x better
    cross share           60.4 %            54.3 %

(An earlier note in this file compared 54.3 % against 79.8 %; that 79.8 % was the GEOMETRIC
arm at leaf 512, not the criterion at leaf 1024. The criterion's own figures are 60.4 % and
54.3 %.)

### The angular-momentum gap between the two runs, and why it is not a defect

    step   6c/21.0M dt 2.5e-4     4c/17.8M dt 5e-4
       2   dL/L 1.0133e-09        dL/L 7.3644e-07
       4   dL/L 1.4626e-09        dL/L 1.0270e-06

727x at step 2, which dt alone (2x) does not explain. But COM drift differs by 2.24x and dP
by 11.6x, so dL is out of line with BOTH. Un-normalising gives the resolution:

    |dL| / |dP|     6 cards: 1.85e-03      4 cards: 0.116

An effective lever arm of 1.85e-03 is far below any real particle radius, i.e. the 6-domain
geometry happened to cancel cross-domain torques almost exactly; 0.116 is of order the actual
radii and is what dP predicts. **The 6-card number was fortuitously good, rather than the
4-card number being degraded** -- consistent with dP differing 5.8x per unit time while dL/dP
differs 63x. Neither is a defect, and the distinction only matters for not chasing one.

The 4-card drift also DECELERATES: 3.68e-07 per step over steps 0-2, then 1.45e-07 per step
over 2-4, so the opening steps carry a transient. At the post-transient rate the projection
at step 489 is **~7e-05**, i.e. 0.009 % of the true total |L| (|L|/lscale = 0.804). That
projection, not the ratio against the other run, is what gated the launch.

### Launched

    17,825,792 particles = disc 14,854,827 + bulge 2,970,965, equal particle masses
    4 x A100-40GB (cards 0, 2, 5, 7, all idle and uncontended)
    leaf 1024, theta 0.7, order 6, fp32, nearfield_accum=wide
    dehnen_error, adaptive_eps 1e-5, cross criterion ON, caps DERIVED
    dt 5e-4, 489 steps = 0.2442 code = 36.4 Myr = a quarter orbit
    softening 0.008988 (89.9 pc)
    started 15:51, ~21.6 h expected including compile, repartitions and checkpoints

The IC (`/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz`, 406 MB) was REGENERATED at
the new N rather than trimmed from the 21 M file: the rollout trims a prefix without
renormalising mass, so trimming would have dropped total baryon mass from 7.2 to 6.1 against
an unchanged analytic halo of 100 -- a different galaxy, not a coarser sampling of the same
one. Structure confirms it is the same galaxy: disc r50 0.5654 against 0.5655, bulge r50
0.2446 against 0.2447, and the same quarter orbit of 489 steps.

**RUN IN PROGRESS at the time of writing.** Outcome, final conservation, movies and the
per-component structural comparison are not yet in this file.

## 8. The run failed at step 10, and two guards failed with it

The 17.8 M / 4-card rollout launched 2026-09-01 15:51 reported `dL/L=nan`, `KE=nan` at
**step 10 of 489** and then produced no further output for **seventeen hours** while holding
four A100s at 100 % utilisation. Killed 2026-09-02 09:53.

`com_drift` was finite (3.1996e-06) while `dP`, `dL` and `KE` were not, which localises it:
COM is built from POSITIONS, the other three from VELOCITIES. So the velocities went
non-finite first, which is the signature of a force that returned NaN AFTER the drift --
`kick(vh, an, dt)` poisons `v` while `x` is still finite.

**Why it then hung rather than crashed.** Once the state is non-finite the tree's bounding box
is too, Morton keys degenerate, every leaf becomes every other leaf's neighbour, and the
traversal grinds without terminating. 100 % GPU utilisation with zero progress is what that
looks like from outside, and it is indistinguishable from healthy work unless something is
checking the numbers.

### Guard 1: there was no NaN check at all

A non-finite state is the cheapest failure to detect and the most expensive to miss, and the
step loop checked for capacity overflow and for a constant force scale but never for NaN. It
printed `nan` and carried on.

Fixed: `jnp.isfinite` over accel, positions and velocities every step -- three device
reductions and one host sync against a ~155 s step, and the loop already blocks on `X` each
step so nothing is pipelined away. On failure it writes the non-finite counts per array, the
force-scale range, the overflow flags and a snapshot, then exits.

### Guard 2: the periodic overflow check was VACUOUS

`kdk` discarded the diagnostic vector:

    an, _, _, _, _ = force(xn)          # the diag went in the bin

so `diag` inside the step loop stayed bound to the **first force** forever. `--overflow-every`
therefore re-validated step 0 on every cadence and could never have seen a capacity that
overflowed later -- which is the only reason the check exists, since caps are static and pair
counts are not. It was reported here (section 3) as mid-run protection. It was not.

Fixed: `kdk` returns the diag and the loop rebinds it every step. **Verified rather than
asserted**: the force-scale range now evolves per step (max 25.78 -> 25.82 -> 25.87 -> 25.88
-> 25.90 -> 26.26 on the CPU smoke) where before it was frozen at the first force's value.
That is the test for staleness -- a constant diagnostic across steps means it is not live.

These two compound: a cap that overflows mid-run under the criterion truncates the prepass,
which the record says drives `f_b` DOWN, and eq (16a) divides by `eps * min_b f_b`. A floor
reaching zero gives a non-finite accept mask. The vacuous check could not see the overflow and
the missing NaN check could not see the result.

### dL/L at the 1e-6 level is NOT reproducible in this lane

The diagnostic re-run gives `dL/L = 1.867e-06` at step 2 where the earlier probe of the
**identical** config and IC gave `7.364e-07` -- a factor 2.5 apart. Together with the 727x gap
between the 4- and 6-card runs, and the lever-arm analysis in section 7 showing near-perfect
torque cancellation in one geometry, the conclusion is that dL/L here is a small residual of
cancelling torques and is dominated by fp32 force non-determinism -- the same effect that
moves `rel_l2` by 1 part in 29 000 between identical runs.

**So do not compare dL/L between runs at this level, and do not read a ratio as a finding.**
Only its finiteness and order of magnitude are meaningful. Section 7's discussion of the 727x
gap should be read with that in mind: the lever-arm explanation stands as arithmetic, but the
underlying numbers are not stable enough to support a conclusion drawn from their ratio.
