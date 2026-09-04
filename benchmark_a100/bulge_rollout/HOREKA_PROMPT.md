# Paste this into a fresh Claude Code session on HoreKa

Copy everything in the fenced block. It is self-contained: it states the goal, the exact
configuration that was measured, what has to be verified before a 20-hour job, and the
traps that cost time here.

---

```
I want to run a 21,012,480-particle galaxy simulation (self-gravitating disc + bulge,
analytic NFW halo) for a quarter of an orbital time on HoreKa GPUs, using jaccpot's
distributed FMM through ODISSEO's mesh lane. The configuration is already measured on
8xA100-40GB elsewhere; I moved here for the scheduler. Do not re-derive it -- port it.

## Repos and where the setup is written down

  ~/Odisseo   5d8dc0f on branch horeka/disc-bulge-21m-quarter-orbit  <-- committed and pushed
  ~/jaccpot   c1cede7 (main)
  ~/yggdrax   a5262ae (main)

Three repos, and only three -- the mesh lane imports jaccpot and yggdrax and nothing else.
nornax is NOT needed (it belongs to the block-step lane; verified: no mesh-lane file imports
it, and odisseo.option_classes / odisseo.render_callback pull in only jax and numpy).

  git clone git@github.com:vepe99/Odisseo.git \
    && (cd Odisseo && git checkout horeka/disc-bulge-21m-quarter-orbit)
  git clone git@github.com:TobiBu/jaccpot.git && (cd jaccpot && git checkout c1cede7)
  git clone git@github.com:TobiBu/yggdrax.git && (cd yggdrax && git checkout a5262ae)

jaccpot pins jax >=0.10.2,<0.11 (0.9.1/0.9.2 core-dump on its CPU backend); the runs above
were taken with jax 0.9.0 in the odisseo env, importing jaccpot/yggdrax from the checkouts.
Put all three on PYTHONPATH rather than installing.

READ FIRST, both in Odisseo:
  benchmark_a100/bulge_rollout/SETUP.md    exact commands, ICs, provenance, memory table
  benchmark_a100/bulge_rollout/findings.md why each choice is what it is, and 5 measured
                                          traps -- read section 4 and the round 1-4 results
A ready SLURM script is at benchmark_a100/bulge_rollout/horeka_quarter_orbit.sbatch with
every site-specific field marked CONFIRM.

## The run

  N 21,012,480 = 17,510,400 disc + 3,502,080 bulge, equal particle masses
  leaf 1024, theta 0.7, order 6, fp32 state, nearfield_accum=wide
  mac_type=dehnen_error, adaptive_eps=1e-5, cross criterion ON, caps DERIVED (pin nothing)
  dt 5e-4, 489 steps = 0.2445 code time = 36.4 Myr = a quarter orbit at the half-mass radius
  softening 0.008278 (82.8 pc)
  code units: length 1 = 10 kpc, mass 1 = 1e10 Msun, G = 1, time 1 = 149.1 Myr

Measured on 6xA100-40GB: 136.3 s/step, 18.5 h for 489 steps, force rel_l2 4.03e-03 against
an fp64 direct sum, dL/L 2.3e-09 after 6 steps, every overflow flag clear, peak 40,105 MiB
on the coordinator card and ~31,450 MiB on each worker.

## Why the MAC is not negotiable (settled 2026-09-04)

`mac_type=dehnen_error` is the ONLY configuration whose force is the same number twice. Four
calls of the same input through the same compiled program, each vs one fp64 direct sum:
criterion 3.111689e-03..3.111703e-03 (spread 4.6e-06); geometric `dehnen` 4.97e-03 on the
first call then 0.539 / 0.310 / 0.521 -- 30-54 % wrong on every later call, identical accept
mask, all overflow flags clear, nothing non-finite. **No guard sees it, and every accuracy
number ever taken for the geometric MAC was on the first call.** Do not run `dehnen` for a
rollout, and do not accept a t=0 probe as validation: use `--probe-every 50` so the force is
scored on later steps too. Details: findings.md sections 10-12.

## The one thing that decides whether it runs at all

This lane is SINGLE-PROCESS: one shard_map mesh over jax.devices(), and the host side (RCB
partition, the probe, checkpointing, the realignment map) assumes one process holding all
21M rows. There is no jax.distributed.initialize(), so it CANNOT span nodes. Device count is
capped by GPUs per node, and memory per device is set by the leaf COUNT (N/ndev/leaf):

  ndev  leaf   leaves/dev   est. coordinator   fits 40 GB   fits 80-94 GB
     4  1024        5,130         60,881 MiB           no             YES
     4  2048        2,565         15,220 MiB          YES             YES  (~2x slower/step)
     6  1024        3,420    40,105 (measured)        98 %             YES
     8  1024        2,565         27,396 MiB          YES             YES  (8 GPUs, one node)

So: find out how many GPUs per node and how much memory per GPU each HoreKa GPU partition
gives, and pick from that table. On 4x>=80GB use leaf 1024 unchanged. On 4x40GB either use
leaf 2048 (fits, but expect ~250-270 s/step so ~35 h, and its accuracy is UNMEASURED) or
fall back to --mac-type dehnen --leaf 512, which fits at ~31.4 GB and is 3.77x worse on
force error but 63.66 s/step. Tell me which partitions exist and what you recommend before
submitting anything long.

## Initial conditions

Either copy the 478 MB file, or regenerate (needs agama; deterministic given the same agama
build -- 1.0.159 was used):

  python tools/agama_generate_scm_disk_ic.py \
    --output $WORK/odisseo_ic/disk_bulge_21m_v2.npz \
    --n-particles 21012480 --quantum 30720 --seed 7 --state-dtype float32 \
    --iterations 8 --bulge-mass-code 1.2 --bulge-scale-code 0.08 --bulge-gamma 1.0 \
    --rmax-code 20.0

Verify after loading: N 21,012,480, total mass 7.2, particle mass 3.426535e-07 with
max/min = 1.000000 across BOTH components, disc prograde fraction 1.0000, bulge 0.4999,
bulge rmax <= 20. If particle masses differ between disc and bulge the IC is wrong -- that
was a real bug here (AGAMA's sample(k) spreads the DF's whole mass over k, so concatenating
draws of different sizes gives a two-population IC).

## Validate cheaply before queueing 20 hours

Submit a 4-step job first (--steps 4, no --render-every, keep --probe 256) and check:
  1. the "# caps:" line -- confirms derived vs pinned
  2. "# force_scale range [...]" is NOT constant. Constant means the criterion silently fell
     back to eps*1, which accepts more and runs FASTER. Expect a spread of ~2e4 on this IC.
  3. overflow={...} all zero. A truncated walk reads FASTER and computes a wrong force.
  4. "# PROBE step=0 ... rel_l2" ~= 4.03e-03 (6 cards/21M) or 2.88e-03 (4 cards/17.8M); only
     comparable at --probe 256 --probe-seed 20260901. Then "# PROBE step=50 ..." and later
     MUST stay at that level -- a later-step probe drifting to 1e-1 is the geometric-MAC
     defect and means the configuration is not the one specified here.
  5. "realignment verified against scatter_to_input_order"
  6. peak memory per card, against the table above
Only then submit the full 489 steps.

## Traps that cost time on the previous system

- Never compare rel_l2 across different --probe values; the same config reads 3.13e-3 at 192
  and 4.42e-3 at 256.
- The angular-momentum diagnostic must reduce in float64. A float32 device reduction over
  21M cross products has its own error of ~3e-06, which is the same size as the signal -- two
  runs whose forces agreed to 7 significant figures appeared to differ 23x in dL/L. It is
  already fixed in tools/mesh_galaxy_run.py (host float64); do not "optimise" it back.
- Do not pass --cross-queue/--cross-interactions at leaf >=1024. Those are a leaf<=512
  workaround, and pinning caps the criterion needs buys a run that truncates.
- --m2l-chunk and --nearfield-chunk do NOT reduce the buffer that OOMs at leaf 512; measured
  byte-identical OOM (46.17 GiB) with chunks 8x and 4x smaller.
- Request exclusive nodes. A shared card ruins the timing and can OOM a 20-hour job.
- The rollout script has a per-step finiteness gate, --restart-from (baseline carried in the
  snapshot) and a rolling checkpoint every 20 steps; the reference wrapper is
  benchmark_a100/bulge_rollout/results/launch_used.sh's successor `run_25m_8card.sh` pattern:
  retry ONLY a "NON-FINITE STATE" abort, never an OOM or timeout.

## What I want out of it

A completed quarter orbit with: qorbit_final.npz (self-describing snapshot with component
labels), face-on AND edge-on movies (edge-on is the only view that shows a bulge), the
float64 conservation trace, and tools/mesh_rollout_analysis.py output comparing disc and
bulge structure initial vs final. Report step time, peak memory and rel_l2 so they can be
compared with the A100 numbers above.
```
