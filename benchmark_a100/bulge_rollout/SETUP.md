# Exact setup: a 21-million-particle disc + bulge, quarter orbit

Everything needed to reproduce or relaunch the run, on this box or elsewhere. Measurements
behind these choices are in `findings.md`; this file is the operational half.

## 0. Provenance

    repo      commit     branch
    jaccpot   c1cede7    main
    yggdrax   a5262ae    main
    nornax    8fe9dbd    main       (NOT needed by the mesh lane; block-step lane only)
    Odisseo   5d8dc0f    horeka/disc-bulge-21m-quarter-orbit

    odisseo env  python 3.13.12  jax 0.9.0   numpy 2.4.4  agama 1.0.159
    jaccpot venv python 3.12.3   jax 0.10.2

**Environment requirement (2026-09-04, findings §14): jax ≥ 0.10.2 in a STANDALONE venv.** The
`odisseo` micromamba env (jax 0.9.0) is what took every number in §§1–13 of findings.md, and on
it the distributed forward silently drops the cross-domain near field on most force calls after
the first (XLA `ragged_all_to_all` stale-address defect; fixed in the 0.9.1 XLA:GPU plugin). The
production venv is `/export/scratch/tbuck/venv_prod_jax0102`: `python -m venv` (NO
`--system-site-packages` — that resolves `jax_plugins.xla_cuda12` to the old env's plugin `.so`
while `pip list` says 0.10.2), `pip install "jax[cuda12]==0.10.2" numpy scipy jaxtyping beartype
matplotlib imageio pillow diffrax equinox astropy autocvd`, then `pip install --no-deps -e` for
yggdrax-main-wt, jaccpot, nornax, Odisseo. Verify with
`python -c "import jax_plugins.xla_cuda12 as p; print(p.__file__)"` — it must be inside the venv.
Any run on jax < 0.9.1 must pass `--halo-exchange buf` (accuracy-identical, no measurable cost at
4 cards) and is otherwise computing garbage.

Hardware these numbers were taken on: 8 x A100-PCIE-40GB, **no NVLink** (PIX/NODE/SYS only),
GPUs 0-3 on NUMA 0 and 4-7 on NUMA 1, so any set of five or more crosses the CPU
interconnect. Cards shared with other users throughout.

## 1. The initial conditions

    micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py \
      --output /export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz \
      --n-particles 21012480 --quantum 30720 --seed 7 --state-dtype float32 \
      --iterations 8 \
      --bulge-mass-code 1.2 --bulge-scale-code 0.08 --bulge-gamma 1.0 \
      --rmax-code 20.0

Runtime ~9 min (CPU only, heavily threaded). Output 478 MB. Verified contents:

    N 21 012 480 = disc 17 510 400 (M 6.0) + bulge 3 502 080 (M 1.2), total 7.2
    particle mass 3.426535e-07, IDENTICAL across components (max/min = 1.000000)
    disc prograde fraction 1.0000, median v_phi 2.5335
    bulge prograde fraction 0.4999 (pressure supported, by construction)
    r_half: disc 0.5655, bulge 0.2447    rmax: disc 6.671, bulge 19.9998
    scm_converged True, shuffled True, rmax_code 20.0

**Why each non-obvious flag is there** (all three were found by measuring, see `findings.md`
section 2):

- `--quantum 30720` emits an N divisible by `ndev * leaf_size` for ndev in {5,6} and leaf in
  {256,512,1024}, so the rollout never has to trim and no device is padded. Without it the
  rollout trims a PREFIX, which on a disc-then-bulge concatenation deletes bulge particles
  only and silently changes the mass ratio.
- `--rmax-code 20.0` clips the Hernquist tail. A Hernquist profile has `M(>r) ~ 2a/r`, so the
  expected largest radius among 3.5e6 particles is `~2aN = 5.6e5` code units -- measured
  3 656 740 on the first build. One particle at that radius sets the tree's bounding box and
  collapses Morton resolution for the other 21 million. The clip discards 0.798 % of the
  bulge's mass against an analytic prediction of 0.795 %.
- Component counts are split BY MASS so particle masses are equal across components. Unequal
  masses would make the heavier species sink by dynamical friction -- a numerical effect that
  mimics the bulge growth the run is meant to measure.

**Code units**: length 1 = 10 kpc, mass 1 = 1e10 Msun, G = 1, so time 1 = 149.1 Myr.
The NFW halo (`halo_mvir_code 100`, `halo_rs_code 2.0`) is **analytic and not sampled**; the
rollout adds it per particle. The bulge IS sampled and therefore self-gravitating.

Reproducibility: the generator seeds numpy's global RNG (which AGAMA draws from) and uses a
seeded `default_rng` for the shuffle, so the same agama build gives the same file. A
DIFFERENT agama version may not. Prefer copying the `.npz` over regenerating if the numbers
have to match exactly.

## 1a. RESOLVED (2026-09-04 22:55): the wrong forces were the halo exchange on jax 0.9.0

Both MACs were ~45–50 % wrong on force calls after the first because XLA's `ragged_all_to_all`
returned its fill value once buffers moved (donation). Pure-JAX repro, full-pipeline confirmation
with `--halo-exchange buf`, and the production configuration (jax 0.10.2, native) all in
findings.md §14. The MACs, kernels, aligner and partition were innocent. **Every run must keep
`--probe-every N` on**: it is the only instrument that saw this, and a later-step probe must sit
in the step-0 class (~3e-3 rel-L2 at eps 1e-5) or the run is wrong.

## 1b. (SUPERSEDED by 1a -- kept as the record of an experiment that was blind) the MAC is a reproducibility requirement, not a trade-off

`mac_type="dehnen_error"` is the ONLY configuration whose force is the same number twice.
Measured 2026-09-04 (`which_eval_is_right.py`, four calls of the same input through the same
compiled program, each against one fp64 direct sum, 17,825,792 / 4 cards / leaf 1024):

    call     dehnen_error        dehnen (geometric)
      0      3.111693e-03        4.967e-03
      1      3.111689e-03        0.539
      2      3.111693e-03        0.310
      3      3.111703e-03        0.521

**The geometric MAC is 30-54 % wrong on every call after the first**, differently each time,
with an identical accept mask, all overflow flags clear and nothing non-finite -- no guard
sees it, and every accuracy figure ever taken for it was on the first call. Do not run
`mac_type="dehnen"` for anything that evaluates the force more than once. The criterion's
own non-determinism is a +-1-pair tie-break at the acceptance boundary with a force effect at
round-off (pairwise rel_l2 2.2e-07) and zero effect on accuracy.

Score a call other than the first: `--probe-every N` re-scores the force against a fresh fp64
direct sum on a cadence. A t=0 probe alone is what let this go unnoticed.

`findings.md` sections 10-12 have the full record.

## 2. The run

    N 21 012 480, 6 x A100, RCB partitioner
    leaf 1024, theta 0.7, order 6, fp32 state, nearfield_accum = wide
    mac_type dehnen_error, adaptive_eps 1e-5, cross criterion ON, caps DERIVED
                                        ^ REQUIRED for reproducibility -- see section 1b
    dt 5e-4, 489 steps = 0.2445 code = 36.4 Myr
    softening 0.008278 (82.8 pc, the script's derived value = 0.5*rdisk/sqrt(N/1e5))

A quarter orbit is defined at the **baryonic half-mass radius**: r = 0.5301 (5.30 kpc),
v_circ = 3.4088, so `T_orb = 2*pi*r/v = 0.9770` code = 145.7 Myr and a quarter is 0.2443.

    CUDA_VISIBLE_DEVICES=<six cards> XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -u tools/mesh_galaxy_run.py \
      --ic /export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz \
      --ndev 6 --leaf 1024 --theta 0.7 --order 6 --dtype float32 \
      --nearfield-accum wide --mac-type dehnen_error --adaptive-eps 1e-5 \
      --dt 5e-4 --steps 489 \
      --probe 256 --probe-seed 20260901 \
      --render-every 10 --projection xy,xz --render-res 800 --render-extent 1.2 \
      --repartition-every 100 --checkpoint-every 100 \
      --diag-every 10 --overflow-every 10 \
      --max-hours 23 \
      --out-prefix <outdir>/qorbit

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is not cosmetic: see `findings.md` section 4 for a
defect it avoids at small scale. It does NOT affect this configuration (forces agree to seven
significant figures with it either way), but leave it off anyway -- it also keeps the run from
reserving 30 GB it may not need.

**Do NOT pass `--cross-queue` / `--cross-interactions` at leaf 1024.** The derived caps fit
there and are what the criterion needs. Pinning them is only for leaf <= 512, where the
derived caps ask for a single 46-64 GiB buffer -- and pinning them there still OOMs
(`findings.md` sections 5 and round 4).

### Measured behaviour of exactly this configuration

    first force incl. compile   1003.8 s   (the criterion runs two self walks + a prepass)
    median step                  136.3 s    (154 169 particles/s)
    peak memory                  40 105 MiB coordinator, ~31 450 MiB each worker, STABLE
    force_scale range            [0.008366, 179.2] -- a 21 420x spread
    every overflow flag          0
    probe (fp64 direct sum, 256) rel_l2 4.0344e-03, median 6.717e-04, p99 7.481e-03
    dL/L after 6 steps           2.266e-09  (float64, host)
    self_near 12 548 308         cross_near 19 162 479
    489 steps                    18.5 h

`rel_l2` comes from a SUBSAMPLED reference and is comparable only at the same `--probe`. The
same config reads 3.134e-3 at probe 192 and 4.417e-3 at probe 256 in jaccpot's own record --
so keep `--probe 256 --probe-seed 20260901` if the number is to be compared with the above.

### What to check before trusting a long run

1. `# caps:` line -- confirms what was actually used, derived or pinned.
2. `# force_scale range [...]` -- under `dehnen_error` a CONSTANT scale means the criterion
   silently fell back to `eps * 1`, which accepts far more and runs FASTER. The run refuses
   to start on that, but read the spread anyway.
3. `overflow={...}` all zero, on the first force AND every `--overflow-every` steps. A
   truncated walk reads faster and computes a wrong force.
4. `# PROBE ... rel_l2` -- should be ~4.03e-03 for this configuration.
5. `realignment verified against scatter_to_input_order` -- the evaluator's rows are permuted
   even at zero padding; this is the check that the aligner is the same map.

## 3. Outputs

    qorbit_diag.json          config, per-step times, float64 invariants, probe, first_diag
    qorbit_ckpt.npz           rolling snapshot, replaced atomically every 100 steps (~610 MB)
    qorbit_final.npz          final snapshot, same format
    qorbit_frames_xy.npz      face-on surface density, 800^2 per frame
    qorbit_frames_xz.npz      edge-on -- the only view that shows the bulge

Snapshots carry `state0` in the caller's ORIGINAL row order plus `mass`, `component`,
`step`, `t`, `dt`, `softening` and the halo/bulge parameters, so they are self-describing.

Post-processing:

    python tools/mesh_frames_to_movie.py --prefix <outdir>/qorbit \
        --extent 1.2 --length-unit-kpc 10 --dt 5e-4 --render-every 10 --fps 20

    python tools/mesh_rollout_analysis.py \
        --ic /export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz \
        --final <outdir>/qorbit_final.npz --json-out <outdir>/qorbit_analysis.json

The analysis separates disc from bulge and reports, per component, half-mass radius, scale
height, rotation, dispersions and surface-density profile, initial against final -- plus
conservation in float64.

## 3b. The 4-card variant (17,825,792 particles), and the cap cliff

A second IC exists for four cards:

    micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py \
      --output /export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz \
      --n-particles 17825792 --quantum 8192 --seed 7 --state-dtype float32 \
      --iterations 8 --bulge-mass-code 1.2 --bulge-scale-code 0.08 --bulge-gamma 1.0 \
      --rmax-code 20.0

    N 17,825,792 = disc 14,854,827 + bulge 2,970,965, equal particle masses, total 7.2
    `--quantum 8192` so ndev 4 works at leaf 512, 1024 AND 2048
    softening 0.008988 (89.9 pc, derived); quarter orbit still 489 steps at dt 5e-4

    CUDA_VISIBLE_DEVICES=0,2,5,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -u tools/mesh_galaxy_run.py --ic <the 17m8 file> \
      --ndev 4 --leaf 1024 --theta 0.7 --order 6 --dtype float32 \
      --nearfield-accum wide --mac-type dehnen_error --adaptive-eps 1e-5 \
      --dt 5e-4 --steps 489 --probe 256 --probe-seed 20260901 \
      --render-every 10 --projection xy,xz --render-res 800 --render-extent 1.2 \
      --repartition-every 100 --checkpoint-every 100 \
      --diag-every 10 --overflow-every 10 --max-hours 23 --out-prefix <outdir>/qorbit

Measured: **27,692 MiB coordinator / 17,540 MiB worker**, 154.43 s/step, rel_l2 2.8765e-03,
all overflow flags clear, force_scale spread 17,478x. 489 steps ~= 21.0 h.

**Why 17,825,792 and not more.** `cross_max_interactions_per_node` DOUBLES (16,384 ->
32,768) once leaves-per-device passes ~4,608, taking the estimated footprint from 76 % of a
40 GB card to 131 %. At leaf 1024 on four cards that puts a hard ceiling at 4,352 leaves per
device:

    N            leaves/dev   cross_int   est. MiB   verdict
    16,777,216        4,096      16,384     29,226   fits
    17,825,792        4,352      16,384     30,963   fits -- the ceiling
    18,874,368        4,608      32,768     53,534   OOM

Do NOT pick an N between these by interpolating; the limit is a discrete cap threshold. To go
above it on four cards you must change leaf, not N.

**Four cards are lighter per card than six** at these sizes, because the cross caps carry a
factor of (ndev - 1). That also makes four cards MORE accurate here (rel_l2 2.88e-03 against
4.03e-03 at six cards / 21.0 M), since the error is cross-domain limited.

**Do not trim the 21 M IC to reach a smaller N.** The rollout trims a PREFIX and does not
renormalise mass, so trimming 21.0 M -> 17.8 M drops total baryon mass from 7.2 to 6.1
against an unchanged analytic halo of 100. Regenerate at the target N instead; it takes
~8 min on CPU and sets both component masses to target by construction.

## 4. Porting to a queued system (HoreKa)

**The blocking constraint is that this lane is single-process.** It builds one `shard_map`
mesh over `jax.devices()`, and the host-side work (RCB partition, the probe, checkpointing,
the realignment map) assumes one process holding all 21 million rows. There is no
`jax.distributed.initialize()`, so **it cannot span nodes as written**. Device count is
therefore capped by GPUs per node.

Memory scales with the traversal caps, which scale with the LEAF COUNT per device
(`num_leaves = N / ndev / leaf`), not directly with N. Estimates below are the cap proxy
`interactions_per_node*leaves (self+cross) + both wavefront queues`, scaled to the one
measured point (ndev 6 / leaf 1024 = 40 105 MiB):

    ndev  leaf   N/device    leaves   est. coordinator   fits 40 GB   fits 94 GB
       4  1024  5 253 120     5 130         60 881 MiB           no          YES
       4  2048  5 253 120     2 565         15 220 MiB          YES          YES
       6  1024  3 502 080     3 420         40 105 MiB (meas.)   no*         YES
       8  1024  2 626 560     2 565         27 396 MiB          YES          YES

  \* 40 105 of 40 960 MiB is 98 %: it ran here only because the cards were otherwise empty.

The ndev-4 row is the one to watch: at leaf 1024 it is only reachable below 4,608 leaves per
device (N <= 17,825,792), because `cross_max_interactions_per_node` doubles above that. See
section 3b -- the limit is a discrete cap threshold, and 21,012,480 on four cards lands the
wrong side of it. MEASURED on four cards at 17,825,792: 27,692 MiB coordinator, 17,540 MiB
worker, so the proxy over-estimates by ~12 %.

**A 4-GPU node of 40 GB cards will not run leaf 1024** (~61 GB needed). The options, in order
of how little they change:

1. **A node with >= 64 GB per GPU** (H100-80/94GB, or A100-80GB). ndev 4, leaf 1024,
   unchanged configuration. Confirm the partition's per-GPU memory before submitting.
2. **leaf 2048 on 4 x 40 GB.** Fits easily (~15 GB), but near-field cost goes as
   `leaf_pairs x leaf^2`; each doubling of leaf has cost ~2x here (measured 63.7 -> 136.3
   s/step for 512 -> 1024). Expect ~250-270 s/step, i.e. ~35 h for 489 steps. Also unmeasured
   for accuracy, and jaccpot's record notes leaf 2048 was previously unmeasurable in fp32 --
   which the `nearfield_accum=wide` fix may now have lifted, but that has not been checked.
3. **Multi-node.** Needs `jax.distributed.initialize()`, one process per node, and a rework of
   every host-side step that currently assumes all rows locally. The mesh lane has never run
   multi-node. This is real work, not a flag.

**There is no geometric fallback.** `--mac-type dehnen` was measured 30-54 % wrong on every
force evaluation after the first (section 1b); its 63.66 s/step and rel_l2 1.52e-02 are
first-call numbers. If only 40 GB cards are available and the criterion does not fit, the
answer is a larger leaf or fewer particles, not the geometric MAC.

Also note the derived softening depends only on N, so it is unchanged by ndev or leaf; and
`--steps` must be recomputed if `--dt` changes, as `ceil(0.2443 / dt)`.

## 5. What this branch changes

All committed on `horeka/disc-bulge-21m-quarter-orbit` (5d8dc0f), branched from
`feat/odisseo-mesh-lane` (0a75785):

    tools/agama_generate_scm_disk_ic.py   bulge component, disc-only rotation gate, shuffle,
                                          --rmax-code clip, uniform per-component mass
    tools/mesh_galaxy_run.py              --nearfield-accum (default wide), --mac-type,
                                          --adaptive-eps, --mac-cross-criterion,
                                          --overflow-every, --probe/--probe-seed,
                                          --checkpoint-every, --projection (multi-view),
                                          cap overrides, repartition implemented,
                                          float64 host invariants, DIAG_FIELDS import
    tools/mesh_frames_to_movie.py         new
    tools/mesh_rollout_analysis.py        new
    odisseo/mesh_coupling.py              mac_type / adaptive_eps / mac_cross_criterion

The two modified notebooks in the working tree (`notebooks/jaccpot_fmm_first_integration.ipynb`,
`notebooks/scalability/galaxy_disk_fmm_large_n.py`) predate this work and were deliberately
NOT included.

Plus a stash in the `jaccpot` checkout (`stash@{0}`, "local: distributed order=6/theta=0.7
defaults + test theta pins") which raises `DistributedFMMConfig`'s defaults from order 3 /
theta 0.4. It applies cleanly to `c1cede7` and is NOT needed for the run, which passes both
explicitly. Landing it requires the test theta pins in that same stash, because upstream's
cap-sizing tests read the defaults.

`tests/test_mesh_lane.py` is 15/15 green on CPU (`JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=2`). On GPU one test is flaky for a reason
that is not this work -- `findings.md` section 4.
