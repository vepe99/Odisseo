# The distributed mesh rollout: provenance and evidence

`tools/mesh_galaxy_run.py` is committed here **byte-identical** to the script that produced
every number below, so the measurements can be re-derived without arguing about which version
ran. Do not reformat it before reproducing them.

It exists because `jaccpot.distributed.fmm` exports a *force* and nothing else: no leapfrog, no
Verlet, no scan over steps. ODISSEO's four lanes (`direct`, `fmm_forward`, `fmm_differentiable`,
`fmm_blockstep`) are all single-device, and `odisseo.render_callback` streams only from the
single-GPU `strict_run_v2` scan. This is the missing loop plus the renderer that goes with it.

## What is validated

| config | result |
|---|---|
| 262 144 / 2 cards / 40 steps, leaf 64, θ 0.7, order 6, fp32 | dL/L **2.4e-06**, COM drift 3.3e-04, zero overflow flags, **1.06 s/step** (`val262k.log`) |
| 20 971 520 / 5 cards, leaf 512, θ 0.7, order 6, fp32 | first force clean, no overflow; self_near 23 089 698, cross_near 30 131 451 (`probe21m.log`) |
| 20 971 520 / 5 cards, 125 steps with rendering | dL/L **2.6e-06**, COM drift 5.1e-05, **median 69.2 s/step** (min 55.4, max 80.9) (`run21m.log`) |

The 21 M rows are a snapshot of a run stopped on its `--max-hours` budget, not a completed
integration. Seconds on this box carry ~20 % spread — the cards are shared. Pair counts, overflow
flags and the conservation figures are contention-independent.

## Two traps this file already solves. Preserve both.

**1. The output rows are permuted even with ZERO padding.** `make_force_evaluator` returns rows in
per-device Morton order. `scatter_to_input_order`'s docstring says the maps "agree whenever no
device is padded" — true of them as *maps*, false of *row order*, and an easy sentence to misread.
Reading the force on that assumption gives every particle a Morton neighbour's acceleration:
smooth, plausible, and wrong by tens of percent. Measured again on 2 CPU devices at
cap == count == 512: a naive `gid_flat` read is wrong on **1022 of 1024 rows**.

The fix is an on-device realignment (argsort of the returned gids + two gathers inside one
`shard_map`; `rank_in` precomputed on the host because the partition is frozen), checked against
`scatter_to_input_order` on the first force. **Never delete that check**, and keep it on a value
that has not been through an add — subtracting the halo term back off reintroduces fp32 rounding
and turns an exact check into a fake ~1e-6 mismatch on 80 % of rows.

**2. Do not fuse the force and the integrator into one jit.** At 21 M on five cards that makes XLA
hold the traversal buffers and the integrator temporaries in one live range; one device fails an
allocation, never joins the `AllGather`, and the other four hang at the rendezvous **forever** — a
deadlock at 0 % GPU utilisation, not an OOM message. Split into drift / force / kick as three
dispatches with `donate_argnums`; peak becomes the max of the two, not the sum.

## Regenerating the initial conditions

The `.npz` ICs are not committed (474 MB for the 21 M one, and `notebooks/scalability/ic_cache/`
is gitignored). Regenerate with AGAMA — 10^7 disc particles sample in ~131 s:

```bash
micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py \
  --output notebooks/scalability/ic_cache/disk_21m.npz \
  --n-particles 20971520 --seed 7 --state-dtype float32
```

Defaults are the SCM disc used throughout: `disk_mass_code 6.0`, `halo_mvir_code 100.0`,
`halo_rs_code 2.0`, `rdisk_code 0.24`, `hdisk_code 0.03`. The halo is **analytic and not
sampled** — the IC carries `halo_mass_code` / `halo_rs_code` for exactly this reason, and the
rollout adds that NFW term per particle. Evolving the disc under self-gravity alone is a
different system.

## Known gaps

- **`--repartition-every` is declared but not implemented.** Over a Gyr the frozen RCB partition
  degrades. Cheap to add: `cap` depends only on `(n, ndev, leaf_size)` and never on positions, so
  a repartition is a host permutation plus a `device_put` with **no recompile**.
- **No energy diagnostic.** The distributed evaluator returns accelerations only
  (`compute_potential=False` is hardwired), so |ΔE/E| is unavailable. Momentum, angular momentum,
  COM drift and KE are exact and are what is reported. An estimator is not a conservation check.
- **Overflow is checked only on the first force.** Caps are static but pair counts grow as the
  disc clusters, so an overflow can switch on mid-run and silently truncate the force. The diag
  vector is returned every call; checking on a cadence is free.
- **N must be `ndev * k * leaf`** so `cap == count` and no device is padded. The script hard-errors
  otherwise; the aligner's `rank_in` construction assumes no padding.
- **ODISSEO has no lane that reaches this.** `resolve_lane` returns four single-device lanes.

## The two offline audits alongside it

`mac_sphere_audit.py` (+ `mac_audit_small.json`, `mac_audit_10M.json`) tests whether the disc's
superlinear near field is an artefact of the MAC radius being a box circumsphere about the box
centre rather than a COM-centred particle radius. **It is not.** Median `r_box / r_com` = 0.970 at
10 485 760 — the shipped radius is already 3 % *tighter* — and switching to the COM makes the near
field 12 % **worse**. The N^1.45 growth is physical; no tree or geometry work follows.

The all-pairs proxy the script uses is trustworthy: it predicts the real recorded
`self_near_pairs` to 1.04x at θ 0.4 and 0.99x at θ 0.7.

One real lever did surface and is not free: the box (L-infinity) MAC is worth 1.42x fewer near
pairs, but `jaccpot/docs/treecode_mac_stability.md` records box extents as statically as accurate
and **dynamically unstable**. Re-measure it over many steps before believing it.
