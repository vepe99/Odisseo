# Globular clusters in a live Milky Way

`odisseo.globular_clusters` puts the Milky Way's globular clusters on their observed orbits in a live
N-body galaxy. It is a thin layer: the clusters' integration (direct-sum Hermite-4 inside each cluster,
the galaxy coupled through the FMM, stripping into the galaxy's rows, energy ledgers, checkpoints) is
nornax's hybrid Hermite scheme (`nornax.adapters.jaccpot_runner.ClustersRunnerScheme`, nornax
`docs/hybrid_hermite.md` sections 8, 9, 13 and 14) running inside jaccpot's `ProductionRunner`.
Odisseo turns catalogue rows into the cluster block that scheme takes and builds the scheme with
documented defaults.

## The API

```python
from odisseo import globular_clusters as gc

cat = gc.load_baumgardt(directory="...")            # or $ODISSEO_GC_DATA
xc, vc, mc, table = gc.clusters_from_catalogue(
    ["Pal 5", "NGC 5466"], M=1000, model="king",     # or "plummer"
    catalogue=cat, return_table=True)                # [K, M, 3], [K, M, 3], [K, M]: code units
eps_c = gc.default_cluster_softening(table)          # r_h,min / 1.305 / 20
gal, clu, settings = gc.build_solvers(eps_g=0.01, eps_c=eps_c)
scheme = gc.clusters_scheme((xc, vc, mc), eps_c=eps_c, cluster_solver=clu)

from jaccpot.production import ProductionRunner, SegmentConfig
runner = ProductionRunner(gal, settings=settings, segment=SegmentConfig(steps=50), scheme=scheme)
result = runner.run(galaxy_state, galaxy_masses, dt=1e-3, num_steps=...)
```

| function | what |
|---|---|
| `load_baumgardt(orbits=, structure=, directory=)` | the catalogue, `{key: GCRecord}` (the orbit table joined with the structural table) |
| `normalize_name(name)` | `"Pal 5"`, `"pal5"` -> `"PAL_5"`; `"47 Tuc"`, `"M68"` -> their NGC numbers |
| `observed_to_galactocentric(records, frame=)` | positions [kpc] and velocities [km/s] with astropy |
| `galactocentric_frame(**overrides)`, `BAUMGARDT_FRAME` | the frame (astropy's defaults), or the catalogue's own |
| `to_code_units(pos_kpc, vel_kms, units=GALAXY_UNITS)` | kpc, km/s -> code units |
| `KingModel(w0)`, `king_w0_from_concentration(c)` | King (1966) models; W0 from `c = log10(r_t / r_0)` |
| `sample_king(n, w0=, mass=, r_half=)`, `sample_plummer(...)` | centred realisations (the Plummer sampler is `odisseo.initial_condition.Plummer_sphere`) |
| `cluster_table(names, catalogue)` | per cluster: centre, mass, half-mass radius, W0 in code units |
| `clusters_from_catalogue(names, M, model=, centres=)` | the `(xc, vc, mc)` block; `centres=` puts the clusters elsewhere (e.g. an orbit integrated back) |
| `integrate_orbit(x0, v0, accel, t, dt)` | leapfrog test-particle orbits (negative `t`: backwards) |
| `build_solvers(eps_g, eps_c)` | the galaxy solver (Odisseo's runner lane, `build_galaxy_solver`), the clusters' star solver and the runner's settings |
| `clusters_scheme(clusters, eps_c=, cluster_solver=, ...)` | `ClustersRunnerScheme` with the defaults below |

**The scheme's defaults** (`clusters_scheme`), all nornax's or HH3a/HH3b's 1e8 configuration:

- `ClusterHybridHermite(n_sub=16, order=4, cluster_batch=1)`: fixed internal steps `dt / n_sub`, one
  cluster's pair tensor at a time; `MembershipSplit(DirectSumGravity(G, eps_c, block_size=1024))`
  (Plummer softening `eps_c`). The demo runs `dt = 1e-3`, `n_sub = 8`: the same internal step.
- `StrippingPolicy()`: strip radius 2 r_J, the tidal tensor fitted to the stars' slow field, Omega about
  the galaxy's centre of mass, dwell 1, dissolution below 8 members. These are the physics choices of
  nornax `docs/hybrid_hermite.md` section 12 item 6 and stay at nornax's defaults here.
- `reserve = default_reserve(K, M)` = `K M / 8` rounded up to 1024 free galaxy rows for stripped stars
  (grown, with a re-plan, when full); a run that dissolves its clusters should pass `K * M`.
- `options = dict(cross="fmm", coupling="fmm", cluster_leaf_size=64)`.
- `default_n_sub(dt, table)`: one fixed `n_sub` serves every cluster, so the most compact cluster sets it
  (20 internal steps per half-mass crossing time; Pal 5 alone: 8 at `dt = 1e-3`; the 165-cluster bundle:
  54, set by Liller 1). The softening `eps_c` is likewise one value for all clusters
  (`default_cluster_softening`: the smallest r_h / 26).

**Requirements: nornax >= 3a75a41 and jaccpot >= a2add480** (HH5, 2026-11-25; `require_hh5()` refuses older
ones). They build the clusters' star tree over the FILLED slots only, grow a re-plan only where it failed (by 2x
its need), and restore a carried run with the writer's tree and static target-block preflight. Before them:

- a stripped star's empty slot, parked where the star left, stretched the star tree as the cluster moved, and
  the star tree re-planned and recompiled at almost every stripping edge;
- a restored large-N run refused its first segment ("fused payload static target-block cap was not
  preflighted").

The first version of this layer (Odisseo #49, 2026-11-25) worked around both: it re-parked the empty slots
co-moving inside the member radius (`repark_empty`), and the demo patched jaccpot's preflight check on restore.
Both workarounds are gone.

**Correction.** The re-park was claimed physics-neutral, on a bitwise test with nornax's direct reference force.
That force has no tree, so the test could not see the effect. On the fused lane the re-parked slots still shaped
the star tree: HH5 measured galaxy accelerations differing by 1.1e-6 with vs without the re-park (jaccpot
`docs/production_2026-11.md` `## HH5`, nornax `docs/hybrid_hermite.md` section 17). With HH5's star tree, empty
slots are out of the tree, so their positions do not matter.

## The catalogue

Holger Baumgardt's Galactic globular cluster database,
<https://people.smp.uq.edu.au/HolgerBaumgardt/globular/>: the orbit table `orbits_table.txt` (Gaia EDR3
proper motions, line-of-sight velocities, distances; Vasiliev & Baumgardt 2021, Baumgardt & Vasiliev
2021) and the structural table `combined_table.txt` (N-body fits: masses, core, half-mass and tidal
radii; Baumgardt & Hilker 2018, Baumgardt et al. 2023). Cite these when you use them.

- `python tools/fetch_baumgardt.py --out DIR` downloads both and writes `DIR/SOURCE.json` (URL, UTC date,
  size, sha256). Then `export ODISSEO_GC_DATA=DIR`.
- The site states no licence. The repository carries only a four-row test fixture
  (`tests/data/baumgardt_*_fixture.txt`: NGC 104, NGC 4590, NGC 5466, Pal 5), never the tables.
- Columns used: RA, Dec, distance, <RV>, the proper motions (orbit table); mass, r_c, r_h,m (3D
  half-mass radius), r_t (structure table). K = 165 clusters have both.
- **King W0** comes from `c = log10(r_t / r_c)`. The table's r_c is Spitzer's core radius, which is
  King's radius r_0; its r_t is a Jacobi radius (Webb et al. 2013), not King's truncation, so W0 is an
  estimate (Pal 5: c = 0.51, W0 = 2.0; 47 Tuc: c = 2.31, W0 = 9.8). The model is then scaled to the
  catalogue's mass and 3D half-mass radius.

## Units and frames

- **Code units** `GALAXY_UNITS`: 10 kpc, 1e10 Msun, G = 1 (the units of `tools/a4_galaxy_ics.py`'s
  galaxies); time 149.1 Myr, velocity 65.58 km/s.
- **Galactocentric frame**: astropy's `Galactocentric()` with the running astropy's defaults (the `"v4.0"`
  set: R0 = 8.122 kpc, z_sun = 20.8 pc, v_sun = (12.9, 245.6, 7.78) km/s; astropy 7.2 and 8.0 tested).
  `galactocentric_frame(**overrides)` changes them; `frame_parameters(frame)` records them.
- `BAUMGARDT_FRAME` (R0 = 8.178 kpc, z_sun = 0, v_sun = (11.1, 252.24, 7.25) km/s, no roll) reproduces
  the catalogue's own X, Y, Z, U, V, W to its rounding (inferred by least squares over all 165 clusters:
  0.01 kpc, 0.1 km/s at the 95th percentile). The catalogue's X points from the centre to the Sun:
  X = -x, U = -v_x. The unit test checks the transform against these published values.
- Axes: x from the Sun towards the Galactic centre, y along the Sun's motion, z to the north Galactic
  pole. The Milky Way's disc therefore rotates with L_z < 0 in this frame. The a4 galaxy's disc rotates
  with L_z > 0, so the demo's loader (`gcmw.load_galaxy`) turns the galaxy by 180 degrees about x (a
  rotation; the model is symmetric under it) and puts it at rest at the origin: the GCs' observed orbits
  keep their sense relative to the disc (Pal 5 is prograde).

## The demo: Pal 5 in a live Milky Way

`notebooks/gc_in_mw/Pal5_live_MW.ipynb` drives five scripts (all resumable; outputs under
`$ODISSEO_GC_RUNS`):

1. `tools/fetch_baumgardt.py` (the catalogue) and `tools/a4_galaxy_ics.py galaxy --export-potential`
   (the live galaxy and its agama potential);
2. `prepare.py` (agama): fits Odisseo's static MW (NFW + MN3 disc + Hernquist bulge) to the live model's
   potential where the orbit goes, and integrates Pal 5 back in time in the live model's potential;
3. `run_live.py`: Pal 5 as a hybrid Hermite cluster in the live galaxy through `ProductionRunner`,
   forward to the present (rows per edge, snapshots, `.npz` restarts);
4. `run_static.py`: the same stars as a plain Odisseo N-body system in the static fit;
5. `analyse.py`: conservation, stripping, the stream track live vs static.

`GC_SMALL=1` (the default) runs everything on CPU: galaxy 1e5 subsampled to 5000, M = 64, 2 Myr; 12-14 min
end to end with nbconvert on this host, compile-dominated. `GC_SMALL=0 GC_RUN=0` only plots existing GPU
outputs; that is how the committed figures were made:

```bash
ODISSEO_GC_RUNS=... ODISSEO_GC_DATA=... GC_SMALL=0 GC_RUN=0 \
  jupyter nbconvert --to notebook --execute --inplace notebooks/gc_in_mw/Pal5_live_MW.ipynb
```

### Results (2026-11-27/28, HH5 mains, local shared A100s: functional, no timings)

**The live Milky Way.** The a4 `galaxy` model of `tools/a4_galaxy_ics.py` (seed 7) at 1e6 particles:

- the disc has 6e10 Msun, a halo particle 1.4e6 Msun, softening 100 pc;
- v_c at 8 kpc is 250 km/s.

The galaxy is rotated into the Milky Way's sense, as described above.

**The static MW.** `prepare.py` fitted Odisseo's static MW to the live model's potential where Pal 5's orbit
goes:

- free parameters: the halo and the disc; the bulge is the model's own Hernquist;
- result: halo 1.14e12 Msun, r_s 22.9 kpc; disc 6.8e10 Msun, h_R 3.9 kpc, h_z 1 kpc (h_z at its bound);
- force difference along the orbit: median 1.6 %, p90 5.2 %; v_c(8 kpc) 239 vs 250 km/s.

The N-body galaxy's own field differs from its agama potential by as much: 1.5 % median along the orbit
(`field_check.py`). Pal 5 integrated back 1 Gyr in the live model's potential and forward again in each
potential ends 2.74 kpc apart at the present: this is the orbit-level size of the live-vs-static potential
difference.

**The IC matters more than anything else here.** Pal 5 at its present catalogue mass (1.34e4 Msun) and r_h
(27.6 pc), put on its orbit 1 Gyr back, dissolves within 300 Myr in both arms (`dissolution.py`: bound mass by
an iterative energy cut). Today's mass and radius, carried back without the mass lost since, are too light for
the orbit.

- **The catalogue's `mass="initial"`** (6.0e4 Msun) is the formation mass about 12 Gyr ago: from there the
  static arm keeps 47k.
- **Calibration on the static arm** at the catalogue r_h:

  | start mass [Msun] | 2.0e4 | 3.0e4 | 3.5e4 | 4.0e4 | 5.0e4 |
  |---|---|---|---|---|---|
  | static arm | dissolved ~450 Myr | dissolved ~850 Myr | dissolved ~990 Myr | 12.3k at 1 Gyr | 29.3k at 1 Gyr |

  Survival is a steep threshold in mass.
- **The live arm loses mass faster than the static arm.**
  - From 4e4 it dissolves at ~900 Myr (static: 12.3k left at 1 Gyr).
  - From 5e4 it ends at **17.0k** (static: 29.3k). That is close to Pal 5 today, so **the demo starts Pal 5 at
    5e4 Msun** (`--mass 50000`).
  - Why the live arm loses more: not the strip rule (at most ~70 stripped stars stay bound to the members),
    and not disc shocking (a static disc as thin as the live one changes nothing). N-body heating by the 1e6
    galaxy's massive particles is the leading candidate; a 1e7 run would test it.

**Pal 5** (King, W0 2.03; 5e4 Msun; r_h 27.6 pc; M = 4096 stars; eps_c 0.11 pc; `n_sub = 8` at `dt = 1e-3`), from
1 Gyr back to the present (`runs/pal5_live_cal5e4` vs `runs/calib_static_J`):

- **The run.** 67 segments in 4 claims; **1 capacity re-plan in the whole run** (HH5's targeted re-plans).
  Restores continue bitwise.
- **Conservation (live).** E - ledger drift <= 1.4e-7, |ΔP| <= 9.2e-9, device peak 0.87 GB. The static arm's
  energy drift is 1.3e-5.
- **Today.** The live Pal 5 keeps a bound core of 17.0k Msun and 1500 member stars; 2596 stars are in its
  stream.
- **Live vs static at the present.**
  - The progenitors are **0.40 kpc** apart, against 2.74 kpc between the two potentials' point orbits:
    **agreement within the live-vs-static potential difference**.
  - The stream tracks in each progenitor's orbital frame agree to a median |Δφ2| of **0.08 deg** (max 0.37),
    inside the stream's width of 0.29 deg; |Δr|/r 1.3e-2.
  - The live stream is shorter: φ1 from -4.1 to 3.5 deg, against -5.1 to 5.7 deg static.
  - The static progenitor sits on its potential's point orbit to 14 pc. The live progenitor is 2.4 kpc off its
    agama point orbit: the N-body galaxy is not its smooth potential.
- **At 745 Myr:** progenitors 0.51 kpc apart (point orbits 2.43 kpc); |Δφ2| 0.40 deg against a width of
  0.44 deg.

## The K ~ 150 bundle

`notebooks/gc_in_mw/bundle.sh functional|long`: every catalogue cluster with a mass and a half-mass radius
(K = 165, M = 1000, King) at today's positions in the live galaxy (`run_live.py --names ALL`). The
functional run is a few segments; the long run is resumable. Scaling numbers are HoreKa's (jaccpot job
`hh3`), not this page's.

**The functional run** (2026-11-25; local shared A100: functional, no timings). 1e7 galaxy, K = 165 King
clusters x 1000 stars, `n_sub = 54` (Liller 1 sets it), `eps_c` 0.06 pc, reserve 21504, `dt = 1e-3`, 4 segments
x 10 steps (6 Myr):

| edge | t [Myr] | stripped | E - ledger drift | \|ΔP\| | retries / re-plans | peak |
|---|---|---|---|---|---|---|
| 1 | 1.5 | 183 | 5.0e-8 | 3.8e-10 | 0 / 0 | 4.84 GB |
| 2 | 3.0 | 295 | 7.5e-8 | 7.5e-10 | 0 / 0 | 4.84 GB |
| 3 | 4.5 | 373 | 6.5e-8 | 1.1e-9 | 0 / 0 | 4.51 GB |
| 4 | 6.0 | 389 | 5.3e-8 | 1.5e-9 | 0 / 0 | 4.51 GB |

No cluster dissolved. Thirteen clusters lost stars, and they fall into two groups:

- bulge clusters (R_GC < 2 kpc): Gran 1 12 %, Liller 1, Terzan 2, NGC 6528;
- diffuse outer clusters whose catalogue King extent is larger than 2 r_J at today's position: Whiting 1 8 %,
  Pal 13 6 %, AM 4 -- an initial-condition transient of putting today's structure on today's orbit.

The run resumed once from its restart: with the pre-HH5 code and the preflight workaround, jaccpot warned
`galaxy_provider_key` differs and the continuation was not bitwise; the ledger stayed in class. HH5 traced the
warning to the same root cause as the preflight refusal and fixed both.
`notebooks/gc_in_mw/analyse_bundle.py` makes the figure and `bundle_summary.json`. A long run is
`bundle.sh long`. Expect each capacity re-plan to recompile the whole program set, and
`cluster_batch=1` runs the 165 clusters' Hermite substeps one after the other.
