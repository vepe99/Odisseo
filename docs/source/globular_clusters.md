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
- `repark_empty=True` (TEMPORARY, until nornax keeps empty slots out of the star tree's box): after every
  transfer and at every edge's end, `install_empty_slot_repark` moves each cluster's empty (stripped) slots
  inside its member radius, co-moving. It wraps nornax's private `_edge_progs(slow).transfer` on the
  scheme instance and raises (naming `repark_empty=False`) if nornax changes those. nornax builds the clusters'
  star tree over all `[K, M]` slots and leaves an emptied slot where its star left, at rest; as the cluster
  moves on, that tree's box grows and the star tree re-planned -- a full prepare and a recompilation -- at
  every edge (22 min per segment at 1e7 + 4096). Empty slots are massless: over four stripping edges of an
  orbiting cluster the galaxy, its accelerations, the members, their accelerations and the ledger equal
  `repark_empty=False`'s bitwise (`tests/test_globular_clusters.py`), while the slots' box stays inside the
  members' sphere instead of growing with the cluster's travel. Deterministic, so restarts stay bitwise.
- `default_n_sub(dt, table)`: one fixed `n_sub` serves every cluster, so the most compact cluster sets it
  (20 internal steps per half-mass crossing time; Pal 5 alone: 8 at `dt = 1e-3`; the 165-cluster bundle:
  54, set by Liller 1). The softening `eps_c` is likewise one value for all clusters
  (`default_cluster_softening`: the smallest r_h / 26).

**Continuing a run** (`ProductionRunner.run(restart_from=...)`, the `.npz` restarts `run_live.py` writes):
on a GPU at large N a restored run refused its first segment ("fused payload static target-block cap was not
preflighted"): jaccpot's pinned trace state does not carry its static target-block preflight key. The demo's
`run_live.py` applies a TEMPORARY workaround when it continues a run (`--restore-preflight-patch`, on by
default there; `gcmw.patch_jaccpot_csr_restore_preflight`, not part of the `odisseo` package). It makes that one
check, in jaccpot's `_large_n_pipeline` module only, accept a key that differs only in the leaf count when both
caps are 1: on the CSR near-field lane the cap is forced to 1 and the block rectangle is a never-read
placeholder. Every other mismatch still raises. It goes away once jaccpot carries the key. With it the 1e7
Pal 5 run continued bitwise; the bundle's continuation was not bitwise (jaccpot warned that the galaxy
provider's compile key differs), with its ledger in class.

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

`GC_SMALL=1` (the default) runs everything on CPU in minutes (galaxy 2e4, M = 256, 20 Myr). The
committed figures are the GPU demo; see the results section below.

RESULTS_PLACEHOLDER

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

The run resumed once from its restart (`galaxy_provider_key` warning, not bitwise, ledger in class).
`notebooks/gc_in_mw/analyse_bundle.py` makes the figure and `bundle_summary.json`. A long run is
`bundle.sh long`. Expect each capacity re-plan to recompile the whole program set, and
`cluster_batch=1` runs the 165 clusters' Hermite substeps one after the other.
