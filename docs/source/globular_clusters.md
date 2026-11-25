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
- Solvers: on a GPU jaccpot's `GALAXY_SETTINGS` (leaf 64, p6, `dehnen_error` at 1e-4, fp32); on CPU the
  geometric Dehnen MAC at theta 0.6 (the error-based MAC's force scale needs the Pallas near field, which
  the CPU route only interprets). The star solver is geometric at theta 0.5 with softening `eps_c`.

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

BUNDLE_PLACEHOLDER
