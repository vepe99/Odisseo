# FMM runs on jaccpot's production runner

`integrate()` hands every fixed-step shared-timestep FMM run to jaccpot's
`ProductionRunner` (jaccpot `docs/production.md`): the fused lane, run as K-step
`strict_run_v2(carry="particles")` segments with traced parameters, capacity
re-plans and diagnostics at the segment edges. Odisseo builds the solver, maps its
configuration onto it and packages the result (`odisseo.runner_coupling`); the step
itself is jaccpot's, so it costs what it costs in jaccpot's harness
(`bench/fused_memory_budget.py`).

## Which runs go there

Since 2026-11 (multi-node plan, wave 3; a user decision) `resolve_lane()` names the
lane `"fmm_runner"` for **every** fixed-step `FMM_ACC` run on one device -- any
`N`, CPU or GPU, any preset -- except `fmm_mesh` (the same runner across GPUs,
[below](#across-the-gpus-of-a-node)), `fmm_blockstep` and `fmm_differentiable`,
which keep precedence. `fixed_timestep=False` is the adaptive coupler lane
(`"fmm_forward"`, diffrax steps), unchanged.

A fixed-step run the runner does not do -- `fmm_refresh_every != 1`,
`fmm_refresh_after_position_update=True`, an active-particle schedule -- raises a
`ValueError` naming what still runs it: Odisseo's own FMM leapfrog loop, which
ran these, is gone ([Migration](#migration)).

**On a CPU** the run takes the fused lane's CPU route
(`runner_coupling.fused_lane_scope`): jaccpot gates the fused lane's prepare on
`jax.default_backend() == "gpu"`, and its pure-JAX near field runs on a CPU
unchanged, so for the duration of the build, the run and a force evaluation the
backend is reported as `"gpu"` (what jaccpot's own CPU tests of the lane do), the
Pallas kernels are off, and under `dehnen_error` the CSR near field runs in Pallas
interpret mode (its force scale needs it). The real backend is restored
afterwards. It is process-global while it lasts: run no other JAX work in another
thread meanwhile. A stopgap until jaccpot takes a CPU switch for the fused lane.

### The settings: two profiles

`runner_coupling.runner_profile()` picks how the runner is sized:

| profile | when | settings |
|---|---|---|
| `"galaxy"` | `fmm_galaxy_settings` given; `fmm_preset="galaxy"`; or the automatic large-N profile: preset `"fast"` (the default), `fmm_auto_large_n_profile=True` (the default), a GPU backend and at least `fmm_large_n_min_particles` (200,000) particles | jaccpot's `GALAXY_SETTINGS` -- leaf 64, p6, cell level floor 8, automatic leaf capacity, the preset's MAC, fp32 -- or `fmm_galaxy_settings` (`galaxy_settings()`) |
| `"config"` | every other run (a config that sets nothing, on a CPU or below 200,000 particles; `fmm_preset="large_n_gpu"`, `"fast"` with the auto profile off, ...) | the config's own FMM knobs (`config_settings()`): `leaf_size=fmm_leaf_size` (16), `max_order=fmm_fixed_order or fmm_max_order` (4), `theta=fmm_theta` (0.6), the geometric Dehnen MAC (`fmm_mac_type="dehnen"`) or eq (16a) (`"dehnen_error"` + `fmm_adaptive_eps`), no cell level floor, jaccpot's automatic leaf capacity and cubic Morton box, the state's precision (fp32 for `"large_n_gpu"` with `fmm_large_n_force_fp32`, as on the coupler) |

The coupler's engine knobs have no runner counterpart and are **not read**:
`fmm_basis`, `fmm_farfield_mode`, `fmm_m2l_chunk_size`, `fmm_nearfield_mode`,
`fmm_nearfield_edge_chunk_size`, `fmm_tree_build_mode`, `fmm_tree_leaf_target`
(unless it equals the leaf size), `fmm_jit_tree`, `fmm_jit_traversal`, the pair-queue
and traversal capacities, `fmm_prepare_stage_memory_split_enabled`,
`fmm_upward_leaf_batch_size`, `fmm_enforce_static_shape_contract`,
`fmm_static_shape_warmup_prepares` -- and under the `"galaxy"` profile
`fmm_leaf_size`, `fmm_max_order`, `fmm_fixed_order` and `fmm_theta` too. A config
that sets one to a non-default value gets one `UserWarning` per process and knob
set (`runner_coupling.warn_ignored_knobs`), naming the settings that run. They
tune the removed loop's engine and memory, not the physics, and the adaptive lane
still reads them, so they warn rather than raise. Any MAC other than `"dehnen"` /
`"dehnen_error"` raises: the fused lane runs these two.

**The MAC is the preset's.** Since jaccpot's 2026-10-21 preset change
(`GalaxySettings(mac_type="dehnen_error", adaptive_eps=1e-4)`) that is Dehnen's
eq (16a) at a relative force-accuracy target of 1e-4; before, the geometric
Dehnen MAC at opening angle 0.8. jaccpot measured, on one A100, 1.12x the
geometric step at 25M disc + bulge for 2.8x lower rel-L2 (1.49e-4 against
4.18e-4), and about 22 B/particle more at 1e8 (jaccpot `docs/production.md`,
"The MAC"). Under it `G` and the softening are the solver's: a call with another
`params.G` or `softening` builds a new solver and compiles (below); the opening
angle decides no pair. The geometric MAC back:

```python
from jaccpot.production import GalaxySettings

config = config._replace(
    fmm_galaxy_settings=GalaxySettings(mac_type="dehnen", adaptive_eps=None)
)
```

Under the `"galaxy"` profile `fmm_mac_type="dehnen"` (Odisseo's default) keeps the
settings' MAC, so it cannot ask for the geometric one; `fmm_mac_type="dehnen_error"`
with `fmm_adaptive_eps` sets the target. (Under the `"config"` profile
`fmm_mac_type` is the MAC.)

## Integration schemes (multi-node plan I1)

nornax owns the integration schemes, jaccpot's `ProductionRunner` the run and
Odisseo the science (the plan's decision U1). `config.fmm_scheme` hands the
runner a scheme:

| `fmm_scheme` | the step |
|---|---|
| `None` (default) | the runner's built-in velocity-Verlet KDK, unchanged |
| `"nornax_kdk"` | `nornax.schemes.LeapfrogKDK`: the same KDK written by nornax, **bitwise** the built-in (`tests/test_runner_scheme.py`) |
| an object | any `jaccpot.scheme.SplitStepScheme` (hashable) |

It selects nothing since the 2026-11 switch (every fixed-step run is the
runner's) and changes no setting. The lanes with their own step --
`fmm_blockstep`, `fmm_differentiable`, the adaptive `"fmm_forward"`, the older
`"fmm_mesh"` -- refuse it, and so does `"fmm_mesh_runner"` until the plan's I2
gives the multi-GPU runner the same seam. The scheme is part of the kept
runner's key: switching it builds a new runner and compiles.

**Deprecated** (`DeprecationWarning`, once, on the caller's line; nothing is
removed yet):

- Odisseo's own FMM leapfrog loop: `integrate_leapfrog_jaccpot_active`,
  `build_jitted_leapfrog_jaccpot_active`, and lane `"fmm_forward"` with
  `fixed_timestep=True` -- replaced by `"fmm_runner"` (`fmm_preset="galaxy"`,
  optionally `fmm_scheme="nornax_kdk"`);
- the older mesh lane's KDK (`"fmm_mesh"`, `MeshOptions`,
  `mesh_coupling.integrate_mesh_jaccpot`) -- replaced by `"fmm_mesh_runner"`
  (`docs/source/parallelism.md`).

`integrate()` no longer reaches the deprecated loop. The adaptive diffrax FMM
path (`fixed_timestep=False`), the block-step lane (nornax's block KDK already)
and the differentiable lane are not leapfrog loops the runner replaces and are
not deprecated.

## Settings

The runner lane reads:

| field | |
|---|---|
| `softening`, `softening_kernel`, `params.G` | the solver's softening (Plummer-equivalent), kernel and G, exactly as the coupler passed them |
| `fmm_galaxy_settings` | a `jaccpot.production.GalaxySettings`: the `"galaxy"` profile with these settings |
| `fmm_preset`, `fmm_auto_large_n_profile`, `fmm_large_n_min_particles` | the profile (above) |
| `fmm_leaf_size`, `fmm_max_order`, `fmm_fixed_order`, `fmm_theta` | the `"config"` profile's settings |
| `fmm_mac_type="dehnen_error"`, `fmm_adaptive_eps`, `fmm_mac_force_scale_mode` | Dehnen's eq (16a) MAC; any MAC other than `"dehnen"` / `"dehnen_error"` is refused |
| `fmm_large_n_force_fp32=False` with a float64 state | the fp64 force under `"galaxy"` (without `fmm_galaxy_settings`) and `"large_n_gpu"` |
| `fmm_use_pallas` / `ODISSEO_FMM_USE_PALLAS`, the `fmm_large_n_*` environment knobs | as on the coupler (on a GPU; the CPU route sets its own) |
| `fmm_scheme` | the integration scheme (above) |
| `external_accelerations` and the potentials' fields | the external field |

The solver is `preset="galaxy"`'s own configuration with the settings' leaf size,
cell level floor, leaf capacity, Morton box (`GalaxySettings.cubic_bounds`, from
jaccpot's S3b on), order and MAC put in, so whatever else the preset sets comes
with it; `GALAXY_SETTINGS` builds exactly the preset. `fmm_galaxy_settings` is the
way to set anything else:

```python
from jaccpot.production import GalaxySettings

config = config._replace(
    fmm_galaxy_settings=GalaxySettings(adaptive_eps=1e-5, max_order=8),
    # or the geometric MAC at a tighter angle:
    # GalaxySettings(mac_type="dehnen", adaptive_eps=None, theta=0.6, max_order=8)
)
```

The state keeps its dtype: a float32 state is the production memory footprint
(a float64 state with the fp32 force is jaccpot's mixed mode).

## A second call compiles nothing

jaccpot caches its compiled programs on the solver, keyed by the external
field's identity. The runner lane therefore

- keeps the solver and runner of the most recent configuration (the particle
  count included) between calls (`clear_runner_cache()` drops them; until then
  the runner also holds its last handle's accelerations, 12 B per particle in
  fp32, 24 with an external field);
- passes `dt = t_end / num_timesteps`, `params.G` and the softening squared as
  traced arguments -- under `dehnen_error`, the preset's MAC, `dt` only: G and
  the softening are the solver's, so they are part of the kept solver's key and
  another value builds a new solver, which compiles;
- evaluates the external field through ONE pair of module-level functions,
  `external_acceleration` / `external_potential`, taking an
  `ExternalFieldParams`: the whole `SimulationParams` as traced leaves and the
  static configuration fields the potentials read as pytree metadata.

Another `dt` or set of potential parameters builds no solver and adds no
program to jaccpot's cache, and with the geometric MAC neither does another `G`
(`tests/test_runner_lane.py`). What runs per call is
jaccpot's eager caps-only prepare at the initial state, the start force and the
segments. jaccpot's plan only grows: when a segment's lists outgrow their
capacities, jaccpot re-plans and re-runs it, and the next run from an initial
state compiles once for the grown plan; runs after that compile nothing. A run of `num_timesteps` not a multiple of the 50-step segments
compiles one more scan for its last segment (jaccpot warns), once per length.

Start the process with XLA's fusion autotuner off, as for any production run
(`XLA_FLAGS=--xla_gpu_experimental_enable_fusion_autotuner=false`; jaccpot warns
otherwise).

## Snapshots and diagnostics

With `return_snapshots=True` the segments are `K = num_timesteps //
num_snapshots` steps and snapshot `k` is the state after `k * K` steps, `k = 0
.. num_snapshots - 1`: the initial state first, the final state not among them,
at times `k * K * dt` -- the direct lane's snapshots (and, since 2026-10, the
coupler's and the differentiable lane's). The states are copied to the host at
the segment edges, so the device never holds the history; `SnapshotData`'s arrays
are NumPy arrays. `total_energy` (`K + W_self + W_ext`, jaccpot's O(N) fused-lane
potential: about one force evaluation per snapshot) and `angular_momentum`
(`sum m x x v`) are the runner's fp64 diagnostics.

A plain run returns the final state only and computes no diagnostics. For them,
or for restart files and jaccpot's `SnapshotWriter` / `DiagnosticsLog` callbacks,
call the lane directly:

```python
from jaccpot import SegmentConfig
from jaccpot.production import DiagnosticsLog
from odisseo.runner_coupling import integrate_runner_jaccpot

log = DiagnosticsLog("run/diag.jsonl")
out = integrate_runner_jaccpot(
    state, mass, config, params,
    segment=SegmentConfig(steps=50, restart_every=10, restart_path="run/restart.npz"),
    callbacks=(log,),
)
out.state          # final state
out.run.edges      # every edge: step, time, retries, wall time, diagnostics
out.run.baselines  # step 0's momentum, angular momentum, energy
```

## Across the GPUs of a node

With `fmm_mesh=True` the same runner runs in its multi-GPU mode (jaccpot
`docs/production.md`, "Multi-GPU mode"; lane `"fmm_mesh_runner"`,
`odisseo.mesh_runner_coupling`): one process drives every card of a 1-D mesh, the
particles are decomposed into Morton domains, each segment is one
`jit(shard_map(scan))` program on the fused multi-GPU lane, the domains are
re-balanced between segments, and a segment that outgrows a capacity on any card is
re-planned and re-run on all of them. `resolve_lane()` picks it for

- `fmm_mesh=True` with `fmm_preset="galaxy"`, or
- `fmm_mesh=True` with a `MeshRunnerOptions` as `config.mesh_options`;

a `MeshOptions` there keeps the older mesh lane (`"fmm_mesh"`,
`odisseo.mesh_coupling`), whatever the preset (`docs/source/parallelism.md`).

```python
from odisseo.mesh_runner_coupling import MeshRunnerOptions

config = config._replace(
    fmm_preset="galaxy",
    fmm_mesh=True,
    mesh_options=MeshRunnerOptions(ndev=2),  # None: every visible device
)
final = integrate(state, mass, config, params)  # a host NumPy array, input order
```

Everything this page says about the settings, the external field, the units,
snapshots and the kept solver holds there, with these differences:

| | one card (`"fmm_runner"`) | the mesh (`"fmm_mesh_runner"`) |
|---|---|---|
| `params.G`, `softening` | traced (geometric MAC) or the solver's (`dehnen_error`) | the solver's: another value builds another solver and compiles; `dt` and the potentials' parameters stay traced |
| the external field | `[N, 2, 3]` | each device's `(cap, 2, 3)` rows, padding rows masked out: every Odisseo potential is per particle and works unchanged |
| the final state | a device array | a host NumPy array in input order (gathered by id once, at the end) |
| snapshots | copied to the host at the edges | gathered by id to the host at the edges |
| `total_energy` | the fused lane's potential | the mesh potential, step 0 included; **NaN until jaccpot's fused mesh force has one** (S11 session 2, track E): the run warns once, the edges' diagnostics say why (`energy_unavailable`) |
| `step_callback` | passed through | refused (the mesh scan streams nothing) |
| a second call | compiles nothing | no new segment, start or repartition program (jaccpot caches them on the solver's engine, keyed by the plan), but each fresh run's decomposition and per-device prepares compile small programs again (46 at 1e6 on two A100s) |

`MeshRunnerOptions` holds only what the mesh adds: `ndev` (the first `ndev` of
`jax.devices()`; which cards a process sees is `CUDA_VISIBLE_DEVICES`, i.e.
`autocvd`), `repartition_every` (1: before every segment after the first, jaccpot's
default; 0: never), `cap_headroom` (rows per device over an even split; `None`:
jaccpot's 1.15) and `force` (`None`: the fused force; jaccpot's `DirectMeshForce`
is a partition-independent direct sum for small-N references, which also runs on
CPU devices). The fused force prepares on a GPU only, so without an injected force
a CPU backend is refused, as on one card. Gradients, block steps, active-particle
schedules and the options the single-card runner refuses are refused too.

For the edges' diagnostics and the mesh plan (shard capacity, live counts per
device, repartitions, cross-domain capacities, retries), restart files and jaccpot's
callbacks, call the lane directly:

```python
from jaccpot.production import DiagnosticsLog
from odisseo.mesh_runner_coupling import integrate_mesh_runner_jaccpot

out = integrate_mesh_runner_jaccpot(
    state, mass, config, params, callbacks=(DiagnosticsLog("run/diag.jsonl"),)
)
out.state                        # [N, 2, 3] on the host, input order
out.run.edges[-1].plan["mesh"]   # ndev, cap, live counts
```

Launch it as one process with the cards visible and `XLA_FLAGS` set before jax
starts (`docs/source/parallelism.md`, "One large simulation across the GPUs of a
node").

## Migration

What the 2026-11 switch (multi-node plan, wave 3) changed for a fixed-step
`FMM_ACC` run on one device:

| before | now |
|---|---|
| a config that sets nothing, on a CPU or below 200,000 particles: Odisseo's own leapfrog loop over jaccpot's coupler (`fmm_forward`), at `fmm_leaf_size` 16, `fmm_max_order` 4, `fmm_theta` 0.6 | the runner (`fmm_runner`) at the same leaf size, order, opening angle and MAC (profile `"config"`): no cell level floor, jaccpot's automatic leaf capacity and cubic Morton box, the fused lane's force; numbers move ([below](#what-moved)) |
| `fmm_preset="large_n_gpu"`: the coupler's loop on the large-N lane | the runner, profile `"config"`, fp32 force as before |
| `fmm_preset="galaxy"`, or the auto large-N profile on a GPU | unchanged: the runner at jaccpot's galaxy settings |
| the coupler's engine knobs (tree, near field, traversal, static shapes) | not read; warned once (above) |
| `fmm_refresh_every > 1`, `fmm_refresh_after_position_update=True`, an active-particle schedule | a `ValueError` naming what still runs them (`fmm_blockstep=True` for individual timesteps, `fixed_timestep=False` for the adaptive lane) |
| a MAC other than `"dehnen"` / `"dehnen_error"` | a `ValueError` (the adaptive lane keeps the others) |
| CPU backend: the runner refused it | the fused lane's CPU route (above) |
| `fixed_timestep=False` | unchanged: the adaptive coupler lane |
