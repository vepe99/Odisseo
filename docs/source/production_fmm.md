# Galaxy-scale runs on jaccpot's production runner

`integrate()` hands large shared-timestep FMM runs to jaccpot's
`ProductionRunner` (jaccpot `docs/production.md`): the fused single-GPU lane,
run as K-step `strict_run_v2(carry="particles")` segments with traced
parameters, capacity re-plans and diagnostics at the segment edges. Odisseo
builds the solver, maps its configuration onto it and packages the result
(`odisseo.runner_coupling`); the step itself is jaccpot's, so it costs what it
costs in jaccpot's harness (`bench/fused_memory_budget.py`).

## Which runs go there

`resolve_lane()` names the lane `"fmm_runner"` for an `FMM_ACC` run (not
`fmm_mesh`, `fmm_blockstep` or `fmm_differentiable`, which keep precedence) when

- `config.fmm_preset == "galaxy"`, or
- the automatic large-N profile applies: preset `"fast"` (the default),
  `fmm_auto_large_n_profile=True` (the default), a GPU backend and at least
  `fmm_large_n_min_particles` (200,000) particles -- **and** the runner runs the
  call: `fixed_timestep=True`, `fmm_refresh_every=1`,
  `fmm_refresh_after_position_update=False` and no active-particle schedule.
  Otherwise the auto profile keeps the coupler's `"large_n_gpu"` loop, as before.

`fmm_preset="large_n_gpu"` always keeps the coupler's loop. An explicit
`"galaxy"` with an option the runner does not run raises `NotImplementedError`
instead of falling back, and so does a CPU backend: the fused lane's prepare runs
on a GPU only (jaccpot's own CPU tests report the backend as `"gpu"` to run its
pure-JAX route; `tests/test_runner_lane.py` does the same). On a CPU, and below
the threshold, nothing changes.

**So the default changed:** `SimulationConfig()` with `FMM_ACC` on a GPU at
200,000 particles or more now runs jaccpot's galaxy settings -- leaf 64,
expansion order 6, opening angle 0.8, cell level floor 8, automatic leaf
capacity, the geometric Dehnen MAC, an fp32 force -- not `fmm_leaf_size=16`,
`fmm_max_order=4`, `fmm_theta=0.6`.

## Settings

The runner lane reads:

| field | |
|---|---|
| `softening`, `softening_kernel`, `params.G` | the solver's softening (Plummer-equivalent), kernel and G, exactly as the coupler passes them |
| `fmm_galaxy_settings` | a `jaccpot.production.GalaxySettings`; `None` is `GALAXY_SETTINGS` |
| `fmm_mac_type="dehnen_error"`, `fmm_adaptive_eps`, `fmm_mac_force_scale_mode` | Dehnen's eq (16a) MAC; any MAC other than `"dehnen"` / `"dehnen_error"` is refused |
| `fmm_large_n_force_fp32=False` with a float64 state | the fp64 force (without `fmm_galaxy_settings`) |
| `fmm_use_pallas` / `ODISSEO_FMM_USE_PALLAS`, the `fmm_large_n_*` environment knobs | as on the coupler |
| `external_accelerations` and the potentials' fields | the external field |

It does **not** read `fmm_leaf_size`, `fmm_max_order`, `fmm_theta`, the tree,
near-field, traversal or static-shape knobs: every `SimulationConfig` field has a
default, so a value "set by the caller" cannot be told from the default.
`fmm_galaxy_settings` is the way to change them:

```python
from jaccpot.production import GalaxySettings

config = config._replace(
    fmm_preset="galaxy",
    fmm_galaxy_settings=GalaxySettings(theta=0.6, max_order=8),
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
  traced arguments (not under `dehnen_error`, which takes the solver's own G);
- evaluates the external field through ONE pair of module-level functions,
  `external_acceleration` / `external_potential`, taking an
  `ExternalFieldParams`: the whole `SimulationParams` as traced leaves and the
  static configuration fields the potentials read as pytree metadata.

Another `dt`, `G` or set of potential parameters builds no solver and adds no
program to jaccpot's cache (`tests/test_runner_lane.py`). What runs per call is
jaccpot's eager caps-only prepare at the initial state, the start force and the
segments. A run of `num_timesteps` not a multiple of the 50-step segments
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
