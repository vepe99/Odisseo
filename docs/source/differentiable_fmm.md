# Gradients through the FMM

ODISSEO's direct acceleration schemes have always been differentiable: a
simulation run with `DIRECT_ACC` or `NO_SELF_GRAVITY` can be differentiated with
respect to any field of `SimulationParams`, including external-potential
parameters such as `params.NFW_params.Mvir`. The `FMM_ACC` scheme could not —
which mattered, because it is the only scheme that reaches galaxy N.

`odisseo.differentiable` closes that gap. With jaccpot's differentiable FMM
underneath, `jax.grad` now flows through an FMM simulation onto

* **external-potential parameters** — any field of `SimulationParams`,
* the **initial state** (positions *and* velocities), and
* particle **masses**.

## Why the forward lane cannot be differentiated

The production lane in `odisseo.jaccpot_coupling` is built for forward
throughput, and two of its choices are incompatible with autodiff:

1. It passes `params` as a **static** `jax.jit` argument, so a traced `params`
   fails with `Non-hashable static arguments are not supported` before any
   physics runs.
2. It drives jaccpot's `prepare_state` / `evaluate_prepared_state` pair from a
   host-side Python loop. `evaluate_prepared_state` reads the **prebaked**
   expansions of the cached prepared state and never touches the live positions,
   so even if a tracer did reach it, the self-gravity term would contribute
   *exactly zero* sensitivity.

Rather than silently return that zero, `integrate` now raises with a pointer to
this lane when a tracer reaches the forward FMM path.

## Quick start

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)   # the jaccpot lanes require it
from odisseo.differentiable import (
    prepare_differentiable_fmm,
    integrate_leapfrog_differentiable,
)
from odisseo.option_classes import FMM_ACC, SimulationConfig, SimulationParams

config = SimulationConfig(
    N_particles=len(mass),
    num_timesteps=20,
    acceleration_scheme=FMM_ACC,
    external_accelerations=(NFW_POTENTIAL,),
    fmm_differentiable=True,   # only needed for the integrate() entry point
)

# Build the tree ONCE, from concrete inputs, OUTSIDE the differentiated function.
plan = prepare_differentiable_fmm(state0, mass, config, params)

def loss(mvir):
    p = params._replace(NFW_params=params.NFW_params._replace(Mvir=mvir))
    final = integrate_leapfrog_differentiable(state0, mass, config, p, plan=plan)
    return jnp.sum((final[:, 0] - observed_positions) ** 2)

dloss_dmvir = jax.grad(loss)(mvir0)
```

The same run is available through the unified entry point, which also handles
snapshots:

```python
from odisseo.integration_api import integrate

final = integrate(state0, mass, config, params, fmm_plan=plan)
```

`config.fixed_timestep=False` routes to an adaptive diffrax solve
(`integrate_diffrax_differentiable`) instead, differentiated through
`config.diffrax_adjoint_method`.

## The fixed-topology contract

Gradients are **exact for the numeric pipeline at frozen topology**. The reverse
pass differentiates P2M, the centre-of-mass expansion centres, the M2M/M2L/L2L
translations, L2P and the near-field P2P, while treating every integer index array
as a constant: the Morton permutation, node membership, the M2L interaction list,
the near-field neighbour lists, and every MAC accept/reject decision. What is
dropped is the force's *implicit* dependence on position through "which cell a
particle lands in", which is nonzero only on the measure-zero set where a pair
crosses a MAC boundary. See `jaccpot/docs/differentiable_fmm.md` for the full
statement and its accuracy measurements.

Three practical consequences:

* **`prepare_differentiable_fmm` needs concrete inputs.** jaccpot's tree build is
  host-side and not traceable. Build the plan once outside the differentiated
  function; it raises rather than degrading if a tracer reaches it.
* **One `jax.grad` call integrates at fixed topology.** The tree cannot be
  refreshed inside the differentiated window — the analogue of
  `fmm_refresh_every` in the forward lane, except the refresh has to happen
  outside `jax.grad`. Keep `num_steps` short enough that particles do not stream
  out of their cells, and check with `topology_drift`:

  ```python
  from odisseo.differentiable import topology_drift
  print(topology_drift(plan, final[:, 0]))
  # {'max_displacement': 0.093, 'rms_displacement': 0.037,
  #  'max_displacement_over_leaf_extent': 0.014}
  ```

  Watch `max_displacement_over_leaf_extent`. Well below 1 means the frozen tree
  still describes the distribution; at or above 1 particles have crossed cells and
  the *forward* force is degraded too, not only the gradient.
* **A finite-difference reference must perturb the same plan.** FD over a run
  that rebuilds the tree disagrees wherever a pair crosses a MAC boundary.

## Configuration

| `SimulationConfig` field | default | effect |
|---|---|---|
| `fmm_differentiable` | `False` | Route `FMM_ACC` through this lane in `integrate`. |
| `fmm_grad_nearfield_lane` | `"auto"` | jaccpot's near-field reverse: `"bucketed"` below 100k particles, the leaf-major `"fast_lane"` at or above. The bucketed reverse OOMs at galaxy scale, so leave it on `"auto"` unless measuring. |
| `fmm_grad_fused_m2l_pallas` | `None` | Opt into the fused-Pallas M2L on the gradient path (Ampere+). |

Every other `fmm_*` knob is shared with the forward lane, so a differentiable run
and a forward run of the same `config` execute the same FMM configuration. For
anything finer, pass a `jaccpot.GradConfig` straight through:

```python
from jaccpot import GradConfig
plan = prepare_differentiable_fmm(
    state0, mass, config, params,
    grad_config=GradConfig(nearfield_lane="fast_lane", reverse_tiers=4),
)
```

## Accuracy

The FMM gradient is an exact gradient *of the FMM force*, so it matches the
direct-sum lane's gradient to the FMM's own force accuracy — not to machine
precision. Measured on the test problem in `tests/test_differentiable_fmm.py`
(N=96, order 6, θ=0.4, fp64, self-gravity dominant):

| comparison | relative agreement |
|---|---|
| AD vs finite differences of the **same frozen plan** | ~3e-8 |
| FMM-lane gradient vs direct-sum-lane gradient | ~1e-6 |

A tighter gradient needs a more accurate force: raise `fmm_max_order` or lower
`fmm_theta`. Note that the shipped `large_n_gpu` preset carries ~6 % force error
at order 4 on a clustered disc — that is the accuracy your gradients are gradients
*of*.

On GPU in fp32, a finite-difference cross-check is limited by the FD, not by the
gradient: with a loss of order 1e5–1e6, the loss *change* from a small parameter
perturbation lands at or below fp32 resolution and the central difference
quantises (or cancels to exactly zero). Measured at N=2000: AD and FD agree to
2.6e-3 with the FD pinned at a quantisation step. Do accuracy work in fp64 on CPU,
and use the GPU for the runs that need it.

## Cost

Measured on one A100 40 GB, fp32, clustered disc, order 4, θ=0.6, 3 leapfrog steps
(= 4 force evaluations), `fmm_grad_nearfield_lane="auto"`:

| N | leaf | prepare | forward (steady) | forward+backward (steady) | peak GPU memory |
|---|---|---|---|---|---|
| 2 000 | 32 | 34 s | 26 s | 191 s | — |
| 20 000 | 64 | 73 s | 42 s | 1 140 s | ~30 GB |

Two things to plan around:

* **The reverse is ~7–27× the forward**, and both `prepare` and the reverse are
  dominated by *compilation* — the second call is far cheaper than the first. Set
  `jax_compilation_cache_dir` for a cold process.
* **30 GB at N=20 000** says the bucketed near-field reverse gets memory-hungry
  well below jaccpot's `"auto"` crossover of 100 000 particles for this geometry
  and leaf size. If you are near an OOM at N ≳ 2e4, set
  `fmm_grad_nearfield_lane="fast_lane"` rather than waiting for `"auto"` to switch
  for you.

## Forward mode (`jax.jvp`, `jax.jacfwd`)

jaccpot's FMM kernels are `custom_vjp`s, so by default a jvp through them raises
(`can't apply forward-mode autodiff (jvp) to a custom_vjp function`). jaccpot's
`ad_mode="forward"` (jaccpot main 1bb30620 on) gives the **large-N lane**
(`fmm_preset="large_n_gpu"`) `custom_jvp` siblings whose primal is the production
kernel, bit for bit. Forward mode pays when the parameters are few (one tangent per
parameter, `jax.jacfwd` batches them) or the run is long (a jvp stores no residuals,
so its memory does not grow with the steps).

```python
config = config._replace(fmm_preset="large_n_gpu")      # the radix lane is reverse-only
plan = prepare_differentiable_fmm(state0, mass, config, params)

def observables(theta):                                  # theta = (Mvir, r_s)
    p = params._replace(NFW_params=params.NFW_params._replace(Mvir=theta[0], r_s=theta[1]))
    final = integrate_leapfrog_differentiable(state0, mass, config, p, plan=plan,
                                              ad_mode="forward")
    return final[:, 0].reshape(-1)

J = jax.jacfwd(observables)(theta0)                      # [n_obs, 2], one pass
fisher = J.T @ (J / sigma[:, None] ** 2)                  # Gauss-Newton Fisher
```

* **Selecting it.** `ad_mode="forward"` on `integrate_leapfrog_differentiable`,
  `differentiable_total_acceleration` or `differentiable_fmm_self_acceleration`.
  `integrate_diffrax_differentiable` derives it from
  `diffrax_adjoint_method=FORWARDMODE` (diffrax's `ForwardMode` adjoint
  differentiates by jvp). Through `integrate`, or anywhere else, wrap the call in
  `jaccpot.runtime.grad_options.ad_mode_override("forward")`: `ad_mode=None`, the
  default, follows it. The mode is read when the force is traced and is part of
  `jax.jit`'s cache key, so one plan serves `jax.grad` and `jax.jacfwd` side by side.
* **The radix lane is reverse-only.** Forward mode on a radix plan (`"fast"` /
  `"accurate"` presets) raises `NotImplementedError` up front.
* **First order only.** `jax.grad` of a forward-mode program and `jacfwd(jacfwd)` /
  `jax.hessian` are refused. So **the FMM Fisher matrix is `J^T C^-1 J`** from one
  `jax.jacfwd`: the Gauss-Newton (expected) Fisher of a Gaussian likelihood, not the
  Hessian of the log-likelihood that Odisseo's direct-sum Fisher notebooks
  (`notebooks/dev/albastross/Fisher_contour/`) take with `jacfwd(jacfwd)`. The two
  agree at the best fit for a well-specified model; away from it, the Hessian adds
  a residual-weighted second-derivative term that `J^T J` leaves out.
* **Gates** (`tests/test_differentiable_fmm.py::TestForwardModeOnTheLargeNLane`,
  slow; CPU with jaccpot's Pallas kernels in interpret mode and
  `jax.default_backend` patched to `"gpu"`, fp64, N=96): a diffrax `ForwardMode`
  jvp against the `RecursiveCheckpointAdjoint` gradient and central differences;
  `jax.jacfwd` over (`Mvir`, `r_s`) against `jax.jacrev`; the Fisher `J^T J`
  symmetric, positive semi-definite, equal to the `jacrev` one; `jacfwd(jacfwd)`
  refused.

## Limits

* **Active-particle scheduling is not supported.** It is a forward-throughput
  optimisation; the gradient lane integrates all particles every step.
* **Potentials are not available on the gradient path** — accelerations only.
* **Bare `jax.grad` is the supported usage.** jaccpot's inner kernels are already
  jit-compiled. An outer `jax.jit` around the whole integration works at moderate
  N but can hit host-side ops in jaccpot's re-run sweeps at large N.
* **Only the real basis exists.** `"real"` (the default; `"solidfmm"` is an alias)
  is differentiable. `"complex"` and `"cartesian"` were removed in jaccpot's
  2026-10 cleanup and raise at solver construction.
* The large-N path additionally needs the frozen M2L pair list retained. The plan
  builder requests it for you (`retain_far_pairs_for_grad=True`) — the forward
  lane still discards it to save memory.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Non-hashable static arguments` mentioning `SimulationParams` | differentiating the forward FMM lane | set `fmm_differentiable=True` and pass `fmm_plan=` |
| `NotImplementedError: traced params reached the forward FMM lane` | same, now reported deliberately | as above |
| `NotImplementedError: ... needs CONCRETE state/mass` | the plan was built inside the differentiated function | hoist `prepare_differentiable_fmm` out |
| gradient w.r.t. masses is exactly zero | the FMM self-gravity term is not being differentiated | check the plan came from `prepare_differentiable_fmm`, not the forward coupler |
| OOM in the backward at N ≳ 10⁵ | bucketed near-field reverse | leave `fmm_grad_nearfield_lane="auto"` |
| AD and FD disagree | FD perturbed a run that rebuilt the tree | FD the same frozen plan |
| `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` | `jax.jvp` / `jax.jacfwd` / diffrax `ForwardMode` through the reverse-mode kernels | a `large_n_gpu` plan and `ad_mode="forward"` (or `ad_mode_override("forward")`); see [Forward mode](#forward-mode-jaxjvp-jaxjacfwd) |
| `NotImplementedError: forward mode ... large-N lane only` | forward mode on a radix plan | build the plan with `fmm_preset="large_n_gpu"`, or use `jax.jacrev` |
