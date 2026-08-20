# Momentum-conserving individual timesteps (Nornax + Jaccpot)

`odisseo.blockstep_coupling` runs a **block-power-of-two individual-timestep KDK
leapfrog** on a **mutual (momentum-conserving) FMM**. It sits alongside
`odisseo.jaccpot_coupling`, which stays the right choice for shared-timestep
runs and is unchanged by this work.

The two lanes differ in one structural way, and everything else follows from it:

| | `jaccpot_coupling` | `blockstep_coupling` |
|---|---|---|
| force | target-centric FMM | mutual FMM (Dehnen 2014) |
| near field | gather: each target sums its neighbours | each leaf pair once, `+f` / `-f` |
| momentum residual | ~1e-17 measured, but not by construction | ~1e-17, structurally |
| per-level antisymmetric split | **not expressible** | `level_accelerations(level=k)` |
| timestepping | one global `dt` | rungs `0 … k_max`, `dt_k = dt_max / 2**k` |
| legal for a block step | **no** | yes |
| external potentials | yes | no — self-gravity only |

## Why the existing coupler cannot be reused

A block-step KDK splits interactions by level `k = max(rung_i, rung_j)` and
requires each level's contribution to be applied *antisymmetrically*, so that an
**inactive coarse partner of an active fine interaction still receives its
equal-and-opposite kick**. That is not a diagnostic to check afterwards; it is
the scheme's defining correctness property (Dehnen 2014; Farr & Bertschinger
2007).

The load-bearing gap is that the production coupler cannot **express** that
split at all. It computes one total acceleration per target; restricting
*sources* to a rung subset, and delivering the reaction to a partner that is not
being integrated this sub-step, has no place in its API.
`jaccpot.BlockStepFMM` evaluates each pair once, applies both signs, and takes a
per-level weight vector into the traversal.

### A claim this integration did *not* reproduce

jaccpot's own documentation motivates the mutual restructure by stating that the
target-centric force's momentum residual sits at the force accuracy
(~1e-3 … 1e-5), because every pair is evaluated twice and the two evaluations
round differently. **That did not reproduce here.** Measured through
`evaluate_acceleration_jaccpot`, `theta = 0.7`, order 4, leaf 32, float64:

| N = 20 000 Plummer | mutual | target-centric |
|---|---|---|
| momentum residual, equal masses | 2.98e-17 | 2.55e-17 |
| momentum residual, unequal masses | 2.62e-17 | 3.11e-17 |
| force difference between the two | — | 2.8e-3 |

The 2.8e-3 confirms the far field really is active and approximate, so this is
not a degenerate direct-sum configuration; the residual simply cancels on both
paths. The same holds at N = 256 in the test suite. A plausible reading is that
jaccpot's dual-tree traversal already evaluates each accepted cell pair once and
seeds both endpoints' local expansions, which is the mechanism the mutual far
field relies on — but this integration has not gone and confirmed that in
jaccpot's traversal, so it is stated as a measurement, not an explanation.

Two consequences, both acted on here:

* the test suite asserts the **direction** (the target-centric lane must not come
  out *better*) rather than an order-of-magnitude gap that measurement does not
  support, and
* the case for this lane rests on the per-level split, not on total momentum. A
  force can conserve total momentum and still be illegal for a block step.
  `test_each_level_conserves_momentum_on_its_own` is therefore the assertion that
  matters, since it is the one the production coupler could not satisfy even in
  principle.

## Quick start

One entry point, and the fast lane by default:

```python
from odisseo import construct_initial_state
from odisseo.blockstep_coupling import BlockStepOptions, integrate_blockstep_jaccpot
from odisseo.option_classes import SimulationConfig, SimulationParams

config = SimulationConfig(N_particles=n, softening=1e-3)
params = SimulationParams(G=1.0)
options = BlockStepOptions(dt_max=1.2e-2, k_max=3, theta=0.7, max_order=4)

result = integrate_blockstep_jaccpot(
    state, mass, config, params, options=options, n_base=32
)
print(result.momentum_drift.max())    # ~1e-17
print(result.energy_drift[-1])        # bounded, oscillating
print(result.seconds_per_base_step)
```

`options.topology_backend` defaults to `"auto"`, which is `"device"` wherever the
installed jaccpot can build the topology in JAX. `integrate_blockstep_jaccpot`
then **routes to the fully jitted lane**: the topology is rebuilt on device inside
one `lax.scan` over the whole rollout, so there is no host round trip anywhere in
the loop. That is 56x per base step at N = 20 000 and 96x at N = 100 000 against
the host lane, with the same momentum residual.

The three values are not interchangeable:

| `topology_backend` | behaviour |
|---|---|
| `"auto"` (default) | `"device"` if jaccpot provides it, else `"host"` |
| `"device"` | device, and **raises** if jaccpot cannot provide it |
| `"host"` | the NumPy dual-tree traversal, a Python loop with a host round trip per base step |

`"auto"` and `"device"` differ in exactly one way, and it is the one that matters:
`"auto"` may quietly fall back, `"device"` may not. A silent fallback presents as
a slow machine, not as a missing dependency.

Call `integrate_blockstep_jitted` directly if you want to be explicit, or to reach
`time_steady_state=True` (report the warm timing rather than the compile):

```python
from odisseo.blockstep_coupling import integrate_blockstep_jitted

result = integrate_blockstep_jitted(
    state, mass, config, params,
    options=BlockStepOptions(
        dt_max=4.3e-2, k_max=3, theta=0.7, max_order=4,
        topology_backend="device",
    ),
    n_base=32,
    time_steady_state=True,
)
```

The host lane stays reachable with `topology_backend="host"`, and is the only one
that works against a jaccpot without the device topology. It is also the only lane
that takes `block_state`, `record_every` and `progress` -- those exist because it
re-enters the host once per base step, which is precisely what the jitted lane
removes, so passing them with a device backend raises rather than being ignored.

### Through the unified `integrate()`

`integrate()` reaches this lane too, selected the same way the differentiable FMM
lane is:

```python
from odisseo.integration_api import integrate
from odisseo.option_classes import FMM_ACC

config = SimulationConfig(
    N_particles=n, softening=1e-3,
    acceleration_scheme=FMM_ACC,
    fmm_blockstep=True,                        # the selector
    blockstep_options=BlockStepOptions(...),   # required: no default dt_max
    num_timesteps=32,                          # BASE steps -- see below
)
final_state = integrate(state, mass, config, params)
```

Two things to know before using it.

**`blockstep_options` is required.** There is no default, because
`BlockStepOptions` has no default `dt_max`, and a `dt_max` below every
particle's own criterion puts the whole system on rung 0 -- the block scheme
then collapses to a shared timestep with no symptom at all. Size it from the
acceleration distribution; `tools/blockstep_fmm_demo.py` takes the 90th
percentile of `eta*sqrt(softening/|a_i|)`.

**`num_timesteps` counts base steps here**, and one base step contains `2**k_max`
sub-steps of the finest rung. At `k_max=3`, `num_timesteps=32` is 32 base steps,
not 32 force evaluations and not 32 advances of `dt_max`. Nothing about the physics
changes; the accounting does, and it is the one thing that does not carry over from
the other lanes.

**It returns a plain final state**, like every other `integrate()` lane, so a
caller never has to branch on the config to know what it was handed. That means the
per-base-step diagnostics -- momentum and energy drift, the rung histograms, the
timings -- are *dropped*, and they cannot be recovered from the state afterwards.
Call `integrate_blockstep_jaccpot` directly whenever you want them; it is the
richer interface, not a lower-level one.

`fmm_blockstep` takes precedence over `fmm_differentiable`, and the external
potential guard still applies -- this lane is self-gravity only.

A worked example with both lanes side by side, a Plummer or Hernquist IC, and
the full diagnostic table:

```bash
python tools/blockstep_fmm_demo.py --n 100000 --k-max 3 --n-base 12
python tools/blockstep_fmm_demo.py --n 20000 --ic hernquist --backend pallas
python tools/blockstep_fmm_demo.py --n 100000 --lane host --no-shared
```

## Which nornax this needs

The fused-boundary primitive lives on nornax **`main`**, at or after commit
`8fe9dbd` — *"Differentiable individual-timestep KDK leapfrog integrator (#7)"*,
which squash-merged the block-step integrator **and** the fused-boundary work
(#8) and the scanned fused path. Earlier `main` commits (`4a72780` and before)
have neither, and `nornax.solvers.fused_boundary_model` is the thing to probe
for:

```python
from nornax.solvers import fused_boundary_model  # ImportError on older nornax
```

The test module skips itself when that import fails, rather than erroring.

Note that the topic branch `feat/block-step-kdk-leapfrog` is *behind* `main` for
this purpose — `main` carries the same block-step files verbatim plus the
Hermite-6/8 adapter work. Build against `main`.

## The things that are easy to get wrong

### 1. Topology lifetime

`BlockStepFMM.prepare(positions, masses)` runs a host-side dual-tree traversal
and **cannot be traced**. It is called once per base step, matching the cadence
at which nornax reassigns rungs — the two discrete refreshes are meant to line
up. Within a base step every one of the `2**k_max + 1` boundaries reuses the same
frozen tree.

That host call is also why the rollout is driven from a Python loop over base
steps: nornax's `block_kdk_rollout` scans its base steps, so a tree rebuild
cannot happen inside it. `rebuild_every` sets how many base steps share one tree,
and the driver walks them with `block_kdk_base_step`. Setting
`scan_base_steps=True` hands each interval to `block_kdk_rollout` instead — the
same trajectory, in one `lax.scan`; it is not the default for the reason in §3
below, and `test_scanned_and_eager_base_step_drivers_agree` pins that the two
agree.

`test_the_tree_is_rebuilt_once_per_rebuild_interval` pins the count, because
getting it wrong is the difference between an O(N) run and one that rebuilds a
tree per sub-step — and it is invisible in the results.

### 2. Fusion must be *selected*, not merely available

`BlockStepFMM` satisfies nornax's `FusedMutualForceModel` with no adapter
changes, and `advance_base_step` opts in automatically via
`fused_boundary_model(force, k_max)`. A silent fallback to the per-level path
still produces correct answers while paying one tree traversal per active level
instead of one per boundary — 19 against 9 at `k_max = 3`. **No correctness test
can catch that**, so every entry point calls `assert_fused_boundary_selected`
before it steps.

### 3. The boundary walk — where nornax's default is wrong for this backend

Having opted into fusion, nornax asks a *second* question: does the model accept
a **traced** `level_weights` vector? If so it walks the `2**k_max + 1` boundaries
with a `lax.scan` instead of unrolling them. `BlockStepFMM.boundary_kick` does
accept one, so nornax's signature probe says yes.

For a tree rebuilt every base step that is the wrong answer. The scan inlines the
whole force into a single program, and because the topology constants change at
every rebuild, that program is **recompiled every base step**. Measured here:

| | s / base step | s / prepare |
|---|---|---|
| boundaries scanned | 18.6 | 6.2 |
| boundaries unrolled | **10.4** | **2.1** |

*(N = 512, `k_max = 2`, float64, CPU, 5 base steps.)* On an A100 at N = 20 000 the
scanned path spent 218.6 s on its first base step against 35.7 s unrolled, at
otherwise identical settings.

`BlockStepOptions.traced_boundary_weights` therefore defaults to `False`, which
declares the answer to nornax explicitly and restores the unrolled loop. Set it
to `True` when *trace size* is what binds (an outer `jax.jit` over the rollout,
or a `k_max` deep enough that `2**k_max` unrolled kicks stop fitting), and to
`None` to hand the decision back to nornax's probe.

Note that `jit_force` below is **not** the same knob pointing the other way. It
compiles one program per *topology* and reuses it across every boundary and every
base step of the interval; the boundary scan compiles one program per *base step*,
because nornax's scan body closes over the schedule rather than the tree. Both can
be on at once, and the two compile counts are independent.

The same trade governs `scan_base_steps`, one level up: handing a rebuild
interval to `block_kdk_rollout` puts a `lax.scan` around the base steps and
inlines the force again. It defaults off for the same reason, so
`block_kdk_base_step` is what the driver calls unless you ask otherwise. Peak
**memory**, not trace size, is what breaks CI on this path: jaccpot measured
2.08 GB unrolled against 2.67 GB scanned for the same rollout, with `N` barely
moving either number — it is compile/executable memory, not data.

### 3b. The single largest performance lever: `jaccpot.mutual` has no `jax.jit`

Not one `@jax.jit` appears in `jaccpot/mutual/`, so every force evaluation
dispatches op by op. Measured on one A100-40GB, N = 20 000, float64,
`theta = 0.7`, order 4, leaf 32:

| one mutual traversal | seconds |
|---|---|
| eager (as jaccpot ships) | 5.1 – 5.6 |
| under `jax.jit` | **0.038 – 0.040** |

**135×**, and 38 ms is the figure jaccpot's own benchmark quotes for this force —
so the eager number is the anomaly, not the jitted one.

End to end through this lane, same GPU, N = 20 000, `k_max = 3`, 8 base steps
under one frozen tree (`--rebuild-every 8`):

| | eager | `--jit-force` |
|---|---|---|
| s / base step (steady state) | 52.46 | **0.475** — 110× |
| total wall, 8 base steps | 433.1 s | **210.5 s** |
| peak host RSS | 4.70 GB | **3.02 GB** |
| momentum drift | 2.9e-19 | 8.7e-19 |
| rung histogram, first & last | identical | identical |

The 110× per step collapses to 2.1× on the total because the ~207 s compile is
paid once and 8 base steps is barely enough to amortise it. Note the peak RSS
went *down*, not up: one compiled program replaces jaccpot's many separately
compiled kernels.

`BlockStepOptions.jit_force` turns it on through `JittedMutualForce`, which
routes every entry point (boundary kick, total acceleration, single level)
through one compiled `sum_k w_k a_k` kernel. Deriving the static
`active_floor`/`half` form into a *weight vector* is what keeps it to one
program: left as static arguments they would key the jit cache and compile a
separate program per floor × half.

It is **off by default** because the frozen topology is baked in as constants, so
each rebuild pays a fresh compile — measured 213 s, then 171 s on the next
topology, so it does not warm up. The crossover is

```
rebuild_every * (2**k_max + 2) > compile / (eager - jitted)  ~= 34 traversals
```

i.e. `rebuild_every >= 4` at `k_max = 3`. Turn it on for long runs at a held
topology and for gradient work, where the topology is frozen anyway.

### 3c. Static shapes — the compile-per-topology ceiling, lifted

Two changes were needed, and only one of them is the `jax.jit`.

**(1) The topology must arrive as a traced pytree argument.** `MutualFMMState`
and `MutualTreeArrays` are now registered pytrees, so the state is a jit
*argument* and the program is keyed on its shapes rather than on constant values.
That works on its own — parity 1.6e-19 against unmodified jaccpot across 42
arrays covering totals, per-level accelerations, boundary kicks, base steps and
gradients — and on its own it buys **no compile reuse at all**, because the
shapes move. Measured over a 12 base-step Hernquist rollout at N = 4096:

| per rebuild | min | max | spread |
|---|---|---|---|
| near-pair count | 8040 | 8118 | 1.0% |
| **far-pair count** | 10 | 74 | **640%** |
| tree depth | 22 | 24 | 9% |
| widest tree level | 20 | 24 | 20% |
| leaves | 128 | 128 | fixed |
| `max_leaf_size` | 32 | 32 | fixed |

**(2) The shapes must be padded to capacities.** `near_a`/`far_a` are padded to
caps (the `near_valid`/`far_valid` masks already existed and were already
honoured — they were simply all-true), and `level_nodes`/`level_parents` went
from variable-arity *tuples* — whose arity is the tree depth, and therefore part
of the pytree structure — to a dense `(depth_cap, width_cap)` block plus a
validity mask. The four Python loops over tree levels became one `lax.scan`.

What makes this viable is that the leaf and node counts are *already* structurally
fixed: `num_leaves == ceil(N / leaf_size)` and `max_leaf_size == leaf_size`,
because the tree builder slices the Morton order into fixed-width buckets. Only
the MAC outcomes and the internal-node linkage are distribution-dependent.

`BlockStepOptions(jit_force=True)` enables both. Measured, A100, N = 20 000,
`k_max = 3`, 6 base steps, tree rebuilt **every** base step:

| | eager | `jit_force` |
|---|---|---|
| seconds / base step | 114.25 | **0.504** — 227× |
| total wall, 6 base steps | 900.6 s | **68.6 s** |
| distinct compiled programs | — | **1** |
| momentum drift | 3.16e-18 | 3.12e-18 |

Only ~3 s of that 68.6 s total is stepping. The rest is the one-time compile plus
the **host-side tree build**, which is what Phase 2 (a device-side mutual
traversal) attacks.

#### Capacity sizing is a real knob, not bookkeeping

The kernels do work proportional to the *capacity*: the M2L chunks its directed
pair list at a fixed budget, so a far cap 8× the occupancy means 8× the M2L
chunks. A first cut used a flat 4× headroom on the far list — sized from the 640%
relative drift above — and resolved a cap of 262144 for 32830 real far pairs at
N = 20 000. Retuning to **additive plus relative** headroom (`+10% + 256`, snapped
onto a 1/1.5/2 × 2^k ladder) gives 49152 instead, and took the per-base-step time
from 1.344 s to 0.504 s — 2.7× on top of the jit.

The additive term is what matters at small counts. In *absolute* terms the
measured drift is +78 near pairs and +64 far pairs; a single multiplicative factor
cannot cover both a list of 10 and a list of 32830.

#### Two traps, both now pinned by tests

**Aux data is part of the jit cache key.** The per-rebuild occupancy counters
(`num_near_pairs`, `num_far_pairs`) were first added as aux data, which re-keyed
the treedef and produced three compiles for three rebuilds — caught by the
compile-count test, not by any correctness test. They are pytree *children* (0-d
arrays) for that reason.

**After padding, `far_a.shape[0]` is the capacity, not the count.** A topology
with zero far pairs still reports a nonzero shape, so the vacuity guard from §
"Two traps the tests are built around" silently stopped guarding. Use
`state.num_far_pairs` / `num_near_pairs`;
`test_a_no_far_pair_configuration_is_rejected` now builds the
occupancy-zero-capacity-nonzero state that separates the two.

> **Retraction.** An earlier version of this page claimed the shapes were
> *already* stable, citing two rebuilds that returned identical pair counts. That
> measurement perturbed positions by adding a scalar to every coordinate — a
> **rigid translation**, which leaves the tree invariant by construction. Any
> future check must perturb non-rigidly;
> `test_a_rigid_translation_does_not_change_the_topology` in jaccpot now asserts
> the invariance so the trap cannot be re-entered by accident.

### 3d. The last host round-trip — the topology now builds on device

With §3b and §3c in place the force costs ~0.55 s per base step at N = 20 000 and
the **host tree build costs 22 s**. It is the whole remaining wall, and it is a
device-to-host round trip: `jaccpot/mutual/topology.py` is NumPy end to end — six
D2H transfers, a scalar Python loop over every node for the centre/radius pass, a
host BFS for the depths, and a NumPy wavefront dual-tree walk.

`BlockStepOptions(topology_backend="device")` replaces all of it with JAX:

| | |
|---|---|
| host tree build | **22.0 s** |
| device tree + topology build, jitted | **0.0062 s** |
| | **3750×** |

and because the device build is traceable it *fuses* with the force, so
build+force together (0.0393 s) come out **below** the force alone against a host
topology (0.0511 s).

The pieces, and what was already there to reuse:

* **`yggdrax.interactions.dual_tree_walk_mutual`** — new. yggdrax's production
  `_dual_tree_walk_impl` was already mutual in every respect that matters:
  canonical `a ≤ b`, the same `(L,L)/(L,R)/(R,R)` diagonal split, the same
  split-the-larger heuristic, and a MAC that *is* the symmetric mutual one. Only
  its **emitter** was wrong — it scatters each accepted pair into *both*
  endpoints' CSR rows, which loses the pairing identity the `+f`/`−f`
  antisymmetry needs. So this is a separate emitter, not a separate algorithm,
  and a separate function rather than a mode, which leaves the production
  traversal and everything built on it untouched by construction. It reuses the
  module-level `_COUNT_REFINE_VM` so the refinement case analysis cannot drift
  between the two.
* **`jaccpot.mutual.device_topology`** — new. `node_centers_and_radii`,
  `node_depths`, `dense_level_schedule`, `leaf_blocks`,
  `build_mutual_state_device`.
* **`yggdrax._tree_impl.rebuild_static_radix_tree_from_template`** — already
  existed and is already traceable (Morton encode → `argsort` →
  `template._replace`). This is what makes the *particle order* refresh on device.

#### Radii must be exact, not a bounding-sphere merge

The obvious way to get per-node radii on device is the bottom-up merge
`r_parent = max_child(|c_child − c_parent| + r_child)`. That is an **upper
bound**, and a looser radius changes MAC outcomes, which changes the accepted
pair set, which changes the force. The host definition is the exact
`max_i |x_i − c_n|`, and it is reproduced by walking *up* from every particle
with one scatter-max per tree level — `O(N · depth)`, vectorised, fixed shapes.

#### Accuracy is unaffected — but do not compare the two lanes to each other

The backends use different trees (LBVH against a static-radix template), so they
differ from *each other* at ~8e-3. That number means nothing. Against an exact
direct sum at N = 20 000, θ = 0.7, order 4:

| | vs exact |
|---|---|
| host (LBVH) | 2.08e-3 |
| device (static-radix) | 2.11e-3 |

Both sit at the FMM's own tolerance. The device tree is *shallower and wider*
(depth 16 / width 768 against 48 / 128), which is what a balanced bisection over
leaf buckets gives, and it costs nothing in accuracy here.

#### Gradients

The topology build sits under `stop_gradient` — but only the copy of `positions`
that feeds the **MAC geometry**. The copy that feeds the upward sweep stays live,
because the expansion centres are recomputed from live positions on every
evaluation and carry a real gradient term; freezing those would silently drop it.
That is the same split `jaccpot/distributed/fmm.py` makes. Measured gradient
parity against the host-built state: **2.5e-16**.

### 3e. `integrate_blockstep_jitted` — the whole rollout as one program

`integrate_blockstep_jaccpot` cannot host an in-scan rebuild: nornax's
`block_kdk_rollout` scans its base steps and threads `args` unchanged, so a
per-base-step topology has nowhere to live — it would have to be in the scan
**carry**, and nornax has no hook for that.

`integrate_blockstep_jitted` therefore drives nornax's *primitives* — `n_sub`,
`boundary_weight_table`, `assign_rungs`, all pure and already device-side and
explicitly meant to be consumed this way — and carries the topology alongside the
state. Measured, A100, N = 20 000, `k_max = 3`, float64, 4 base steps:

| steady state, per base step | |
|---|---|
| eager, host topology (the original baseline) | 52.46 s |
| host lane, `jit_force` (0.55 s stepping + 21.96 s tree build) | 22.51 s |
| **jitted lane** | **0.3988 s** |
| | **56.5×** over the host lane, **132×** over the baseline |

with momentum drift **3.7e-18**, energy drift +2.5e-6, one jit cache entry, and
the rung ladder fully populated (2222 / 2950 / 4980 / 9848). One-off costs: 26 s
to freeze the template, 46 s to compile the rollout.

`test_the_jitted_lane_matches_the_host_loop_lane` pins the two lanes against each
other on the **rung histogram**, which must match exactly — same `assign_rungs`,
same schedule — while allowing the trajectories to differ at FMM tolerance,
because the trees differ.

> **Read the timings carefully.** `seconds_per_base_step` on the jitted lane
> divides the whole measured rollout by `n_base`, so by default it includes the
> one-time compile; on the host lane it *excludes* the host tree build, which is
> reported separately in `prepare_seconds`. Compared naively, the jitted lane
> looks **ten times slower** while actually being sixty times faster. Pass
> `time_steady_state=True` for the warm figure — it is the only one worth
> quoting.

#### Overflow is silent, so it is raised

A capacity that does not cover the topology truncates the pair list or the level
schedule. Nothing downstream notices: no NaN, no shape mismatch, and momentum
stays **exactly** conserved, because dropping a canonical pair drops both of its
halves. It first showed up as an unexplained 2e-2 force error, caused by reusing
the host lane's capacity profile (LBVH tree) for the device lane (static-radix
tree) — different depth and width, so the level schedule truncated.

`MutualFMMState.topology_overflow` now carries the flag out, with
`overflow_causes` naming which of far / near / pair-queue / level-width /
tree-depth blew. `prepare()` raises on it, and the jitted lane reduces the
per-base-step flags and raises after the rollout. **Never reuse a capacity
profile across topology backends.**

### 4. Rung range

`BlockStepFMM` **rejects** rungs outside `[0, k_max]` rather than clamping;
nornax's `assign_rungs` **clips** into `[0, k_max]`. They agree only because
ODISSEO hands both the same `options.k_max`, which
`integrate_blockstep_jaccpot` checks explicitly against the model's own.

## What is asserted

`tests/test_blockstep_fmm.py`:

* **Momentum** — `|sum_i m_i v_i - p_0| / sum_i |m_i v_i| < 1e-13` across a
  multi-rung rollout, and *each level on its own* conserves momentum, since the
  block step applies one level at a time.
* **Momentum is structural, not a tolerance** — the residual is swept over
  `theta ∈ {0.5, 0.6, 0.8}` × order `∈ {2, 4}`, which moves the force error by
  orders of magnitude, and must not move with it. This is the test that catches a
  kernel recomputing `dr` for the second endpoint instead of negating the
  first — the one change that breaks exact cancellation while leaving every
  accuracy number untouched.
* **Energy** — bounded *and* oscillating. A bound alone is satisfied by a slow
  one-way accumulation, so the sign of the drift must change at least once.
* **The rungs must actually buy something** — at a fixed `dt_max` where this IC
  has an under-resolved close encounter, adding one rung takes max `|dE/E|` from
  1.26e-3 to 7.12e-5, an 18× improvement, with 27 of 256 particles on the new
  rung. Every other test here would pass with `k_max = 0`; this one would not.
* **Against the shared-timestep lane** — the two forces agree on the **total**
  acceleration to ~1e-3. Only the total: the mutual far field assigns each cell
  the rung of its finest particle and splits at cell granularity, while any
  per-pair split is finer. Both are genuine partitions, so the totals must match
  even though the decompositions do not.
* **Cheap oracle** — nornax's `MutualDirectSumGravity`, momentum-exact by
  construction, at small N.
* **Differentiability** — `d(summary)/d(IC)` through a two-base-step rollout
  against finite differences of the *same* frozen plan, with `reassign_rungs`
  off. FD over a run that rebuilds the tree or reassigns rungs disagrees whenever
  a pair crosses a MAC boundary, and the disagreement is not a gradient error.
* **The optimisations are not different physics** — the scanned and unrolled
  boundary walks, the scanned and eager base-step drivers, and the jitted and
  eager force lanes are each pinned against one another to round-off, and
  `jit_force` is additionally checked to compile exactly one program for all
  three of its entry points and to drop it on every `prepare`.

### Two traps the tests are built around

**A test that passes for the wrong reason.** A configuration with no far pairs
makes the FMM a direct sum, so accuracy assertions pass at 1e-16 while testing
nothing. Every far-field number goes through `_assert_far_field_is_exercised`
first. This is not hypothetical: a single N = 256 Plummer sphere yields **zero**
far pairs at every `theta` up to 1.1 and every leaf size tried — the tree is
simply too shallow for the MAC to fire.

The fix was structural rather than a looser `theta`. The suite's IC is **two
well-separated Plummer clumps**, which puts a genuinely well-separated node pair
in the tree at ODISSEO's production `theta = 0.6`: 18 far pairs, 96 near, and a
2e-4 force error against an exact direct sum. Loosening `theta` to 0.9 on the
single sphere would have bought far pairs at the price of a 4e-3 force error, at
which point no meaningful accuracy tolerance can be asserted at all.

The clump masses are also deliberately **unequal**. With equal masses and a
near-field-only force, a target-centric gather is accidentally antisymmetric to
the last bit — the prefactor `G m_j / r^3` is the same number for both endpoints
— so a momentum comparison would pass for a reason unrelated to the mutual
restructure.

Note also that the mutual MAC (`theta * |c_B - c_A| > R_A + R_B`) is symmetric
and therefore **stricter** than the target-centric `R_source / d < theta` at the
same numeric value.

**`isinstance(x, jax.core.Tracer)` is not "can I read this value".** A
*concrete* array closed over by a `lax.cond`/`lax.scan` branch is not a Tracer,
yet reducing it still yields one inside the trace, so `int(...)` raises. The rung
validation attempts the read and catches `jax.errors.JAXTypeError`;
`test_rung_validation_survives_a_concrete_array_closed_over_by_a_trace` pins it.

## What the individual timesteps actually buy

Read the wall-clock comparison carefully, because the naive expectation is wrong.
Jaccpot's fused boundary kick applies `level_weights[max(rung_i, rung_j)]` as a
scalar *inside* the kernel — it **weights** pairs, it does not **prune** them —
so every boundary costs a full traversal regardless of how many levels are
active. Per `dt_max` of physical time:

| | traversals | host tree builds |
|---|---|---|
| block step, `k_max = K` | `2**K + 1` | `1` |
| shared timestep at `dt_min` | `2**K` | `2**K / refresh_every` |

So the advantage over a shared step *that resolves the finest particle* is **not**
fewer force evaluations. It is one host-side tree build per `dt_max` instead of
one per sub-step — a real cost, measured at 22 s against 5 s per traversal at
N = 20 000 — plus a scheme whose per-level splitting is exact. A shared-timestep
run is only cheaper if it is allowed to take `dt_max` steps, which is exactly the
accuracy the fine particles cannot afford.

`tools/blockstep_fmm_demo.py` measures all of this rather than asserting it, and
prints the shared-timestep lane over the same physical time for comparison.

### Measured end to end

**With the device topology and the jitted lane** (§3d, §3e) — A100, float64,
`theta = 0.7`, order 4, leaf 32, softening 1e-3, N = 20 000, `k_max = 3`:

| steady state, per base step | |
|---|---|
| eager, host topology | 52.46 s |
| `jit_force`, host topology (0.55 s stepping + 21.96 s tree build) | 22.51 s |
| **`integrate_blockstep_jitted`, device topology** | **0.3988 s** |

momentum drift 3.7e-18 · energy drift +2.5e-6 · one jit cache entry · rungs
2222 / 2950 / 4980 / 9848. One-off: 26 s template freeze + 46 s compile.

The numbers below predate §3d and are kept because they are what the host lane
still costs, and because the shared-timestep comparison was only ever run there.



One A100-40GB (shared with other jobs; the device was idle for these runs),
float64, `theta = 0.7`, order 4, leaf 32, softening 1e-3, eager force lane:

| | N = 20 000 Plummer, `k_max = 2` | N = 100 000 Hernquist, `k_max = 3` |
|---|---|---|
| far / leaf pairs | 32 830 / 99 586 | 341 504 / 729 286 |
| rung occupancy | 2000 / 7711 / 10289 | 10000 / 13169 / 23368 / 53463 |
| **momentum drift** | **3.5e-18** | **1.7e-18** |
| energy drift | +2.3e-6 over 6 base steps | +2.5e-6 over 3 base steps |
| s / base step | 86.5 | 228.6 |
| s / tree build | 22.4 | 31.1 |
| peak host RSS | 6.2 GB | 6.6 GB |

At N = 100 000 one force evaluation replaces 10¹⁰ direct pairs, and the rung
ladder is fully populated — 53 463 of 100 000 particles on the finest rung,
10 000 on the coarsest.

Against the shared-timestep lane over the *same physical time*:

| | N = 8 000, `k_max = 2` | N = 20 000, `k_max = 2` |
|---|---|---|
| block step, wall per `dt_max` | 65.9 s | 86.5 s |
| shared at `dt_min`, wall per `dt_max` | 75.6 s | 75.6 s |
| block step, host tree builds | 3 | 6 |
| shared, host tree builds | 12 | 24 |
| block step, energy drift | −1.7e-7 | +2.3e-6 |
| shared, energy drift | +3.8e-5 | +6.5e-5 |
| the two forces agree to | 2.3e-3 | 2.8e-3 |

At N = 8 000 the block step is both cheaper *and* 224× more accurate in energy;
at N = 20 000 it is 14% dearer for 28× the energy accuracy. The tree-build rows
are where the individual timesteps show up directly: one build per `dt_max`
against one per sub-step, at 21–22 s a build.

The per-base-step numbers are dominated by the eager force dispatch above, not by
the physics: at 5.1 s per traversal, six traversals account for ~31 s of the
86.5 s at N = 20 000, with the rest in the tree build and jaccpot's own host-side
work.

## Limitations

* **Self-gravity only.** An external potential has no equal-and-opposite partner
  inside the system, so it breaks the exact momentum statement. A `config` with
  `external_accelerations` set is rejected rather than silently ignored; use
  `odisseo.jaccpot_coupling` or `odisseo.differentiable` there.
* **float64 is the supported dtype.** In float32 the pair antisymmetry itself is
  unaffected — the negation `-fl(dr)` is exact at any precision — but the
  *reduction order* degrades and the residual lands at ~1e-8 rather than ~1e-17.
* **Gradients are fixed-topology.** `jax.grad` through a rollout works with the
  tree rebuilt per base step and `stop_gradient`-ed, the same treatment nornax
  gives its rung schedule.
* **`backend="pallas"` silently loses `d/d(dt_max)`, `d/d(softening)` and
  `d/d(G)`.** Its reverse rule returns `jnp.zeros_like(level_weights)` (and the
  same for `softening_sq` / `g_value`), on the stated grounds that the level
  table is "discrete or frozen". It is neither: `level_weights[k] ==
  half * dt_max / 2**k` is smooth in `dt_max` and the force is *linear* in it, so
  the near field's entire contribution is dropped. Measured: `d/d(dt_max)` comes
  out **111× too small** on the Pallas backend (only the far field, which stays
  pure JAX, contributes), against an exact match on `backend="jax"`. Forward
  results are unaffected, and
  `test_the_pallas_backend_drops_most_of_the_dt_max_gradient` pins the
  discrepancy so it fails the moment upstream fixes it. Fixing it properly means
  reducing `f_geometric · Fbar` per level inside the reverse kernel, which
  already holds the tile in registers.
