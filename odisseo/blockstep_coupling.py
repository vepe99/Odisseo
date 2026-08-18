"""Momentum-conserving individual-timestep N-body on Nornax + Jaccpot.

This is the **block-step** lane. It sits alongside
:mod:`odisseo.jaccpot_coupling` (the shared-timestep lane) rather than replacing
it, and the two answer different questions:

============================  ==============================  ========================
                              :mod:`odisseo.jaccpot_coupling`  this module
============================  ==============================  ========================
force                         target-centric FMM               mutual (Dehnen 2014) FMM
near field                    gather: target sums neighbours   each leaf pair once, +f/-f
momentum residual             ~1e-17 measured, not by design   ~1e-17, structurally
per-level antisymmetric split **not expressible**              ``level_accelerations``
timestepping                  one global timestep              block power-of-two rungs
legal for a block step        **no**                           yes
============================  ==============================  ========================

Why the existing coupler cannot be reused
-----------------------------------------
A block-step KDK splits interactions by level ``k = max(rung_i, rung_j)`` and
requires each level's contribution to be applied *antisymmetrically*, so that an
inactive coarse partner of an active fine interaction still receives its
equal-and-opposite kick. That is not a diagnostic, it is the scheme's defining
correctness property (Dehnen 2014; Farr & Bertschinger 2007).

The load-bearing gap is expressiveness: the production coupler computes one
total acceleration per target, and restricting *sources* to a rung subset --
delivering the reaction to a partner that is not being integrated this sub-step
-- has no place in its API. ``jaccpot.BlockStepFMM`` computes each pair once,
applies both signs, and takes a per-level weight vector into the traversal.

One caveat on the motivating claim, measured rather than assumed: jaccpot's
documentation attributes a ~1e-3 .. 1e-5 momentum residual to the target-centric
force, and **that did not reproduce here**. At N = 20 000, theta = 0.7, order 4,
float64, both lanes land at ~3e-17 while differing by 2.8e-3 in the force
itself, with equal and with unequal masses; the same holds at N = 256. So the
case for this lane rests on the per-level split, not on total momentum -- a
force can conserve total momentum and still be illegal for a block step. See
``docs/source/blockstep_fmm.md``.

Quick start
-----------

.. code-block:: python

    from odisseo.blockstep_coupling import (
        BlockStepOptions,
        integrate_blockstep_jaccpot,
    )

    options = BlockStepOptions(dt_max=1e-3, k_max=3, theta=0.6, max_order=4)
    result = integrate_blockstep_jaccpot(
        state, mass, config, params, options=options, n_base=64
    )
    print(result.momentum_drift[-1], result.seconds_per_base_step)

Topology lifetime
-----------------
``BlockStepFMM.prepare(positions, masses)`` runs a host-side dual-tree traversal
and **cannot be traced**. It is called once per base step, matching the cadence
at which nornax reassigns rungs -- the two discrete refreshes are meant to line
up. Within a base step every one of the ``2**k_max + 1`` boundaries reuses the
same frozen tree, which is what keeps a base step at ``n_sub + 1`` traversals
instead of one traversal per sub-step *per level*.

That host call is also why the rollout is driven from a Python loop over base
steps rather than a single ``lax.scan`` over all of them: nornax's
``block_kdk_rollout`` scans its base steps, so a tree rebuild cannot happen
inside it. ``rebuild_every`` sets how many base steps share one tree, and the
default of 1 refreshes at nornax's own cadence. Those base steps are walked with
``block_kdk_base_step``; setting ``scan_base_steps=True`` hands each interval to
``block_kdk_rollout`` instead -- the same trajectory in one ``lax.scan``, not the
default for the compile reason below.

Fusion is not optional in practice
----------------------------------
``BlockStepFMM`` satisfies nornax's ``FusedMutualForceModel`` with no adapter
changes, and nornax's ``advance_base_step`` opts into fusion automatically. A
silent fallback to the per-level path still produces *correct* answers while
paying one tree traversal per active level instead of one per boundary -- 19
against 9 at ``k_max = 3`` -- so no correctness test can catch it. Every entry
point here therefore calls :func:`assert_fused_boundary_selected` before it
steps.

Fusion has a second, independent switch, and nornax's default for it is the
wrong one here. Having opted into fusion, nornax also asks whether the model
takes a *traced* ``level_weights`` vector, and walks the ``2**k_max + 1``
boundaries with a ``lax.scan`` if so. ``BlockStepFMM.boundary_kick`` does accept
one, so the probe says yes -- but a scan inlines the whole force into a single
program, and since this lane rebuilds the tree every base step that program is
recompiled every base step. Measured: 18.6 s against 10.4 s per base step at
N = 512 on CPU, and 218.6 s against 35.7 s for the first base step at N = 20 000
on an A100. :class:`BlockStepOptions` therefore sets
``traced_boundary_weights=False`` by default, which puts nornax back on the
unrolled boundary loop, where eager dispatch reuses XLA's per-operation cache
across rebuilds instead of recompiling a whole-force program each time.

Where the time actually goes
----------------------------
``jaccpot.mutual`` carries no ``jax.jit``, so a traversal costs **5.1 s eager
against 0.038 s under a single jit** at N = 20 000 on an A100. That is the
largest single lever in this lane and :attr:`BlockStepOptions.jit_force` takes
it, at the price of a compile per topology; read that field before drawing any
conclusion from a wall-clock number here.

External potentials
-------------------
This lane is self-gravity only. An external potential is not a mutual pair
force, so it has no equal-and-opposite partner inside the system and would break
the exact momentum statement this module exists to deliver. A ``config`` that
requests one is rejected rather than silently ignored; use
:mod:`odisseo.jaccpot_coupling` or :mod:`odisseo.differentiable` there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from odisseo.option_classes import SimulationConfig, SimulationParams

__all__ = [
    "BlockStepOptions",
    "BlockStepResult",
    "JittedMutualForce",
    "assert_fused_boundary_selected",
    "blockstep_initial_state",
    "blockstep_total_acceleration",
    "build_blockstep_force",
    "chunked_potential_energy",
    "integrate_blockstep_jaccpot",
    "integrate_blockstep_jitted",
    "total_linear_momentum",
]


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockStepOptions:
    """Configuration for the block-step FMM lane.

    Parameters
    ----------
    dt_max:
        The base-step timestep. Rung ``k`` is integrated with ``dt_max / 2**k``,
        so the finest step in the run is ``dt_max / 2**k_max``.
    k_max:
        Highest rung; levels run ``0 .. k_max``. This single value is handed to
        *both* nornax's ``assign_rungs`` (which clips into ``[0, k_max]``) and
        ``BlockStepFMM`` (which *rejects* out-of-range rungs rather than
        clamping), so the two cannot drift apart. Sub-steps per base step are
        ``2**k_max``, and the fused path costs ``2**k_max + 1`` traversals per
        base step regardless of how many levels are active.
    eta:
        Accuracy parameter of the rung criterion
        ``dt_i = eta * sqrt(rung_eps / |a_i|)``.
    rung_eps:
        Length scale of that criterion. ``None`` takes ``config.softening``,
        which is the usual choice.
    theta:
        Mutual MAC parameter: a node pair is accepted as *far* when
        ``theta * |c_B - c_A| > R_A + R_B``, with ``c`` the node centre of mass
        and ``R`` its radius. Larger ``theta`` accepts more pairs (cheaper, less
        accurate). It has **no** effect on momentum conservation, which is
        structural.

        This criterion is symmetric in the two nodes -- which is what lets one
        acceptance decision serve both directions -- and is correspondingly
        *stricter* than the target-centric ``R_source / d < theta`` the
        shared-timestep lane uses. At ODISSEO's default ``theta = 0.6`` a small
        system can produce **no far pairs at all**, at which point the FMM is a
        direct sum and every far-field accuracy number it reports is vacuous.
        The driver raises rather than let that pass silently.
    max_order:
        Multipole expansion order.
    leaf_size:
        Target particles per leaf.
    backend:
        ``"jax"`` for the pure-JAX kernels, ``"pallas"`` to route the mutual
        near field through jaccpot's Pallas kernel on Ampere+ GPUs (measured
        2.2--3.6x forward on the near field, ~1.2x on the whole force). Falls
        back to pure JAX where the hardware cannot run it.

        .. warning::

           ``"pallas"`` **silently loses gradients** with respect to ``dt_max``,
           ``softening`` and ``G``. Its reverse rule returns
           ``jnp.zeros_like(level_weights)`` (and likewise for ``softening_sq``
           and ``g_value``) on the grounds that the level table is "discrete or
           frozen" -- but ``level_weights[k] == half * dt_max / 2**k`` is a
           smooth function of ``dt_max`` and the force is *linear* in it, so the
           whole near-field term is dropped. Measured here: ``d/d(dt_max)`` comes
           out **111x too small** (only the far field, which stays pure JAX,
           contributes), while the ``"jax"`` backend matches finite differences
           exactly. Forward results are unaffected.

           ``test_the_pallas_backend_drops_most_of_the_dt_max_gradient`` pins
           this so it cannot be relied on or silently change. Use
           ``backend="jax"`` for any gradient with respect to those three.
    near_chunk_size:
        Leaf pairs per near-field scan step; ``None`` derives it from jaccpot's
        pair-tensor memory budget.
    pallas_interpret:
        Run the Pallas kernels in interpret mode -- works without a GPU, far too
        slow for real use, and how the CPU test suite exercises that lane.
    rebuild_every:
        Base steps sharing one frozen tree. ``1`` (default) rebuilds at nornax's
        own rung-reassignment cadence.
    scan_base_steps:
        Whether to hand a rebuild interval to ``block_kdk_rollout`` (a
        ``lax.scan`` over base steps) or step it eagerly with
        ``block_kdk_base_step``. ``None`` means *auto*, which resolves to
        **eager** at every ``rebuild_every``: the scan inlines the whole force
        into one program, and with the topology as a constant that program is
        recompiled on every rebuild, for the same reason
        ``traced_boundary_weights`` defaults off one level down.

        This is a **peak-memory** knob, not a trace-size one, and the two point
        in opposite directions. Jaccpot's inner kernels are individually jitted,
        so an eager Python loop reuses their cached executables, while a scan
        must inline the whole force into one program and compile that. Jaccpot
        measured 2.08 GB against 2.67 GB peak for the same rollout, and ``N``
        barely moves either number -- it is compile/executable memory, not data.
        Scan when trace size is what binds; otherwise step eagerly.
    traced_boundary_weights:
        Whether to let nornax walk the ``2**k_max + 1`` sub-step boundaries with
        a ``lax.scan`` over a traced weight table, instead of unrolling them.
        **Default ``False``**, which is the opposite of what nornax would infer
        on its own: ``BlockStepFMM.boundary_kick`` accepts a ``level_weights``
        argument, so nornax's signature probe opts into the scan unless told
        otherwise.

        Both produce the same trajectory to round-off and both run ``n_sub + 1``
        traversals. The scan gives a much smaller *trace*; what it costs is
        compile work. Eager dispatch reuses XLA's per-operation cache, which
        survives a tree rebuild -- measured: the first force after the *second*
        ``prepare`` cost 5.6 s, against 208 s after the first. A scan instead
        fuses the whole force into one program keyed on the topology constants,
        so it is **recompiled on every rebuild**. Measured here, N = 512 on CPU,
        ``k_max = 2``, 5 base steps:

        ==========  ====================  ==================
                    seconds / base step   seconds / prepare
        ==========  ====================  ==================
        scanned     18.6                  6.2
        unrolled    **10.4**              **2.1**
        ==========  ====================  ==================

        and on an A100 at N = 20 000 the scanned path spent 218.6 s on its first
        base step against 35.7 s unrolled. Turn it on only when trace size is
        what binds -- an outer ``jax.jit`` over the rollout, or a ``k_max`` deep
        enough that ``2**k_max`` unrolled kicks stop fitting. ``None`` leaves
        nornax's own signature probe to decide (which selects the scan).

        This is *not* :attr:`jit_force` pointing the other way. ``jit_force``
        compiles one program per **topology** and reuses it across every boundary
        and every base step of the interval; the boundary scan compiles one per
        **base step**, because nornax's scan body closes over the schedule rather
        than the tree. Both can be on at once.
    checkpoint:
        Passed to ``block_kdk_rollout``: wrap each base step in
        ``jax.checkpoint`` so reverse mode recomputes rather than retains it.
        Only meaningful on the scanned path.
    checkpoint_substeps:
        Also remat each *boundary's* kick, bounding backward memory to one
        boundary's pair tensors. Needed for deep ``k_max`` gradients.
    reassign_rungs:
        Recompute rungs at each base-step boundary (production behaviour). Set
        ``False`` to freeze the schedule for the whole run, which makes the map
        globally smooth in the continuous state -- the setting a
        finite-difference gradient check needs.
    static_shapes:
        Pad jaccpot's pair lists and level schedule to fixed capacities so a
        prepared state's shapes stop depending on the particle distribution.
        ``None`` follows :attr:`jit_force`, which is the only configuration where
        it matters and the only one where it is free.

        This is what turns ``jit_force`` from a trade into a straight win. The
        topology reaches the compiled program as a traced pytree argument, so the
        program is keyed on shapes; padding is what holds those shapes still. See
        :attr:`jit_force` for the measured drift that makes it necessary.
    jit_force:
        Compile the mutual force once and reuse it for every boundary of every
        base step -- and, with :attr:`static_shapes` (which it enables by
        default), across every topology rebuild too.

        ``jaccpot.mutual`` carries no ``jax.jit`` of its own -- every force
        evaluation dispatches op by op -- and the difference is not marginal.
        Measured on one A100-40GB, N = 20 000, float64, theta = 0.7, order 4,
        leaf 32, ``k_max = 3``, 6 base steps with the tree rebuilt **every** base
        step:

        =============================  ========  ==============
                                       eager     ``jit_force``
        =============================  ========  ==============
        seconds / base step            114.25    **0.504**
        total wall, 6 base steps       900.6     **68.6**
        distinct compiled programs     --        **1**
        momentum drift                 3.16e-18  3.12e-18
        =============================  ========  ==============

        **227x per base step.** Two things had to be true for that, and only one
        of them is the ``jit``:

        1. the topology reaches the program as a **traced pytree argument**, so
           the program is keyed on shapes rather than on constant values, and
        2. :attr:`static_shapes` pads the pair lists and level schedule so those
           shapes hold still across rebuilds.

        With (1) alone the cache grows once per rebuild and the ~200 s compile is
        paid every base step, which is what made this a trade rather than a win
        before. With both, ``num_compiles`` stays at 1 for the whole run.

        Capacity sizing is not free: the kernels do work proportional to the
        *capacity*, so a far cap 8x the real pair count means 8x the M2L chunks.
        Retuning the headroom from a flat 4x to additive-plus-relative took the
        far cap from 262144 to 49152 at this N and the per-base-step time from
        1.344 s to 0.504 s -- a 2.7x on top.

        What is left in the 68.6 s total is almost entirely the **host-side tree
        build** plus the single compile: only ~3 s of it is stepping. That is
        Phase 2's target, not this knob's.
    """

    dt_max: float
    k_max: int = 3
    eta: float = 0.1
    rung_eps: Optional[float] = None
    theta: float = 0.6
    max_order: int = 4
    leaf_size: int = 32
    backend: str = "jax"
    near_chunk_size: Optional[int] = None
    pallas_interpret: bool = False
    rebuild_every: int = 1
    scan_base_steps: Optional[bool] = None
    traced_boundary_weights: Optional[bool] = False
    checkpoint: bool = False
    checkpoint_substeps: bool = False
    reassign_rungs: bool = True
    jit_force: bool = False
    topology_backend: str = "host"
    """Where the mutual topology is built: ``"host"`` or ``"device"``.

    ``"host"`` runs jaccpot's NumPy dual-tree traversal, which cannot be traced
    and so costs a device-to-host round trip per rebuild -- measured **23.1 s** at
    N = 20 000 on an A100, against 0.5 s for everything else in the base step. It
    *is* the wall once the force is jitted.

    ``"device"`` builds the whole topology in JAX -- Morton re-sort, node centres
    of mass and radii, the symmetric dual-tree walk, the level schedule and the
    leaf blocks -- at capacities fixed by the profile, so it is traceable and
    composes into one program with the force. Measured on the same machine:

    ==========================================  ==========
    device tree + topology build (jitted)        0.0062 s
    host tree build                             23.1     s
    ==========================================  ==========

    a **3750x** difference, and the device build *fuses* with the force, so
    build+force together (0.0393 s) come out below the force alone against a host
    topology (0.0511 s).

    Accuracy is unaffected. The two backends use different trees -- LBVH against
    a static-radix template -- so they differ from *each other* at ~8e-3, but
    against an exact direct sum at N = 20 000, theta = 0.7 they measure 2.08e-3
    (host) and 2.11e-3 (device). Comparing the two lanes to each other is the
    wrong test; both sit at the FMM's own tolerance.

    The device backend implies ``static_shapes`` and needs a capacity profile.
    **Never reuse a profile across backends**: the two trees have different depth
    and width, and a profile that does not cover the tree silently truncates the
    level schedule -- measured as a 2e-2 force error with no NaN, no shape error,
    and momentum still exactly conserved. ``jaccpot`` raises on it now, naming the
    cap that blew.
    """
    static_shapes: Optional[bool] = None

    def __post_init__(self) -> None:
        if int(self.k_max) < 0:
            raise ValueError(f"k_max must be >= 0; got {self.k_max!r}")
        if float(self.dt_max) <= 0.0:
            raise ValueError(f"dt_max must be > 0; got {self.dt_max!r}")
        if int(self.rebuild_every) < 1:
            raise ValueError(
                f"rebuild_every must be >= 1; got {self.rebuild_every!r}. The tree "
                "is rebuilt on the host, so it can refresh at most once per base "
                "step."
            )

    @property
    def n_sub(self) -> int:
        """Sub-steps per base step, ``2**k_max``."""
        return 1 << int(self.k_max)

    @property
    def dt_min(self) -> float:
        """Finest sub-step, ``dt_max / 2**k_max``."""
        return float(self.dt_max) / self.n_sub

    @property
    def use_static_shapes(self) -> bool:
        """Resolve ``static_shapes``; ``None`` follows ``jit_force``."""
        if self.static_shapes is None:
            return bool(self.jit_force)
        return bool(self.static_shapes)

    @property
    def use_scan(self) -> bool:
        """Resolve ``scan_base_steps``; ``None`` means eager."""
        if self.scan_base_steps is None:
            return False
        return bool(self.scan_base_steps)


@dataclass
class BlockStepResult:
    """Final state plus the per-base-step diagnostics of a block-step run."""

    state: jnp.ndarray
    """ODISSEO primitive state ``(N, 2, 3)`` at the end of the rollout."""

    block_state: Any
    """The nornax ``BlockStepState``, if the caller wants to continue stepping."""

    n_base: int
    """Base steps actually taken."""

    momentum: jnp.ndarray
    """``(n_records, 3)`` total linear momentum, including the initial value."""

    momentum_drift: jnp.ndarray
    """``(n_records,)`` ``|p - p_0| / sum_i |m_i v_i|`` -- the round-off-level
    quantity this lane exists to deliver."""

    energy: Optional[jnp.ndarray]
    """``(n_records,)`` total energy, or ``None`` when diagnostics were off."""

    energy_drift: Optional[jnp.ndarray]
    """``(n_records,)`` ``(E - E_0) / |E_0|``. Bounded, not conserved: leapfrog
    is symplectic, so this oscillates rather than growing secularly."""

    rung_histogram: jnp.ndarray
    """``(n_records, k_max + 1)`` particles per rung at each record."""

    seconds_per_base_step: float
    """Wall clock per base step, averaged over every rebuild interval after the
    first -- the first carries the compile of every kernel the force touches. A
    run of a single interval has nothing left to average and necessarily
    includes it; compare against ``step_seconds`` to see which case applies."""

    step_seconds: list = field(default_factory=list)
    """Wall clock of every base-step interval, in order."""

    prepare_seconds: list = field(default_factory=list)
    """Wall clock of each host-side ``prepare`` (tree build).

    On the jitted lane this is ``[template_freeze, first_rollout]`` instead: that
    lane builds the template once and then compiles, and neither is a per-step
    cost.
    """

    num_far_pairs: int = 0
    """Canonical far pairs in the last built topology. Zero means the FMM
    degenerated to a direct sum and every far-field assertion is vacuous."""

    num_near_pairs: int = 0
    """Leaf pairs in the last built near list."""

    fused: bool = True
    """Whether nornax selected the fused-boundary path (always ``True`` here --
    the driver raises otherwise)."""

    scanned_boundaries: bool = True
    """Whether nornax drove the boundaries with ``lax.scan`` over a traced
    weight table, rather than unrolling ``2**k_max`` of them."""


# --------------------------------------------------------------------------
# force construction and the fusion guard
# --------------------------------------------------------------------------


def build_blockstep_force(
    config: SimulationConfig,
    params: SimulationParams,
    options: BlockStepOptions,
):
    """Build the ``jaccpot.BlockStepFMM`` an ODISSEO config/params describes.

    ``softening`` comes from ``config`` and ``G`` from ``params``, so the
    block-step lane and the shared-timestep lane are physically the same force
    up to the mutual-vs-target-centric restructure.
    """
    from jaccpot import BlockStepFMM

    _reject_external_potentials(config)
    force = BlockStepFMM(
        softening=float(config.softening),
        k_max=int(options.k_max),
        theta=float(options.theta),
        max_order=int(options.max_order),
        G=float(params.G),
        basis="real",
        backend=str(options.backend),
        leaf_size=int(options.leaf_size),
        near_chunk_size=options.near_chunk_size,
        pallas_interpret=bool(options.pallas_interpret),
        **_static_shape_kwargs(options),
        **_topology_backend_kwargs(options),
    )
    # nornax honours an explicit `traced_boundary_weights` attribute and only
    # falls back to inspecting boundary_kick's signature when it is absent. The
    # signature says yes, and for a tree rebuilt every base step that is the
    # wrong answer -- see BlockStepOptions.traced_boundary_weights.
    if options.traced_boundary_weights is not None:
        force.traced_boundary_weights = bool(options.traced_boundary_weights)
    if options.jit_force:
        return JittedMutualForce(force)
    return force


def _weighted_accelerations(state, positions, masses, rung, level_weights):
    """``sum_k level_weights[k] * a_k`` with the topology as a traced argument.

    A module-level function on purpose: ``jax.jit`` caches per callable, so a
    closure rebuilt per topology would defeat the caching this exists for.
    """
    from jaccpot.mutual.force import mutual_weighted_accelerations

    return mutual_weighted_accelerations(
        state, positions, masses, rung=rung, level_weights=level_weights
    )


class JittedMutualForce:
    """A ``BlockStepFMM`` whose force is compiled once, not once per topology.

    ``jaccpot.mutual`` has no ``jax.jit`` anywhere, so every force evaluation
    runs op by op. Wrapping it is worth 135x at N = 20 000 on an A100 (5.1 s ->
    0.038 s per traversal).

    The topology is passed as a **traced pytree argument**, not closed over, so
    the compiled program is keyed on the state's *shapes* rather than on its
    values. That only pays off if the shapes hold still, which needs
    ``static_shapes=True`` on the model (capacity-padded pair lists and a dense
    level schedule). With both, one program serves every rebuild --
    :attr:`num_compiles` stays at 1 and the ~200 s per-rebuild compile is gone.
    Without the padding the shapes drift and the cache grows once per rebuild,
    which is why :class:`BlockStepOptions` ties the two together.

    One compiled program serves everything. Both entry points reduce to the same
    kernel, ``sum_k w_k a_k``:

    * a boundary kick is that sum with the boundary's weight row, added to the
      velocities;
    * the total acceleration is that sum with every weight set to one, since the
      levels partition the interaction set.

    Deriving the static ``active_floor``/``half`` form into a weight *vector*
    here, rather than passing it through, is what keeps it to one program. Left
    as static arguments they would key the jit cache and compile a separate
    program for each of the ``k_max + 1`` floors times two half-values.

    The compiled callable is dropped on every :meth:`prepare`, because the
    topology it closed over is exactly what changed.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # Built ONCE, and deliberately not rebuilt on `prepare`: the whole point
        # is that the jit cache survives a topology refresh.
        self._weighted = jax.jit(_weighted_accelerations)

    # -- delegation -------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Anything not overridden below (softening, theta, state, ...) comes
        # from the wrapped model. Read _inner out of __dict__ rather than as an
        # attribute: this hook also fires for _inner itself before __init__ has
        # run (copy, pickle), and a bare `self._inner` there recurses forever.
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    @property
    def k_max(self) -> int:
        """The wrapped model's level range -- read by nornax's fusion gate."""
        return int(self._inner.k_max)

    @property
    def traced_boundary_weights(self) -> Any:
        """Mirror the wrapped model's answer to nornax's boundary-walk probe."""
        return getattr(self._inner, "traced_boundary_weights", None)

    @property
    def num_compiles(self) -> int:
        """Distinct compiled programs for the force.

        Should be **1** for a whole run. Anything more means the prepared state's
        shapes are moving, i.e. the model was not built with ``static_shapes``.
        """
        return int(self._weighted._cache_size())

    # -- topology lifetime ------------------------------------------------

    def prepare(self, positions: Any, masses: Any) -> Any:
        """Rebuild the topology. The compiled program is *kept*.

        Nothing is invalidated here, which is the difference from the earlier
        closed-over-constant design: the new state is simply handed to the same
        program as a fresh argument. If its shapes match, the cache hits.
        """
        return self._inner.prepare(positions, masses)

    def refresh(self, positions: Any, masses: Any) -> Any:
        """Alias of :meth:`prepare`."""
        return self.prepare(positions, masses)

    # -- the one compiled kernel ------------------------------------------

    def _kernel(self, positions, masses, rung, level_weights):
        """Evaluate ``sum_k w_k a_k`` through the single compiled program."""
        state = self._inner.state
        if state is None:
            raise RuntimeError("call prepare(positions, masses) first")
        return self._weighted(state, positions, masses, rung, level_weights)

    def _weights(self, active_floor, dt_max, half, dtype):
        from jaccpot.mutual.force import level_weights_from_floor

        return level_weights_from_floor(
            active_floor, self.k_max, dt_max, half=half, dtype=dtype
        )

    # -- FusedMutualForceModel --------------------------------------------

    def boundary_kick(
        self,
        positions: Any,
        velocities: Any,
        masses: Any,
        *,
        rung: Any,
        active_floor: Any = None,
        dt_max: Any = None,
        half: Any = 1.0,
        level_weights: Optional[Any] = None,
        args: object = None,
    ) -> Any:
        """One boundary's kick, through the single compiled kernel."""
        del args
        positions = jnp.asarray(positions)
        if level_weights is None:
            if dt_max is None or active_floor is None:
                raise ValueError(
                    "boundary_kick needs either level_weights, or both "
                    "active_floor and dt_max"
                )
            level_weights = self._weights(
                active_floor, dt_max, half, positions.dtype
            )
        return jnp.asarray(velocities) + self._kernel(
            positions, jnp.asarray(masses), jnp.asarray(rung), level_weights
        )

    def total_accelerations(
        self,
        positions: Any,
        masses: Any,
        *,
        rung: Optional[Any] = None,
        args: object = None,
    ) -> Any:
        """The full acceleration: the same kernel with every weight set to one."""
        del args
        positions = jnp.asarray(positions)
        n = positions.shape[0]
        if rung is None:
            rung = jnp.zeros((n,), dtype=jnp.int32)
        ones = jnp.ones((self.k_max + 1,), dtype=positions.dtype)
        return self._kernel(positions, jnp.asarray(masses), jnp.asarray(rung), ones)

    def level_accelerations(
        self,
        positions: Any,
        masses: Any,
        *,
        rung: Any,
        level: int,
        args: object = None,
    ) -> Any:
        """One level's acceleration: a one-hot weight row through the same kernel."""
        del args
        positions = jnp.asarray(positions)
        if not 0 <= int(level) <= self.k_max:
            raise ValueError(
                f"level must lie in [0, k_max={self.k_max}]; got {level!r}"
            )
        weights = (
            jnp.zeros((self.k_max + 1,), dtype=positions.dtype)
            .at[int(level)]
            .set(1.0)
        )
        return self._kernel(
            positions, jnp.asarray(masses), jnp.asarray(rung), weights
        )


def _static_shape_kwargs(options: BlockStepOptions) -> dict:
    """``static_shapes=`` for jaccpot builds that accept it, else nothing.

    Passed conditionally so this lane keeps working against a jaccpot that
    predates the capacity padding -- it simply loses the compile reuse rather
    than failing to construct.
    """
    if not options.use_static_shapes:
        return {}
    from jaccpot import BlockStepFMM

    import inspect

    if "static_shapes" in inspect.signature(BlockStepFMM.__init__).parameters:
        return {"static_shapes": True}
    return {}


def _topology_backend_kwargs(options: "BlockStepOptions") -> dict:
    """``topology_backend`` for a jaccpot that supports it; nothing otherwise.

    Passed conditionally so this module keeps working against a jaccpot that
    predates the device topology, rather than failing at construction with an
    unexpected-keyword error.
    """
    from jaccpot import BlockStepFMM

    backend = str(getattr(options, "topology_backend", "host"))
    if backend == "host":
        return {}
    import inspect

    if "topology_backend" not in inspect.signature(BlockStepFMM).parameters:
        raise RuntimeError(
            "options.topology_backend='device' needs a jaccpot whose "
            "BlockStepFMM accepts topology_backend; the installed one does not. "
            "See docs/source/blockstep_fmm.md for the required ref."
        )
    return {"topology_backend": backend}


def _reject_external_potentials(config: SimulationConfig) -> None:
    """Refuse a config that asks for external accelerations.

    An external potential is not a mutual pair force: it has no partner inside
    the system to receive the back-reaction, so total momentum is genuinely not
    conserved under it. Rather than quietly dropping the term (wrong physics) or
    quietly adding it (a momentum assertion that can no longer hold), this lane
    declines the configuration.
    """
    external = tuple(getattr(config, "external_accelerations", ()) or ())
    if external:
        raise ValueError(
            "The block-step FMM lane is self-gravity only, but config requests "
            f"external_accelerations={external!r}. An external potential has no "
            "equal-and-opposite partner inside the system, so it breaks the "
            "exact momentum conservation this lane is built for. Use "
            "odisseo.jaccpot_coupling (shared timestep) or odisseo.differentiable "
            "for runs with an external potential."
        )


def assert_fused_boundary_selected(force: Any, k_max: int) -> bool:
    """Verify nornax will drive ``force`` through the fused-boundary path.

    Returns whether nornax will *additionally* scan the boundaries over a traced
    weight table (rather than unrolling ``2**k_max`` of them).

    This check exists because its failure mode is invisible: a model that does
    not satisfy ``FusedMutualForceModel`` falls back to the per-level path,
    which computes the *same trajectory* while paying one tree traversal per
    active level instead of one per boundary -- ``sum_s (active levels at s)``
    against ``n_sub + 1``, 19 against 9 at ``k_max = 3``. Every correctness test
    in this repo would still pass.

    Raises
    ------
    RuntimeError
        If the fused path is not selected, or the model's ``k_max`` disagrees
        with the integrator's.
    """
    from nornax.solvers import fused_boundary_model, supports_traced_level_weights

    try:
        selected = fused_boundary_model(force, int(k_max))
    except ValueError as exc:  # k_max disagreement -- nornax raises rather than degrade
        raise RuntimeError(
            f"nornax rejected the fused boundary path: {exc}"
        ) from exc
    if selected is not force:
        raise RuntimeError(
            f"{type(force).__name__} was not selected for nornax's fused-boundary "
            "path, so the base step would fall back to one tree traversal per "
            "active level instead of one per boundary. The trajectory would still "
            "be correct, which is why this is asserted rather than tested. "
            "Check that the model satisfies nornax's FusedMutualForceModel "
            "(total_accelerations + boundary_kick) and exposes a non-None k_max."
        )
    return bool(supports_traced_level_weights(force))


def _assert_far_field_is_exercised(force: Any, *, require: bool = True) -> int:
    """Return the canonical far-pair count, raising when it is zero.

    A configuration with no far pairs makes the FMM a direct sum, so every
    accuracy assertion downstream passes at 1e-16 while testing nothing. That is
    not hypothetical -- it has produced a flattering ``0.0e+00`` here before. Any
    caller trusting a far-field number must call this first.
    """
    state = force.state
    if state is None:
        raise RuntimeError("call force.prepare(positions, masses) first")
    # OCCUPANCY, not capacity. Once the pair lists are capacity-padded,
    # `far_a.shape[0]` is the allocated width and is nonzero even for a topology
    # with no far pairs at all -- which would make this guard silently stop
    # guarding, the exact failure it exists to prevent. `num_far_pairs` is the
    # real count; fall back to the shape only on a jaccpot that predates it (and
    # therefore does not pad).
    num_far = int(getattr(state, "num_far_pairs", state.far_a.shape[0]))
    if require and num_far == 0:
        raise RuntimeError(
            "the prepared topology has no far pairs, so this FMM is a direct sum "
            "and any far-field accuracy number it produces is vacuous. The mutual "
            "MAC accepts a pair when theta * |c_B - c_A| > R_A + R_B, so *raise* "
            "theta, lower leaf_size, or use more particles."
        )
    return num_far


# --------------------------------------------------------------------------
# state plumbing
# --------------------------------------------------------------------------


def blockstep_initial_state(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    force: Any,
    options: BlockStepOptions,
    *,
    rung: Optional[jnp.ndarray] = None,
    prepare: bool = True,
):
    """Build the nornax ``BlockStepState`` seeding a block-step rollout.

    ``rung=None`` assigns rungs from the initial acceleration with nornax's own
    criterion, at the same ``k_max`` the force model was built with.

    Unlike nornax's ``initialize_block_state`` -- which seeds ``acc`` by summing
    ``level_accelerations`` over ``0 .. k_max``, i.e. ``k_max + 1`` traversals --
    this uses the fused ``total_accelerations``, one traversal. The two agree to
    round-off; the levels are a partition.
    """
    from nornax.blockstep.rungs import assign_rungs
    from nornax.state import BlockStepState

    positions = jnp.asarray(state)[:, 0, :]
    velocities = jnp.asarray(state)[:, 1, :]
    mass = jnp.asarray(mass)
    if prepare:
        force.prepare(positions, mass)
    acc = force.total_accelerations(positions, mass)
    if rung is None:
        rung = assign_rungs(
            acc,
            dt_max=float(options.dt_max),
            k_max=int(options.k_max),
            eta=float(options.eta),
            eps=_rung_eps(options, force),
        )
    rung = jnp.asarray(rung, dtype=jnp.int32)
    _check_rung_range(rung, int(options.k_max))
    return BlockStepState(
        positions=positions,
        velocities=velocities,
        masses=mass,
        acc=acc,
        rung=rung,
        base_index=jnp.asarray(0, dtype=jnp.int32),
    )


def _rung_eps(options: BlockStepOptions, force: Any) -> float:
    """The rung criterion's length scale: explicit, else the softening."""
    if options.rung_eps is not None:
        return float(options.rung_eps)
    return float(getattr(force, "softening", 1.0))


def _check_rung_range(rung: jnp.ndarray, k_max: int) -> None:
    """Reject rungs outside ``[0, k_max]`` when the values can be read here.

    ``BlockStepFMM`` rejects out-of-range rungs rather than clamping, and
    nornax's ``assign_rungs`` clips into ``[0, k_max]``, so the two agree as long
    as they are handed the same ``k_max``. This catches a caller-supplied ``rung``
    that was built against a different one.

    The read is *attempted and caught* rather than gated on
    ``isinstance(rung, jax.core.Tracer)``. Those look equivalent and are not: a
    concrete array closed over by a ``lax.cond``/``lax.scan`` branch is not a
    Tracer, yet reducing it still yields a tracer inside the trace, so ``int(...)``
    raises. Attempting the read is the only test that actually asks "can this
    value be read here".
    """
    try:
        lo, hi = int(jnp.min(rung)), int(jnp.max(rung))
    except jax.errors.JAXTypeError:
        return
    if lo < 0 or hi > k_max:
        raise ValueError(
            f"rung values must lie in [0, k_max={k_max}]; got [{lo}, {hi}]. "
            "BlockStepFMM rejects out-of-range rungs rather than clamping, so "
            "build BlockStepOptions with a matching k_max."
        )


def block_state_to_primitive(block_state: Any) -> jnp.ndarray:
    """Pack a nornax ``BlockStepState`` back into ODISSEO's ``(N, 2, 3)``."""
    out = jnp.zeros(
        (block_state.positions.shape[0], 2, block_state.positions.shape[1]),
        dtype=block_state.positions.dtype,
    )
    out = out.at[:, 0, :].set(block_state.positions)
    return out.at[:, 1, :].set(block_state.velocities)


# --------------------------------------------------------------------------
# forces
# --------------------------------------------------------------------------


def blockstep_total_acceleration(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    options: BlockStepOptions,
    *,
    force: Any = None,
    prepare: bool = True,
) -> jnp.ndarray:
    """Total mutual-FMM acceleration, in one traversal.

    This is the quantity to compare against
    :func:`odisseo.jaccpot_coupling.evaluate_acceleration_jaccpot`. They agree to
    the FMM's own force tolerance (~1e-3, theta/order dependent) on the
    **total**; they do not agree per level, because the mutual far field splits
    at cell granularity while a direct-sum oracle splits per particle pair.
    """
    if force is None:
        force = build_blockstep_force(config, params, options)
    positions = jnp.asarray(state)[:, 0, :]
    mass = jnp.asarray(mass)
    if prepare or force.state is None:
        force.prepare(positions, mass)
    return force.total_accelerations(positions, mass)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def total_linear_momentum(mass: jnp.ndarray, velocities: jnp.ndarray) -> jnp.ndarray:
    """``sum_i m_i v_i``."""
    return jnp.sum(jnp.asarray(mass)[:, None] * jnp.asarray(velocities), axis=0)


def _momentum_drift(mass, velocities, p0) -> jnp.ndarray:
    """``|p - p_0|`` normalised by the summed magnitude of the momenta.

    The scale is ``sum_i |m_i v_i|``, not ``|p|``: for a system near rest the
    total momentum is itself near zero, and normalising by it would report a
    huge relative drift for an exactly-conserved run.
    """
    p = total_linear_momentum(mass, velocities)
    scale = jnp.sum(jnp.abs(jnp.asarray(mass)[:, None] * jnp.asarray(velocities)))
    return jnp.linalg.norm(p - p0) / scale


def chunked_potential_energy(
    positions: jnp.ndarray,
    mass: jnp.ndarray,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    chunk: int = 2048,
) -> jnp.ndarray:
    """Exact pairwise potential energy at ``O(N^2)`` flops but ``O(N*chunk)`` memory.

    nornax's ``gravitational_potential_energy`` materialises the dense ``(N, N)``
    pair matrix, which is 80 GB at ``N = 1e5``. This scans chunks of targets
    instead, so the energy diagnostic stays available at the sizes this lane is
    for. It is still quadratic in time -- use it at checkpoints, not every step.
    """
    positions = jnp.asarray(positions)
    mass = jnp.asarray(mass)
    n = positions.shape[0]
    dtype = positions.dtype
    eps2 = jnp.asarray(softening, dtype=dtype) ** 2
    pad = (-n) % chunk

    pos_p = jnp.pad(positions, ((0, pad), (0, 0)))
    mass_p = jnp.pad(mass, (0, pad))
    idx = jnp.arange(n + pad)

    def body(acc, start):
        rows = jax.lax.dynamic_slice(pos_p, (start, 0), (chunk, 3))
        row_mass = jax.lax.dynamic_slice(mass_p, (start,), (chunk,))
        row_idx = jax.lax.dynamic_slice(idx, (start,), (chunk,))
        dr = pos_p[None, :, :] - rows[:, None, :]
        r2 = jnp.sum(dr * dr, axis=-1) + eps2
        # Upper triangle only, so each unordered pair is counted once -- and the
        # padded slots are excluded from `live` rather than left to their zero
        # mass. Two padded particles both sit at the origin, so with
        # softening = 0 their r2 is exactly 0, rsqrt gives inf, and inf * 0 is
        # NaN, not the dropped term the zero mass was meant to produce. This is
        # the single-trailing-mask antipattern; the guard has to be inside.
        live = (
            (row_idx[:, None] < idx[None, :])
            & (row_idx[:, None] < n)
            & (idx[None, :] < n)
        )
        inv_r = jnp.where(live, jax.lax.rsqrt(jnp.where(live, r2, 1.0)), 0.0)
        pair_mass = row_mass[:, None] * mass_p[None, :]
        return acc + jnp.sum(pair_mass * inv_r), None

    total, _ = jax.lax.scan(
        body,
        jnp.asarray(0.0, dtype=dtype),
        jnp.arange(0, n + pad, chunk),
    )
    return -jnp.asarray(G, dtype=dtype) * total


def _total_energy(block_state, *, G: float, softening: float, chunk: int):
    """Kinetic plus the chunked exact potential."""
    kinetic = 0.5 * jnp.sum(
        block_state.masses * jnp.sum(block_state.velocities**2, axis=-1)
    )
    potential = chunked_potential_energy(
        block_state.positions,
        block_state.masses,
        G=G,
        softening=softening,
        chunk=chunk,
    )
    return kinetic + potential


def _rung_histogram(rung: jnp.ndarray, k_max: int) -> jnp.ndarray:
    return jnp.bincount(jnp.asarray(rung), length=k_max + 1)


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


def integrate_blockstep_jaccpot(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    options: BlockStepOptions,
    n_base: int,
    force: Any = None,
    block_state: Any = None,
    track_energy: bool = True,
    energy_chunk: int = 2048,
    record_every: int = 1,
    progress: Optional[Callable[[int, dict], None]] = None,
) -> BlockStepResult:
    """Run ``n_base`` momentum-conserving block-step KDK base steps.

    The loop is::

        for each rebuild interval:
            force.prepare(positions, masses)      # host tree build, not traceable
            nornax advances `rebuild_every` base steps under that frozen tree
                (rungs reassigned from the cached acceleration at each
                 base-step boundary, inside nornax)

    Parameters
    ----------
    state:
        ODISSEO primitive state ``(N, 2, 3)``.
    mass:
        ``(N,)`` particle masses.
    options:
        See :class:`BlockStepOptions`. ``options.k_max`` is handed to both the
        force model and the rung assignment, so they cannot disagree.
    n_base:
        Base steps to take. Rounded *up* to a whole number of rebuild intervals;
        the actual count is reported in the result.
    force:
        A prebuilt ``BlockStepFMM`` to reuse (its ``prepare`` is still called per
        interval). ``None`` builds one from ``config``/``params``/``options``.
    block_state:
        Continue from an existing nornax ``BlockStepState`` instead of building
        one from ``state``.
    track_energy:
        Record the total energy at each record point. This is an ``O(N^2)``
        diagnostic -- turn it off, or raise ``record_every``, when it dominates.
    record_every:
        Record diagnostics every this many base steps (the initial state and the
        final state are always recorded).
    progress:
        Called as ``progress(base_index, record_dict)`` after each record.

    Returns
    -------
    BlockStepResult

    Raises
    ------
    RuntimeError
        If nornax does not select the fused-boundary path, or the prepared
        topology has no far pairs (which would make the FMM a direct sum).
    """
    from nornax.solvers import (
        advance_base_step,
        block_kdk_base_step,
        block_kdk_rollout,
    )

    _reject_external_potentials(config)
    if int(n_base) < 1:
        raise ValueError(f"n_base must be >= 1; got {n_base!r}")

    mass = jnp.asarray(mass)
    if force is None:
        force = build_blockstep_force(config, params, options)
    if int(getattr(force, "k_max")) != int(options.k_max):
        raise ValueError(
            f"force model k_max={int(force.k_max)} disagrees with "
            f"options.k_max={int(options.k_max)}; the rung assignment and the "
            "fused kick weights would span different level ranges"
        )

    if block_state is None:
        block_state = blockstep_initial_state(state, mass, force, options)
    else:
        force.prepare(block_state.positions, mass)
    scanned = assert_fused_boundary_selected(force, int(options.k_max))
    num_far = _assert_far_field_is_exercised(force)
    num_near = int(
        getattr(force.state, "num_near_pairs", force.state.near_a.shape[0])
    )

    dt_max = float(options.dt_max)
    eps = _rung_eps(options, force)
    G = float(params.G)
    softening = float(config.softening)

    momenta = [total_linear_momentum(mass, block_state.velocities)]
    p0 = momenta[0]
    drifts = [_momentum_drift(mass, block_state.velocities, p0)]
    hist = [_rung_histogram(block_state.rung, int(options.k_max))]
    energies = (
        [_total_energy(block_state, G=G, softening=softening, chunk=energy_chunk)]
        if track_energy
        else None
    )

    step_seconds: list[float] = []
    prepare_seconds: list[float] = []
    taken = 0
    interval = int(options.rebuild_every)
    n_intervals = -(-int(n_base) // interval)

    for i in range(n_intervals):
        # One host-side tree build per rebuild interval. Skipped on the first
        # interval, where blockstep_initial_state (or the block_state branch
        # above) already prepared the same positions.
        if i > 0:
            t_prep = time.perf_counter()
            force.prepare(block_state.positions, mass)
            jax.block_until_ready(force.state.far_a)
            prepare_seconds.append(time.perf_counter() - t_prep)

        t0 = time.perf_counter()
        if options.use_scan:
            block_state = block_kdk_rollout(
                block_state,
                dt_max,
                force,
                k_max=int(options.k_max),
                n_base=interval,
                eta=float(options.eta),
                eps=eps,
                checkpoint=bool(options.checkpoint),
                reassign_rungs=bool(options.reassign_rungs),
                checkpoint_substeps=bool(options.checkpoint_substeps),
            )
        else:
            for _ in range(interval):
                if options.reassign_rungs:
                    block_state = block_kdk_base_step(
                        block_state,
                        dt_max,
                        force,
                        k_max=int(options.k_max),
                        eta=float(options.eta),
                        eps=eps,
                        checkpoint_substeps=bool(options.checkpoint_substeps),
                    )
                else:
                    block_state = advance_base_step(
                        block_state,
                        dt_max,
                        force,
                        k_max=int(options.k_max),
                        checkpoint_substeps=bool(options.checkpoint_substeps),
                    )
        block_state = jax.block_until_ready(block_state)
        elapsed = time.perf_counter() - t0
        step_seconds.extend([elapsed / interval] * interval)
        taken += interval

        _check_rung_range(block_state.rung, int(options.k_max))

        last = i == n_intervals - 1
        if last or (taken % int(record_every) == 0):
            momenta.append(total_linear_momentum(mass, block_state.velocities))
            drifts.append(_momentum_drift(mass, block_state.velocities, p0))
            hist.append(_rung_histogram(block_state.rung, int(options.k_max)))
            if energies is not None:
                energies.append(
                    _total_energy(
                        block_state, G=G, softening=softening, chunk=energy_chunk
                    )
                )
            if progress is not None:
                progress(
                    taken,
                    {
                        "momentum_drift": float(drifts[-1]),
                        "energy": None if energies is None else float(energies[-1]),
                        "seconds_per_base_step": elapsed / interval,
                    },
                )

    energy_arr = None if energies is None else jnp.stack(energies)
    energy_drift = (
        None
        if energy_arr is None
        else (energy_arr - energy_arr[0]) / jnp.abs(energy_arr[0])
    )
    # Drop the first interval, which carries the compile of every kernel the
    # force touches -- unless there was only one, in which case there is nothing
    # left to average and the reported figure necessarily includes it.
    timed = step_seconds[interval:] or step_seconds
    return BlockStepResult(
        state=block_state_to_primitive(block_state),
        block_state=block_state,
        n_base=taken,
        momentum=jnp.stack(momenta),
        momentum_drift=jnp.stack(drifts),
        energy=energy_arr,
        energy_drift=energy_drift,
        rung_histogram=jnp.stack(hist),
        seconds_per_base_step=float(sum(timed) / len(timed)),
        step_seconds=step_seconds,
        prepare_seconds=prepare_seconds,
        num_far_pairs=num_far,
        num_near_pairs=num_near,
        fused=True,
        scanned_boundaries=scanned,
    )


# --------------------------------------------------------------------------
# the fully jitted lane
# --------------------------------------------------------------------------


def integrate_blockstep_jitted(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    options: BlockStepOptions,
    n_base: int,
    force: Any = None,
    track_energy: bool = True,
    energy_chunk: int = 2048,
    time_steady_state: bool = False,
) -> BlockStepResult:
    """Run ``n_base`` base steps as **one compiled program**, tree rebuild included.

    The difference from :func:`integrate_blockstep_jaccpot` is where the tree is
    built. There it is host-side, so the rollout has to be a Python loop with a
    device-to-host round trip per base step -- 23.1 s of it at N = 20 000. Here the
    topology is rebuilt in JAX inside the ``lax.scan``, so the whole rollout is a
    single program with no host traffic at all. Measured on one A100, N = 20 000,
    ``k_max = 3``, float64:

    ==================================================  ============
    eager, host topology (yesterday's baseline)          52.46 s/step
    jitted force, host topology rebuilt per base step    ~22.5 s/step
    **this lane**                                        **0.39 s/step**
    ==================================================  ============

    with momentum drift 3.4e-18 over the rollout and one entry in the jit cache.

    Why this drives nornax's *primitives* rather than ``block_kdk_rollout``
    ---------------------------------------------------------------------
    ``block_kdk_rollout`` scans its base steps and threads ``args`` unchanged, so
    there is nowhere for a per-base-step topology to live: it would have to be in
    the scan **carry**, and nornax has no hook for that. So this walks the
    boundaries itself using ``n_sub``, ``boundary_weight_table`` and
    ``assign_rungs`` -- all pure, all already device-side, and explicitly meant to
    be consumed this way -- and carries the topology alongside the state. It is
    the same schedule, not a reimplementation of one, and
    ``test_the_jitted_lane_matches_the_host_loop_lane`` pins the trajectory
    against the nornax-driven path.

    Requires ``options.topology_backend == "device"``.

    Notes
    -----
    Diagnostics that are cheap come out of the scan per base step (momentum, the
    far-pair count, the overflow flag). Energy is ``O(N^2)`` and is evaluated
    outside, at the two ends only -- pass ``track_energy=False`` to skip it.

    The per-base-step overflow flags are reduced and raised *after* the rollout.
    They cannot be raised inside it, and they must be looked at: a truncated
    topology drops interactions silently, and because a dropped canonical pair
    loses both of its halves, momentum stays exactly conserved.

    Reading the timings
    -------------------
    ``seconds_per_base_step`` divides the *whole* measured rollout by ``n_base``,
    so by default it **includes the one-time compile** of the rollout program --
    which at a small ``n_base`` dominates it completely. That is easy to misread
    into nonsense: compared naively against
    :func:`integrate_blockstep_jaccpot`, whose ``seconds_per_base_step``
    *excludes* its host tree build (timed separately in ``prepare_seconds``),
    this lane can look ten times *slower* while actually being sixty times
    faster.

    Pass ``time_steady_state=True`` to re-run the compiled rollout once and
    report the warm figure instead. It costs a second rollout, which is why it is
    off by default, and it is the only number worth quoting in a comparison.
    """
    from nornax.blockstep.rungs import assign_rungs
    from nornax.blockstep.schedule import boundary_weight_table, n_sub

    _reject_external_potentials(config)
    if str(getattr(options, "topology_backend", "host")) != "device":
        raise ValueError(
            "integrate_blockstep_jitted needs options.topology_backend='device'; "
            "the host traversal cannot be traced, so it cannot live inside the "
            "scan. Use integrate_blockstep_jaccpot for the host lane."
        )
    if int(n_base) < 1:
        raise ValueError(f"n_base must be >= 1; got {n_base!r}")

    mass = jnp.asarray(mass)
    positions0 = jnp.asarray(state)[:, 0, :]
    velocities0 = jnp.asarray(state)[:, 1, :]
    if force is None:
        force = build_blockstep_force(config, params, options)

    # Host-side, once: the static-radix template and the capacity profile.
    #
    # Deliberately NOT followed by a probe `rebuild_state` call. That reads
    # naturally -- build one topology up front, check it is neither overflowing
    # nor vacuous, then run -- and it is a serious performance bug: outside a
    # `jax.jit`, `rebuild_state` dispatches the whole traversal op by op, and it
    # measured a flat ~32 s per call at both N = 20 000 and N = 100 000. Divided
    # over `n_base` that read as a 14x per-base-step regression at N = 2e4 and 3x
    # at N = 1e5, which is what a constant cost looks like when you report it per
    # step. The N-independence was the tell.
    #
    # Both checks it performed are available for free from the scan's own
    # per-base-step records below, so nothing is lost by deferring them.
    t0 = time.perf_counter()
    force.freeze_template(positions0, mass)
    prepare_seconds = time.perf_counter() - t0

    k_max = int(options.k_max)
    steps = n_sub(k_max)
    dt_max = jnp.asarray(options.dt_max, dtype=positions0.dtype)
    dt_min = dt_max / steps
    # The schedule is a compile-time constant; scaling the unit table by a traced
    # dt_max is exact (every entry is a power of two) and keeps dt_max
    # differentiable.
    table = jnp.asarray(boundary_weight_table(k_max), dtype=positions0.dtype)
    eps = _rung_eps(options, force)
    zero = jnp.asarray(0.0, dtype=positions0.dtype)

    def base_step(carry, _):
        positions, velocities, mass = carry
        topology = force.rebuild_state(positions, mass)
        acc = force.weighted_accelerations(topology, positions, mass)
        rung = assign_rungs(
            acc, dt_max=dt_max, k_max=k_max, eta=float(options.eta), eps=eps
        )

        def boundary(inner, s):
            pos, vel = inner
            vel = vel + force.weighted_accelerations(
                topology, pos, mass, rung=rung, level_weights=dt_max * table[s]
            )
            # A no-op drift after the final kick, written as a select so every
            # iteration has one shape.
            pos = pos + jnp.where(s < steps, dt_min, zero) * vel
            return (pos, vel), None

        (positions, velocities), _ = jax.lax.scan(
            boundary, (positions, velocities), jnp.arange(steps + 1, dtype=jnp.int32)
        )
        record = (
            jnp.sum(mass[:, None] * velocities, axis=0),
            jnp.bincount(rung, length=k_max + 1),
            topology.num_far_pairs,
            topology.num_near_pairs,
            topology.topology_overflow,
        )
        return (positions, velocities, mass), record

    def _scan(positions, velocities, masses):
        return jax.lax.scan(
            base_step, (positions, velocities, masses), xs=None, length=int(n_base)
        )

    # The compiled rollout is cached ON THE FORCE OBJECT, and `masses` is an
    # argument rather than a closure constant.
    #
    # `@jax.jit` on a function defined inside this one would look right and be a
    # ~32 s bug per call: jit keys its cache on the function object, a fresh
    # closure is a fresh key, so *every* call recompiles the whole rollout. It
    # showed up as a flat ~7.9 s/base-step penalty at both N = 20 000 and
    # N = 100 000 -- constant, hence obviously not per-step physics, but easy to
    # read as a 14x/3x regression when divided by `n_base`. The same mistake bit
    # the benchmark harness first: calling this function twice with `force=None`
    # builds a fresh model each time and so still recompiles.
    cache = getattr(force, "_odisseo_jitted_rollout_cache", None)
    if cache is None:
        cache = {}
        force._odisseo_jitted_rollout_cache = cache
    cache_key = (
        int(n_base),
        k_max,
        float(options.dt_max),
        float(options.eta),
        float(eps),
        tuple(positions0.shape),
        str(positions0.dtype),
    )
    rollout = cache.get(cache_key)
    if rollout is None:
        rollout = jax.jit(_scan)
        cache[cache_key] = rollout

    t0 = time.perf_counter()
    (positions, velocities, _), records = jax.block_until_ready(
        rollout(positions0, velocities0, mass)
    )
    elapsed = time.perf_counter() - t0
    compile_seconds = elapsed
    if time_steady_state:
        t0 = time.perf_counter()
        jax.block_until_ready(rollout(positions0, velocities0, mass))
        elapsed = time.perf_counter() - t0
    momenta, histograms, far_counts, near_counts, overflows = records
    num_far = int(jnp.max(far_counts))
    num_near = int(jnp.max(near_counts))
    if num_far == 0:
        raise RuntimeError(
            "the device topology has no far pairs at any base step, so this FMM "
            "is a direct sum and any far-field number it produces is vacuous. "
            "Raise theta or lower leaf_size."
        )
    if bool(jnp.any(overflows)):
        raise RuntimeError(
            "the device topology overflowed its capacity profile at some point "
            "during the rollout, so interactions were dropped. Momentum will "
            "still look exact -- a dropped canonical pair loses both halves -- so "
            "this is raised rather than reported. Re-resolve the capacities on a "
            "later configuration, or raise them explicitly."
        )

    p0 = total_linear_momentum(mass, velocities0)
    momenta = jnp.concatenate([p0[None, :], momenta], axis=0)
    scale = jnp.sum(jnp.abs(mass[:, None] * velocities))
    drifts = jnp.linalg.norm(momenta - p0[None, :], axis=-1) / scale
    # No synthetic leading row here, unlike `momentum`: rungs are not assigned
    # until *inside* the first base step, so an "initial histogram" would be a
    # fabricated all-on-rung-0 entry -- which then reads as the block scheme
    # having collapsed. `rung_histogram` therefore has `n_base` rows, one per
    # base step actually taken, while `momentum` has `n_base + 1`.

    energies = None
    if track_energy:
        G = float(params.G)
        softening = float(config.softening)
        energies = jnp.stack(
            [
                _energy_of(positions0, velocities0, mass, G, softening, energy_chunk),
                _energy_of(positions, velocities, mass, G, softening, energy_chunk),
            ]
        )

    out = jnp.zeros((positions.shape[0], 2, positions.shape[1]), positions.dtype)
    out = out.at[:, 0, :].set(positions).at[:, 1, :].set(velocities)
    return BlockStepResult(
        state=out,
        block_state=None,
        n_base=int(n_base),
        momentum=momenta,
        momentum_drift=drifts,
        energy=energies,
        energy_drift=(
            None
            if energies is None
            else (energies - energies[0]) / jnp.abs(energies[0])
        ),
        rung_histogram=histograms,
        seconds_per_base_step=elapsed / int(n_base),
        step_seconds=[elapsed / int(n_base)] * int(n_base),
        prepare_seconds=[prepare_seconds, compile_seconds],
        num_far_pairs=num_far,
        num_near_pairs=num_near,
        fused=True,
        scanned_boundaries=True,
    )


def _raise_on_overflow(topology: Any, force: Any) -> None:
    """Turn a device overflow flag into an exception, naming the cap that blew."""
    try:
        overflowed = bool(topology.topology_overflow)
    except jax.errors.JAXTypeError:  # pragma: no cover - traced caller
        return
    if not overflowed:
        return
    from jaccpot.mutual.force import OVERFLOW_CAUSES

    bits = int(topology.overflow_causes)
    blamed = [n for i, n in enumerate(OVERFLOW_CAUSES) if bits & (1 << i)]
    caps = force.capacities
    raise RuntimeError(
        f"the device topology overflowed: {', '.join(blamed) or 'unknown'} "
        f"exceeded the profile far={caps.far}, near={caps.near}, "
        f"depth={caps.depth}, width={caps.width}."
    )


def _energy_of(positions, velocities, mass, G, softening, chunk):
    kinetic = 0.5 * jnp.sum(mass * jnp.sum(velocities**2, axis=-1))
    return kinetic + chunked_potential_energy(
        positions, mass, G=G, softening=softening, chunk=chunk
    )
