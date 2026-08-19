"""Gradients through an FMM simulation, including external-potential parameters.

ODISSEO's production FMM lane (:mod:`odisseo.jaccpot_coupling`) is built for
forward throughput: it drives jaccpot's host-side ``prepare_state`` /
``evaluate_prepared_state`` pair from a Python loop and passes ``params`` as a
*static* jit argument. Both choices block ``jax.grad``:

* a traced ``params`` cannot be hashed as a static argument, so
  ``jax.grad(..., wrt=params)`` fails before any physics runs, and
* ``evaluate_prepared_state`` reads the *prebaked* expansions of the cached
  prepared state, so even if a tracer did reach it the self-gravity term would
  contribute exactly zero sensitivity.

This module adds the differentiable counterpart, built on jaccpot's
``differentiable_accelerations`` seam. Self-gravity is re-evaluated from the live
positions at **frozen tree topology**, the external potential is evaluated with
``params`` *traced*, and the leapfrog/diffrax loop is plain JAX. The result is
end-to-end gradients of any scalar loss with respect to

* external-potential parameters (``params.NFW_params.Mvir``, ``params.MN_params.a``,
  ... — any field of :class:`~odisseo.option_classes.SimulationParams`),
* the initial state (positions *and* velocities), and
* particle masses.

Quick start
-----------

.. code-block:: python

    from odisseo.differentiable import (
        prepare_differentiable_fmm,
        integrate_leapfrog_differentiable,
    )

    # Build the tree ONCE, from concrete inputs, outside the differentiated
    # function: jaccpot's prepare_state is host-side and not traceable.
    plan = prepare_differentiable_fmm(state0, mass, config, params)

    def loss(mvir):
        p = params._replace(NFW_params=params.NFW_params._replace(Mvir=mvir))
        final = integrate_leapfrog_differentiable(state0, mass, config, p, plan=plan)
        return jnp.sum((final[:, 0] - target_positions) ** 2)

    dloss_dmvir = jax.grad(loss)(mvir0)

The fixed-topology contract
---------------------------

Gradients are exact for the numeric pipeline at **frozen topology**: the Morton
permutation, node membership, the M2L interaction list, the near-field neighbour
lists and every MAC accept/reject decision are held constant, while P2M, the
centre-of-mass expansion centres, the M2M/M2L/L2L translations, L2P and the
near-field P2P are all differentiated through. See
``jaccpot/docs/differentiable_fmm.md``.

Two consequences for a *simulation* (as opposed to a single force call):

1. The topology cannot be rebuilt inside the differentiated window, so one
   ``jax.grad`` call integrates at fixed topology. Choose ``num_steps`` such
   that particles do not stream far out of their cells; check with
   :func:`topology_drift`. This is the same trade the production lane makes with
   ``fmm_refresh_every``, except the refresh must happen *outside* ``jax.grad``.
2. A finite-difference reference must perturb the *same* frozen plan. FD over a
   run that rebuilds the tree disagrees whenever a pair crosses a MAC boundary.

Bare ``jax.grad`` is the supported usage; jaccpot's inner kernels are already
jit-compiled. An outer ``jax.jit`` around the whole integration works at moderate
N but can hit host-side ops in jaccpot's re-run sweeps at large N.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

from odisseo.option_classes import (
    DOPRI5,
    DOPRI8,
    FORWARDMODE,
    RECURSIVECHECKPOINTADJOING,
    TSIT5,
    SimulationConfig,
    SimulationParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch

__all__ = [
    "DifferentiableFMMPlan",
    "prepare_differentiable_fmm",
    "differentiable_fmm_self_acceleration",
    "differentiable_total_acceleration",
    "integrate_leapfrog_differentiable",
    "integrate_diffrax_differentiable",
    "topology_drift",
]


def _contains_tracer(value: Any) -> bool:
    """Return ``True`` when a pytree contains JAX tracers."""
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


@dataclass(frozen=True)
class DifferentiableFMMPlan:
    """Frozen FMM topology plus the solver that owns it.

    Build with :func:`prepare_differentiable_fmm` from concrete inputs, then reuse
    it for every gradient evaluation of an optimisation/inference loop. The plan
    is a plain Python object (not a pytree): pass it through closures or as a
    keyword argument, not as a differentiated argument.

    Attributes
    ----------
    solver:
        The ``jaccpot.FastMultipoleMethod`` instance the topology belongs to.
    prepared_state:
        The frozen topology from ``solver.prepare_state``.
    reference_positions:
        The positions the topology was built from, kept for :func:`topology_drift`.
    grad_config:
        ``jaccpot.GradConfig`` forwarded to every differentiable evaluation.
    large_n_grad_plan:
        Pre-validated large-N gradient plan, when the prepared state is a
        ``LargeNPreparedState``; ``None`` otherwise.
    """

    solver: Any
    prepared_state: Any
    reference_positions: jnp.ndarray
    n_particles: int
    leaf_size: int
    max_order: int
    working_dtype: Any
    grad_config: Optional[Any] = None
    large_n_grad_plan: Optional[Any] = None
    preset: str = "fast"
    basis: str = "real"


def _resolve_grad_config(
    config: SimulationConfig,
    grad_config: Optional[Any],
) -> Optional[Any]:
    """Return the jaccpot ``GradConfig`` for this run, or ``None``.

    An explicitly passed ``grad_config`` wins outright. Otherwise the two
    ``SimulationConfig`` knobs that matter are translated, and if neither is set
    away from its default we return ``None`` so jaccpot's own measured defaults
    (and any ``JACCPOT_*`` environment variables) apply untouched.
    """
    if grad_config is not None:
        return grad_config

    lane = str(getattr(config, "fmm_grad_nearfield_lane", "auto")).strip().lower()
    fused_m2l = getattr(config, "fmm_grad_fused_m2l_pallas", None)
    if lane == "auto" and fused_m2l is None:
        return None

    try:
        from jaccpot import GradConfig
    except ImportError:  # pragma: no cover - jaccpot without GradConfig
        raise NotImplementedError(
            "config.fmm_grad_nearfield_lane / fmm_grad_fused_m2l_pallas need a "
            "jaccpot that exports GradConfig (the differentiable-FMM branch). "
            "Leave them at their defaults to use the installed jaccpot as-is."
        )
    return GradConfig(
        nearfield_lane=lane,
        fused_m2l_pallas=(None if fused_m2l is None else bool(fused_m2l)),
    )


def _solver_kwargs_from_config(
    config: SimulationConfig,
    params: SimulationParams,
    *,
    working_dtype,
    preset: str,
    runtime_path: str,
    leaf_size: int,
) -> dict[str, Any]:
    """Map a ``SimulationConfig`` onto ``_build_fmm_solver`` keyword arguments."""
    return dict(
        working_dtype=working_dtype,
        config=config,
        params=params,
        fmm_preset=str(preset),
        fmm_basis=str(config.fmm_basis),
        fmm_theta=float(config.fmm_theta),
        fmm_runtime_path=str(runtime_path),
        fmm_mac_type=str(config.fmm_mac_type),
        fmm_farfield_mode=str(config.fmm_farfield_mode),
        fmm_m2l_chunk_size=(
            None if config.fmm_m2l_chunk_size is None else int(config.fmm_m2l_chunk_size)
        ),
        fmm_nearfield_mode=str(config.fmm_nearfield_mode),
        fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
        fmm_tree_build_mode=str(config.fmm_tree_build_mode),
        fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
        fmm_fixed_order=(
            None if config.fmm_fixed_order is None else int(config.fmm_fixed_order)
        ),
        leaf_size=int(leaf_size),
        fmm_jit_tree=(
            None if config.fmm_jit_tree is None else bool(config.fmm_jit_tree)
        ),
        # The gradient path does not use the jitted traversal (jaccpot keeps
        # jit_traversal=False on the grad seam), so do not request it here.
        fmm_jit_traversal=False,
        fmm_max_pair_queue=(
            None if config.fmm_max_pair_queue is None else int(config.fmm_max_pair_queue)
        ),
        fmm_pair_process_block=(
            None
            if config.fmm_pair_process_block is None
            else int(config.fmm_pair_process_block)
        ),
        fmm_max_interactions_per_node=(
            None
            if config.fmm_max_interactions_per_node is None
            else int(config.fmm_max_interactions_per_node)
        ),
        fmm_max_neighbors_per_leaf=(
            None
            if config.fmm_max_neighbors_per_leaf is None
            else int(config.fmm_max_neighbors_per_leaf)
        ),
        fmm_prepare_stage_memory_split_enabled=(
            None
            if config.fmm_prepare_stage_memory_split_enabled is None
            else bool(config.fmm_prepare_stage_memory_split_enabled)
        ),
        fmm_upward_leaf_batch_size=getattr(config, "fmm_upward_leaf_batch_size", None),
    )


def prepare_differentiable_fmm(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    grad_config: Optional[Any] = None,
    leaf_size: Optional[int] = None,
    max_order: Optional[int] = None,
    bounds: Optional[tuple] = None,
) -> DifferentiableFMMPlan:
    """Build the frozen FMM topology that gradient evaluations differentiate at.

    Must be called with **concrete** ``state`` and ``mass``: jaccpot's tree build
    is host-side and rejects tracers. Call it once, outside the differentiated
    function, and pass the returned plan into
    :func:`integrate_leapfrog_differentiable` or
    :func:`differentiable_total_acceleration`.

    Args:
        state: Primitive state, shape ``(N, 2, 3)``.
        mass: Particle masses, shape ``(N,)``.
        config: Simulation configuration; supplies every FMM tuning knob.
        params: Simulation parameters; only ``params.G`` is read here.
        grad_config: Optional ``jaccpot.GradConfig``. Overrides the
            ``config.fmm_grad_*`` knobs.
        leaf_size: Override ``config.fmm_leaf_size``.
        max_order: Override ``config.fmm_max_order``.
        bounds: Optional ``(lo, hi)`` root-cell bounds. Fixing the root cell
            keeps the topology comparable across separate plan builds.

    Returns:
        DifferentiableFMMPlan: the frozen topology plus its solver.

    Raises:
        NotImplementedError: if ``state`` or ``mass`` is traced, or if the
            installed jaccpot has no differentiable seam.
    """
    if _contains_tracer(state) or _contains_tracer(mass):
        raise NotImplementedError(
            "prepare_differentiable_fmm needs CONCRETE state/mass: jaccpot's "
            "prepare_state builds the tree on the host and is not traceable. "
            "Build the plan once outside the differentiated function, then pass "
            "plan=... into the integrator. This is the fixed-topology contract, "
            "not a limitation of the coupling."
        )

    from odisseo.jaccpot_coupling import (
        _build_fmm_solver,
        _temporary_large_n_environment,
    )

    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    if state_arr.ndim != 3 or state_arr.shape[1:] != (2, 3):
        raise ValueError("state must have shape (N, 2, 3)")

    leaf_size_i = int(config.fmm_leaf_size if leaf_size is None else leaf_size)
    max_order_i = int(config.fmm_max_order if max_order is None else max_order)

    # Same preset/dtype resolution as the forward lane, so a differentiable run
    # and a forward run of the same config execute the same FMM configuration.
    from odisseo.integration_api import _resolve_fmm_runtime_profile

    preset, runtime_path, working_dtype = _resolve_fmm_runtime_profile(
        state_arr, config
    )

    solver_kwargs = _solver_kwargs_from_config(
        config,
        params,
        working_dtype=working_dtype,
        preset=preset,
        runtime_path=runtime_path,
        leaf_size=leaf_size_i,
    )
    if _supports_retain_far_pairs(_build_fmm_solver):
        # The large-N gradient path re-runs the downward sweep against the frozen
        # M2L pair list, which the forward lane discards to save memory
        # (~24 B/pair). Without it the differentiable call raises rather than
        # returning a far field with no mass sensitivity. The radix (non-large-N)
        # path does not need it, so an older jaccpot simply goes without.
        solver_kwargs["retain_far_pairs_for_grad"] = True
    solver = _build_fmm_solver(**solver_kwargs)

    if not callable(getattr(solver, "differentiable_accelerations", None)):
        raise NotImplementedError(
            "the installed jaccpot has no FastMultipoleMethod."
            "differentiable_accelerations; gradients through the FMM need the "
            "differentiable-FMM jaccpot (feat/differentiable-fmm or later). "
            "Use a direct acceleration scheme (DIRECT_ACC / NO_SELF_GRAVITY) "
            "for gradient runs against an older jaccpot."
        )

    positions = state_arr[:, 0, :].astype(working_dtype)
    masses = mass_arr.astype(working_dtype)
    with _temporary_large_n_environment(config, fmm_preset=preset):
        prepared_state = solver.prepare_state(
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size_i,
            max_order=max_order_i,
            theta=float(config.fmm_theta),
        )

    large_n_grad_plan = _maybe_large_n_grad_plan(solver, prepared_state)

    return DifferentiableFMMPlan(
        solver=solver,
        prepared_state=prepared_state,
        reference_positions=positions,
        n_particles=int(state_arr.shape[0]),
        leaf_size=leaf_size_i,
        max_order=max_order_i,
        working_dtype=working_dtype,
        grad_config=_resolve_grad_config(config, grad_config),
        large_n_grad_plan=large_n_grad_plan,
        preset=str(preset),
        basis=str(config.fmm_basis),
    )


def _supports_retain_far_pairs(build_solver: Any) -> bool:
    """Whether both ODISSEO's builder and jaccpot's FarFieldConfig take the knob.

    Probed rather than caught: a ``TypeError`` from the solver constructor has
    many other causes, and swallowing it would turn a real misconfiguration into
    a silently non-differentiable far field.
    """
    import inspect

    if "retain_far_pairs_for_grad" not in inspect.signature(build_solver).parameters:
        return False
    try:
        from jaccpot import FarFieldConfig
    except ImportError:  # pragma: no cover - no jaccpot at all
        return False
    return any(
        f.name == "retain_far_pairs_for_grad" for f in dataclasses.fields(FarFieldConfig)
    )


def _maybe_large_n_grad_plan(solver: Any, prepared_state: Any) -> Optional[Any]:
    """Pre-validate a large-N prepared state, hoisting setup out of the loop.

    Returns ``None`` for the radix path, which needs no plan. A large-N state
    that cannot be differentiated raises here — at plan-build time, with a
    concrete message — rather than on the first gradient evaluation.
    """
    try:
        from jaccpot.runtime._large_n_grad import (
            LargeNPreparedState,
            prepare_large_n_grad_plan,
        )
    except ImportError:
        return None
    if not isinstance(prepared_state, LargeNPreparedState):
        return None
    return prepare_large_n_grad_plan(solver, prepared_state)


def differentiable_fmm_self_acceleration(
    plan: DifferentiableFMMPlan,
    positions: jnp.ndarray,
    mass: jnp.ndarray,
) -> jnp.ndarray:
    """Self-gravity acceleration, differentiable w.r.t. ``positions`` and ``mass``.

    Args:
        plan: Frozen topology from :func:`prepare_differentiable_fmm`.
        positions: Live positions, shape ``(N, 3)``, in the original particle order.
        mass: Live masses, shape ``(N,)``.

    Returns:
        jnp.ndarray: ``(N, 3)`` accelerations in the original particle order.
    """
    out_dtype = jnp.asarray(positions).dtype
    acc = plan.solver.differentiable_accelerations(
        plan.prepared_state,
        jnp.asarray(positions).astype(plan.working_dtype),
        jnp.asarray(mass).astype(plan.working_dtype),
        grad_plan=plan.large_n_grad_plan,
        grad_config=plan.grad_config,
    )
    return jnp.asarray(acc).astype(out_dtype)


def differentiable_total_acceleration(
    plan: DifferentiableFMMPlan,
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
) -> jnp.ndarray:
    """Total acceleration: fixed-topology FMM self-gravity plus external potentials.

    Differentiable with respect to ``state``, ``mass`` and — the point of this
    module — every field of ``params``.
    """
    acc = differentiable_fmm_self_acceleration(plan, state[:, 0, :], mass)
    if len(config.external_accelerations) > 0:
        acc = acc + combined_external_acceleration_vmpa_switch(state, config, params)
    return acc


def _resolve_plan(
    plan: Optional[DifferentiableFMMPlan],
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    grad_config: Optional[Any],
) -> DifferentiableFMMPlan:
    """Return the caller's plan, or build one if the inputs are still concrete."""
    if plan is not None:
        return plan
    if _contains_tracer(state) or _contains_tracer(mass):
        raise NotImplementedError(
            "no plan= was given and state/mass are traced, so the topology "
            "cannot be built here (jaccpot's prepare_state is host-side). Call "
            "plan = prepare_differentiable_fmm(state0, mass, config, params) "
            "OUTSIDE the differentiated function and pass plan=plan."
        )
    return prepare_differentiable_fmm(
        state, mass, config, params, grad_config=grad_config
    )


def integrate_leapfrog_differentiable(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    plan: Optional[DifferentiableFMMPlan] = None,
    num_steps: Optional[int] = None,
    dt: Optional[Any] = None,
    return_history: bool = False,
    grad_config: Optional[Any] = None,
):
    """Fixed-step leapfrog with FMM self-gravity, differentiable end to end.

    Velocity-Verlet, identical in form to :func:`odisseo.integrators.leapfrog`.
    The trailing acceleration of each step is carried into the next one: the
    force depends on positions only, so this is bit-identical to re-evaluating it
    and halves the number of FMM calls (and of reverse passes).

    Args:
        state: Initial primitive state, shape ``(N, 2, 3)``.
        mass: Particle masses, shape ``(N,)``.
        config: Simulation configuration.
        params: Simulation parameters; differentiable.
        plan: Frozen topology from :func:`prepare_differentiable_fmm`. Built here
            when omitted, which requires concrete ``state``/``mass``.
        num_steps: Number of steps; defaults to ``config.num_timesteps``.
        dt: Timestep; defaults to ``params.t_end / num_steps`` (differentiable in
            ``params.t_end``).
        return_history: Return every intermediate state, shape
            ``(num_steps + 1, N, 2, 3)``, instead of only the final one.
        grad_config: Optional ``jaccpot.GradConfig``, used only when this call
            builds the plan itself.

    Returns:
        jnp.ndarray: the final state, or the full history if ``return_history``.

    Notes:
        The topology is frozen for the whole call. For a long integration, run
        several calls and rebuild the plan between them — outside ``jax.grad``,
        which necessarily truncates the gradient at each rebuild — or keep
        ``num_steps`` short enough that :func:`topology_drift` stays small.
    """
    plan_resolved = _resolve_plan(plan, state, mass, config, params, grad_config)

    steps = int(config.num_timesteps if num_steps is None else num_steps)
    if steps < 0:
        raise ValueError("num_steps must be non-negative")
    if plan_resolved.n_particles != int(jnp.asarray(state).shape[0]):
        raise ValueError(
            "plan was built for "
            f"{plan_resolved.n_particles} particles but state has "
            f"{int(jnp.asarray(state).shape[0])}"
        )

    state_curr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    if steps == 0:
        return state_curr[jnp.newaxis, ...] if return_history else state_curr

    dt_val = (params.t_end / steps) if dt is None else dt
    dt_arr = jnp.asarray(dt_val, dtype=state_curr.dtype)

    acc = differentiable_total_acceleration(
        plan_resolved, state_curr, mass_arr, config, params
    )

    history = [state_curr] if return_history else None
    for _ in range(steps):
        pos_new = (
            state_curr[:, 0] + state_curr[:, 1] * dt_arr + 0.5 * acc * (dt_arr**2)
        )
        state_curr = state_curr.at[:, 0].set(pos_new)

        acc_new = differentiable_total_acceleration(
            plan_resolved, state_curr, mass_arr, config, params
        )

        vel_new = state_curr[:, 1] + 0.5 * (acc + acc_new) * dt_arr
        state_curr = state_curr.at[:, 1].set(vel_new)
        acc = acc_new

        if return_history:
            history.append(state_curr)

    if return_history:
        return jnp.stack(history, axis=0)
    return state_curr


def integrate_diffrax_differentiable(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    plan: Optional[DifferentiableFMMPlan] = None,
    t_end: Optional[Any] = None,
    dt0: Optional[Any] = None,
    return_history: bool = False,
    grad_config: Optional[Any] = None,
    max_steps: int = 100_000,
):
    """Adaptive-step diffrax integration with FMM self-gravity, differentiable.

    The vector field is :func:`differentiable_total_acceleration`, so the solve is
    differentiable in ``params`` through diffrax's adjoint
    (``config.diffrax_adjoint_method``).

    Args:
        state: Initial primitive state, shape ``(N, 2, 3)``.
        mass: Particle masses, shape ``(N,)``.
        config: Simulation configuration; selects solver and adjoint.
        params: Simulation parameters; differentiable.
        plan: Frozen topology; built here when omitted.
        t_end: End time; defaults to ``params.t_end``.
        dt0: Initial step; defaults to ``t_end / config.num_timesteps``.
        return_history: Save ``config.num_snapshots`` states instead of only the
            final one.
        grad_config: Optional ``jaccpot.GradConfig`` for a plan built here.
        max_steps: diffrax step budget.

    Returns:
        jnp.ndarray: final state ``(N, 2, 3)``, or ``(num_snapshots, N, 2, 3)``.

    Raises:
        NotImplementedError: for the split-term symplectic solvers, which need a
            two-term formulation this lane does not build.

    Notes:
        diffrax traces the vector field inside its own loop, so this lane runs the
        FMM under an outer trace -- the situation jaccpot flags as workable at
        moderate N but liable to hit host-side ops in its re-run sweeps at large N.
        Prefer :func:`integrate_leapfrog_differentiable` at galaxy scale.
    """
    import diffrax

    plan_resolved = _resolve_plan(plan, state, mass, config, params, grad_config)

    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)

    if config.diffrax_solver == DOPRI5:
        solver = diffrax.Dopri5()
    elif config.diffrax_solver == TSIT5:
        solver = diffrax.Tsit5()
    elif config.diffrax_solver == DOPRI8:
        solver = diffrax.Dopri8()
    else:
        raise NotImplementedError(
            "the differentiable FMM diffrax lane supports DOPRI5, TSIT5 and "
            "DOPRI8; the split-term symplectic solvers need a two-term "
            "formulation that is not wired here. Use "
            "integrate_leapfrog_differentiable for symplectic stepping."
        )

    if config.diffrax_adjoint_method == RECURSIVECHECKPOINTADJOING:
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    elif config.diffrax_adjoint_method == FORWARDMODE:
        adjoint = diffrax.ForwardMode()
    else:
        raise ValueError("unsupported diffrax_adjoint_method")

    def vector_field(t, y, args):
        del t
        mass_in = args
        acc = differentiable_total_acceleration(
            plan_resolved, y, mass_in, config, params
        )
        return jnp.stack((y[:, 1], acc), axis=1)

    t1 = params.t_end if t_end is None else t_end
    step0 = (t1 / int(config.num_timesteps)) if dt0 is None else dt0

    if return_history:
        saveat = diffrax.SaveAt(
            ts=jnp.linspace(0.0, t1, int(config.num_snapshots), endpoint=True)
        )
    else:
        saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=solver,
        t0=0.0,
        t1=t1,
        dt0=step0,
        y0=state_arr,
        args=mass_arr,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=diffrax.PIDController(
            rtol=float(config.fmm_adaptive_rtol),
            atol=float(config.fmm_adaptive_atol),
        ),
        max_steps=int(max_steps),
    )

    ys = jnp.asarray(sol.ys)
    if return_history:
        return ys
    return ys[-1]


def topology_drift(
    plan: DifferentiableFMMPlan,
    positions: jnp.ndarray,
) -> dict[str, float]:
    """How far particles have moved since the plan's topology was built.

    A diagnostic for the fixed-topology window: the frozen tree stays a good
    description of the particle distribution only while displacements are small
    compared with a cell. Call it on the *final* positions of a differentiable
    run (outside ``jax.grad``, on concrete arrays) to decide whether the window
    was too long.

    Args:
        plan: The plan whose topology was frozen.
        positions: Concrete positions, shape ``(N, 3)``.

    Returns:
        dict: ``max_displacement`` and ``rms_displacement`` in position units,
        plus ``max_displacement_over_leaf_extent`` — the same displacement in
        units of the mean leaf-cell extent, which is the number to watch. Values
        well below 1 mean the frozen topology still describes the distribution;
        values around or above 1 mean particles have crossed cells and the
        forward force (not only the gradient) is degraded.
    """
    if _contains_tracer(positions):
        raise NotImplementedError(
            "topology_drift is a host-side diagnostic; call it on concrete "
            "positions, outside jax.grad."
        )
    pos = jnp.asarray(positions).astype(plan.reference_positions.dtype)
    disp = jnp.linalg.norm(pos - plan.reference_positions, axis=1)
    extent = _mean_leaf_extent(plan)
    max_disp = float(jnp.max(disp))
    return {
        "max_displacement": max_disp,
        "rms_displacement": float(jnp.sqrt(jnp.mean(disp**2))),
        "max_displacement_over_leaf_extent": (
            float("nan") if extent is None else max_disp / extent
        ),
    }


def _mean_leaf_extent(plan: DifferentiableFMMPlan) -> Optional[float]:
    """Mean leaf-cell edge length of the frozen tree, or ``None`` if unavailable.

    Falls back to the root-extent/leaf-count estimate when the prepared state
    does not expose per-node bounds, so the diagnostic degrades to an estimate
    rather than to nothing.
    """
    reference = plan.reference_positions
    lo = jnp.min(reference, axis=0)
    hi = jnp.max(reference, axis=0)
    root_extent = float(jnp.max(hi - lo))
    n_leaves = max(1, int(plan.n_particles) // max(1, int(plan.leaf_size)))
    if root_extent <= 0.0:
        return None
    # Cells subdivide in 3D, so linear extent shrinks as the cube root.
    return root_extent / (n_leaves ** (1.0 / 3.0))


def _plan_summary(plan: DifferentiableFMMPlan) -> dict[str, Any]:
    """Small, printable description of a plan (for logs and test diagnostics)."""
    grad_config = plan.grad_config
    return {
        "n_particles": plan.n_particles,
        "leaf_size": plan.leaf_size,
        "max_order": plan.max_order,
        "preset": plan.preset,
        "basis": plan.basis,
        "working_dtype": str(plan.working_dtype),
        "large_n": plan.large_n_grad_plan is not None,
        "grad_config": (
            None
            if grad_config is None
            else {
                f.name: getattr(grad_config, f.name)
                for f in dataclasses.fields(grad_config)
            }
        ),
    }
