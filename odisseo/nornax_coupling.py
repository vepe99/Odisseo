"""Couple ODISSEO to the nornax higher-order Hermite integrators.

nornax (https://github.com/TobiBu/nornax) provides backend-agnostic Hermite
predictor-corrector integrators (orders 4/6/8) that consume a ``ForceModel``
returning acceleration + jerk (+ snap/...). This module adapts ODISSEO's
``(N, 2, 3)`` state and ``SimulationConfig``/``SimulationParams`` onto that API.

Backend selection follows ODISSEO's existing config semantics:

* ``config.integrator == HERMITE`` selects the nornax Hermite integrator.
* ``config.acceleration_scheme`` selects the self-gravity backend it is fed:
    - ``FMM_ACC``          -> nornax ``JaccpotForceModel`` (jaccpot FMM, forward
                              only; jaccpot's FMM has no autodiff rule today).
    - ``NO_SELF_GRAVITY``  -> zero self-gravity (external potentials only).
    - any direct scheme    -> nornax ``DirectSumGravity`` (differentiable O(N^2)).

When ``config.external_accelerations`` is non-empty the self-gravity backend is
wrapped so that ODISSEO's analytic external potentials are added on top, both to
the acceleration and (via ``jax.jvp`` along the velocity) to the jerk. External
potentials are pure, differentiable JAX functions of position, so this composed
model stays autodiff-safe on the direct-sum path.

nornax is imported lazily so ODISSEO keeps importing without the optional
dependency (mirrors ``require_diffrax`` in nornax's own code).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from odisseo.option_classes import (
    FMM_ACC,
    NO_SELF_GRAVITY,
    SimulationConfig,
    SimulationParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch


def require_nornax():
    """Import and return the nornax module, or raise a helpful error."""
    try:
        import nornax  # noqa: PLC0415

        return nornax
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "The HERMITE integrator requires the optional 'nornax' package. "
            "Install it with `pip install -e <nornax checkout>` or "
            "`pip install git+https://github.com/TobiBu/nornax`."
        ) from exc


# ---------------------------------------------------------------------------
# Force models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZeroSelfGravity:
    """nornax ForceModel returning zero self-gravity (external-only runs)."""

    def derivatives(self, t, positions, velocities, masses, *, max_order, args=None):
        nornax = require_nornax()
        zeros = jnp.zeros_like(positions)
        jerk = zeros if max_order >= 2 else None
        return nornax.ForceDerivatives(acc=zeros, jerk=jerk)


@dataclass(frozen=True)
class _ComposedForceModel:
    """Wrap a self-gravity ForceModel and add ODISSEO external potentials.

    The external acceleration is summed onto the inner acceleration, and its
    jerk ``d/dt a_ext = J_ext . v`` is obtained from a forward-mode ``jvp`` of
    the external-acceleration field along the velocity.
    """

    inner: Any
    config: SimulationConfig
    params: SimulationParams

    def derivatives(self, t, positions, velocities, masses, *, max_order, args=None):
        nornax = require_nornax()
        base = self.inner.derivatives(
            t, positions, velocities, masses, max_order=max_order, args=args
        )
        acc_ext, jerk_ext = _external_acc_and_jerk(
            positions, velocities, self.config, self.params
        )
        acc = base.acc + acc_ext
        jerk = base.jerk
        if jerk is not None:
            jerk = jerk + jerk_ext
        # Higher derivatives (snap/...) of the external field are not composed
        # yet; they stay as provided by the inner model.
        return base._replace(acc=acc, jerk=jerk)


def _external_acc_and_jerk(positions, velocities, config, params):
    """Return (external acceleration, external jerk) via jvp along velocity.

    ``combined_external_acceleration_vmpa_switch`` takes the full ODISSEO
    ``(N, 2, 3)`` state but only reads positions, so we build a state from the
    perturbed positions and differentiate w.r.t. positions with tangent
    ``velocities`` (``da/dt = da/dx . dx/dt = J . v``).
    """
    vel_const = jax.lax.stop_gradient(velocities)

    def acc_of_positions(pos):
        state = jnp.stack((pos, vel_const), axis=1)
        return combined_external_acceleration_vmpa_switch(state, config, params)

    acc_ext, jerk_ext = jax.jvp(acc_of_positions, (positions,), (velocities,))
    return acc_ext, jerk_ext


def _fmm_solver_kwargs(config: SimulationConfig, working_dtype) -> dict:
    """Map SimulationConfig FMM knobs onto _build_fmm_solver's kwargs."""
    return dict(
        working_dtype=working_dtype,
        fmm_preset=str(config.fmm_preset),
        fmm_basis=str(config.fmm_basis),
        fmm_theta=float(config.fmm_theta),
        fmm_runtime_path=str(config.fmm_runtime_path),
        fmm_mac_type=str(config.fmm_mac_type),
        fmm_farfield_mode=str(config.fmm_farfield_mode),
        fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
        fmm_nearfield_mode=str(config.fmm_nearfield_mode),
        fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
        fmm_tree_build_mode=str(config.fmm_tree_build_mode),
        fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
        fmm_fixed_order=config.fmm_fixed_order,
        leaf_size=int(config.fmm_leaf_size),
        fmm_jit_tree=config.fmm_jit_tree,
        fmm_jit_traversal=config.fmm_jit_traversal,
        fmm_max_pair_queue=config.fmm_max_pair_queue,
        fmm_pair_process_block=config.fmm_pair_process_block,
        fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=(
            config.fmm_prepare_stage_memory_split_enabled
        ),
        fmm_upward_leaf_batch_size=config.fmm_upward_leaf_batch_size,
    )


def _build_self_gravity_model(config: SimulationConfig, params: SimulationParams, *, working_dtype):
    """Return the nornax ForceModel for self-gravity per acceleration_scheme."""
    nornax = require_nornax()
    scheme = int(config.acceleration_scheme)

    if scheme == int(FMM_ACC):
        from odisseo.jaccpot_coupling import _build_fmm_solver

        solver = _build_fmm_solver(
            config=config,
            params=params,
            **_fmm_solver_kwargs(config, working_dtype),
        )
        options = nornax.JaccpotOptions(
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
            theta=float(config.fmm_theta),
            jerk_mode=str(config.hermite_jerk_mode),
        )
        return nornax.JaccpotForceModel(solver, options)

    if scheme == int(NO_SELF_GRAVITY):
        return _ZeroSelfGravity()

    # Every other direct scheme -> differentiable direct-sum backend.
    from nornax.forces.direct import DirectSumGravity

    return DirectSumGravity(
        G=float(params.G),
        softening=float(config.softening),
    )


def build_force_model(config: SimulationConfig, params: SimulationParams, *, working_dtype=None):
    """Build the (possibly composed) nornax ForceModel for a Hermite run."""
    inner = _build_self_gravity_model(config, params, working_dtype=working_dtype)
    if len(config.external_accelerations) > 0:
        return _ComposedForceModel(inner=inner, config=config, params=params)
    return inner


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------


# Cap on the diffrax adaptive step buffer (nornax's solve_adaptive_to_time uses
# SaveAt(steps=True), sizing an array of ~ (t_end / min_dt) states). We clamp the
# controller's min_dt so this stays bounded regardless of the configured value.
_ADAPTIVE_MAX_STEPS = 200_000

# Empirical opening-angle below which fixed-step Hermite-4 tracks exact
# direct-summation on the solidfmm (p=4, Dehnen-MAC) backend; above it the
# integrator can diverge. See _warn_fmm_hermite_stability.
_FMM_HERMITE_STABLE_THETA = 0.3


class HermiteFMMStabilityWarning(UserWarning):
    """Hermite integration driven by approximate FMM forces may be unstable.

    High-order predictor-corrector (Hermite) schemes assume the acceleration and
    its time derivatives are mutually consistent along the trajectory. FMM
    forces are only *piecewise*-smooth in position (they jump as particles cross
    tree-cell / MAC boundaries), so at a loose opening angle the integrator can
    diverge even though the instantaneous acceleration and jerk are accurate.
    This mirrors established practice: Dehnen's falcON FMM integrates with
    leapfrog, and P3T/PeTar reserve Hermite for the *exact direct-summation*
    short-range forces while using leapfrog for the tree/FMM long-range part.
    """


def _warn_fmm_hermite_stability(config: SimulationConfig) -> None:
    """Warn when pairing a Hermite integrator with the approximate FMM backend."""
    theta = float(config.fmm_theta)
    detail = (
        f" your fmm_theta={theta:g} is above the empirically stable threshold "
        f"(~{_FMM_HERMITE_STABLE_THETA:g} for solidfmm p=4); expect divergence"
        if theta > _FMM_HERMITE_STABLE_THETA
        else f" your fmm_theta={theta:g} is at/below the empirically stable "
        f"threshold (~{_FMM_HERMITE_STABLE_THETA:g} for solidfmm p=4)"
    )
    warnings.warn(
        "Hermite integration with the jaccpot FMM backend is EXPERIMENTAL and "
        "can be unstable: FMM forces are only piecewise-smooth in position, so a "
        "high-order predictor-corrector may diverge even when the instantaneous "
        "acceleration/jerk are accurate." + detail + ". For approximate "
        "(tree/FMM) forces prefer integrator=LEAPFROG; reserve Hermite for exact "
        "acceleration_scheme=DIRECT_ACC (or a tight-theta, high-order FMM). "
        "Tighten fmm_theta and/or raise fmm_max_order to improve stability.",
        HermiteFMMStabilityWarning,
        stacklevel=3,
    )


def _validate(config: SimulationConfig) -> None:
    order = int(config.hermite_order)
    if order not in (4, 6, 8):
        raise ValueError(f"hermite_order must be 4, 6, or 8; got {order}")
    scheme = int(config.acceleration_scheme)
    if scheme == int(FMM_ACC) and order != 4:
        raise NotImplementedError(
            "The jaccpot FMM backend supports only Hermite-4 today "
            "(nornax's JaccpotForceModel is capped at order 4). Use a direct "
            "acceleration_scheme for Hermite-6/8."
        )
    if bool(config.fixed_timestep) and order != 4:
        raise NotImplementedError(
            "Fixed-timestep Hermite uses the differentiable Hermite-4 scan "
            "(hermite_order must be 4). Set fixed_timestep=False for "
            "Hermite-6/8 (adaptive stepping)."
        )


def _pack(final_state) -> jnp.ndarray:
    """Repack a nornax NBodyState into ODISSEO's (N, 2, 3) layout."""
    return jnp.stack((final_state.positions, final_state.velocities), axis=1)


def _make_controller(config: SimulationConfig, params: SimulationParams):
    """Build an Aarseth controller with a min_dt clamped for the diffrax buffer."""
    nornax = require_nornax()
    floor = float(params.t_end) / _ADAPTIVE_MAX_STEPS
    min_dt = max(float(config.hermite_min_dt), floor)
    return nornax.AarsethController(
        eta=float(config.hermite_eta),
        min_dt=min_dt,
        max_dt=float(config.hermite_max_dt),
    )


def _fixed_step(positions, velocities, mass, force_model, config, params):
    """Fixed-dt Hermite-4 predictor-corrector to t_end (dt = t_end/num_timesteps).

    Traceable (direct-sum) backends run through ``jax.lax.scan`` so the rollout
    is reverse-mode differentiable and memory-light. The jaccpot FMM backend is
    not JAX-traceable (Python-driven, tracer-guarded), so it is stepped in a
    host loop instead.
    """
    from nornax.initialize import initialize_state
    from nornax.solvers.hermite4 import hermite4_step

    n_steps = int(config.num_timesteps)
    dt = jnp.asarray(params.t_end, dtype=positions.dtype) / n_steps
    state = initialize_state(positions, velocities, mass, force_model, max_order=2)

    if int(config.acceleration_scheme) == int(FMM_ACC):
        for _ in range(n_steps):
            state = hermite4_step(state, dt, force_model, args=None)
        return state

    def body(s, _):
        return hermite4_step(s, dt, force_model, args=None), None

    final_state, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final_state


def _adaptive_segment(positions, velocities, mass, force_model, config, params, controller, t0, t1):
    """Advance from t0 to t1 with the backend-appropriate adaptive solver.

    FMM self-gravity is host-driven and forward-only, so it uses nornax's
    Python-loop ``hermite4_adaptive_solve``; the differentiable direct-sum
    backend uses the diffrax-backed ``solve_adaptive_to_time`` (orders 4/6/8).
    """
    nornax = require_nornax()
    order = int(config.hermite_order)

    if int(config.acceleration_scheme) == int(FMM_ACC):
        from nornax.initialize import initialize_state
        from nornax.solvers.hermite4 import hermite4_adaptive_solve

        state = initialize_state(
            positions, velocities, mass, force_model, time=float(t0), max_order=2
        )
        return hermite4_adaptive_solve(
            state,
            force_model,
            controller,
            t_final=float(t1),
            atol=float(config.hermite_atol),
        ).final_state

    return nornax.solve_adaptive_to_time(
        positions,
        velocities,
        mass,
        force_model,
        t_final=float(t1),
        order=order,
        controller=controller,
        atol=float(config.hermite_atol),
        time=float(t0),
    ).final_state


def integrate_hermite_nornax(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    return_history: bool = False,
) -> jnp.ndarray:
    """Integrate an ODISSEO state with a nornax Hermite integrator.

    Args:
        state: ODISSEO primitive state, shape ``(N, 2, 3)``.
        mass: Particle masses, shape ``(N,)``.
        config: SimulationConfig; ``integrator`` must be ``HERMITE``.
        params: SimulationParams (provides ``G`` and ``t_end``).
        return_history: If True, return a snapshot stack ``(num_snapshots, N, 2, 3)``
            sampled at evenly spaced times; otherwise return the final ``(N, 2, 3)``.

    Returns:
        Final state ``(N, 2, 3)`` or a snapshot stack ``(S, N, 2, 3)``.

    Notes:
        * ``fixed_timestep=True`` -> fixed-dt Hermite-4 scan to ``t_end``
          (differentiable, memory-light). Requires ``hermite_order == 4``.
        * ``fixed_timestep=False`` -> adaptive stepping to ``t_end``. FMM runs
          forward via a host loop; the direct-sum backend runs the diffrax
          solver (orders 4/6/8, differentiable).
    """
    _validate(config)
    if int(config.acceleration_scheme) == int(FMM_ACC):
        _warn_fmm_hermite_stability(config)

    state = jnp.asarray(state)
    mass = jnp.asarray(mass)
    positions = state[:, 0]
    velocities = state[:, 1]

    force_model = build_force_model(config, params, working_dtype=state.dtype)

    if return_history:
        num_snapshots = int(config.num_snapshots)
        if num_snapshots <= 0:
            raise ValueError("num_snapshots must be positive")
        controller = _make_controller(config, params)
        times = jnp.linspace(0.0, float(params.t_end), num_snapshots, endpoint=True)
        snapshots = [state]
        pos, vel = positions, velocities
        for i in range(1, num_snapshots):
            fs = _adaptive_segment(
                pos, vel, mass, force_model, config, params,
                controller, times[i - 1], times[i],
            )
            pos, vel = fs.positions, fs.velocities
            snapshots.append(_pack(fs))
        return jnp.stack(snapshots, axis=0)

    if bool(config.fixed_timestep):
        final_state = _fixed_step(
            positions, velocities, mass, force_model, config, params
        )
        return _pack(final_state)

    controller = _make_controller(config, params)
    final_state = _adaptive_segment(
        positions, velocities, mass, force_model, config, params,
        controller, 0.0, params.t_end,
    )
    return _pack(final_state)
