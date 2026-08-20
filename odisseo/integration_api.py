from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from odisseo.jaccpot_coupling import (
    integrate_diffrax_jaccpot_active,
    integrate_leapfrog_jaccpot_active,
)
from odisseo.option_classes import (
    DIRECT_ACC,
    DIRECT_ACC_FOR_LOOP,
    DIRECT_ACC_LAXMAP,
    DIRECT_ACC_MATRIX,
    DIRECT_ACC_SHARDING,
    FMM_ACC,
    NO_SELF_GRAVITY,
    SimulationConfig,
    SimulationParams,
)
from odisseo.time_integration import SnapshotData, time_integration


def _resolve_fmm_runtime_profile(
    state: jnp.ndarray,
    config: SimulationConfig,
) -> tuple[str, str, jnp.dtype]:
    """Resolve preset/runtime-path/dtype for jaccpot FMM execution."""
    preset = str(config.fmm_preset).strip().lower()
    runtime_path = str(config.fmm_runtime_path).strip().lower()
    effective_dtype = jnp.dtype(state.dtype)

    auto_large_n = bool(config.fmm_auto_large_n_profile)
    min_particles = max(1, int(config.fmm_large_n_min_particles))
    on_gpu = str(jax.default_backend()).strip().lower() == "gpu"
    if (
        auto_large_n
        and preset == "fast"
        and int(state.shape[0]) >= min_particles
        and on_gpu
    ):
        preset = "large_n_gpu"
        if runtime_path == "auto":
            runtime_path = "large_n"

    if preset == "large_n_gpu" and bool(config.fmm_large_n_force_fp32):
        effective_dtype = jnp.dtype(jnp.float32)
        if runtime_path == "auto":
            runtime_path = "large_n"

    return preset, runtime_path, effective_dtype


def integrate(
    primitive_state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    active_indices_fn: Optional[
        Callable[[int, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ] = None,
    active_indices_schedule: Optional[jnp.ndarray] = None,
    active_mask_schedule: Optional[jnp.ndarray] = None,
    fmm_plan: Optional[object] = None,
):
    """Unified integration API across direct and Jaccpot-FMM backends.

    Selector
    --------
    ``config.acceleration_scheme``:
    - direct schemes (`DIRECT_ACC`, `DIRECT_ACC_LAXMAP`, `DIRECT_ACC_MATRIX`,
      `DIRECT_ACC_FOR_LOOP`, `DIRECT_ACC_SHARDING`, `NO_SELF_GRAVITY`)
      route to legacy ``time_integration``. Those are already differentiable in
      ``params``.
    - ``FMM_ACC`` routes to the Jaccpot coupler workflow -- or, with
      ``config.fmm_differentiable=True``, to the differentiable FMM lane in
      :mod:`odisseo.differentiable`, which supports ``jax.grad`` with respect to
      external-potential parameters, the initial state and masses.
    - ``FMM_ACC`` with ``config.fmm_blockstep=True`` routes to the
      momentum-conserving individual-timestep lane in
      :mod:`odisseo.blockstep_coupling`, configured by
      ``config.blockstep_options``, which is **required** (there is no default
      ``dt_max``). ``config.num_timesteps`` is read as the number
      of **base steps** there, each containing ``2**k_max`` sub-steps. It returns a
      plain final state like every other lane, so the per-base-step diagnostics
      (momentum and energy drift, rung histograms, timings) are dropped -- call
      ``integrate_blockstep_jaccpot`` directly if you need them.
      ``fmm_blockstep`` takes precedence over ``fmm_differentiable``.

    Parameters
    ----------
    fmm_plan:
        Optional :class:`~odisseo.differentiable.DifferentiableFMMPlan`, honoured
        only by the differentiable FMM lane. Build it once with
        ``prepare_differentiable_fmm`` outside the differentiated function and
        pass it here; otherwise the lane builds one per call, which requires
        concrete ``primitive_state``/``mass`` and repeats the tree build on every
        gradient evaluation.
    """
    direct_schemes = {
        DIRECT_ACC,
        DIRECT_ACC_LAXMAP,
        DIRECT_ACC_MATRIX,
        DIRECT_ACC_FOR_LOOP,
        DIRECT_ACC_SHARDING,
        NO_SELF_GRAVITY,
    }

    if int(config.acceleration_scheme) in direct_schemes:
        return time_integration(primitive_state, mass, config, params)

    if int(config.acceleration_scheme) == int(FMM_ACC):
        if bool(getattr(config, "fmm_blockstep", False)):
            return _integrate_fmm_blockstep(primitive_state, mass, config, params)

        if bool(getattr(config, "fmm_differentiable", False)):
            return _integrate_fmm_differentiable(
                primitive_state,
                mass,
                config,
                params,
                active_indices_fn=active_indices_fn,
                active_indices_schedule=active_indices_schedule,
                active_mask_schedule=active_mask_schedule,
                fmm_plan=fmm_plan,
            )

        _reject_traced_forward_fmm_inputs(primitive_state, mass, params)

        fmm_preset, fmm_runtime_path, fmm_working_dtype = _resolve_fmm_runtime_profile(
            primitive_state,
            config,
        )

        common_kwargs = dict(
            state=primitive_state,
            mass=mass,
            config=config,
            params=params,
            num_steps=int(config.num_timesteps),
            active_indices_fn=active_indices_fn,
            active_indices_schedule=active_indices_schedule,
            active_mask_schedule=active_mask_schedule,
            refresh_every=int(config.fmm_refresh_every),
            refresh_after_position_update=bool(
                config.fmm_refresh_after_position_update
            ),
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
            fmm_preset=str(fmm_preset),
            fmm_basis=str(config.fmm_basis),
            fmm_theta=float(config.fmm_theta),
            fmm_runtime_path=str(fmm_runtime_path),
            fmm_working_dtype=fmm_working_dtype,
            fmm_mac_type=str(config.fmm_mac_type),
            fmm_farfield_mode=str(config.fmm_farfield_mode),
            fmm_m2l_chunk_size=(
                None
                if config.fmm_m2l_chunk_size is None
                else int(config.fmm_m2l_chunk_size)
            ),
            fmm_nearfield_mode=str(config.fmm_nearfield_mode),
            fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
            fmm_tree_build_mode=str(config.fmm_tree_build_mode),
            fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
            fmm_fixed_order=(
                None if config.fmm_fixed_order is None else int(config.fmm_fixed_order)
            ),
            fmm_jit_tree=(
                None if config.fmm_jit_tree is None else bool(config.fmm_jit_tree)
            ),
            fmm_jit_traversal=(
                None
                if config.fmm_jit_traversal is None
                else bool(config.fmm_jit_traversal)
            ),
            fmm_max_pair_queue=(
                None
                if config.fmm_max_pair_queue is None
                else int(config.fmm_max_pair_queue)
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
            enforce_static_shape_contract=bool(
                config.fmm_enforce_static_shape_contract
            ),
            static_shape_warmup_prepares=int(
                config.fmm_static_shape_warmup_prepares
            ),
            rematerialize_between_refresh=bool(
                config.fmm_rematerialize_between_refresh
            ),
            return_history=bool(config.return_snapshots),
        )

        if bool(config.fixed_timestep):
            states_or_final = integrate_leapfrog_jaccpot_active(**common_kwargs)
        else:
            states_or_final = integrate_diffrax_jaccpot_active(**common_kwargs)

        if bool(config.return_snapshots):
            states = jnp.asarray(states_or_final)
            target_snaps = int(config.num_snapshots)
            if target_snaps <= 0:
                raise ValueError("num_snapshots must be positive")
            stride = max(1, int(states.shape[0]) // target_snaps)
            snap_states = states[::stride][:target_snaps]
            times = jnp.linspace(0.0, params.t_end, snap_states.shape[0], endpoint=True)
            return SnapshotData(
                times=times,
                states=snap_states,
            )

        return states_or_final

    raise ValueError(
        "acceleration_scheme must be a direct scheme or FMM_ACC"
    )


def _integrate_fmm_blockstep(primitive_state, mass, config, params):
    """Run the block-step lane and return a plain final state.

    Returns the same shape as every other :func:`integrate` lane rather than the
    richer ``BlockStepResult``, so a caller does not have to branch on the config
    to know what it was handed. The diagnostics that are dropped here -- momentum
    and energy drift per base step, the rung histograms, the per-step timings --
    are not recoverable from the state, so anything that needs them should call
    ``integrate_blockstep_jaccpot`` directly.

    ``config.num_timesteps`` is the number of **base steps**; see the note on
    ``SimulationConfig.fmm_blockstep``.

    Parameters
    ----------
    primitive_state:
        ``(N, 2, 3)`` positions and velocities.
    mass:
        ``(N,)`` particle masses.
    config:
        Must carry ``fmm_blockstep=True`` and a ``blockstep_options``.
    params:
        Simulation parameters; ``params.G`` is used.

    Returns
    -------
    The final ``(N, 2, 3)`` state.
    """
    from odisseo.blockstep_coupling import (
        BlockStepOptions,
        integrate_blockstep_jaccpot,
    )

    options = getattr(config, "blockstep_options", None)
    if options is None:
        raise ValueError(
            "config.fmm_blockstep=True needs config.blockstep_options set; there "
            "is no default, because BlockStepOptions has no default dt_max and "
            "picking one here would be the worst kind of guess. A dt_max below "
            "every particle's own criterion puts the whole system on rung 0, and "
            "the block scheme then silently collapses to a shared timestep -- a "
            "run that looks healthy while exercising none of this lane. Size it "
            "from the acceleration distribution, e.g. a high percentile of "
            "eta*sqrt(softening/|a_i|); tools/blockstep_fmm_demo.py does exactly "
            "that."
        )
    if not isinstance(options, BlockStepOptions):
        raise TypeError(
            "config.blockstep_options must be a BlockStepOptions or None; got "
            f"{type(options).__name__}"
        )

    n_base = int(config.num_timesteps)
    if n_base < 1:
        raise ValueError(
            "config.num_timesteps is the number of base steps for the block-step "
            f"lane and must be >= 1; got {n_base!r}"
        )

    # `track_energy=False`: the energy diagnostic is an exact O(N^2) pair sum, and
    # this entry point discards it anyway, so computing it would be pure cost --
    # 34.6 s per evaluation at N = 1e6.
    result = integrate_blockstep_jaccpot(
        primitive_state,
        mass,
        config,
        params,
        options=options,
        n_base=n_base,
        track_energy=False,
    )
    return result.state


def _reject_traced_forward_fmm_inputs(
    primitive_state: jnp.ndarray,
    mass: jnp.ndarray,
    params: SimulationParams,
) -> None:
    """Fail loudly when a gradient is attempted on the forward FMM lane.

    The forward coupler passes ``params`` as a *static* jit argument and drives
    jaccpot's host-side ``prepare_state``/``evaluate_prepared_state`` pair, which
    reads prebaked expansions. Under ``jax.grad`` that produces either an opaque
    "non-hashable static argument" error or -- worse -- a self-gravity term with
    no sensitivity at all. Point at the differentiable lane instead.
    """
    from odisseo.differentiable import _contains_tracer

    traced_params = _contains_tracer(params)
    traced_state = _contains_tracer(primitive_state) or _contains_tracer(mass)
    if not (traced_params or traced_state):
        return

    traced_what = ", ".join(
        part
        for part, hit in (("params", traced_params), ("state/mass", traced_state))
        if hit
    )
    raise NotImplementedError(
        f"traced {traced_what} reached the forward FMM lane, which cannot be "
        "differentiated: it passes params as a static jit argument and evaluates "
        "jaccpot's PREBAKED prepared-state expansions, so a gradient would "
        "either fail on hashing or come back with no self-gravity sensitivity. "
        "Set config.fmm_differentiable=True (and pass a plan from "
        "odisseo.differentiable.prepare_differentiable_fmm) to differentiate the "
        "FMM lane, or use a direct acceleration scheme."
    )


def _integrate_fmm_differentiable(
    primitive_state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    active_indices_fn,
    active_indices_schedule,
    active_mask_schedule,
    fmm_plan,
):
    """Run the differentiable FMM lane and package snapshots like the forward one."""
    from odisseo.differentiable import (
        integrate_diffrax_differentiable,
        integrate_leapfrog_differentiable,
    )

    if (
        active_indices_fn is not None
        or active_indices_schedule is not None
        or active_mask_schedule is not None
    ):
        raise NotImplementedError(
            "the differentiable FMM lane integrates all particles every step; "
            "active-particle scheduling is a forward-throughput optimisation "
            "and is not wired into the gradient path."
        )

    return_history = bool(config.return_snapshots)
    if bool(config.fixed_timestep):
        out = integrate_leapfrog_differentiable(
            primitive_state,
            mass,
            config,
            params,
            plan=fmm_plan,
            return_history=return_history,
        )
    else:
        out = integrate_diffrax_differentiable(
            primitive_state,
            mass,
            config,
            params,
            plan=fmm_plan,
            return_history=return_history,
        )

    if not return_history:
        return out

    states = jnp.asarray(out)
    target_snaps = int(config.num_snapshots)
    if target_snaps <= 0:
        raise ValueError("num_snapshots must be positive")
    stride = max(1, int(states.shape[0]) // target_snaps)
    snap_states = states[::stride][:target_snaps]
    times = jnp.linspace(0.0, params.t_end, snap_states.shape[0], endpoint=True)
    return SnapshotData(times=times, states=snap_states)
