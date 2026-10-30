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
    """Resolve preset/runtime-path/dtype for jaccpot FMM execution.

    The runtime path is resolved for Odisseo's own records only: it is accepted
    and ignored since the jaccpot 2026-10 cleanup (X4), where the preset alone
    selects the lane, and ``_build_fmm_solver`` no longer passes it on.
    """
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


#: The lane names :func:`resolve_lane` can return, in dispatch-precedence order.
INTEGRATION_LANES = (
    "direct",
    "fmm_mesh",
    "fmm_mesh_runner",
    "fmm_blockstep",
    "fmm_differentiable",
    "fmm_runner",
    "fmm_forward",
)


def _runner_selected(
    config: SimulationConfig,
    *,
    num_particles: Optional[int],
    backend: Optional[str],
    scheduled: bool,
) -> bool:
    """Whether an ``FMM_ACC`` forward run goes to the runner lane.

    ``fmm_preset="galaxy"`` asks for it (options it does not support then raise
    there, rather than run another lane). The automatic large-N profile -- preset
    ``"fast"`` on a GPU with at least ``fmm_large_n_min_particles`` particles, which
    picked ``"large_n_gpu"`` on the coupler before -- picks it when the runner
    supports the run, and keeps the coupler's ``"large_n_gpu"`` loop otherwise.
    """
    from odisseo.runner_coupling import runner_unsupported

    preset = str(config.fmm_preset).strip().lower()
    if preset == "galaxy":
        return True
    n = int(config.N_particles) if num_particles is None else int(num_particles)
    on_gpu = (
        str(jax.default_backend() if backend is None else backend).strip().lower()
        == "gpu"
    )
    auto_large_n = (
        bool(config.fmm_auto_large_n_profile)
        and preset == "fast"
        and on_gpu
        and n >= max(1, int(config.fmm_large_n_min_particles))
    )
    return auto_large_n and runner_unsupported(config, scheduled=scheduled) is None


def _mesh_runner_selected(config: SimulationConfig) -> bool:
    """Whether an ``fmm_mesh`` run goes to the runner's multi-GPU mode.

    A ``MeshOptions`` as ``config.mesh_options`` keeps the older mesh lane
    (:mod:`odisseo.mesh_coupling`), whatever the preset, exactly as before. Otherwise
    the production settings ask for the runner, as on one card: ``fmm_preset=
    "galaxy"``, or a ``MeshRunnerOptions`` (:mod:`odisseo.mesh_runner_coupling`);
    options it does not support then raise there, rather than run another lane.
    """
    from odisseo.mesh_coupling import MeshOptions
    from odisseo.mesh_runner_coupling import MeshRunnerOptions

    options = getattr(config, "mesh_options", None)
    if isinstance(options, MeshOptions):
        return False
    return (
        isinstance(options, MeshRunnerOptions)
        or str(config.fmm_preset).strip().lower() == "galaxy"
    )


def resolve_lane(
    config: SimulationConfig,
    *,
    num_particles: Optional[int] = None,
    backend: Optional[str] = None,
    scheduled: bool = False,
) -> str:
    """Name the lane :func:`integrate` will dispatch this config to.

    :func:`integrate` returns a plain state on every lane, deliberately, so the
    return value cannot tell a caller which one ran -- and a selector that fell
    through to the wrong lane still returns a plausible ``(N, 2, 3)``. This is
    the single source of truth for that decision: ``integrate`` dispatches on it
    rather than re-deriving the conditions, so what is reported here cannot drift
    from what actually runs.

    Pure and cheap -- it reads flags and integrates nothing, so it is also the way
    to check what a config would do before committing to a long run.

    ``"fmm_runner"`` is jaccpot's ``ProductionRunner`` (:mod:`odisseo.runner_coupling`):
    ``fmm_preset="galaxy"``, or the automatic large-N profile (preset ``"fast"`` on a
    GPU at ``fmm_large_n_min_particles`` or more) for a run the runner does; that
    choice depends on the particle count, the backend and the active-particle
    schedule, which :func:`integrate` passes here.

    With ``fmm_mesh=True``: ``"fmm_mesh_runner"`` -- the same runner in its
    multi-GPU mode (:mod:`odisseo.mesh_runner_coupling`) -- for ``fmm_preset=
    "galaxy"`` or a ``MeshRunnerOptions`` as ``mesh_options``; ``"fmm_mesh"``, the
    older mesh lane (:mod:`odisseo.mesh_coupling`), for a ``MeshOptions`` and
    otherwise.

    Parameters
    ----------
    config:
        The configuration to resolve.
    num_particles:
        The particle count; ``None`` reads ``config.N_particles``.
    backend:
        ``jax.default_backend()`` when ``None``.
    scheduled:
        Whether the call carries an active-particle schedule
        (``active_indices_fn``, ``active_indices_schedule`` or
        ``active_mask_schedule``).

    Returns
    -------
    str
        One of :data:`INTEGRATION_LANES`.
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
        return "direct"
    if int(config.acceleration_scheme) == int(FMM_ACC):
        # fmm_blockstep takes precedence over fmm_differentiable.
        if bool(getattr(config, "fmm_mesh", False)):
            if _mesh_runner_selected(config):
                return "fmm_mesh_runner"
            return "fmm_mesh"
        if bool(getattr(config, "fmm_blockstep", False)):
            return "fmm_blockstep"
        if bool(getattr(config, "fmm_differentiable", False)):
            return "fmm_differentiable"
        if _runner_selected(
            config,
            num_particles=num_particles,
            backend=backend,
            scheduled=scheduled,
        ):
            return "fmm_runner"
        return "fmm_forward"
    raise ValueError(
        f"unknown acceleration_scheme {int(config.acceleration_scheme)}"
    )


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
    return_lane: bool = False,
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
    - ``FMM_ACC`` with ``config.fmm_preset="galaxy"`` -- and, by default, preset
      ``"fast"`` on a GPU with at least ``fmm_large_n_min_particles`` particles --
      routes to jaccpot's ``ProductionRunner`` (lane ``"fmm_runner"``,
      :mod:`odisseo.runner_coupling`): jaccpot's galaxy settings (leaf 64, p6,
      the preset's MAC, or ``config.fmm_galaxy_settings``), not ``fmm_leaf_size`` /
      ``fmm_max_order`` / ``fmm_theta``. The preset's MAC is Dehnen's eq (16a) at
      ``adaptive_eps`` 1e-4 since jaccpot's 2026-10-21 preset change (the geometric
      MAC at theta 0.8 before): under it G and the softening are the solver's, so a
      call with another ``params.G`` or ``softening`` compiles again (``dt`` and the
      potentials' parameters stay traced); ``fmm_galaxy_settings=GalaxySettings(
      mac_type="dehnen", adaptive_eps=None)`` is the geometric MAC
      (``docs/source/production_fmm.md``). ``fmm_preset="large_n_gpu"`` keeps the
      coupler's own loop. History runs return the state every
      ``num_timesteps // num_snapshots`` steps from the initial one, as host
      arrays, with the runner's energy and angular momentum; call
      ``integrate_runner_jaccpot`` for the diagnostics of a plain run.
    - ``FMM_ACC`` with ``config.fmm_mesh=True`` and ``fmm_preset="galaxy"`` (or a
      ``MeshRunnerOptions`` as ``config.mesh_options``) routes to the same runner in
      its multi-GPU mode (lane ``"fmm_mesh_runner"``,
      :mod:`odisseo.mesh_runner_coupling`): one process drives every card of the
      mesh (``MeshRunnerOptions.ndev``; ``None``: every visible device), the same
      settings, field and snapshots as ``"fmm_runner"``; ``params.G`` and the
      softening are the solver's, and the final state is a host NumPy array in
      input order. A ``MeshOptions`` keeps the older mesh lane (``"fmm_mesh"``,
      :mod:`odisseo.mesh_coupling`) unchanged (``docs/source/parallelism.md``).

    Parameters
    ----------
    return_lane:
        When ``True``, return ``(result, lane)`` instead of bare ``result``,
        where ``lane`` is the :func:`resolve_lane` name of the lane that ran.
        Off by default so the return type is unchanged for existing callers.

        Every lane returns a plain state, so the result alone cannot tell you
        which one ran, and a selector that fell through to the wrong lane still
        returns a plausible ``(N, 2, 3)``. This is how a caller -- or a test --
        pins dispatch on something exact and integer-like rather than by
        comparing float arrays, which cannot distinguish two lanes on GPU at any
        tolerance (see the control measurement in
        ``tests/test_blockstep_fmm.py``).

    fmm_plan:
        Optional :class:`~odisseo.differentiable.DifferentiableFMMPlan`, honoured
        only by the differentiable FMM lane. Build it once with
        ``prepare_differentiable_fmm`` outside the differentiated function and
        pass it here; otherwise the lane builds one per call, which requires
        concrete ``primitive_state``/``mass`` and repeats the tree build on every
        gradient evaluation.
    """
    scheduled = (
        active_indices_fn is not None
        or active_indices_schedule is not None
        or active_mask_schedule is not None
    )
    lane = resolve_lane(
        config,
        num_particles=int(jnp.shape(primitive_state)[0]),
        scheduled=scheduled,
    )

    def _tagged(value):
        return (value, lane) if return_lane else value

    if lane == "direct":
        return _tagged(time_integration(primitive_state, mass, config, params))

    if lane == "fmm_mesh":
        return _tagged(_integrate_fmm_mesh(primitive_state, mass, config, params))

    if lane == "fmm_blockstep":
        return _tagged(
            _integrate_fmm_blockstep(primitive_state, mass, config, params)
        )

    if lane == "fmm_differentiable":
        return _tagged(
            _integrate_fmm_differentiable(
                primitive_state,
                mass,
                config,
                params,
                active_indices_fn=active_indices_fn,
                active_indices_schedule=active_indices_schedule,
                active_mask_schedule=active_mask_schedule,
                fmm_plan=fmm_plan,
            )
        )

    _reject_traced_forward_fmm_inputs(primitive_state, mass, params)

    if lane == "fmm_runner":
        from odisseo.runner_coupling import integrate_runner_jaccpot

        out = integrate_runner_jaccpot(
            primitive_state, mass, config, params, scheduled=scheduled
        )
        return _tagged(
            out.snapshots if bool(config.return_snapshots) else out.state
        )

    if lane == "fmm_mesh_runner":
        from odisseo.mesh_runner_coupling import integrate_mesh_runner_jaccpot

        out = integrate_mesh_runner_jaccpot(
            primitive_state, mass, config, params, scheduled=scheduled
        )
        return _tagged(
            out.snapshots if bool(config.return_snapshots) else out.state
        )

    # lane == "fmm_forward" -- resolve_lane raises on anything else.

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

    if bool(config.return_snapshots) and bool(config.fixed_timestep):
        # the coupler's history is the state after each step, without the initial
        return _tagged(
            _fixed_step_snapshots(
                states_or_final,
                config,
                params,
                initial=primitive_state,
            )
        )
    if bool(config.return_snapshots):
        states = jnp.asarray(states_or_final)
        target_snaps = int(config.num_snapshots)
        if target_snaps <= 0:
            raise ValueError("num_snapshots must be positive")
        stride = max(1, int(states.shape[0]) // target_snaps)
        snap_states = states[::stride][:target_snaps]
        times = jnp.linspace(0.0, params.t_end, snap_states.shape[0], endpoint=True)
        return _tagged(
            SnapshotData(
                times=times,
                states=snap_states,
            )
        )

    return _tagged(states_or_final)


def _fixed_step_snapshots(
    history: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    initial: Optional[jnp.ndarray] = None,
) -> SnapshotData:
    """Package a fixed-step history as every FMM lane and the direct lane do.

    Snapshot ``k`` is the state after ``k * K`` steps, ``K = num_timesteps //
    num_snapshots``, for ``k = 0 .. num_snapshots - 1``: the initial state first,
    the final one not among them, at times ``k * K * dt``. That is the direct lane
    (``time_integration``) when ``num_snapshots`` divides ``num_timesteps``, and the
    runner lane (:mod:`odisseo.runner_coupling`). Until 2026-10 the coupler's
    snapshots were the states after steps ``1, 1 + K, ...`` -- one step late, the
    initial state missing -- and both FMM lanes labelled them ``linspace(0, t_end,
    num_snapshots)``, ``t_end`` on a state ``K`` steps short of it.

    Parameters
    ----------
    history:
        ``[steps(+1), N, 2, 3]`` -- the state after every step, preceded by the
        initial state unless ``initial`` is given.
    config:
        ``num_timesteps`` and ``num_snapshots``.
    params:
        ``t_end``: ``dt = t_end / num_timesteps``.
    initial:
        The initial state, when ``history`` does not start with it.

    Returns
    -------
    SnapshotData
        ``times`` and ``states``.
    """
    states = jnp.asarray(history)
    if initial is not None:
        states = jnp.concatenate(
            [jnp.asarray(initial, dtype=states.dtype)[None], states], axis=0
        )
    target_snaps = int(config.num_snapshots)
    if target_snaps <= 0:
        raise ValueError("num_snapshots must be positive")
    num_steps = int(config.num_timesteps)
    stride = max(1, num_steps // target_snaps)
    snap_states = states[::stride][:target_snaps]
    dt = params.t_end / num_steps
    times = jnp.arange(snap_states.shape[0]) * (stride * dt)
    return SnapshotData(times=times, states=snap_states)


def _integrate_fmm_mesh(primitive_state, mass, config, params):
    """Run the multi-GPU mesh lane and return a plain final state.

    Returns the same shape as every other :func:`integrate` lane rather than the
    richer ``MeshResult``, so a caller does not have to branch on the config to know
    what it was handed. The dropped diagnostics -- per-step timings, the first
    force's per-device pair counts and overflow flags, the conserved quantities --
    are not recoverable from the state, so anything that needs them should call
    ``mesh_coupling.integrate_mesh_jaccpot`` directly.

    Parameters
    ----------
    primitive_state:
        ``(N, 2, 3)`` positions and velocities.
    mass:
        ``(N,)`` particle masses.
    config:
        Must carry ``fmm_mesh=True`` and a ``mesh_options``.
    params:
        Simulation parameters; ``params.G`` is used.

    Returns
    -------
    The final ``(N, 2, 3)`` state.
    """
    from odisseo.mesh_coupling import MeshOptions, integrate_mesh_jaccpot

    options = getattr(config, "mesh_options", None)
    if options is None:
        raise ValueError(
            "config.fmm_mesh=True needs config.mesh_options set; there is no "
            "default, because MeshOptions has no default dt and this lane runs a "
            "FIXED-step KDK -- a guessed dt would be silently wrong rather than "
            "loudly wrong. Size it from the dynamical time at the smallest scale "
            "you care to resolve."
        )
    if not isinstance(options, MeshOptions):
        raise TypeError(
            "config.mesh_options must be a MeshOptions or None; got "
            f"{type(options).__name__}"
        )
    if bool(getattr(config, "fmm_differentiable", False)):
        raise NotImplementedError(
            "the mesh lane has no gradient path. `make_force_evaluator` does accept "
            "differentiable=True, but the rollout has no adjoint and the halo "
            "exchange needs jax >= 0.9.1 for its reverse pass. Use "
            "fmm_differentiable on a single device, or drop it here."
        )
    if bool(getattr(config, "fmm_blockstep", False)):
        raise NotImplementedError(
            "the mesh lane is a fixed-step KDK and has no rung machinery; "
            "fmm_mesh and fmm_blockstep cannot both be set. DistributedBlockStepFMM "
            "exists but is a separate lane on jaccpot's mutual path."
        )

    n_steps = int(config.num_timesteps)
    if n_steps < 1:
        raise ValueError(
            f"config.num_timesteps must be >= 1 for the mesh lane; got {n_steps!r}"
        )
    return integrate_mesh_jaccpot(
        primitive_state, mass, config, params, options=options, n_steps=n_steps
    ).state


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
    if bool(config.fixed_timestep):
        # this lane's history starts with the initial state
        return _fixed_step_snapshots(out, config, params)

    states = jnp.asarray(out)
    target_snaps = int(config.num_snapshots)
    if target_snaps <= 0:
        raise ValueError("num_snapshots must be positive")
    stride = max(1, int(states.shape[0]) // target_snaps)
    snap_states = states[::stride][:target_snaps]
    times = jnp.linspace(0.0, params.t_end, snap_states.shape[0], endpoint=True)
    return SnapshotData(times=times, states=snap_states)
