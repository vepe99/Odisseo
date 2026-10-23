"""Odisseo on jaccpot's production runner across GPUs: the multi-GPU lane as a thin client.

:func:`odisseo.integration_api.integrate` sends an ``FMM_ACC`` run with
``config.fmm_mesh=True`` here (lane ``"fmm_mesh_runner"``) when
``config.fmm_preset == "galaxy"`` or ``config.mesh_options`` is a
:class:`MeshRunnerOptions` -- and not when it is a
:class:`~odisseo.mesh_coupling.MeshOptions`, which keeps the older mesh lane (below).
jaccpot's :class:`~jaccpot.production.ProductionRunner` in its multi-GPU mode
(``ProductionRunner(..., mesh=MeshRunConfig(...))``, jaccpot ``docs/production.md``,
"Multi-GPU mode") then owns the run: one process drives every card of a 1-D mesh, the
particles are decomposed into Morton domains, each segment is ONE
``jit(shard_map(scan))`` velocity-Verlet program on the fused multi-GPU lane, the
domains are re-balanced between segments, a segment whose lists outgrow their
capacities is re-planned and re-run on every card, and the diagnostics come at the
segment edges. This module is :mod:`odisseo.runner_coupling` with a mesh: the same
solver and settings (:func:`~odisseo.runner_coupling.galaxy_settings`,
:func:`~odisseo.runner_coupling.build_galaxy_solver`), the same external field
(:class:`~odisseo.runner_coupling.ExternalFieldParams`), the same snapshots, units and
kept solver and runner. What differs:

* **G and the softening are the solver's.** The mesh force is static in them (as one
  card is under ``dehnen_error``, the preset's MAC), so another ``params.G`` or
  ``softening`` builds another solver and compiles; ``dt`` and the potentials'
  parameters stay traced. The runner checks the force against them
  (``force_params``), so an injected force (``MeshRunnerOptions.force``) with
  another G or softening is refused.
* **The external field sees one device's rows.** jaccpot calls
  :func:`~odisseo.runner_coupling.external_acceleration` on each device's ``(cap, 2,
  3)`` rows, padding rows included (masked out of the force and the energy): every
  Odisseo potential is per particle and works unchanged.
* **The final state is a host NumPy array** in input order, gathered by id (one
  device-to-host copy at the end), like the snapshots, which are gathered at the
  edges. Nothing of the run stays on one card.
* **The energy** is ``K + W_self + W_ext`` from jaccpot's mesh potential, evaluated
  at step 0 itself (one card's step-0 energy waits for the first segment). The
  fused mesh force has no potential until jaccpot's S11 session 2 track E lands:
  until then ``total_energy`` is NaN, the run warns once and every edge's
  diagnostics say why (``energy_unavailable``); this module passes it through as
  soon as it exists. The angular momentum is exact either way.
* No in-scan ``step_callback``, no gradients, no block steps, no active-particle
  schedules (:func:`mesh_runner_unsupported`).

LAUNCHING. One process, every card of the mesh visible to it. Pick the cards (on a
shared node with ``autocvd``) and set ``XLA_FLAGS`` BEFORE jax starts -- XLA reads it
once: the fusion-autotuner flag of every production run plus the three
ragged-exchange flags (``jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS``:
XLA's NCCL send/recv path for the cross-domain exchange; without them the
exchange is several times slower across PCIe, or deadlocks on jax 0.11). jaccpot
warns when they are missing. ``docs/source/parallelism.md`` has the commands.

WHY A SECOND CALL IS CHEAP. As on one card: the solver and runner of the most recent
configuration are kept (:data:`_CACHE`; :func:`clear_mesh_runner_cache`), and jaccpot
caches the segment, start, potential and repartition programs on the solver's
engine (on an injected force: the force), keyed by the force's plan, the mesh, the
shard capacity, the external field's identity and the parameters' structure. What
runs per call is the decomposition, the per-device caps-only prepares at the new
initial state, the start force and the segments.

THE OLDER MESH LANE. :mod:`odisseo.mesh_coupling` (lane ``"fmm_mesh"``, a
``MeshOptions`` as ``config.mesh_options``) runs jaccpot's earlier distributed force
(``jaccpot.distributed.fmm``: RCB domains, a halo exchange, three dispatches per step
-- drift, force, kick --, no potential) with its own leaf size, order, opening angle
and ``dt``. It is unchanged and stays reachable while its users need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import jax
import numpy as np

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.runner_coupling import (
    ExternalFieldParams,
    RunnerResult,
    _collected_snapshots,
    _in_dtype,
    _segment_plan,
    build_galaxy_solver,
    external_acceleration,
    external_potential,
    galaxy_settings,
    runner_unsupported,
)

__all__ = [
    "MeshRunnerOptions",
    "clear_mesh_runner_cache",
    "integrate_mesh_runner_jaccpot",
    "mesh_runner_options",
    "mesh_runner_unsupported",
]


@dataclass(frozen=True)
class MeshRunnerOptions:
    """The multi-GPU runner lane's mesh-only knobs (``config.mesh_options``).

    Everything else -- leaf size, order, MAC, precision, G, softening, kernel, the
    external field -- is read from ``SimulationConfig`` and ``SimulationParams`` as
    on the single-card runner lane. Frozen, so it is hashable inside
    ``SimulationConfig``; ``None`` as ``config.mesh_options`` is the defaults.

    Parameters
    ----------
    ndev : int or None
        Devices of the mesh: the first ``ndev`` of ``jax.devices()``; ``None``, all
        of them. WHICH cards a process sees is ``CUDA_VISIBLE_DEVICES`` (``autocvd``),
        set before jax starts.
    repartition_every : int
        Re-balance the domains before the segment after every ``repartition_every``
        completed segments; ``1`` (jaccpot's default) before every segment after the
        first, ``0`` never after the initial decomposition.
    cap_headroom : float or None
        Rows per device over an even split (the decomposition and every repartition
        must fit them; a repartition that would not is declined); ``None``:
        jaccpot's ``MESH_CAP_HEADROOM`` (1.15).
    force : Any
        ``None``: the fused multi-GPU force on the solver (GPU only). A jaccpot
        mesh-force object is used as given -- ``jaccpot.distributed.segment.
        DirectMeshForce`` is the partition-independent direct sum, a small-N
        reference that also runs on CPU devices; its G and softening must be the
        run's.
    """

    ndev: Optional[int] = None
    repartition_every: int = 1
    cap_headroom: Optional[float] = None
    force: Any = None

    def __post_init__(self) -> None:
        """Validate the options.

        Raises
        ------
        ValueError
            If any option is outside its range.
        """
        if self.ndev is not None and int(self.ndev) < 1:
            raise ValueError(f"ndev must be >= 1 or None, got {self.ndev!r}")
        if int(self.repartition_every) < 0:
            raise ValueError(
                f"repartition_every must be >= 0, got {self.repartition_every!r}"
            )
        if self.cap_headroom is not None and not float(self.cap_headroom) >= 1.0:
            raise ValueError(
                f"cap_headroom must be >= 1 or None, got {self.cap_headroom!r}"
            )


def mesh_runner_options(config: SimulationConfig) -> MeshRunnerOptions:
    """``config.mesh_options`` as this lane's options (``None``: the defaults).

    Parameters
    ----------
    config : SimulationConfig
        The configuration.

    Returns
    -------
    MeshRunnerOptions
        The options.

    Raises
    ------
    TypeError
        For anything else (a ``MeshOptions`` selects the older mesh lane, so it
        never gets here through ``integrate``).
    """
    options = getattr(config, "mesh_options", None)
    if options is None:
        return MeshRunnerOptions()
    if not isinstance(options, MeshRunnerOptions):
        raise TypeError(
            "the multi-GPU runner lane takes config.mesh_options = a "
            "MeshRunnerOptions or None; got "
            f"{type(options).__name__} (a MeshOptions selects the older mesh lane, "
            "odisseo.mesh_coupling)"
        )
    return options


def mesh_runner_unsupported(
    config: SimulationConfig, *, scheduled: bool = False
) -> Optional[str]:
    """Why the multi-GPU runner lane cannot run ``config``, or ``None`` when it can.

    The single-card runner's limits (:func:`~odisseo.runner_coupling.
    runner_unsupported`: velocity Verlet at a fixed step, every particle on every
    step, the force refreshed after every drift), plus what the mesh mode has not:
    a gradient path and block steps.

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    scheduled : bool
        Whether the call carries an active-particle schedule.

    Returns
    -------
    Optional[str]
        The first reason, or ``None``.
    """
    if bool(getattr(config, "fmm_differentiable", False)):
        return (
            "fmm_differentiable=True (the mesh mode has no gradient path; "
            "odisseo.differentiable runs on one device)"
        )
    if bool(getattr(config, "fmm_blockstep", False)):
        return (
            "fmm_blockstep=True (the mesh mode steps every particle at one dt; "
            "odisseo.blockstep_coupling runs block steps)"
        )
    return runner_unsupported(config, scheduled=scheduled)


# --------------------------------------------------------------------------
# The kept solver and runner.
# --------------------------------------------------------------------------

#: ``{key: (solver, runner)}`` for the most recent configuration; see the module
#: docstring. Between calls the runner also holds its last edge -- the segment
#: runner's sharded state, masses and total acceleration on every card --, which
#: :func:`clear_mesh_runner_cache` releases.
_CACHE: dict = {}


def clear_mesh_runner_cache() -> None:
    """Drop the kept solver and mesh runner, their compiled programs and last state."""
    _CACHE.clear()


def _solver_and_runner(
    config: SimulationConfig,
    params: SimulationParams,
    settings: Any,
    options: MeshRunnerOptions,
    ndev: int,
    num_particles: int,
    external: bool,
) -> tuple:
    """The kept ``(solver, runner)`` for this configuration, built when it changed.

    The key holds what the solver is built with -- ``params.G`` and the softening
    included: the mesh force is static in them --, the options and the device
    count (the mesh), and the particle count (the shard capacity and the automatic
    leaf capacity are sized from it).
    """
    from jaccpot import ProductionRunner
    from jaccpot.production import MeshRunConfig
    from yggdrax.distributed import make_mesh

    from odisseo.jaccpot_coupling import (
        _large_n_environment_overrides,
        _resolve_use_pallas,
        _temporary_large_n_environment,
    )

    key = (
        settings,
        float(config.softening),
        float(params.G),
        str(config.fmm_mac_force_scale_mode),
        _resolve_use_pallas(config),
        tuple(sorted(_large_n_environment_overrides(config).items())),
        options,
        int(ndev),
        int(num_particles),
        bool(external),
    )
    kept = _CACHE.get(key)
    if kept is None:
        _CACHE.clear()  # one configuration at a time: its programs and last state
        with _temporary_large_n_environment(config):
            solver = build_galaxy_solver(
                settings,
                G=float(params.G),
                softening=float(config.softening),
                use_pallas=_resolve_use_pallas(config),
                mac_force_scale_mode=str(config.fmm_mac_force_scale_mode),
            )
        # None: jaccpot's default headroom
        layout = (
            {}
            if options.cap_headroom is None
            else {"cap_headroom": float(options.cap_headroom)}
        )
        runner = ProductionRunner(
            solver,
            settings=settings,
            external_acceleration_fn=external_acceleration if external else None,
            external_potential_fn=external_potential if external else None,
            mesh=MeshRunConfig(
                make_mesh(int(ndev)),
                repartition_every=int(options.repartition_every),
                force=options.force,
                **layout,
            ),
        )
        kept = (solver, runner)
        _CACHE[key] = kept
    return kept


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------


def integrate_mesh_runner_jaccpot(
    state: Any,
    mass: Any,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    segment: Optional[Any] = None,
    callbacks: Sequence[Callable[[Any], Optional[bool]]] = (),
    step_callback: Optional[Callable[[Any, Any], None]] = None,
    scheduled: bool = False,
) -> RunnerResult:
    """Run ``config`` on jaccpot's runner across the mesh (the ``"fmm_mesh_runner"`` lane).

    ``config.num_timesteps`` velocity-Verlet steps of ``dt = params.t_end /
    config.num_timesteps`` on the fused multi-GPU lane, in segments, on the devices
    of ``config.mesh_options`` (:class:`MeshRunnerOptions`; ``None``: every device).

    SNAPSHOTS as on the single-card runner lane
    (:func:`~odisseo.runner_coupling.integrate_runner_jaccpot`): with
    ``config.return_snapshots`` the segments are ``K = num_timesteps //
    num_snapshots`` steps and snapshot ``k`` is the state after ``k * K`` steps,
    gathered by id to the host at the edge; ``total_energy`` and
    ``angular_momentum`` are the runner's fp64 diagnostics (the energy is NaN while
    the mesh force has no potential: see the module docstring).

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]`` positions and velocities (host or device; copied to the host
        and decomposed over the mesh, never consumed). Its dtype is the state's.
    mass : Any
        ``[N]`` masses, passed in the state's dtype.
    config : SimulationConfig
        The configuration: the single-card runner lane's fields
        (:mod:`odisseo.runner_coupling`) and ``mesh_options``.
    params : SimulationParams
        ``G``, ``t_end`` and the external potentials' parameters.
    segment : Optional[SegmentConfig]
        jaccpot's ``SegmentConfig`` (segment length, diagnostics, restart files);
        ``None``: 50 steps without diagnostics, or with snapshots ``K`` steps with
        every diagnostic at every edge. With snapshots its ``steps`` must divide
        ``K``.
    callbacks : Sequence[Callable[[MeshSegmentEdge], Optional[bool]]]
        Further runner callbacks (``jaccpot.production.SnapshotWriter``,
        ``DiagnosticsLog``, ...), called at every edge after the snapshot collector;
        an edge's ``state_host()`` gathers its state by id.
    step_callback : None
        Refused unless ``None``: the mesh mode has no in-scan hook.
    scheduled : bool
        Whether the caller passed an active-particle schedule (refused).

    Returns
    -------
    RunnerResult
        The final state (a host NumPy array in input order), the snapshots, and
        jaccpot's ``RunResult`` (``mode="mesh"``) with the diagnostics of every edge
        and the mesh plan (shard capacity, live counts, repartitions, cross
        capacities) in each edge's ``plan``.

    Raises
    ------
    NotImplementedError
        For a configuration the lane does not run (:func:`mesh_runner_unsupported`),
        a ``step_callback``, or the fused force off a GPU backend.
    TypeError
        For a ``config.mesh_options`` that is not a :class:`MeshRunnerOptions`.
    ValueError
        For a non-positive ``num_timesteps`` / ``num_snapshots``, a ``segment``
        whose length does not divide the snapshot interval, more devices than
        there are, or (jaccpot's) an injected force whose G or softening is not the
        run's.
    """
    from jaccpot import ForceParams

    from odisseo.jaccpot_coupling import _temporary_large_n_environment

    reason = mesh_runner_unsupported(config, scheduled=scheduled)
    if reason is not None:
        raise NotImplementedError(
            f"the multi-GPU runner lane (fmm_mesh=True, fmm_preset='galaxy') does "
            f"not run {reason}."
        )
    if step_callback is not None:
        raise NotImplementedError(
            "the multi-GPU runner lane has no in-scan step_callback (the mesh "
            "segment's scan streams nothing); use a callback at the segment edges"
        )
    options = mesh_runner_options(config)
    if options.force is None and str(jax.default_backend()).strip().lower() != "gpu":
        raise NotImplementedError(
            "the multi-GPU runner lane's fused force prepares on a GPU backend "
            f"only; got {jax.default_backend()!r}. MeshRunnerOptions(force="
            "DirectMeshForce(...)) runs the direct-sum reference on CPU devices."
        )
    available = jax.device_count()
    ndev = available if options.ndev is None else int(options.ndev)
    if ndev > available:
        raise ValueError(
            f"MeshRunnerOptions(ndev={ndev}) but this process sees {available} "
            "device(s): pick the cards (CUDA_VISIBLE_DEVICES, autocvd) before jax "
            "starts"
        )
    num_steps = int(config.num_timesteps)
    if num_steps <= 0:
        raise ValueError("num_timesteps must be positive")
    # host arrays: the runner decomposes from the host, so nothing of the full
    # system is placed on one card
    state_np = np.asarray(state)
    mass_np = np.asarray(mass, dtype=state_np.dtype)
    settings = galaxy_settings(config, state_np.dtype)
    external = len(config.external_accelerations) > 0
    segment, collector = _segment_plan(config, segment, num_steps)

    _, runner = _solver_and_runner(
        config,
        params,
        settings,
        options,
        ndev,
        int(state_np.shape[0]),
        external,
    )
    # the mesh force is static in G and the softening: passing them makes the
    # runner check that the force's own are the run's
    force_params = ForceParams(G=float(params.G), eps2=float(config.softening) ** 2)
    external_params = (
        ExternalFieldParams.from_config(config, _in_dtype(params, state_np.dtype))
        if external
        else None
    )
    runner.segment = segment
    runner.callbacks = ((collector,) if collector is not None else ()) + tuple(
        callbacks
    )
    try:
        with _temporary_large_n_environment(config):
            run = runner.run(
                state_np,
                mass_np,
                dt=float(params.t_end) / float(num_steps),
                num_steps=num_steps,
                force_params=force_params,
                external_params=external_params,
            )
    finally:
        runner.callbacks = ()
    snapshots = _collected_snapshots(collector, config, num_steps, run)
    return RunnerResult(state=run.state, snapshots=snapshots, run=run)
