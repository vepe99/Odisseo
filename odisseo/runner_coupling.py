"""Odisseo on jaccpot's production runner: the fused single-GPU lane as a thin client.

:func:`odisseo.integration_api.integrate` sends an ``FMM_ACC`` run here (lane
``"fmm_runner"``) when ``config.fmm_preset == "galaxy"``, or when the automatic
large-N profile would have picked ``"large_n_gpu"`` -- preset ``"fast"`` on a GPU
with at least ``fmm_large_n_min_particles`` particles -- and the run is one the
runner does (:func:`runner_unsupported`). jaccpot's
:class:`~jaccpot.production.ProductionRunner` then owns the run: K-step
``strict_run_v2(carry="particles")`` segments chained through their handle, with
``dt``, G, the softening and the external field's parameters as traced arguments,
capacity re-plans at segment starts, and diagnostics at the segment edges. What the
fused lane costs there is what it costs in jaccpot's own harness
(``bench/fused_memory_budget.py``): this module builds the solver, maps Odisseo's
configuration onto it and packages the result, nothing else.

WHY A SECOND CALL IS CHEAP. jaccpot caches its compiled programs on the solver, keyed
by the external field's identity, so a fresh solver per call (what the coupler does)
recompiles everything, and a parameter passed as a Python float recompiles per
value. Here:

* the solver and its runner are kept between calls (:data:`_CACHE`, the most recent
  configuration only; :func:`clear_runner_cache` drops it);
* ``dt = t_end / num_timesteps``, ``params.G`` and the softening squared are traced
  (``SegmentConfig.traced_params``), so other values compile nothing -- under
  ``dehnen_error``, the preset's MAC (THE SETTINGS), ``dt`` only: G and the softening
  are the solver's;
* the external field is ONE pair of module-level functions,
  :func:`external_acceleration` and :func:`external_potential`, taking an
  :class:`ExternalFieldParams`: Odisseo's whole ``SimulationParams`` as traced leaves
  and the few static configuration fields the potentials read as pytree metadata.

What still runs per call is the eager (caps-only) prepare at the new initial state,
the start force and the segments.

UNITS. As the coupler (``jaccpot_coupling._build_fmm_solver``): the solver's ``G`` is
``params.G``, its softening ``config.softening`` (the Plummer-equivalent length for
every kernel) and its kernel ``config.softening_kernel``; the masses are passed as
they are.

THE SETTINGS. The numerical configuration is jaccpot's ``GalaxySettings`` -- leaf 64,
p6, cell level floor 8, automatic leaf capacity, the preset's MAC, fp32 -- or
``config.fmm_galaxy_settings``. The preset's MAC is Dehnen's eq (16a) at
``adaptive_eps`` 1e-4 since jaccpot's 2026-10-21 preset change (the geometric Dehnen
MAC at theta 0.8 before); under it G and the softening are the solver's, so another
value builds another solver and compiles, and ``GalaxySettings(mac_type="dehnen",
adaptive_eps=None)`` as ``fmm_galaxy_settings`` is the geometric MAC. This lane does
not read ``fmm_leaf_size``, ``fmm_max_order``, ``fmm_theta`` or any other coupler
knob; it does honour ``fmm_mac_type="dehnen_error"`` with ``fmm_adaptive_eps`` (and
``fmm_mac_force_scale_mode``), and ``fmm_large_n_force_fp32=False`` with a float64
state selects the fp64 force (:func:`galaxy_settings`).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.time_integration import SnapshotData

#: Steps per runner segment when ``integrate()`` returns only the final state.
#: jaccpot's ``SegmentConfig`` default: a segment edge costs little next to 50 steps,
#: and a capacity retry re-runs at most one segment. A run whose length is not a
#: multiple compiles a second scan for its last segment (jaccpot warns).
RUNNER_SEGMENT_STEPS = 50

#: The ``SimulationConfig`` fields ``odisseo.potentials`` reads, besides
#: ``external_accelerations``: compile-time constants of the external field.
_FIELD_CONFIG_FIELDS = (
    "external_accelerations",
    "glorder",
    "sech2_z",
    "MN3_positive_density",
)


# --------------------------------------------------------------------------
# The external field: one function object for every run.
# --------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class ExternalFieldParams:
    """What :func:`external_acceleration` and :func:`external_potential` take.

    A pytree: ``params`` (Odisseo's ``SimulationParams``) are its leaves, traced in
    the runner's programs, so other parameter values compile nothing; ``static`` --
    the configuration fields ``odisseo.potentials`` reads (which potentials, the
    quadrature order, the MN3 switches) -- is its metadata, part of the compile key.

    Parameters
    ----------
    params : SimulationParams
        The parameters (floats or arrays).
    static : tuple
        ``((name, value), ...)`` for the fields in :data:`_FIELD_CONFIG_FIELDS`.
    """

    def __init__(self, params: SimulationParams, static: tuple) -> None:
        self.params = params
        self.static = static

    @classmethod
    def from_config(
        cls, config: SimulationConfig, params: SimulationParams
    ) -> "ExternalFieldParams":
        """The field of ``config`` with ``params``.

        Parameters
        ----------
        config : SimulationConfig
            Its ``external_accelerations`` (and the fields the potentials read).
        params : SimulationParams
            The parameters.

        Returns
        -------
        ExternalFieldParams
            The pytree.
        """
        static = tuple(
            (
                name,
                (
                    tuple(int(i) for i in config.external_accelerations)
                    if name == "external_accelerations"
                    else getattr(config, name)
                ),
            )
            for name in _FIELD_CONFIG_FIELDS
        )
        return cls(params, static)

    def config(self) -> SimulationConfig:
        """A ``SimulationConfig`` carrying only the field's static fields.

        Returns
        -------
        SimulationConfig
            Every other field at its default (the potentials read none of them).
        """
        return SimulationConfig(**dict(self.static))

    def tree_flatten(self) -> tuple:
        """Leaves ``(params,)``, metadata ``static``."""
        return (self.params,), self.static

    @classmethod
    def tree_unflatten(cls, static: tuple, children: tuple) -> "ExternalFieldParams":
        """Rebuild from :meth:`tree_flatten`'s output."""
        return cls(children[0], static)

    def __repr__(self) -> str:
        return f"ExternalFieldParams(static={self.static!r})"


def external_acceleration(
    state: jnp.ndarray, field: ExternalFieldParams
) -> jnp.ndarray:
    """The external acceleration at the packed state's positions (the runner's field).

    Parameters
    ----------
    state : jnp.ndarray
        ``[N, 2, 3]``; positions only are read.
    field : ExternalFieldParams
        The potentials and their parameters.

    Returns
    -------
    jnp.ndarray
        ``[N, 3]`` in the state's dtype.
    """
    return combined_external_acceleration_vmpa_switch(
        state, field.config(), field.params
    )


def external_potential(state: jnp.ndarray, field: ExternalFieldParams) -> jnp.ndarray:
    """The external potential per unit mass (the runner's energy diagnostic).

    Parameters
    ----------
    state : jnp.ndarray
        ``[N, 2, 3]``; positions only are read.
    field : ExternalFieldParams
        The potentials and their parameters.

    Returns
    -------
    jnp.ndarray
        ``[N]`` in the state's dtype.
    """
    _, potential = combined_external_acceleration_vmpa_switch(
        state, field.config(), field.params, return_potential=True
    )
    return potential


def _in_dtype(params: Any, dtype: Any) -> Any:
    """``params`` with every floating leaf an array of ``dtype``.

    A Python float and a float64 array would otherwise be different compile keys
    (the runner keys the traced parameters by dtype), and the parameters of a float32
    run are float32 arguments either way.
    """

    def cast(leaf: Any) -> Any:
        if jnp.issubdtype(jnp.result_type(leaf), jnp.floating):
            return jnp.asarray(leaf, dtype)
        return leaf

    return jax.tree_util.tree_map(cast, params)


# --------------------------------------------------------------------------
# Which runs the runner takes, and with what settings.
# --------------------------------------------------------------------------


def runner_unsupported(
    config: SimulationConfig, *, scheduled: bool = False
) -> Optional[str]:
    """Why the runner cannot run ``config``, or ``None`` when it can.

    The runner is endpoint-correct velocity Verlet on the fused lane with every
    particle on every step, the force refreshed after every drift.

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    scheduled : bool
        Whether the call carries an active-particle schedule (``active_indices_fn``,
        ``active_indices_schedule`` or ``active_mask_schedule``).

    Returns
    -------
    Optional[str]
        The first reason, or ``None``.
    """
    if not bool(config.fixed_timestep):
        return "fixed_timestep=False (adaptive diffrax steps)"
    if int(config.fmm_refresh_every) != 1:
        return (
            f"fmm_refresh_every={int(config.fmm_refresh_every)} (the runner "
            "refreshes every step)"
        )
    if bool(config.fmm_refresh_after_position_update):
        return "fmm_refresh_after_position_update=True"
    if scheduled:
        return "an active-particle schedule (every particle moves on every step)"
    return None


def galaxy_settings(config: SimulationConfig, state_dtype: Any) -> Any:
    """The ``GalaxySettings`` the runner lane runs ``config`` with.

    ``config.fmm_galaxy_settings``, or jaccpot's ``GALAXY_SETTINGS``; then

    * ``fmm_mac_type="dehnen_error"`` selects that MAC with ``fmm_adaptive_eps``
      (``"dehnen"``, Odisseo's default, keeps the settings' MAC -- jaccpot's
      ``GALAXY_SETTINGS``: ``dehnen_error`` at 1e-4 since its 2026-10-21 preset
      change --, so the geometric MAC is ``fmm_galaxy_settings=GalaxySettings(
      mac_type="dehnen", adaptive_eps=None)``; any other MAC is refused: the fused
      lane runs these two);
    * without ``fmm_galaxy_settings``, ``fmm_large_n_force_fp32=False`` with a float64
      state selects the fp64 force (what that flag selected on the coupler's
      large-N lane);
    * the softening kernel is ``config.softening_kernel``, as on every lane.

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    state_dtype : Any
        The state's dtype.

    Returns
    -------
    GalaxySettings
        The settings.

    Raises
    ------
    TypeError
        If ``fmm_galaxy_settings`` is not a ``GalaxySettings``.
    ValueError
        For a MAC the fused lane does not run, or ``dehnen_error`` without an eps.
    """
    from jaccpot.production import GALAXY_SETTINGS, GalaxySettings

    settings = config.fmm_galaxy_settings
    if settings is None:
        settings = GALAXY_SETTINGS
        if (
            not bool(config.fmm_large_n_force_fp32)
            and jnp.dtype(state_dtype) == jnp.float64
        ):
            settings = dataclasses.replace(settings, precision="fp64")
    elif not isinstance(settings, GalaxySettings):
        raise TypeError(
            "config.fmm_galaxy_settings must be a jaccpot.production.GalaxySettings "
            f"or None; got {type(settings).__name__}"
        )
    mac = str(config.fmm_mac_type).strip().lower()
    if mac == "dehnen_error":
        if config.fmm_adaptive_eps is None:
            raise ValueError(
                "fmm_mac_type='dehnen_error' needs config.fmm_adaptive_eps, the "
                "relative force-accuracy target of Dehnen's eq (16a)"
            )
        settings = dataclasses.replace(
            settings,
            mac_type="dehnen_error",
            adaptive_eps=float(config.fmm_adaptive_eps),
        )
    elif mac != "dehnen":
        raise ValueError(
            f"the runner lane runs the Dehnen MAC ('dehnen' or 'dehnen_error'); got "
            f"fmm_mac_type={config.fmm_mac_type!r}. Use fmm_preset='fast' for another."
        )
    if settings.mac_type == "dehnen_error" and settings.adaptive_eps is None:
        raise ValueError("GalaxySettings(mac_type='dehnen_error') needs adaptive_eps")
    return dataclasses.replace(settings, softening_kernel=str(config.softening_kernel))


def build_galaxy_solver(
    settings: Any,
    *,
    G: float,
    softening: float,
    use_pallas: Optional[bool] = None,
    mac_force_scale_mode: str = "paper_fb",
) -> Any:
    """``FastMultipoleMethod(preset="galaxy")`` with ``settings``.

    The preset's own advanced configuration with the settings' leaf size, cell level
    floor (``None``: none), leaf capacity and headroom, Morton box shape (where the
    jaccpot has ``GalaxySettings.cubic_bounds``), expansion order and MAC put in, so
    ``GALAXY_SETTINGS`` builds exactly the preset (the same engine state), and
    whatever else the preset sets -- a field this builder does not map -- comes with
    it. ``tests/test_runner_lane.py`` pins both, and fails when ``GalaxySettings``
    gains a field this builder does not know.

    Parameters
    ----------
    settings : GalaxySettings
        :func:`galaxy_settings`.
    G : float
        Gravitational constant.
    softening : float
        Softening length (Plummer-equivalent).
    use_pallas : Optional[bool]
        ``jaccpot_coupling._resolve_use_pallas``.
    mac_force_scale_mode : str
        Under ``mac_type="dehnen_error"``: how eq (16)'s force scale is obtained.

    Returns
    -------
    FastMultipoleMethod
        The solver.
    """
    from jaccpot import FastMultipoleMethod

    base = FastMultipoleMethod(preset="galaxy").advanced
    tree = dataclasses.replace(
        base.tree,
        leaf_target=int(settings.leaf_size),
        leaf_capacity=settings.leaf_capacity,
        cell_min_level=(
            None if settings.cell_min_level is None else int(settings.cell_min_level)
        ),
        leaf_headroom=(
            float(settings.leaf_headroom) if settings.leaf_capacity == "auto" else None
        ),
    )
    if hasattr(settings, "cubic_bounds"):
        # S3b: the preset's cubic Morton box, TreeConfig.cubic_bounds
        tree = dataclasses.replace(tree, cubic_bounds=bool(settings.cubic_bounds))
    advanced = dataclasses.replace(
        base,
        tree=tree,
        runtime=dataclasses.replace(base.runtime, fixed_order=int(settings.max_order)),
        mac_type=str(settings.mac_type),
    )
    criterion: dict = {}
    if settings.mac_type == "dehnen_error":
        criterion = dict(
            adaptive_eps=float(settings.adaptive_eps),
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode=str(mac_force_scale_mode),
        )
    return FastMultipoleMethod(
        preset="galaxy",
        theta=float(settings.theta),
        G=float(G),
        softening=float(softening),
        softening_kernel=str(settings.softening_kernel),
        precision=settings.precision,
        use_pallas=use_pallas,
        advanced=advanced,
        **criterion,
    )


# --------------------------------------------------------------------------
# The kept solver and runner.
# --------------------------------------------------------------------------

#: ``{key: (solver, runner)}`` for the most recent configuration; see the module
#: docstring. Between calls it also holds the runner's last edge -- its handle's
#: accelerations (12 B per particle in fp32, 24 with an external field) --, which
#: :func:`clear_runner_cache` releases.
_CACHE: dict = {}


def clear_runner_cache() -> None:
    """Drop the kept solver and runner, their compiled programs and last handle."""
    _CACHE.clear()


def _solver_and_runner(
    config: SimulationConfig,
    params: SimulationParams,
    settings: Any,
    num_particles: int,
    external: bool,
) -> tuple:
    """The kept ``(solver, runner)`` for this configuration, built when it changed.

    The key holds what the solver is built with, except ``params.G`` (passed per
    call as a traced force parameter) unless the MAC is ``dehnen_error``, which
    takes the solver's own G only. The particle count is in it: the automatic leaf
    capacity is sized at the first prepare and only grows afterwards.
    """
    from jaccpot import ProductionRunner

    from odisseo.jaccpot_coupling import (
        _large_n_environment_overrides,
        _resolve_use_pallas,
        _temporary_large_n_environment,
    )

    scale_carry = settings.mac_type == "dehnen_error"
    key = (
        settings,
        float(config.softening),
        float(params.G) if scale_carry else None,
        str(config.fmm_mac_force_scale_mode) if scale_carry else None,
        _resolve_use_pallas(config),
        tuple(sorted(_large_n_environment_overrides(config).items())),
        int(num_particles),
        bool(external),
    )
    kept = _CACHE.get(key)
    if kept is None:
        _CACHE.clear()  # one configuration at a time: its programs and last handle
        with _temporary_large_n_environment(config):
            solver = build_galaxy_solver(
                settings,
                G=float(params.G),
                softening=float(config.softening),
                use_pallas=_resolve_use_pallas(config),
                mac_force_scale_mode=str(config.fmm_mac_force_scale_mode),
            )
        runner = ProductionRunner(
            solver,
            settings=settings,
            external_acceleration_fn=external_acceleration if external else None,
            external_potential_fn=external_potential if external else None,
        )
        kept = (solver, runner)
        _CACHE[key] = kept
    return kept


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------


class RunnerResult(NamedTuple):
    """What :func:`integrate_runner_jaccpot` returns (and the multi-GPU lane's
    :func:`~odisseo.mesh_runner_coupling.integrate_mesh_runner_jaccpot`)."""

    #: ``[N, 2, 3]`` final state (device; on the multi-GPU lane a host NumPy array,
    #: gathered by id).
    state: Any
    #: With ``config.return_snapshots``: the snapshots (host arrays), else ``None``.
    snapshots: Optional[SnapshotData]
    #: jaccpot's ``RunResult``: the edges (step, time, segment, retries, wall time,
    #: diagnostics), ``baselines`` (step 0's moments and energy), ``wall_s``.
    run: Any


class _SnapshotCollector:
    """A runner callback keeping every ``every``-th step's state on the host.

    The first ``count`` edges at a multiple of ``every`` steps below ``num_steps``:
    steps ``0, every, ..., (count - 1) * every``.
    """

    def __init__(self, every: int, count: int, num_steps: int) -> None:
        self.every = int(every)
        self.count = int(count)
        self.num_steps = int(num_steps)
        self.steps: list = []
        self.times: list = []
        self.states: list = []
        self.energy: list = []
        self.angular_momentum: list = []

    def __call__(self, edge: Any) -> None:
        if (
            edge.step % self.every
            or edge.step >= self.num_steps
            or len(self.steps) >= self.count
        ):
            return None
        diagnostics = edge.diagnostics or {}
        self.steps.append(int(edge.step))
        self.times.append(float(edge.time))
        self.states.append(edge.state_host())
        self.energy.append(diagnostics.get("energy"))
        self.angular_momentum.append(diagnostics.get("angular_momentum"))
        return None

    def snapshots(self, run: Any) -> SnapshotData:
        """The collected states as Odisseo's ``SnapshotData``.

        Step 0's energy is the runner's baseline when its step-0 edge has none: on
        one card the energy needs the first segment's provider (the multi-GPU
        mode's step-0 edge carries its own).
        """
        energy = list(self.energy)
        if energy and energy[0] is None and run.baselines is not None:
            energy[0] = run.baselines.get("energy")
        nan3 = [np.nan] * 3
        return SnapshotData(
            times=np.asarray(self.times, np.float64),
            states=np.stack(self.states),
            total_energy=np.asarray(
                [np.nan if e is None else e for e in energy], np.float64
            ),
            angular_momentum=np.asarray(
                [nan3 if v is None else v for v in self.angular_momentum], np.float64
            ),
            runtime=float(run.wall_s),
            num_iterations=int(run.step),
        )


def _segment_plan(
    config: SimulationConfig, segment: Optional[Any], num_steps: int
) -> tuple:
    """A call's segment plan and snapshot collector (both runner lanes).

    With ``config.return_snapshots``: segments of ``K = num_timesteps //
    num_snapshots`` steps with every diagnostic at every edge (or ``segment``, whose
    length must divide ``K``) and a :class:`_SnapshotCollector`; otherwise
    ``segment``, or :data:`RUNNER_SEGMENT_STEPS` steps without diagnostics, and no
    collector.

    Parameters
    ----------
    config : SimulationConfig
        ``return_snapshots`` and ``num_snapshots``.
    segment : Optional[SegmentConfig]
        The caller's, or ``None``.
    num_steps : int
        The run's steps.

    Returns
    -------
    tuple
        ``(SegmentConfig, Optional[_SnapshotCollector])``.

    Raises
    ------
    ValueError
        For a non-positive ``num_snapshots``, or a ``segment`` whose length does not
        divide the snapshot interval.
    """
    from jaccpot import SegmentConfig

    if not bool(config.return_snapshots):
        if segment is None:
            segment = SegmentConfig(steps=RUNNER_SEGMENT_STEPS, diagnostics_every=0)
        return segment, None
    count = int(config.num_snapshots)
    if count <= 0:
        raise ValueError("num_snapshots must be positive")
    interval = max(1, num_steps // count)
    if segment is None:
        segment = SegmentConfig(
            steps=interval, diagnostics_every=1, diagnostics_every_energy=1
        )
    elif interval % int(segment.steps):
        raise ValueError(
            f"segment.steps={segment.steps} must divide the snapshot interval "
            f"num_timesteps // num_snapshots = {interval}"
        )
    return segment, _SnapshotCollector(interval, count, num_steps)


def _collected_snapshots(
    collector: Optional[_SnapshotCollector],
    config: SimulationConfig,
    num_steps: int,
    run: Any,
) -> Optional[SnapshotData]:
    """The collector's ``SnapshotData``, checked complete; ``None`` without one.

    Parameters
    ----------
    collector : Optional[_SnapshotCollector]
        :func:`_segment_plan`'s.
    config : SimulationConfig
        ``num_snapshots``.
    num_steps : int
        The run's steps.
    run : RunResult
        jaccpot's result (the step-0 energy baseline, the wall time).

    Returns
    -------
    Optional[SnapshotData]
        The snapshots.

    Raises
    ------
    RuntimeError
        If an edge the snapshots need was not seen.
    """
    if collector is None:
        return None
    if len(collector.steps) != int(config.num_snapshots) and num_steps >= int(
        config.num_snapshots
    ):
        raise RuntimeError(
            f"collected snapshots at steps {collector.steps}, expected "
            f"{int(config.num_snapshots)} every {collector.every} steps"
        )
    return collector.snapshots(run)


def integrate_runner_jaccpot(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    segment: Optional[Any] = None,
    callbacks: Sequence[Callable[[Any], Optional[bool]]] = (),
    step_callback: Optional[Callable[[Any, Any], None]] = None,
    step_callback_stride: int = 1,
    scheduled: bool = False,
) -> RunnerResult:
    """Run ``config`` on jaccpot's ``ProductionRunner`` (the ``"fmm_runner"`` lane).

    ``config.num_timesteps`` velocity-Verlet steps of ``dt = params.t_end /
    config.num_timesteps`` on the fused lane, in segments.

    SNAPSHOTS. With ``config.return_snapshots`` the segments are ``K =
    num_timesteps // num_snapshots`` steps and snapshot ``k`` is the state after
    ``k * K`` steps, ``k = 0 .. num_snapshots - 1`` (the initial state first, the
    final state not among them, the times ``k * K * dt``): the direct lane's
    snapshots (``time_integration``) when ``num_snapshots`` divides
    ``num_timesteps``. They are copied to the host at the segment edges, so the device
    holds one state at a time; ``SnapshotData``'s arrays are NumPy arrays.
    ``total_energy`` (``K + W_self + W_ext``, the fused lane's O(N) potential, one
    extra force-like evaluation per snapshot) and ``angular_momentum`` (``sum m x x
    v``) come from the runner's fp64 diagnostics.

    Parameters
    ----------
    state : jnp.ndarray
        ``[N, 2, 3]`` positions and velocities; kept (the first segment does not
        consume it). Its dtype is the state's; a float32 state is the production
        memory footprint.
    mass : jnp.ndarray
        ``[N]`` masses, passed in the state's dtype.
    config : SimulationConfig
        The configuration (see the module docstring for what this lane reads).
    params : SimulationParams
        ``G``, ``t_end`` and the external potentials' parameters.
    segment : Optional[SegmentConfig]
        jaccpot's ``SegmentConfig`` (segment length, diagnostics, restart files);
        ``None``: :data:`RUNNER_SEGMENT_STEPS` steps without diagnostics, or with
        snapshots ``K`` steps with every diagnostic at every edge. With snapshots its
        ``steps`` must divide ``K``.
    callbacks : Sequence[Callable[[SegmentEdge], Optional[bool]]]
        Further runner callbacks (``jaccpot.production.SnapshotWriter``,
        ``DiagnosticsLog``, ...), called at every edge after the snapshot collector.
    step_callback, step_callback_stride
        ``strict_run_v2``'s in-scan hook, passed through.
    scheduled : bool
        Whether the caller passed an active-particle schedule (refused).

    Returns
    -------
    RunnerResult
        The final state, the snapshots, and jaccpot's ``RunResult`` with the
        diagnostics of every edge.

    Raises
    ------
    NotImplementedError
        For a configuration the runner does not run (:func:`runner_unsupported`), or
        off a GPU backend: the fused lane's prepare runs on a GPU only.
    ValueError
        For a non-positive ``num_timesteps`` / ``num_snapshots``, or a ``segment``
        whose length does not divide the snapshot interval.
    """
    from jaccpot import ForceParams

    from odisseo.jaccpot_coupling import _temporary_large_n_environment

    reason = runner_unsupported(config, scheduled=scheduled)
    if reason is not None:
        raise NotImplementedError(
            f"the runner lane (fmm_preset='galaxy') does not run {reason}; use "
            "fmm_preset='large_n_gpu' for the coupler's own loop on the fused lane."
        )
    if str(jax.default_backend()).strip().lower() != "gpu":
        raise NotImplementedError(
            "the runner lane (fmm_preset='galaxy') is jaccpot's fused single-GPU "
            "lane, whose prepare runs on a GPU backend only; got "
            f"{jax.default_backend()!r}. Use fmm_preset='fast' on a CPU."
        )
    num_steps = int(config.num_timesteps)
    if num_steps <= 0:
        raise ValueError("num_timesteps must be positive")
    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass, dtype=state_arr.dtype)
    settings = galaxy_settings(config, state_arr.dtype)
    external = len(config.external_accelerations) > 0

    segment, collector = _segment_plan(config, segment, num_steps)

    solver, runner = _solver_and_runner(
        config, params, settings, int(state_arr.shape[0]), external
    )
    # the runner traces dt, G and eps^2 (SegmentConfig.traced_params); dehnen_error
    # takes the solver's own G and softening only
    force_params = (
        None
        if settings.mac_type == "dehnen_error"
        else ForceParams(G=float(params.G), eps2=float(config.softening) ** 2)
    )
    external_params = (
        ExternalFieldParams.from_config(config, _in_dtype(params, state_arr.dtype))
        if external
        else None
    )
    # this call's segment plan and callbacks on the kept runner; the callbacks are
    # dropped afterwards, so the kept runner holds no snapshot between calls
    runner.segment = segment
    runner.callbacks = ((collector,) if collector is not None else ()) + tuple(
        callbacks
    )
    runner.step_callback = step_callback
    runner.step_callback_stride = int(step_callback_stride)
    try:
        with _temporary_large_n_environment(config):
            run = runner.run(
                state_arr,
                mass_arr,
                dt=float(params.t_end) / float(num_steps),
                num_steps=num_steps,
                force_params=force_params,
                external_params=external_params,
            )
    finally:
        runner.callbacks = ()
        runner.step_callback = None
    snapshots = _collected_snapshots(collector, config, num_steps, run)
    return RunnerResult(state=run.state, snapshots=snapshots, run=run)
