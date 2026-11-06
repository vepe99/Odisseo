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
configuration onto it and packages the result, nothing else. With
``config.fmm_mesh=True`` the same runner runs across the GPUs of a node
(:mod:`odisseo.mesh_runner_coupling`, lane ``"fmm_mesh_runner"``).

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

THE SCHEME (multi-node plan I1). nornax owns the integration schemes, jaccpot the
force and the run: ``config.fmm_scheme`` is handed to the runner as
``ProductionRunner(scheme=...)`` (:func:`runner_scheme`). ``None`` keeps the runner's
built-in velocity-Verlet KDK; ``"nornax_kdk"`` is ``nornax.schemes.LeapfrogKDK``, the
same KDK written by nornax, bitwise to the built-in. Setting it also selects this
lane for a fixed-step ``FMM_ACC`` run, as ``fmm_preset="galaxy"`` does.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence

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


def unsupported_message(reason: str) -> str:
    """The error for a fixed-step ``FMM_ACC`` run the runner does not do.

    Since the 2026-11 switch the runner is the only fixed-step single-device FMM
    lane: the coupler's leapfrog loop, which ran these options, is removed.

    Parameters
    ----------
    reason : str
        :func:`runner_unsupported`'s.

    Returns
    -------
    str
        The message, naming what still runs it.
    """
    return (
        f"the fixed-step FMM lane (fmm_runner, jaccpot's ProductionRunner) does not "
        f"run {reason}. Odisseo's own FMM leapfrog loop, which did, was removed in "
        "2026-11 (docs/source/production_fmm.md, 'Migration'). Use "
        "fmm_refresh_every=1 and fmm_refresh_after_position_update=False on the "
        "runner; fmm_blockstep=True (individual timesteps) for active particles; or "
        "fixed_timestep=False (the adaptive lane, which keeps these options)."
    )


#: The names ``config.fmm_scheme`` takes for nornax's schemes.
RUNNER_SCHEMES = ("nornax_kdk",)


def runner_scheme(config: SimulationConfig) -> Any:
    """The integration scheme the runner lane hands jaccpot's runner.

    Parameters
    ----------
    config : SimulationConfig
        The configuration; ``config.fmm_scheme`` is read.

    Returns
    -------
    Any
        ``None`` (the runner's built-in KDK), ``nornax.schemes.LeapfrogKDK()`` for
        ``"nornax_kdk"``, or the object given.

    Raises
    ------
    ValueError
        For a name that is not in :data:`RUNNER_SCHEMES`.
    TypeError
        For an object that is not a ``jaccpot.scheme.SplitStepScheme``.
    """
    scheme = getattr(config, "fmm_scheme", None)
    if scheme is None:
        return None
    if not isinstance(scheme, str):
        from jaccpot.scheme import check_scheme

        return check_scheme(scheme)
    name = scheme.strip().lower()
    if name == "nornax_kdk":
        from nornax.schemes import LeapfrogKDK

        return LeapfrogKDK()
    raise ValueError(
        f"config.fmm_scheme={scheme!r}: one of {RUNNER_SCHEMES}, None (the runner's "
        "built-in KDK), or a jaccpot.scheme.SplitStepScheme object"
    )


#: The two ways :func:`runner_settings` sizes the runner (:func:`runner_profile`).
RUNNER_PROFILES = ("galaxy", "config")


def _backend_name(backend: Optional[str] = None) -> str:
    """``backend``, or ``jax.default_backend()``, lower-case."""
    return str(jax.default_backend() if backend is None else backend).strip().lower()


def runner_profile(
    config: SimulationConfig,
    *,
    num_particles: Optional[int] = None,
    backend: Optional[str] = None,
) -> str:
    """Which settings the runner lane runs ``config`` with.

    * ``"galaxy"``: jaccpot's ``GalaxySettings`` (:func:`galaxy_settings`) -- for
      ``config.fmm_galaxy_settings`` given, ``fmm_preset="galaxy"``, or the
      automatic large-N profile (preset ``"fast"`` with
      ``fmm_auto_large_n_profile`` on a GPU at ``fmm_large_n_min_particles`` or
      more particles; the runner's settings there before the 2026-11 switch too).
    * ``"config"``: ``SimulationConfig``'s own FMM knobs (:func:`config_settings`)
      -- every other run, so a config that sets nothing runs at Odisseo's small-N
      defaults (leaf 16, p4) with jaccpot's preset MAC (eq (16a) at 1e-4).

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    num_particles : Optional[int]
        The particle count; ``None`` reads ``config.N_particles``.
    backend : Optional[str]
        ``jax.default_backend()`` when ``None``.

    Returns
    -------
    str
        One of :data:`RUNNER_PROFILES`.
    """
    if getattr(config, "fmm_galaxy_settings", None) is not None:
        return "galaxy"
    preset = str(config.fmm_preset).strip().lower()
    if preset == "galaxy":
        return "galaxy"
    n = int(config.N_particles) if num_particles is None else int(num_particles)
    if (
        bool(config.fmm_auto_large_n_profile)
        and preset == "fast"
        and _backend_name(backend) == "gpu"
        and n >= max(1, int(config.fmm_large_n_min_particles))
    ):
        return "galaxy"
    return "config"


def config_settings(config: SimulationConfig, state_dtype: Any) -> Any:
    """``GalaxySettings`` from ``SimulationConfig``'s FMM knobs (profile ``"config"``).

    What the removed coupler loop read, mapped onto the runner's settings:

    * ``leaf_size`` = ``fmm_leaf_size``; ``max_order`` = ``fmm_fixed_order`` when
      set, else ``fmm_max_order``; ``theta`` = ``fmm_theta``;
    * the MAC: ``fmm_mac_type="dehnen"`` (Odisseo's default, which cannot be told
      from "not set") keeps jaccpot's preset MAC (``GALAXY_SETTINGS``: Dehnen's eq
      (16a) at ``adaptive_eps`` 1e-4), as under the ``"galaxy"`` profile;
      ``"dehnen_error"`` is eq (16a) at ``fmm_adaptive_eps``; the fused lane runs
      no other MAC (refused). The fused lane's GEOMETRIC Dehnen MAC is not the
      coupler's at the same angle: at leaf 16, p4, theta 0.6 its force rel-L2 was
      4.3e-3 on the FMM notebook's N = 512 cube (the coupler's 1.2e-4; eq (16a) at
      1e-4: 4.3e-5), and at leaf 64 it overflowed a fixed capacity in that cube's
      collapse, where eq (16a) ran (2026-11-06, CPU). It stays reachable through
      ``fmm_galaxy_settings=GalaxySettings(mac_type="dehnen", adaptive_eps=None,
      ...)``. Under eq (16a) ``theta`` decides no pair;
    * the precision: the state's (an fp64 state an fp64 force), except
      ``fmm_preset="large_n_gpu"`` with ``fmm_large_n_force_fp32`` (the default),
      which ran fp32 on the coupler too;
    * no cell level floor (``cell_min_level=None``: level 8 cuts 1-3 particles per
      leaf at small N); the leaf capacity, its headroom and the cubic Morton box
      are jaccpot's defaults; the softening kernel is ``config.softening_kernel``.

    The coupler's engine knobs have no runner counterpart and are not read
    (:data:`RUNNER_IGNORED_KNOBS`; :func:`warn_ignored_knobs` says so once).

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
    ValueError
        For a MAC the fused lane does not run, or ``dehnen_error`` without an eps.
    """
    from jaccpot.production import GALAXY_SETTINGS, GalaxySettings

    mac = str(config.fmm_mac_type).strip().lower()
    if mac == "dehnen_error":
        if config.fmm_adaptive_eps is None:
            raise ValueError(
                "fmm_mac_type='dehnen_error' needs config.fmm_adaptive_eps, the "
                "relative force-accuracy target of Dehnen's eq (16a)"
            )
        mac_fields = dict(
            mac_type="dehnen_error", adaptive_eps=float(config.fmm_adaptive_eps)
        )
    elif mac == "dehnen":
        mac_fields = dict(
            mac_type=GALAXY_SETTINGS.mac_type, adaptive_eps=GALAXY_SETTINGS.adaptive_eps
        )
    else:
        raise ValueError(
            f"the runner lane runs the Dehnen MAC ('dehnen' or 'dehnen_error'); got "
            f"fmm_mac_type={config.fmm_mac_type!r}. The adaptive lane "
            "(fixed_timestep=False) still runs the coupler's other MACs."
        )
    fp32 = jnp.dtype(state_dtype) != jnp.float64 or (
        str(config.fmm_preset).strip().lower() == "large_n_gpu"
        and bool(config.fmm_large_n_force_fp32)
    )
    order = (
        config.fmm_max_order if config.fmm_fixed_order is None else config.fmm_fixed_order
    )
    return GalaxySettings(
        leaf_size=int(config.fmm_leaf_size),
        max_order=int(order),
        theta=float(config.fmm_theta),
        cell_min_level=None,
        softening_kernel=str(config.softening_kernel),
        precision="fp32" if fp32 else "fp64",
        **mac_fields,
    )


def runner_settings(
    config: SimulationConfig,
    state_dtype: Any,
    *,
    num_particles: Optional[int] = None,
    backend: Optional[str] = None,
) -> Any:
    """The ``GalaxySettings`` the runner lanes run ``config`` with.

    :func:`galaxy_settings` under profile ``"galaxy"``, :func:`config_settings`
    under ``"config"`` (:func:`runner_profile`).

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    state_dtype : Any
        The state's dtype.
    num_particles : Optional[int]
        The particle count; ``None`` reads ``config.N_particles``.
    backend : Optional[str]
        ``jax.default_backend()`` when ``None``.

    Returns
    -------
    GalaxySettings
        The settings.
    """
    profile = runner_profile(config, num_particles=num_particles, backend=backend)
    if profile == "galaxy":
        return galaxy_settings(config, state_dtype)
    return config_settings(config, state_dtype)


#: The coupler knobs of ``SimulationConfig`` the runner lanes do not read, with
#: their defaults: engine and memory tuning of the removed fixed-step loop, which
#: the adaptive lane (``fixed_timestep=False``) still reads. Under profile
#: ``"galaxy"`` the settings' own leaf size, order and opening angle replace
#: ``fmm_leaf_size``, ``fmm_max_order``, ``fmm_fixed_order`` and ``fmm_theta`` too.
RUNNER_IGNORED_KNOBS = (
    ("fmm_basis", "real"),
    ("fmm_farfield_mode", "auto"),
    ("fmm_m2l_chunk_size", None),
    ("fmm_nearfield_mode", "auto"),
    ("fmm_nearfield_edge_chunk_size", 256),
    ("fmm_tree_build_mode", "static_radix"),
    ("fmm_jit_tree", None),
    ("fmm_jit_traversal", True),
    ("fmm_max_pair_queue", None),
    ("fmm_pair_process_block", None),
    ("fmm_max_interactions_per_node", None),
    ("fmm_max_neighbors_per_leaf", None),
    ("fmm_prepare_stage_memory_split_enabled", None),
    ("fmm_upward_leaf_batch_size", None),
    ("fmm_enforce_static_shape_contract", False),
    ("fmm_static_shape_warmup_prepares", 0),
)

#: The knobs (with their defaults) the ``"galaxy"`` profile does not read either.
_GALAXY_IGNORED_KNOBS = (
    ("fmm_leaf_size", 16),
    ("fmm_max_order", 4),
    ("fmm_fixed_order", None),
    ("fmm_theta", 0.6),
)

#: ``fmm_theta``: no effect under the error-based MAC (eq (16a) accepts by error)
_THETA_KNOB = ("fmm_theta", 0.6)

#: The ignored-knob sets already warned about in this process.
_WARNED_IGNORED: set = set()


def ignored_knobs(config: SimulationConfig, settings: Any, profile: str) -> tuple:
    """The knobs ``config`` sets that the runner lane does not read.

    A knob counts when its value differs from its default:
    :data:`RUNNER_IGNORED_KNOBS` always, plus, under profile ``"galaxy"``, the leaf
    size, order and opening angle, and under the error-based MAC (``settings.
    mac_type == "dehnen_error"``: Odisseo's default ``fmm_mac_type="dehnen"`` keeps
    jaccpot's preset one, eq (16a)) the opening angle ``fmm_theta``, which decides
    no pair there. ``fmm_tree_leaf_target`` counts only when it is not the leaf
    size the run uses (the coupler tied the two).

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    settings : GalaxySettings
        :func:`runner_settings`'.
    profile : str
        :func:`runner_profile`'s.

    Returns
    -------
    tuple
        ``((name, value), ...)``.
    """
    knobs = RUNNER_IGNORED_KNOBS + (
        _GALAXY_IGNORED_KNOBS
        if profile == "galaxy"
        else (_THETA_KNOB,) if settings.mac_type == "dehnen_error" else ()
    )
    found = [
        (name, getattr(config, name))
        for name, default in knobs
        if getattr(config, name) != default
    ]
    if int(config.fmm_tree_leaf_target) not in (32, int(settings.leaf_size)):
        found.append(("fmm_tree_leaf_target", config.fmm_tree_leaf_target))
    return tuple(found)


def warn_ignored_knobs(
    config: SimulationConfig, settings: Any, profile: str, *, stacklevel: int = 2
) -> tuple:
    """Warn once per process and knob set that the runner lane ignores them.

    Not an error: the knobs tune the removed coupler loop's engine and memory, not
    the physics the runner computes, and the adaptive lane still reads them. Not
    silent either: the run's numbers do not depend on them any more.

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    settings : GalaxySettings
        :func:`runner_settings`'.
    profile : str
        :func:`runner_profile`'s.
    stacklevel : int
        ``warnings.warn``'s.

    Returns
    -------
    tuple
        :func:`ignored_knobs`' (also when nothing is warned).
    """
    import warnings

    found = ignored_knobs(config, settings, profile)
    key = (profile, found)
    if found and key not in _WARNED_IGNORED:
        _WARNED_IGNORED.add(key)
        names = ", ".join(f"{name}={value!r}" for name, value in found)
        warnings.warn(
            f"the runner lane (fmm_runner, profile {profile!r}) does not read {names}: "
            f"it runs leaf {settings.leaf_size}, p{settings.max_order}, "
            f"theta {settings.theta}, MAC {settings.mac_type!r}, {settings.precision}"
            + (
                " (an error-based MAC: the opening angle decides no pair)"
                if settings.mac_type == "dehnen_error"
                else ""
            )
            + ". "
            "Set config.fmm_galaxy_settings (jaccpot.production.GalaxySettings) to "
            "change those; the knobs above were the removed fixed-step loop's "
            "(docs/source/production_fmm.md, 'Migration').",
            UserWarning,
            stacklevel=stacklevel + 1,
        )
    return found


@contextlib.contextmanager
def fused_lane_scope(config: SimulationConfig, settings: Any) -> Iterator[None]:
    """The scope every runner build, run and force evaluation runs in.

    ``config``'s large-N environment overrides (restored on exit). Nothing else:
    on a CPU backend the solver itself takes jaccpot's fused-lane CPU route
    (:func:`build_galaxy_solver`: ``RuntimePolicyConfig(cpu_fused_lane=True)``),
    which runs the lane with the pure-JAX kernels and, under ``use_pallas=True``
    (:func:`_runner_use_pallas`: ``dehnen_error``), the CSR near field in Pallas
    interpret mode -- jaccpot scopes those switches to its own calls. The process's
    ``jax.default_backend`` is no longer touched.

    Parameters
    ----------
    config : SimulationConfig
        The configuration.
    settings : GalaxySettings
        The run's settings (kept for the callers' signature).

    Yields
    ------
    None
    """
    from odisseo.jaccpot_coupling import _temporary_large_n_environment

    del settings
    with _temporary_large_n_environment(config):
        yield


def _runner_use_pallas(config: SimulationConfig, settings: Any) -> Optional[bool]:
    """``use_pallas`` for the runner's solver.

    On a CPU backend: on under ``dehnen_error`` (its force scale needs the CSR near
    field's Pallas kernel, which jaccpot's CPU route runs in interpret mode), off
    otherwise (the pure-JAX near field). On a GPU: the config's
    (``jaccpot_coupling._resolve_use_pallas``).
    """
    from odisseo.jaccpot_coupling import _resolve_use_pallas

    if _backend_name() == "cpu":
        return settings.mac_type == "dehnen_error"
    return _resolve_use_pallas(config)


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
    cpu_fused_lane: Optional[bool] = None,
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
    cpu_fused_lane : Optional[bool]
        jaccpot's fused-lane CPU route (``RuntimePolicyConfig.cpu_fused_lane``):
        the lane on a CPU backend with the GPU's lane policy and the pure-JAX
        kernels. ``None`` (the default): on when ``jax.default_backend()`` is
        ``"cpu"``. No effect on a GPU.

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
        runtime=dataclasses.replace(
            base.runtime,
            fixed_order=int(settings.max_order),
            cpu_fused_lane=(
                _backend_name() == "cpu"
                if cpu_fused_lane is None
                else bool(cpu_fused_lane)
            ),
        ),
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
    capacity is sized at the first prepare and only grows afterwards. Called inside
    :func:`fused_lane_scope`.
    """
    from jaccpot import ProductionRunner

    from odisseo.jaccpot_coupling import _large_n_environment_overrides

    scale_carry = settings.mac_type == "dehnen_error"
    scheme = runner_scheme(config)
    use_pallas = _runner_use_pallas(config, settings)
    key = (
        settings,
        float(config.softening),
        float(params.G) if scale_carry else None,
        str(config.fmm_mac_force_scale_mode) if scale_carry else None,
        use_pallas,
        tuple(sorted(_large_n_environment_overrides(config).items())),
        int(num_particles),
        bool(external),
        _backend_name(),  # the solver's CPU route is decided at build
        None if scheme is None else scheme.key(),
    )
    kept = _CACHE.get(key)
    if kept is None:
        _CACHE.clear()  # one configuration at a time: its programs and last handle
        solver = build_galaxy_solver(
            settings,
            G=float(params.G),
            softening=float(config.softening),
            use_pallas=use_pallas,
            mac_force_scale_mode=str(config.fmm_mac_force_scale_mode),
        )
        runner = ProductionRunner(
            solver,
            settings=settings,
            external_acceleration_fn=external_acceleration if external else None,
            external_potential_fn=external_potential if external else None,
            scheme=scheme,
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
        For a configuration the runner does not run (:func:`runner_unsupported`).
    ValueError
        For a non-positive ``num_timesteps`` / ``num_snapshots``, or a ``segment``
        whose length does not divide the snapshot interval.

    Notes
    -----
    The settings are :func:`runner_settings`' (:func:`runner_profile`: jaccpot's
    galaxy settings, or the config's own FMM knobs); a knob the lane does not read
    is warned about once (:func:`warn_ignored_knobs`). On a CPU backend the run
    takes jaccpot's fused-lane CPU route (:func:`build_galaxy_solver`).
    """
    from jaccpot import ForceParams

    reason = runner_unsupported(config, scheduled=scheduled)
    if reason is not None:
        raise NotImplementedError(unsupported_message(reason))
    num_steps = int(config.num_timesteps)
    if num_steps <= 0:
        raise ValueError("num_timesteps must be positive")
    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass, dtype=state_arr.dtype)
    n = int(state_arr.shape[0])
    profile = runner_profile(config, num_particles=n)
    settings = runner_settings(config, state_arr.dtype, num_particles=n)
    warn_ignored_knobs(config, settings, profile, stacklevel=3)
    external = len(config.external_accelerations) > 0

    segment, collector = _segment_plan(config, segment, num_steps)

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
    with fused_lane_scope(config, settings):
        solver, runner = _solver_and_runner(config, params, settings, n, external)
        # this call's segment plan and callbacks on the kept runner; the callbacks
        # are dropped afterwards, so the kept runner holds no snapshot between calls
        runner.segment = segment
        runner.callbacks = ((collector,) if collector is not None else ()) + tuple(
            callbacks
        )
        runner.step_callback = step_callback
        runner.step_callback_stride = int(step_callback_stride)
        try:
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


# --------------------------------------------------------------------------
# The lane's self-gravity at one state (odisseo.utils.E_pot, force checks).
# --------------------------------------------------------------------------

#: ``{key: solver}`` for :func:`runner_self_gravity`, the most recent
#: configuration only: a solver of its own, built by the runner's builder from the
#: runner's settings, so that an evaluation never changes the kept run's engine
#: (its recorded walk caps and level widths, its compiled programs).
_FIELD_CACHE: dict = {}


def runner_self_gravity(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    potential: bool = True,
) -> tuple:
    """The runner lane's self-gravity at ``state``: accelerations and potential.

    The force ``integrate()`` would run ``config`` with at these positions --
    :func:`runner_settings`' settings, :func:`build_galaxy_solver`'s solver, G
    ``params.G`` and softening ``config.softening`` -- evaluated once: jaccpot's
    full eager prepare of the fused lane (``FusedForceProvider.prepare``), then its
    accelerations and its O(N) potential (``FusedForceProvider.potential``: the
    near field and the far field's L2P, the program the runner's energy
    diagnostic runs) on that topology. Host-driven: concrete arrays only.

    Parameters
    ----------
    state : jnp.ndarray
        ``[N, 2, 3]``; positions only are read.
    mass : jnp.ndarray
        ``[N]``, cast to the state's dtype.
    config : SimulationConfig
        The configuration (its external field is not included).
    params : SimulationParams
        ``G``.
    potential : bool
        Also evaluate the potential.

    Returns
    -------
    tuple
        ``(acc, phi)``: ``[N, 3]`` accelerations and ``[N]`` potential per unit
        mass (``phi_i = -G sum_{j != i} m_j psi(r_ij)``; ``None`` without
        ``potential``), in the state's dtype, in input order.

    Raises
    ------
    RuntimeError
        If the prepared topology overflowed a capacity (its values are not the
        force then).
    """
    from jaccpot.runtime.fused_provider import FusedForceProvider

    from odisseo.jaccpot_coupling import _large_n_environment_overrides

    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass, dtype=state_arr.dtype)
    n = int(state_arr.shape[0])
    settings = runner_settings(config, state_arr.dtype, num_particles=n)
    with fused_lane_scope(config, settings):
        use_pallas = _runner_use_pallas(config, settings)
        key = (
            settings,
            float(config.softening),
            float(params.G),
            str(config.fmm_mac_force_scale_mode),
            use_pallas,
            tuple(sorted(_large_n_environment_overrides(config).items())),
            n,
            _backend_name(),  # the solver's CPU route is decided at build
        )
        solver = _FIELD_CACHE.get(key)
        if solver is None:
            _FIELD_CACHE.clear()
            solver = build_galaxy_solver(
                settings,
                G=float(params.G),
                softening=float(config.softening),
                use_pallas=use_pallas,
                mac_force_scale_mode=str(config.fmm_mac_force_scale_mode),
            )
            _FIELD_CACHE[key] = solver
        prov, topology = FusedForceProvider.prepare(
            solver,
            state_arr[:, 0],
            mass_arr,
            leaf_size=int(settings.leaf_size),
            max_order=int(settings.max_order),
        )
        if not bool(prov.capacity_ok(topology)):
            raise RuntimeError(
                "runner_self_gravity: the fused lane's prepare at this state "
                f"overflowed a capacity (N={n}, {settings!r})"
            )
        acc = prov.accelerations(topology, None)
        phi = prov.potential(topology, None) if potential else None
    return acc, phi
