from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

import diffrax
import jax
import jax.numpy as jnp
from jax.errors import UnexpectedTracerError

from odisseo.option_classes import (
    DOPRI5,
    DOPRI8,
    LEAPFROGMIDPOINT,
    REVERSIBLEHEUN,
    SEMIIMPLICITEULER,
    TSIT5,
    SimulationConfig,
    SimulationParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch


def _large_n_environment_overrides(
    config: SimulationConfig,
    *,
    fmm_preset: Optional[str] = None,
) -> dict[str, str]:
    """Return the jaccpot large-N env overrides SimulationConfig sets explicitly.

    Only the fields a user set reach the environment. The automatic block that
    used to switch jaccpot's fused strict lane on for the large_n_gpu static-radix
    regime (fused mode, device-only, flat compact far pairs, block size 4, "auto"
    static target blocks, the 131072 far-pair cap, the profile set) is gone:
    since jaccpot's 2026-10 cleanup (D2) those are jaccpot's own defaults, and its
    caps are sized from the particle count. ``fmm_preset`` is kept for the
    callers' signature.
    """
    overrides: dict[str, str] = {}
    if not bool(getattr(config, "fmm_large_n_environment_overrides_enabled", True)):
        return overrides
    target_block_size = getattr(config, "fmm_large_n_target_block_size", None)
    if target_block_size is not None:
        overrides["JACCPOT_LARGE_N_TARGET_BLOCK_SIZE"] = str(int(target_block_size))
    static_target_blocks = getattr(config, "fmm_large_n_static_target_blocks", None)
    if static_target_blocks is not None:
        overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS"] = (
            "1" if bool(static_target_blocks) else "0"
        )
    static_target_blocks_cap = getattr(
        config,
        "fmm_large_n_static_target_blocks_max_per_leaf",
        None,
    )
    if static_target_blocks_cap is not None:
        cap_raw = str(static_target_blocks_cap).strip().lower()
        overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] = (
            "auto" if cap_raw == "auto" else str(int(static_target_blocks_cap))
        )
    return overrides


@contextmanager
def _temporary_large_n_environment(
    config: SimulationConfig,
    *,
    fmm_preset: Optional[str] = None,
):
    overrides = _large_n_environment_overrides(config, fmm_preset=fmm_preset)
    if not overrides:
        yield
        return

    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_use_pallas(config: SimulationConfig) -> Optional[bool]:
    """The ``use_pallas`` a jaccpot solver for ``config`` is built with.

    Fused Pallas near-field/M2L kernels (Ampere+/sm_80+ GPUs). Resolution order:

    1. the ``ODISSEO_FMM_USE_PALLAS`` env var (explicit override), else
    2. ``config.fmm_use_pallas`` (True/False), else
    3. ``None``: jaccpot auto-detects, ON for Ampere sm_80+, pure-JAX on
       sm_75/CPU. jaccpot also falls back automatically on unsupported hardware
       via its ``pallas_*_supported()`` guards.
    """
    pallas_env = os.environ.get("ODISSEO_FMM_USE_PALLAS")
    if pallas_env is not None and pallas_env.strip() != "":
        return pallas_env.strip().lower() in {"1", "true", "yes", "on"}
    if config.fmm_use_pallas is not None:
        return bool(config.fmm_use_pallas)
    return None  # jaccpot auto-detects Ampere


def _build_fmm_solver(
    *,
    working_dtype,
    config: SimulationConfig,
    params: SimulationParams,
    fmm_preset: str,
    fmm_basis: str,
    fmm_theta: float,
    fmm_runtime_path: str,
    fmm_mac_type: str,
    fmm_farfield_mode: str,
    fmm_m2l_chunk_size: Optional[int],
    fmm_nearfield_mode: str,
    fmm_nearfield_edge_chunk_size: int,
    fmm_tree_build_mode: str,
    fmm_tree_leaf_target: int,
    fmm_fixed_order: Optional[int],
    leaf_size: int,
    fmm_jit_tree: Optional[bool],
    fmm_jit_traversal: Optional[bool],
    fmm_max_pair_queue: Optional[int],
    fmm_pair_process_block: Optional[int],
    fmm_max_interactions_per_node: Optional[int],
    fmm_max_neighbors_per_leaf: Optional[int],
    fmm_prepare_stage_memory_split_enabled: Optional[bool],
    fmm_upward_leaf_batch_size: Optional[int],
    retain_far_pairs_for_grad: bool = False,
):
    """Construct the jaccpot solver for one Odisseo FMM configuration.

    ``fmm_runtime_path`` and ``fmm_farfield_mode`` are accepted and ignored since
    jaccpot's 2026-10 cleanup (its ``docs/cleanup_2026-10.md``). ``runtime_path``
    stopped choosing the lane in X4: the ``large_n_gpu`` preset alone opens the
    large-N lane. The far-field modes jaccpot still accepts (``"auto"`` and
    ``"pair_grouped"``) both run the flat far-pair list, and ``"class_major"`` went
    with the grouped far field in X3. Neither is passed on, so jaccpot runs its
    defaults, which is what Odisseo's defaults (``"auto"`` for both) asked for.
    """
    del fmm_runtime_path, fmm_farfield_mode  # accepted and ignored, see above
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )
    from yggdrax.interactions import DualTreeTraversalConfig

    # Apply the large-N env overrides the config sets explicitly before
    # constructing the solver (jaccpot reads some of its knobs in
    # FastMultipoleMethod.__init__). The fused strict lane itself is jaccpot's
    # default since its 2026-10 cleanup. setdefault keeps any explicit shell
    # override winning.
    for _env_key, _env_value in _large_n_environment_overrides(
        config, fmm_preset=fmm_preset
    ).items():
        os.environ.setdefault(_env_key, _env_value)

    use_pallas = _resolve_use_pallas(config)

    traversal_config = None
    if (
        fmm_max_interactions_per_node is not None
        or fmm_max_neighbors_per_leaf is not None
    ):
        traversal_values = (
            fmm_max_pair_queue,
            fmm_pair_process_block,
            fmm_max_interactions_per_node,
            fmm_max_neighbors_per_leaf,
        )
        if not all(value is not None for value in traversal_values):
            raise ValueError(
                "Full jaccpot traversal capacity overrides require "
                "fmm_max_pair_queue, fmm_pair_process_block, "
                "fmm_max_interactions_per_node, and "
                "fmm_max_neighbors_per_leaf."
            )
        traversal_config = DualTreeTraversalConfig(
            max_pair_queue=int(fmm_max_pair_queue),
            process_block=int(fmm_pair_process_block),
            max_interactions_per_node=int(fmm_max_interactions_per_node),
            max_neighbors_per_leaf=int(fmm_max_neighbors_per_leaf),
        )

    # The frozen M2L pair list is discarded by default (the large-N preset targets
    # minimum memory). The differentiable lane needs it retained to re-run the
    # downward sweep against it, so the knob is only passed when requested --
    # keeping the forward production lane byte-for-byte as it was, and keeping
    # this call compatible with a jaccpot that has no such field.
    farfield_kwargs: dict[str, Any] = dict(
        m2l_chunk_size=(
            None if fmm_m2l_chunk_size is None else int(fmm_m2l_chunk_size)
        ),
    )
    if bool(retain_far_pairs_for_grad):
        farfield_kwargs["retain_far_pairs_for_grad"] = True

    criterion_kwargs: dict[str, Any] = {}
    if str(fmm_mac_type) == "dehnen_error":
        adaptive_eps = getattr(config, "fmm_adaptive_eps", None)
        if adaptive_eps is None:
            raise ValueError(
                "fmm_mac_type='dehnen_error' needs config.fmm_adaptive_eps, the "
                "relative force-accuracy target of Dehnen's eq (16a)"
            )
        criterion_kwargs = dict(
            adaptive_eps=float(adaptive_eps),
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode=str(
                getattr(config, "fmm_mac_force_scale_mode", "paper_fb")
            ),
        )

    return FastMultipoleMethod(
        preset=str(fmm_preset),
        basis=str(fmm_basis),
        theta=float(fmm_theta),
        G=float(params.G),
        softening=float(config.softening),
        softening_kernel=getattr(config, "softening_kernel", None),
        working_dtype=working_dtype,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode=str(fmm_tree_build_mode),
                leaf_target=int(fmm_tree_leaf_target),
            ),
            farfield=FarFieldConfig(**farfield_kwargs),
            nearfield=NearFieldConfig(
                mode=str(fmm_nearfield_mode),
                edge_chunk_size=int(fmm_nearfield_edge_chunk_size),
            ),
            runtime=RuntimePolicyConfig(
                jit_tree=None if fmm_jit_tree is None else bool(fmm_jit_tree),
                jit_traversal=(
                    None if fmm_jit_traversal is None else bool(fmm_jit_traversal)
                ),
                max_pair_queue=(
                    None if fmm_max_pair_queue is None else int(fmm_max_pair_queue)
                ),
                pair_process_block=(
                    None
                    if fmm_pair_process_block is None
                    else int(fmm_pair_process_block)
                ),
                traversal_config=traversal_config,
                prepare_stage_memory_split_enabled=(
                    None
                    if fmm_prepare_stage_memory_split_enabled is None
                    else bool(fmm_prepare_stage_memory_split_enabled)
                ),
                upward_leaf_batch_size=(
                    None
                    if fmm_upward_leaf_batch_size is None
                    else int(fmm_upward_leaf_batch_size)
                ),
                fixed_order=(
                    None if fmm_fixed_order is None else int(fmm_fixed_order)
                ),
                # Keep one global leaf-size contract per simulation: tree target
                # and runtime leaf cap are tied to the same value.
                fixed_max_leaf_size=int(leaf_size),
            ),
            mac_type=str(fmm_mac_type),
        ),
        **criterion_kwargs,
    )


def integrate_diffrax_jaccpot_active(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    num_steps: int,
    dt: Optional[float] = None,
    active_indices_fn: Optional[
        Callable[[int, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ] = None,
    active_indices_schedule: Optional[jnp.ndarray] = None,
    active_mask_schedule: Optional[jnp.ndarray] = None,
    refresh_every: int = 1,
    refresh_after_position_update: bool = False,
    leaf_size: int = 16,
    max_order: int = 4,
    fmm_preset: str = "fast",
    fmm_basis: str = "real",
    fmm_theta: float = 0.6,
    fmm_runtime_path: str = "auto",
    fmm_working_dtype=None,
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_m2l_chunk_size: Optional[int] = None,
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_build_mode: str = "lbvh",
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
    fmm_max_pair_queue: Optional[int] = None,
    fmm_pair_process_block: Optional[int] = None,
    fmm_max_interactions_per_node: Optional[int] = None,
    fmm_max_neighbors_per_leaf: Optional[int] = None,
    fmm_prepare_stage_memory_split_enabled: Optional[bool] = None,
    fmm_upward_leaf_batch_size: Optional[int] = None,
    enforce_static_shape_contract: bool = False,
    static_shape_warmup_prepares: int = 0,
    rematerialize_between_refresh: bool = True,
    return_history: bool = False,
    timing_stats: Optional[dict] = None,
) -> jnp.ndarray:
    """Adaptive diffrax integration with FMM acceleration RHS."""
    del refresh_every, rematerialize_between_refresh
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if (
        active_indices_fn is not None
        or active_indices_schedule is not None
        or active_mask_schedule is not None
    ):
        raise NotImplementedError(
            "Adaptive FMM currently supports full-particle updates only (no active-index schedules)."
        )
    if bool(refresh_after_position_update):
        raise NotImplementedError(
            "Adaptive FMM does not support refresh_after_position_update=True."
        )
    if bool(enforce_static_shape_contract):
        raise NotImplementedError(
            "Adaptive FMM does not yet support enforce_static_shape_contract=True."
        )
    if int(static_shape_warmup_prepares) > 0:
        raise NotImplementedError(
            "Adaptive FMM does not yet support static_shape_warmup_prepares>0."
        )

    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    add_external = len(config.external_accelerations) > 0
    dt_val = float(params.t_end) / float(num_steps) if dt is None else float(dt)

    solver = _build_fmm_solver(
        working_dtype=(
            state_arr.dtype
            if fmm_working_dtype is None
            else jnp.dtype(fmm_working_dtype)
        ),
        config=config,
        params=params,
        fmm_preset=fmm_preset,
        fmm_basis=fmm_basis,
        fmm_theta=fmm_theta,
        fmm_runtime_path=fmm_runtime_path,
        fmm_mac_type=fmm_mac_type,
        fmm_farfield_mode=fmm_farfield_mode,
        fmm_m2l_chunk_size=fmm_m2l_chunk_size,
        fmm_nearfield_mode=fmm_nearfield_mode,
        fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
        fmm_tree_build_mode=fmm_tree_build_mode,
        fmm_tree_leaf_target=fmm_tree_leaf_target,
        fmm_fixed_order=fmm_fixed_order,
        leaf_size=leaf_size,
        # Adaptive RHS currently runs inside outer JAX transforms (diffrax/equinox);
        # keep tree build off the jitted LBVH fast path until jaccpot runtime
        # exposes a tracer-safe prepared/evaluate API.
        fmm_jit_tree=False,
        fmm_jit_traversal=fmm_jit_traversal,
        fmm_max_pair_queue=fmm_max_pair_queue,
        fmm_pair_process_block=fmm_pair_process_block,
        fmm_max_interactions_per_node=fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=fmm_prepare_stage_memory_split_enabled,
        fmm_upward_leaf_batch_size=fmm_upward_leaf_batch_size,
    )

    def _pick_diffrax_solver():
        if int(config.diffrax_solver) == DOPRI5:
            return diffrax.Dopri5()
        if int(config.diffrax_solver) == TSIT5:
            return diffrax.Tsit5()
        if int(config.diffrax_solver) == DOPRI8:
            return diffrax.Dopri8()
        if int(config.diffrax_solver) == SEMIIMPLICITEULER:
            return diffrax.SemiImplicitEuler()
        if int(config.diffrax_solver) == REVERSIBLEHEUN:
            return diffrax.ReversibleHeun()
        if int(config.diffrax_solver) == LEAPFROGMIDPOINT:
            return diffrax.LeapfrogMidpoint()
        return diffrax.Dopri5()

    # Keep RHS side-effect free under diffrax/equinox tracing.
    # Prepared-state refresh caching can be reintroduced once jaccpot exposes
    # a fully functional tracer-safe runtime-state API.
    cache: dict[str, Any] = {
        "full_prepare_calls": 0,
        "refresh_prepare_calls": 0,
    }

    def _prepare_full(y_state: jnp.ndarray):
        with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
            try:
                return solver.prepare_state(
                    y_state[:, 0, :],
                    mass_arr,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    cache_policy="stateless",
                )
            except TypeError:
                return solver.prepare_state(
                    y_state[:, 0, :],
                    mass_arr,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                )

    def _rhs(t, y, args):
        del t, args
        cache["full_prepare_calls"] += 1
        prepared = _prepare_full(y)
        acc_self = solver.evaluate_prepared_state(
            prepared,
            target_indices=None,
            return_potential=False,
        )
        if add_external:
            acc = acc_self + combined_external_acceleration_vmpa_switch(
                y, config, params
            )
        else:
            acc = acc_self
        return jnp.stack((y[:, 1, :], acc), axis=1)

    t0 = 0.0
    t1 = float(params.t_end)
    dt0 = float(dt_val)
    rtol = float(getattr(config, "fmm_adaptive_rtol", 1e-3))
    atol = float(getattr(config, "fmm_adaptive_atol", 1e-6))
    min_dt = getattr(config, "fmm_adaptive_min_dt", None)
    max_dt = getattr(config, "fmm_adaptive_max_dt", None)
    controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        dtmin=(None if min_dt is None else float(min_dt)),
        dtmax=(None if max_dt is None else float(max_dt)),
    )
    saveat = diffrax.SaveAt(
        ts=(
            jnp.linspace(t0, t1, int(config.num_timesteps), endpoint=True)
            if bool(return_history)
            else None
        ),
        t1=(not bool(return_history)),
        steps=False,
        dense=bool(getattr(config, "fmm_adaptive_use_dense_output", False)),
    )
    t_start = time.perf_counter()
    try:
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(_rhs),
            solver=_pick_diffrax_solver(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=state_arr,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=100_000,
        )
    except UnexpectedTracerError as exc:
        raise NotImplementedError(
            "Adaptive diffrax FMM hit a jaccpot tracer-leak boundary under JIT. "
            "A pure-JAX prepared/evaluate runtime state API is required in jaccpot "
            "before single-giant-jit adaptive integration can be enabled."
        ) from exc
    elapsed = time.perf_counter() - t_start
    if timing_stats is not None:
        runtime_diag = {}
        get_diag = getattr(solver, "get_runtime_diagnostics", None)
        if callable(get_diag):
            try:
                runtime_diag = dict(get_diag())
            except Exception:
                runtime_diag = {}
        stats = dict(getattr(sol, "stats", {}) or {})
        accepted = int(stats.get("num_accepted_steps", 0))
        rejected = int(stats.get("num_rejected_steps", 0))
        step_attempts = int(accepted + rejected)

        def _normalize_stats_value(value: Any):
            if isinstance(value, (bool, int, float, str)) or value is None:
                return value
            try:
                return float(value)
            except Exception:
                return repr(value)

        raw_stats_normalized = {
            str(k): _normalize_stats_value(v) for k, v in stats.items()
        }

        timing_stats.clear()
        timing_stats.update(
            {
                "integration_mode": "adaptive_diffrax_fmm",
                "adaptive_seconds": float(elapsed),
                "adaptive_num_accepted_steps": accepted,
                "adaptive_num_rejected_steps": rejected,
                "adaptive_total_steps": step_attempts,
                "adaptive_step_attempts_estimate": step_attempts,
                "adaptive_rhs_evals_estimate": step_attempts,
                "adaptive_rhs_evals_estimate_deprecated": int(
                    stats.get("num_steps", step_attempts)
                ),
                "adaptive_diffrax_stats_raw": raw_stats_normalized,
                "adaptive_rejected_step_fraction": (
                    float(rejected) / float(step_attempts) if step_attempts > 0 else 0.0
                ),
                "adaptive_tracing_side_effect_counters_reliable": False,
                "adaptive_refresh_hits": int(cache["full_prepare_calls"]),
                "adaptive_refresh_misses": 0,
                "adaptive_refresh_rhs_calls": 1,
                "adaptive_full_prepare_calls": int(cache["full_prepare_calls"]),
                "adaptive_refresh_prepare_calls": int(cache["refresh_prepare_calls"]),
                "adaptive_tracing_prepare_counter_full": int(
                    cache["full_prepare_calls"]
                ),
                "adaptive_tracing_prepare_counter_refresh": int(
                    cache["refresh_prepare_calls"]
                ),
                "adaptive_profile_key_hits": int(accepted),
                "adaptive_profile_key_misses": 0,
                "adaptive_fail_fast_reject_count": 0,
                "adaptive_rtol": float(rtol),
                "adaptive_atol": float(atol),
                "adaptive_dt0": float(dt0),
                "adaptive_runtime_refresh_total_seconds": float(
                    runtime_diag.get("refresh_total_seconds", 0.0)
                ),
                "adaptive_runtime_refresh_input_seconds": float(
                    runtime_diag.get("refresh_input_seconds", 0.0)
                ),
                "adaptive_runtime_refresh_tree_upward_seconds": float(
                    runtime_diag.get("refresh_tree_upward_seconds", 0.0)
                ),
                "adaptive_runtime_refresh_nearfield_seconds": float(
                    runtime_diag.get("refresh_nearfield_seconds", 0.0)
                ),
                "adaptive_runtime_refresh_compile_or_sync_suspect_seconds": float(
                    runtime_diag.get("refresh_compile_or_sync_suspect_seconds", 0.0)
                ),
                "adaptive_runtime_refresh_timing_calls": int(
                    runtime_diag.get("refresh_timing_calls", 0)
                ),
                "adaptive_runtime_strict_runner_compile_count": int(
                    runtime_diag.get("strict_runner_compile_count", 0)
                ),
                "adaptive_runtime_strict_runner_execute_count": int(
                    runtime_diag.get("strict_runner_execute_count", 0)
                ),
                "adaptive_runtime_strict_v2_compile_count": int(
                    runtime_diag.get("strict_v2_compile_count", 0)
                ),
                "adaptive_runtime_strict_v2_execute_count": int(
                    runtime_diag.get("strict_v2_execute_count", 0)
                ),
            }
        )
        try:
            ts = jnp.asarray(sol.ts)
            if ts.ndim == 1 and ts.size > 1:
                dt_hist = jnp.diff(ts)
                timing_stats.update(
                    {
                        "adaptive_dt_mean": float(jnp.nanmean(dt_hist)),
                        "adaptive_dt_median": float(jnp.nanmedian(dt_hist)),
                        "adaptive_dt_min": float(jnp.nanmin(dt_hist)),
                        "adaptive_dt_max": float(jnp.nanmax(dt_hist)),
                    }
                )
        except Exception:
            pass

    ys = jnp.asarray(sol.ys)
    if bool(return_history):
        return ys
    if ys.ndim >= 1:
        return jnp.asarray(ys[-1])
    return ys


def evaluate_acceleration_jaccpot(
    state: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    active_indices: Optional[jnp.ndarray] = None,
    leaf_size: int = 16,
    max_order: int = 4,
    fmm_preset: str = "fast",
    fmm_basis: str = "real",
    fmm_theta: float = 0.6,
    fmm_runtime_path: str = "auto",
    fmm_working_dtype=None,
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_m2l_chunk_size: Optional[int] = None,
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_build_mode: str = "lbvh",
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
    fmm_max_pair_queue: Optional[int] = None,
    fmm_pair_process_block: Optional[int] = None,
    fmm_max_interactions_per_node: Optional[int] = None,
    fmm_max_neighbors_per_leaf: Optional[int] = None,
    fmm_prepare_stage_memory_split_enabled: Optional[bool] = None,
    fmm_upward_leaf_batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Evaluate one FMM acceleration call for an ODISSEO primitive state."""
    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    solver = _build_fmm_solver(
        working_dtype=(
            state_arr.dtype
            if fmm_working_dtype is None
            else jnp.dtype(fmm_working_dtype)
        ),
        config=config,
        params=params,
        fmm_preset=fmm_preset,
        fmm_basis=fmm_basis,
        fmm_theta=fmm_theta,
        fmm_runtime_path=fmm_runtime_path,
        fmm_mac_type=fmm_mac_type,
        fmm_farfield_mode=fmm_farfield_mode,
        fmm_m2l_chunk_size=fmm_m2l_chunk_size,
        fmm_nearfield_mode=fmm_nearfield_mode,
        fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
        fmm_tree_build_mode=fmm_tree_build_mode,
        fmm_tree_leaf_target=fmm_tree_leaf_target,
        fmm_fixed_order=fmm_fixed_order,
        leaf_size=leaf_size,
        fmm_jit_tree=fmm_jit_tree,
        fmm_jit_traversal=fmm_jit_traversal,
        fmm_max_pair_queue=fmm_max_pair_queue,
        fmm_pair_process_block=fmm_pair_process_block,
        fmm_max_interactions_per_node=fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=(fmm_prepare_stage_memory_split_enabled),
        fmm_upward_leaf_batch_size=(
            getattr(config, "fmm_upward_leaf_batch_size", None)
            if fmm_upward_leaf_batch_size is None
            else int(fmm_upward_leaf_batch_size)
        ),
    )
    with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
        prepared = solver.prepare_state(
            state_arr[:, 0, :],
            mass_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
    return solver.evaluate_prepared_state(
        prepared,
        target_indices=active_indices,
        return_potential=False,
    )


def build_jitted_jaccpot_acceleration(
    config: SimulationConfig,
    params: SimulationParams,
    *,
    active_indices: Optional[jnp.ndarray] = None,
    leaf_size: int = 16,
    max_order: int = 4,
    fmm_preset: str = "fast",
    fmm_basis: str = "real",
    fmm_theta: float = 0.6,
    fmm_runtime_path: str = "auto",
    fmm_working_dtype=None,
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_m2l_chunk_size: Optional[int] = None,
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_build_mode: str = "lbvh",
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
    fmm_max_pair_queue: Optional[int] = None,
    fmm_pair_process_block: Optional[int] = None,
    fmm_max_interactions_per_node: Optional[int] = None,
    fmm_max_neighbors_per_leaf: Optional[int] = None,
    fmm_prepare_stage_memory_split_enabled: Optional[bool] = None,
    fmm_upward_leaf_batch_size: Optional[int] = None,
    outer_jit: bool = False,
):
    """Return a reusable one-call FMM acceleration evaluator.

    Notes
    -----
    By default this wrapper does not apply an additional outer ``jax.jit``.
    The jaccpot runtime already uses internal compiled kernels, and outer-jitting
    full tree build/evaluation can be substantially slower on current runtime
    paths. Set ``outer_jit=True`` only for explicit experimentation.
    """

    def _eager(state: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
        return evaluate_acceleration_jaccpot(
            state,
            mass,
            config,
            params,
            active_indices=active_indices,
            leaf_size=leaf_size,
            max_order=max_order,
            fmm_preset=fmm_preset,
            fmm_basis=fmm_basis,
            fmm_theta=fmm_theta,
            fmm_runtime_path=fmm_runtime_path,
            fmm_working_dtype=fmm_working_dtype,
            fmm_mac_type=fmm_mac_type,
            fmm_farfield_mode=fmm_farfield_mode,
            fmm_m2l_chunk_size=fmm_m2l_chunk_size,
            fmm_nearfield_mode=fmm_nearfield_mode,
            fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
            fmm_tree_build_mode=fmm_tree_build_mode,
            fmm_tree_leaf_target=fmm_tree_leaf_target,
            fmm_fixed_order=fmm_fixed_order,
            fmm_jit_tree=fmm_jit_tree,
            fmm_jit_traversal=fmm_jit_traversal,
            fmm_max_pair_queue=fmm_max_pair_queue,
            fmm_pair_process_block=fmm_pair_process_block,
            fmm_max_interactions_per_node=fmm_max_interactions_per_node,
            fmm_max_neighbors_per_leaf=fmm_max_neighbors_per_leaf,
            fmm_prepare_stage_memory_split_enabled=(
                fmm_prepare_stage_memory_split_enabled
            ),
            fmm_upward_leaf_batch_size=fmm_upward_leaf_batch_size,
        )

    if bool(outer_jit):
        return jax.jit(_eager)
    return _eager
