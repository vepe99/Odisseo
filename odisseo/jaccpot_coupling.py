from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from functools import partial
import hashlib
import inspect
import os
import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.potentials import combined_external_acceleration_vmpa_switch


def _large_n_environment_overrides(
    config: SimulationConfig,
    *,
    fmm_preset: Optional[str] = None,
) -> dict[str, str]:
    """Return jaccpot large-N env overrides requested by SimulationConfig."""
    overrides: dict[str, str] = {}
    if not bool(getattr(config, "fmm_large_n_environment_overrides_enabled", True)):
        return overrides
    target_block_size = getattr(config, "fmm_large_n_target_block_size", None)

    static_target_blocks = getattr(config, "fmm_large_n_static_target_blocks", None)
    auto_static_target_blocks = (
        static_target_blocks is None
        and str(getattr(config, "fmm_tree_build_mode", "")).strip().lower()
        == "static_radix"
        and str(fmm_preset or getattr(config, "fmm_preset", "")).strip().lower()
        == "large_n_gpu"
        and int(getattr(config, "N_particles", 0))
        >= int(getattr(config, "fmm_large_n_min_particles", 200_000))
    )
    if auto_static_target_blocks:
        static_target_blocks = True
        if target_block_size is None:
            target_block_size = 4

    if target_block_size is not None:
        overrides["JACCPOT_LARGE_N_TARGET_BLOCK_SIZE"] = str(int(target_block_size))

    if static_target_blocks is not None:
        overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS"] = (
            "1" if bool(static_target_blocks) else "0"
        )

    static_target_blocks_cap = getattr(
        config,
        "fmm_large_n_static_target_blocks_max_per_leaf",
        None,
    )
    if auto_static_target_blocks and static_target_blocks_cap is None:
        static_target_blocks_cap = 16
    if static_target_blocks_cap is not None:
        overrides["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] = str(
            int(static_target_blocks_cap)
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
):
    from jaccpot import (
        FMMAdvancedConfig,
        FarFieldConfig,
        FastMultipoleMethod,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )
    from yggdrax.interactions import DualTreeTraversalConfig

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

    return FastMultipoleMethod(
        preset=str(fmm_preset),
        basis=str(fmm_basis),
        runtime_path=str(fmm_runtime_path),
        theta=float(fmm_theta),
        G=float(params.G),
        softening=float(config.softening),
        working_dtype=working_dtype,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode=str(fmm_tree_build_mode),
                leaf_target=int(fmm_tree_leaf_target),
            ),
            farfield=FarFieldConfig(
                mode=str(fmm_farfield_mode),
                m2l_chunk_size=(
                    None if fmm_m2l_chunk_size is None else int(fmm_m2l_chunk_size)
                ),
            ),
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
            ),
            mac_type=str(fmm_mac_type),
        ),
        fixed_order=(
            None if fmm_fixed_order is None else int(fmm_fixed_order)
        ),
        # Keep one global leaf-size contract per simulation: tree target and
        # runtime leaf cap are tied to the same value.
        fixed_max_leaf_size=int(leaf_size),
    )


def _scatter_masked_vectors(
    base: jnp.ndarray,
    indices: jnp.ndarray,
    values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Scatter updates for masked rows while leaving others unchanged."""
    safe_idx = jnp.where(mask, indices, 0)
    gathered = base[safe_idx]
    updates = jnp.where(mask[:, None], values, gathered)
    return base.at[safe_idx].set(updates)


def _prepared_state_shape_signature(
    prepared_state: Any,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return dtype+shape signature across array leaves in a prepared state."""
    leaves, _ = jax.tree_util.tree_flatten(prepared_state)
    signature = []
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape is None:
            continue
        dtype = str(getattr(leaf, "dtype", "unknown"))
        signature.append((dtype, tuple(int(d) for d in shape)))
    # Canonicalize ordering so equivalent leaf-shape multisets compare equal
    # even if pytree traversal order changes across topology variants.
    return tuple(sorted(signature))


@partial(jax.jit, static_argnames=("add_external", "config", "params"))
def _leapfrog_step_full_const_self(
    state_curr: jnp.ndarray,
    acc_self_full: jnp.ndarray,
    dt_arr: jnp.ndarray,
    *,
    add_external: bool,
    config: SimulationConfig,
    params: SimulationParams,
) -> jnp.ndarray:
    """One full-particle leapfrog step with fixed self-gravity field."""
    if add_external:
        acc_1 = acc_self_full + combined_external_acceleration_vmpa_switch(
            state_curr,
            config,
            params,
        )
    else:
        acc_1 = acc_self_full

    pos_new = state_curr[:, 0] + state_curr[:, 1] * dt_arr + 0.5 * acc_1 * (dt_arr**2)
    state_pos = state_curr.at[:, 0].set(pos_new)

    if add_external:
        acc_2 = acc_self_full + combined_external_acceleration_vmpa_switch(
            state_pos,
            config,
            params,
        )
    else:
        acc_2 = acc_self_full

    vel_new = state_curr[:, 1] + 0.5 * (acc_1 + acc_2) * dt_arr
    return state_pos.at[:, 1].set(vel_new)


@partial(jax.jit, static_argnames=("steps", "add_external", "config", "params"))
def _run_full_segment_scan(
    state_curr: jnp.ndarray,
    acc_self_full: jnp.ndarray,
    dt_arr: jnp.ndarray,
    *,
    steps: int,
    add_external: bool,
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a jitted scan segment for full-particle updates."""

    def _body(carry, _):
        state_next = _leapfrog_step_full_const_self(
            carry,
            acc_self_full,
            dt_arr,
            add_external=add_external,
            config=config,
            params=params,
        )
        return state_next, state_next

    return jax.lax.scan(_body, state_curr, xs=None, length=int(steps))


@partial(jax.jit, static_argnames=("add_external", "config", "params"))
def _leapfrog_step_active_const_self(
    state_curr: jnp.ndarray,
    acc_self_full: jnp.ndarray,
    active_indices: jnp.ndarray,
    active_mask: jnp.ndarray,
    dt_arr: jnp.ndarray,
    *,
    add_external: bool,
    config: SimulationConfig,
    params: SimulationParams,
) -> jnp.ndarray:
    """One masked active-particle leapfrog step with fixed self-gravity field."""
    safe_idx = jnp.where(active_mask, active_indices, 0)

    pos = state_curr[:, 0]
    vel = state_curr[:, 1]
    acc_self_active = acc_self_full[safe_idx]

    if add_external:
        ext_full = combined_external_acceleration_vmpa_switch(state_curr, config, params)
        acc_1 = acc_self_active + ext_full[safe_idx]
    else:
        acc_1 = acc_self_active

    pos_active_new = pos[safe_idx] + vel[safe_idx] * dt_arr + 0.5 * acc_1 * (dt_arr**2)
    pos_new = _scatter_masked_vectors(pos, safe_idx, pos_active_new, active_mask)
    state_pos = state_curr.at[:, 0].set(pos_new)

    if add_external:
        ext_full_2 = combined_external_acceleration_vmpa_switch(state_pos, config, params)
        acc_2 = acc_self_active + ext_full_2[safe_idx]
    else:
        acc_2 = acc_self_active

    vel_active_new = vel[safe_idx] + 0.5 * (acc_1 + acc_2) * dt_arr
    vel_new = _scatter_masked_vectors(vel, safe_idx, vel_active_new, active_mask)
    return state_pos.at[:, 1].set(vel_new)


@partial(jax.jit, static_argnames=("add_external", "config", "params"))
def _run_active_segment_scan(
    state_curr: jnp.ndarray,
    acc_self_full: jnp.ndarray,
    active_indices_segment: jnp.ndarray,
    active_mask_segment: jnp.ndarray,
    dt_arr: jnp.ndarray,
    *,
    add_external: bool,
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a jitted scan segment for masked active-particle updates."""

    def _body(carry, xs):
        idx_row, mask_row = xs
        state_next = _leapfrog_step_active_const_self(
            carry,
            acc_self_full,
            idx_row,
            mask_row,
            dt_arr,
            add_external=add_external,
            config=config,
            params=params,
        )
        return state_next, state_next

    return jax.lax.scan(
        _body,
        state_curr,
        xs=(active_indices_segment, active_mask_segment),
    )


def integrate_leapfrog_jaccpot_active(
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
    fmm_basis: str = "solidfmm",
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
    """Integrate with Jaccpot FMM using optional active-particle substeps.

    Notes
    -----
    - Source tree is refreshed every ``refresh_every`` steps.
    - ``active_indices_schedule`` + ``active_mask_schedule`` enables a scan/JIT
      path for active subsets with fixed-size index rows.
    - Between refreshes, self-gravity is evaluated with fixed sources, then
      state updates are vectorized in JAX.
    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(refresh_every) <= 0:
        raise ValueError("refresh_every must be positive")
    if active_indices_fn is not None and active_indices_schedule is not None:
        raise ValueError("Provide either active_indices_fn or active_indices_schedule")
    if active_indices_schedule is None and active_mask_schedule is not None:
        raise ValueError("active_mask_schedule requires active_indices_schedule")

    state_curr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    profile = timing_stats is not None
    profile_sync = (
        str(os.environ.get("ODISSEO_PROFILE_SYNC", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    strict_mode_env = str(
        os.environ.get("JACCPOT_STATIC_STRICT_GPU_MODE", "auto")
    ).strip().lower()
    strict_mode_requested = strict_mode_env in {"on", "auto"}
    strict_production_lane = bool(
        strict_mode_requested
        and str(fmm_preset).strip().lower() == "large_n_gpu"
        and str(fmm_runtime_path).strip().lower() in {"large_n", "auto"}
        and str(fmm_tree_build_mode).strip().lower() == "static_radix"
    )
    collect_shape_signatures = bool(profile or enforce_static_shape_contract)
    t_total_start = time.perf_counter() if profile else 0.0
    prepare_seconds = 0.0
    evaluate_seconds = 0.0
    update_seconds = 0.0
    warmup_seconds = 0.0
    prepare_calls = 0
    evaluate_calls = 0
    update_calls = 0
    warmup_prepare_calls = 0
    warmup_evaluate_calls = 0
    refresh_prepare_attempts = 0
    refresh_prepare_successes = 0
    refresh_prepare_fallbacks = 0
    full_prepare_calls = 0
    profiled_full_prepare_calls = 0
    profiled_refresh_prepare_calls = 0
    profiled_refresh_fallback_prepare_calls = 0
    profiled_full_prepare_seconds = 0.0
    profiled_refresh_prepare_seconds = 0.0
    profiled_refresh_fallback_prepare_seconds = 0.0
    profiled_prepare_events: list[dict[str, Any]] = []
    last_prepare_path = "none"
    refresh_disabled = str(
        os.environ.get("ODISSEO_DISABLE_FMM_REFRESH_PREPARED_STATE", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    refresh_fn = getattr(solver, "refresh_prepared_state", None)
    refresh_fn_callable = callable(refresh_fn)
    refresh_fn_signature = (
        inspect.signature(refresh_fn) if refresh_fn_callable else None
    )
    refresh_fn_params = (
        refresh_fn_signature.parameters if refresh_fn_signature is not None else {}
    )
    refresh_fn_accepts_var_kw = bool(
        refresh_fn_signature is not None
        and any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in refresh_fn_params.values()
        )
    )
    refresh_call_mode_index: Optional[int] = None
    prepare_stage_keys = (
        "refresh_input_seconds",
        "refresh_tree_upward_seconds",
        "refresh_tree_build_seconds",
        "refresh_upward_compute_seconds",
        "refresh_upward_geometry_seconds",
        "refresh_upward_mass_moments_seconds",
        "refresh_upward_p2m_seconds",
        "refresh_upward_m2m_seconds",
        "refresh_upward_source_motion_seconds",
        "refresh_dual_downward_seconds",
        "refresh_dual_setup_seconds",
        "refresh_dual_artifact_build_seconds",
        "refresh_dual_split_shared_far_near_seconds",
        "refresh_dual_split_shared_count_seconds",
        "refresh_dual_split_shared_combined_fill_seconds",
        "refresh_dual_split_shared_far_fill_seconds",
        "refresh_dual_split_shared_near_fill_seconds",
        "refresh_dual_split_far_pairs_seconds",
        "refresh_dual_split_leaf_neighbors_seconds",
        "refresh_dual_split_combined_seconds",
        "refresh_dual_raw_combined_seconds",
        "refresh_dual_split_dense_buffers_seconds",
        "refresh_dual_far_pair_plan_seconds",
        "refresh_dual_m2l_autotune_seconds",
        "refresh_dual_select_interactions_seconds",
        "refresh_dual_downward_compute_seconds",
        "refresh_dual_m2l_compute_seconds",
        "refresh_dual_l2l_compute_seconds",
        "refresh_dual_finalize_seconds",
        "refresh_dual_residual_seconds",
        "refresh_nearfield_seconds",
        "refresh_nearfield_leaf_groups_seconds",
        "refresh_nearfield_precompute_seconds",
        "refresh_nearfield_target_blocks_seconds",
        "refresh_nearfield_block_sort_seconds",
        "refresh_nearfield_speed_layout_seconds",
        "refresh_nearfield_overflow_profile_seconds",
        "refresh_nearfield_radix_payload_seconds",
        "refresh_nearfield_neighbor_padding_seconds",
        "refresh_nearfield_state_pack_seconds",
        "refresh_nearfield_residual_seconds",
        "refresh_compile_or_sync_suspect_seconds",
    )
    shape_signature_ref: Optional[tuple[tuple[str, tuple[int, ...]], ...]] = None
    shape_signature_unique: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    shape_drift_events = 0
    shape_checks = 0
    shape_signature_ref_post_warmup: Optional[
        tuple[tuple[str, tuple[int, ...]], ...]
    ] = None
    shape_signature_unique_post_warmup: set[
        tuple[tuple[str, tuple[int, ...]], ...]
    ] = set()
    shape_drift_events_post_warmup = 0
    shape_checks_post_warmup = 0
    shape_signature_hashes_post_warmup: list[str] = []
    shape_signature_diff_post_warmup: list[dict[str, Any]] = []

    def _record_shape_signature(prepared_state, *, warmup_phase: bool = False):
        nonlocal shape_signature_ref
        nonlocal shape_signature_unique
        nonlocal shape_drift_events
        nonlocal shape_checks
        nonlocal shape_signature_ref_post_warmup
        nonlocal shape_signature_unique_post_warmup
        nonlocal shape_drift_events_post_warmup
        nonlocal shape_checks_post_warmup
        nonlocal shape_signature_hashes_post_warmup
        nonlocal shape_signature_diff_post_warmup
        if not collect_shape_signatures:
            return
        signature = _prepared_state_shape_signature(prepared_state)
        shape_checks += 1
        if profile:
            shape_signature_unique.add(signature)
        if shape_signature_ref is None:
            shape_signature_ref = signature
            return
        if signature != shape_signature_ref:
            shape_drift_events += 1
            if bool(enforce_static_shape_contract):
                raise RuntimeError(
                    "Static-shape contract violated in FMM prepared state: "
                    "leaf dtype/shape signature drifted across refresh segments."
                )
        if warmup_phase:
            return
        shape_checks_post_warmup += 1
        if profile:
            sig_hash = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()
            shape_signature_unique_post_warmup.add(signature)
            shape_signature_hashes_post_warmup.append(sig_hash)
        if shape_signature_ref_post_warmup is None:
            shape_signature_ref_post_warmup = signature
            return
        if signature != shape_signature_ref_post_warmup:
            shape_drift_events_post_warmup += 1
            if bool(enforce_static_shape_contract):
                raise RuntimeError(
                    "Static-shape contract violated in FMM prepared state: "
                    "leaf dtype/shape signature drifted after warmup."
                )
            if not profile:
                return
            ref_counter = Counter(shape_signature_ref_post_warmup)
            cur_counter = Counter(signature)
            added = []
            removed = []
            for key, count in (cur_counter - ref_counter).items():
                added.append({"dtype": key[0], "shape": list(key[1]), "count": int(count)})
            for key, count in (ref_counter - cur_counter).items():
                removed.append({"dtype": key[0], "shape": list(key[1]), "count": int(count)})
            shape_signature_diff_post_warmup = [
                {"added": added[:12], "removed": removed[:12]}
            ]

    def _finalize(out_arr):
        if not profile:
            return out_arr
        if profile_sync:
            _ = jax.block_until_ready(out_arr)
        stage_seconds_by_path: dict[str, dict[str, float]] = {}
        for event in profiled_prepare_events:
            path = str(event.get("path", "unknown"))
            stage_seconds = event.get("stage_seconds", {})
            if not isinstance(stage_seconds, dict):
                continue
            path_bucket = stage_seconds_by_path.setdefault(path, {})
            for key, value in stage_seconds.items():
                path_bucket[str(key)] = float(path_bucket.get(str(key), 0.0)) + float(
                    value
                )
        runtime_diag = {}
        get_diag = getattr(solver, "get_runtime_diagnostics", None)
        if callable(get_diag):
            try:
                runtime_diag = dict(get_diag())
            except Exception:
                runtime_diag = {}
        timing_stats.clear()
        timing_stats.update(
            {
                "total_seconds": float(time.perf_counter() - t_total_start),
                "prepare_seconds": float(prepare_seconds),
                "evaluate_seconds": float(evaluate_seconds),
                "update_seconds": float(update_seconds),
                "prepare_calls": int(prepare_calls),
                "evaluate_calls": int(evaluate_calls),
                "update_calls": int(update_calls),
                "num_steps": int(num_steps),
                "refresh_every": int(refresh_every),
                "used_external_potential": bool(add_external),
                "used_schedule_scan_mode": bool(active_indices_schedule is not None),
                "used_fast_full_scan_mode": bool(
                    active_indices_fn is None and not bool(refresh_after_position_update)
                ),
                "shape_contract_enforced": bool(enforce_static_shape_contract),
                "shape_signature_checks": int(shape_checks),
                "shape_signature_unique_count": int(len(shape_signature_unique)),
                "shape_signature_drift_events": int(shape_drift_events),
                "shape_signature_stable": bool(shape_drift_events == 0),
                "shape_signature_checks_post_warmup": int(shape_checks_post_warmup),
                "shape_signature_unique_count_post_warmup": int(
                    len(shape_signature_unique_post_warmup)
                ),
                "shape_signature_drift_events_post_warmup": int(
                    shape_drift_events_post_warmup
                ),
                "shape_signature_stable_post_warmup": bool(
                    shape_drift_events_post_warmup == 0
                ),
                "shape_signature_hashes_post_warmup": list(
                    shape_signature_hashes_post_warmup
                ),
                "shape_signature_diff_post_warmup": list(
                    shape_signature_diff_post_warmup
                ),
                "warmup_seconds": float(warmup_seconds),
                "warmup_prepare_calls": int(warmup_prepare_calls),
                "warmup_evaluate_calls": int(warmup_evaluate_calls),
                "refresh_prepare_attempts": int(refresh_prepare_attempts),
                "refresh_prepare_successes": int(refresh_prepare_successes),
                "refresh_prepare_fallbacks": int(refresh_prepare_fallbacks),
                "full_prepare_calls": int(full_prepare_calls),
                "profiled_full_prepare_calls": int(profiled_full_prepare_calls),
                "profiled_refresh_prepare_calls": int(profiled_refresh_prepare_calls),
                "profiled_refresh_fallback_prepare_calls": int(
                    profiled_refresh_fallback_prepare_calls
                ),
                "profiled_full_prepare_seconds": float(profiled_full_prepare_seconds),
                "profiled_refresh_prepare_seconds": float(
                    profiled_refresh_prepare_seconds
                ),
                "profiled_refresh_fallback_prepare_seconds": float(
                    profiled_refresh_fallback_prepare_seconds
                ),
                "profiled_prepare_events": list(profiled_prepare_events),
                "profiled_prepare_stage_seconds_by_path": {
                    path: {
                        str(key): float(value)
                        for key, value in sorted(stage_seconds.items())
                    }
                    for path, stage_seconds in sorted(stage_seconds_by_path.items())
                },
                "refresh_prepare_method_available": bool(
                    callable(getattr(solver, "refresh_prepared_state", None))
                ),
                "rematerialize_between_refresh": bool(rematerialize_between_refresh),
                "runtime_compiled_profile_fingerprint_last": runtime_diag.get(
                    "compiled_profile_fingerprint_last"
                ),
                "runtime_compiled_profile_transitions": int(
                    runtime_diag.get("compiled_profile_transitions", 0)
                ),
                "runtime_refresh_prepare_calls": int(
                    runtime_diag.get("refresh_prepare_calls", 0)
                ),
                "runtime_max_leaf_size": int(runtime_diag.get("max_leaf_size", 0)),
                "runtime_max_leaves": int(runtime_diag.get("max_leaves", 0)),
                "runtime_refresh_prepare_reuse_tier_full": int(
                    runtime_diag.get("refresh_prepare_reuse_tier_full", 0)
                ),
                "runtime_refresh_prepare_reuse_tier_topology": int(
                    runtime_diag.get("refresh_prepare_reuse_tier_topology", 0)
                ),
                "runtime_refresh_prepare_reuse_tier_overflow": int(
                    runtime_diag.get("refresh_prepare_reuse_tier_overflow", 0)
                ),
                "runtime_large_n_same_topology_refresh_attempts": int(
                    runtime_diag.get("large_n_same_topology_refresh_attempts", 0)
                ),
                "runtime_large_n_same_topology_refresh_hits": int(
                    runtime_diag.get("large_n_same_topology_refresh_hits", 0)
                ),
                "runtime_large_n_same_topology_refresh_misses": int(
                    runtime_diag.get("large_n_same_topology_refresh_misses", 0)
                ),
                "runtime_large_n_same_topology_refresh_miss_no_key": int(
                    runtime_diag.get("large_n_same_topology_refresh_miss_no_key", 0)
                ),
                "runtime_large_n_same_topology_refresh_miss_topology": int(
                    runtime_diag.get("large_n_same_topology_refresh_miss_topology", 0)
                ),
                "runtime_large_n_same_topology_refresh_miss_neighbor": int(
                    runtime_diag.get("large_n_same_topology_refresh_miss_neighbor", 0)
                ),
                "runtime_large_n_same_topology_refresh_miss_traced": int(
                    runtime_diag.get("large_n_same_topology_refresh_miss_traced", 0)
                ),
                "runtime_large_n_same_topology_refresh_last_error": str(
                    runtime_diag.get("large_n_same_topology_refresh_last_error", "")
                ),
                "runtime_static_radix_refresh_hits": int(
                    runtime_diag.get("static_radix_refresh_hits", 0)
                ),
                "runtime_static_radix_refresh_misses": int(
                    runtime_diag.get("static_radix_refresh_misses", 0)
                ),
                "runtime_static_radix_profile_overflows": int(
                    runtime_diag.get("static_radix_profile_overflows", 0)
                ),
                "runtime_large_n_overflow_profile_cap": int(
                    runtime_diag.get("large_n_overflow_profile_cap", 0)
                ),
                "runtime_large_n_overflow_profile_reprofiles": int(
                    runtime_diag.get("large_n_overflow_profile_reprofiles", 0)
                ),
                "runtime_large_n_neighbor_edges_profile_cap": int(
                    runtime_diag.get("large_n_neighbor_edges_profile_cap", 0)
                ),
                "runtime_large_n_neighbor_edges_profile_reprofiles": int(
                    runtime_diag.get("large_n_neighbor_edges_profile_reprofiles", 0)
                ),
                "runtime_interaction_cache_hits": int(
                    runtime_diag.get("interaction_cache_hits", 0)
                ),
                "runtime_interaction_cache_misses": int(
                    runtime_diag.get("interaction_cache_misses", 0)
                ),
                "runtime_refresh_dual_planner_cache_hits": int(
                    runtime_diag.get("refresh_dual_planner_cache_hits", 0)
                ),
                "runtime_refresh_dual_planner_cache_misses": int(
                    runtime_diag.get("refresh_dual_planner_cache_misses", 0)
                ),
                "runtime_refresh_dual_planner_compile_count": int(
                    runtime_diag.get("refresh_dual_planner_compile_count", 0)
                ),
                "runtime_refresh_dual_planner_execute_count": int(
                    runtime_diag.get("refresh_dual_planner_execute_count", 0)
                ),
                "runtime_refresh_dual_planner_steady_timing_bypass_count": int(
                    runtime_diag.get(
                        "refresh_dual_planner_steady_timing_bypass_count",
                        0,
                    )
                ),
                "runtime_refresh_dual_planner_compiled_route_count": int(
                    runtime_diag.get("refresh_dual_planner_compiled_route_count", 0)
                ),
                "runtime_refresh_strict_mode_active_count": int(
                    runtime_diag.get("refresh_strict_mode_active_count", 0)
                ),
                "runtime_strict_profiled_max_pair_queue": int(
                    runtime_diag.get("strict_profiled_max_pair_queue", 0)
                ),
                "runtime_strict_profiled_pair_process_block": int(
                    runtime_diag.get("strict_profiled_pair_process_block", 0)
                ),
                "runtime_strict_profiled_context_key": str(
                    runtime_diag.get("strict_profiled_context_key", "")
                ),
                "runtime_refresh_total_seconds": float(
                    runtime_diag.get("refresh_total_seconds", 0.0)
                ),
                "runtime_refresh_input_seconds": float(
                    runtime_diag.get("refresh_input_seconds", 0.0)
                ),
                "runtime_refresh_tree_upward_seconds": float(
                    runtime_diag.get("refresh_tree_upward_seconds", 0.0)
                ),
                "runtime_refresh_tree_build_seconds": float(
                    runtime_diag.get("refresh_tree_build_seconds", 0.0)
                ),
                "runtime_refresh_upward_compute_seconds": float(
                    runtime_diag.get("refresh_upward_compute_seconds", 0.0)
                ),
                "runtime_refresh_upward_geometry_seconds": float(
                    runtime_diag.get("refresh_upward_geometry_seconds", 0.0)
                ),
                "runtime_refresh_upward_mass_moments_seconds": float(
                    runtime_diag.get("refresh_upward_mass_moments_seconds", 0.0)
                ),
                "runtime_refresh_upward_p2m_seconds": float(
                    runtime_diag.get("refresh_upward_p2m_seconds", 0.0)
                ),
                "runtime_refresh_upward_m2m_seconds": float(
                    runtime_diag.get("refresh_upward_m2m_seconds", 0.0)
                ),
                "runtime_refresh_upward_source_motion_seconds": float(
                    runtime_diag.get("refresh_upward_source_motion_seconds", 0.0)
                ),
                "runtime_refresh_dual_downward_seconds": float(
                    runtime_diag.get("refresh_dual_downward_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_seconds": float(
                    runtime_diag.get("refresh_nearfield_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_leaf_groups_seconds": float(
                    runtime_diag.get("refresh_nearfield_leaf_groups_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_precompute_seconds": float(
                    runtime_diag.get("refresh_nearfield_precompute_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_target_blocks_seconds": float(
                    runtime_diag.get("refresh_nearfield_target_blocks_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_block_sort_seconds": float(
                    runtime_diag.get("refresh_nearfield_block_sort_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_speed_layout_seconds": float(
                    runtime_diag.get("refresh_nearfield_speed_layout_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_overflow_profile_seconds": float(
                    runtime_diag.get("refresh_nearfield_overflow_profile_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_radix_payload_seconds": float(
                    runtime_diag.get("refresh_nearfield_radix_payload_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_neighbor_padding_seconds": float(
                    runtime_diag.get("refresh_nearfield_neighbor_padding_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_state_pack_seconds": float(
                    runtime_diag.get("refresh_nearfield_state_pack_seconds", 0.0)
                ),
                "runtime_refresh_nearfield_residual_seconds": float(
                    runtime_diag.get("refresh_nearfield_residual_seconds", 0.0)
                ),
                "runtime_refresh_profile_accounting_seconds": float(
                    runtime_diag.get("refresh_profile_accounting_seconds", 0.0)
                ),
                "runtime_refresh_compile_or_sync_suspect_seconds": float(
                    runtime_diag.get(
                        "refresh_compile_or_sync_suspect_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_setup_seconds": float(
                    runtime_diag.get("refresh_dual_setup_seconds", 0.0)
                ),
                "runtime_refresh_dual_artifact_build_seconds": float(
                    runtime_diag.get("refresh_dual_artifact_build_seconds", 0.0)
                ),
                "runtime_refresh_dual_split_shared_far_near_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_split_shared_far_near_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_split_shared_count_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_split_shared_count_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_split_shared_combined_fill_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_split_shared_combined_fill_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_split_shared_far_fill_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_split_shared_far_fill_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_split_shared_near_fill_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_split_shared_near_fill_seconds",
                        0.0,
                    )
                ),
                "runtime_refresh_dual_far_pair_plan_seconds": float(
                    runtime_diag.get("refresh_dual_far_pair_plan_seconds", 0.0)
                ),
                "runtime_refresh_dual_m2l_autotune_seconds": float(
                    runtime_diag.get("refresh_dual_m2l_autotune_seconds", 0.0)
                ),
                "runtime_refresh_dual_select_interactions_seconds": float(
                    runtime_diag.get(
                        "refresh_dual_select_interactions_seconds",
                        0.0,
                    )
                ),
                "runtime_recent_dual_node_count": int(
                    runtime_diag.get("recent_dual_node_count", 0)
                ),
                "runtime_recent_dual_leaf_count": int(
                    runtime_diag.get("recent_dual_leaf_count", 0)
                ),
                "runtime_recent_dual_neighbor_count": int(
                    runtime_diag.get("recent_dual_neighbor_count", 0)
                ),
                "runtime_recent_dual_far_pair_count": int(
                    runtime_diag.get("recent_dual_far_pair_count", 0)
                ),
                "runtime_recent_dual_far_pairs_by_gear_counts": tuple(
                    int(v)
                    for v in runtime_diag.get(
                        "recent_dual_far_pairs_by_gear_counts",
                        tuple(),
                    )
                ),
                "runtime_recent_dual_m2l_chunk_size": int(
                    runtime_diag.get("recent_dual_m2l_chunk_size", 0)
                ),
                "runtime_refresh_dual_downward_compute_seconds": float(
                    runtime_diag.get("refresh_dual_downward_compute_seconds", 0.0)
                ),
                "runtime_refresh_dual_m2l_compute_seconds": float(
                    runtime_diag.get("refresh_dual_m2l_compute_seconds", 0.0)
                ),
                "runtime_refresh_dual_l2l_compute_seconds": float(
                    runtime_diag.get("refresh_dual_l2l_compute_seconds", 0.0)
                ),
                "runtime_refresh_dual_final_symmetry_seconds": float(
                    runtime_diag.get("refresh_dual_final_symmetry_seconds", 0.0)
                ),
                "runtime_refresh_dual_source_motion_seconds": float(
                    runtime_diag.get("refresh_dual_source_motion_seconds", 0.0)
                ),
                "runtime_refresh_dual_finalize_seconds": float(
                    runtime_diag.get("refresh_dual_finalize_seconds", 0.0)
                ),
                "runtime_refresh_dual_residual_seconds": float(
                    runtime_diag.get("refresh_dual_residual_seconds", 0.0)
                ),
                "runtime_refresh_timing_calls": int(
                    runtime_diag.get("refresh_timing_calls", 0)
                ),
            }
        )
        return out_arr

    dt_val = float(params.t_end) / float(num_steps) if dt is None else float(dt)
    dt_arr = jnp.asarray(dt_val, dtype=state_curr.dtype)

    solver = _build_fmm_solver(
        working_dtype=(
            state_curr.dtype if fmm_working_dtype is None else jnp.dtype(fmm_working_dtype)
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
        fmm_prepare_stage_memory_split_enabled=(
            fmm_prepare_stage_memory_split_enabled
        ),
        fmm_upward_leaf_batch_size=(
            getattr(config, "fmm_upward_leaf_batch_size", None)
            if fmm_upward_leaf_batch_size is None
            else int(fmm_upward_leaf_batch_size)
        ),
    )
    def _prepare_state(state_in: jnp.ndarray):
        with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
            return solver.prepare_state(
                state_in[:, 0, :],
                mass_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
            )

    def _prepare_or_refresh_state(
        state_in: jnp.ndarray,
        prev_prepared_state: Any | None,
    ) -> Any:
        """Try incremental prepared-state refresh; fallback to full prepare."""
        nonlocal refresh_prepare_attempts
        nonlocal refresh_prepare_successes
        nonlocal refresh_prepare_fallbacks
        nonlocal full_prepare_calls
        nonlocal last_prepare_path

        if prev_prepared_state is None:
            full_prepare_calls += 1
            last_prepare_path = "full"
            return _prepare_state(state_in)

        if bool(refresh_disabled):
            full_prepare_calls += 1
            last_prepare_path = "full"
            return _prepare_state(state_in)

        if not bool(refresh_fn_callable):
            full_prepare_calls += 1
            last_prepare_path = "full"
            return _prepare_state(state_in)

        nonlocal refresh_call_mode_index
        refresh_prepare_attempts += 1
        pos = state_in[:, 0, :]
        leaf = int(leaf_size)
        order = int(max_order)

        # Try several likely public API signatures while keeping behavior
        # backwards-compatible with older jaccpot builds.
        attempts = (
            (
                (prev_prepared_state, pos, mass_arr),
                {"leaf_size": leaf, "max_order": order},
            ),
            (
                (pos, mass_arr, prev_prepared_state),
                {"leaf_size": leaf, "max_order": order},
            ),
            (
                (),
                {
                    "prepared_state": prev_prepared_state,
                    "positions": pos,
                    "masses": mass_arr,
                    "leaf_size": leaf,
                    "max_order": order,
                },
            ),
            (
                (),
                {
                    "previous_prepared_state": prev_prepared_state,
                    "positions": pos,
                    "masses": mass_arr,
                    "leaf_size": leaf,
                    "max_order": order,
                },
            ),
            (
                (prev_prepared_state, pos, mass_arr),
                {},
            ),
            (
                (pos, mass_arr, prev_prepared_state),
                {},
            ),
        )
        if refresh_call_mode_index is None:
            attempt_indices = range(len(attempts))
        else:
            attempt_indices = (int(refresh_call_mode_index),)

        for idx in attempt_indices:
            args, kwargs = attempts[idx]
            try:
                with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
                    if kwargs:
                        if refresh_fn_accepts_var_kw:
                            call_kwargs = kwargs
                        else:
                            call_kwargs = {
                                k: v
                                for k, v in kwargs.items()
                                if k in refresh_fn_params
                            }
                        out = refresh_fn(*args, **call_kwargs)
                    else:
                        out = refresh_fn(*args)
                if refresh_call_mode_index is None:
                    refresh_call_mode_index = int(idx)
                refresh_prepare_successes += 1
                last_prepare_path = "refresh"
                return out
            except TypeError:
                continue

        refresh_prepare_fallbacks += 1
        full_prepare_calls += 1
        last_prepare_path = "refresh_fallback"
        return _prepare_state(state_in)

    def _prepare_stage_snapshot() -> dict[str, float]:
        get_diag = getattr(solver, "get_runtime_diagnostics", None)
        if not callable(get_diag):
            return {}
        try:
            runtime_diag = dict(get_diag())
        except Exception:
            return {}
        return {
            key: float(runtime_diag.get(key, 0.0))
            for key in prepare_stage_keys
        }

    def _profiled_prepare_call(
        state_in: jnp.ndarray,
        prev_prepared_state: Any | None,
    ) -> tuple[Any, float, dict[str, float]]:
        stage_before = _prepare_stage_snapshot()
        timing_target = getattr(solver, "_impl", solver)
        old_timing_active = bool(
            getattr(timing_target, "_refresh_timing_active", False)
        )
        if profile:
            try:
                setattr(timing_target, "_refresh_timing_active", True)
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            prepared_out = _prepare_or_refresh_state(state_in, prev_prepared_state)
            if profile and profile_sync:
                _ = jax.block_until_ready(prepared_out)
            elapsed = time.perf_counter() - t0
        finally:
            if profile:
                try:
                    setattr(
                        timing_target,
                        "_refresh_timing_active",
                        old_timing_active,
                    )
                except Exception:
                    pass
        stage_after = _prepare_stage_snapshot()
        stage_delta = {
            key: float(stage_after.get(key, 0.0) - stage_before.get(key, 0.0))
            for key in prepare_stage_keys
            if abs(float(stage_after.get(key, 0.0) - stage_before.get(key, 0.0)))
            > 0.0
        }
        return prepared_out, float(elapsed), stage_delta

    def _record_profiled_prepare_elapsed(
        elapsed: float,
        stage_delta: Optional[dict[str, float]] = None,
    ) -> None:
        nonlocal profiled_full_prepare_calls
        nonlocal profiled_refresh_prepare_calls
        nonlocal profiled_refresh_fallback_prepare_calls
        nonlocal profiled_full_prepare_seconds
        nonlocal profiled_refresh_prepare_seconds
        nonlocal profiled_refresh_fallback_prepare_seconds
        nonlocal profiled_prepare_events
        event = {
            "index": int(len(profiled_prepare_events)),
            "path": str(last_prepare_path),
            "elapsed_seconds": float(elapsed),
        }
        if stage_delta:
            event["stage_seconds"] = {
                str(key): float(value)
                for key, value in sorted(stage_delta.items())
            }
        profiled_prepare_events.append(event)
        if last_prepare_path == "refresh":
            profiled_refresh_prepare_calls += 1
            profiled_refresh_prepare_seconds += float(elapsed)
        elif last_prepare_path == "refresh_fallback":
            profiled_refresh_fallback_prepare_calls += 1
            profiled_refresh_fallback_prepare_seconds += float(elapsed)
        else:
            profiled_full_prepare_calls += 1
            profiled_full_prepare_seconds += float(elapsed)

    def _eval_prepared(prepared_state, active_indices=None):
        return solver.evaluate_prepared_state(
            prepared_state,
            target_indices=active_indices,
            return_potential=False,
        )

    def _prepare_refresh_and_eval_full(
        state_in: jnp.ndarray,
        prev_prepared_state: Any | None,
    ) -> tuple[Any, jnp.ndarray]:
        """Minimal refresh+evaluate helper for strict full-particle lane."""
        prepared_out = _prepare_or_refresh_state(state_in, prev_prepared_state)
        acc_out = _eval_prepared(prepared_out, active_indices=None)
        return prepared_out, acc_out

    history = []
    add_external = len(config.external_accelerations) > 0

    warmup_prepares = max(0, int(static_shape_warmup_prepares))
    if warmup_prepares > 0:
        prepared_warmup = None
        for _ in range(warmup_prepares):
            tw = time.perf_counter()
            prepared_warmup = _prepare_or_refresh_state(state_curr, prepared_warmup)
            acc_warmup = _eval_prepared(prepared_warmup, active_indices=None)
            if profile_sync:
                _ = jax.block_until_ready(acc_warmup)
            warmup_seconds += time.perf_counter() - tw
            warmup_prepare_calls += 1
            warmup_evaluate_calls += 1
            _record_shape_signature(prepared_warmup, warmup_phase=True)

    if strict_production_lane:
        if active_indices_schedule is not None:
            raise NotImplementedError(
                "Strict production lane does not support active_indices_schedule; "
                "use full-particle scan path."
            )
        if active_indices_fn is not None:
            raise NotImplementedError(
                "Strict production lane does not support active_indices_fn callbacks; "
                "use full-particle scan path."
            )
        if bool(refresh_after_position_update):
            raise NotImplementedError(
                "Strict production lane requires refresh_after_position_update=False."
            )
        if not profile and not bool(return_history):
            # Tight strict lane: avoid generic per-segment profile/history
            # branching and keep host orchestration minimal.
            num_steps_i = int(num_steps)
            refresh_every_i = int(refresh_every)
            full_segments = num_steps_i // refresh_every_i
            tail_segment = num_steps_i % refresh_every_i
            remat_enabled = bool(rematerialize_between_refresh)
            prepared_state = None

            if remat_enabled:
                for _ in range(full_segments):
                    prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                        state_curr, prepared_state
                    )
                    state_curr, _ = _run_full_segment_scan(
                        state_curr,
                        acc_self_full,
                        dt_arr,
                        steps=refresh_every_i,
                        add_external=add_external,
                        config=config,
                        params=params,
                    )
                    state_curr = jnp.asarray(state_curr, dtype=state_curr.dtype)
                if tail_segment > 0:
                    prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                        state_curr, prepared_state
                    )
                    state_curr, _ = _run_full_segment_scan(
                        state_curr,
                        acc_self_full,
                        dt_arr,
                        steps=tail_segment,
                        add_external=add_external,
                        config=config,
                        params=params,
                    )
                    state_curr = jnp.asarray(state_curr, dtype=state_curr.dtype)
            else:
                for _ in range(full_segments):
                    prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                        state_curr, prepared_state
                    )
                    state_curr, _ = _run_full_segment_scan(
                        state_curr,
                        acc_self_full,
                        dt_arr,
                        steps=refresh_every_i,
                        add_external=add_external,
                        config=config,
                        params=params,
                    )
                if tail_segment > 0:
                    prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                        state_curr, prepared_state
                    )
                    state_curr, _ = _run_full_segment_scan(
                        state_curr,
                        acc_self_full,
                        dt_arr,
                        steps=tail_segment,
                        add_external=add_external,
                        config=config,
                        params=params,
                    )
            return _finalize(state_curr)

    if active_indices_schedule is not None:
        active_indices_schedule = jnp.asarray(active_indices_schedule, dtype=jnp.int32)
        if active_indices_schedule.ndim != 2:
            raise ValueError("active_indices_schedule must have shape (num_steps, max_active)")
        if int(active_indices_schedule.shape[0]) != int(num_steps):
            raise ValueError("active_indices_schedule first dimension must equal num_steps")

        if active_mask_schedule is None:
            active_mask_schedule = jnp.ones_like(active_indices_schedule, dtype=bool)
        else:
            active_mask_schedule = jnp.asarray(active_mask_schedule, dtype=bool)
            if active_mask_schedule.shape != active_indices_schedule.shape:
                raise ValueError(
                    "active_mask_schedule must match active_indices_schedule shape"
                )
        if bool(refresh_after_position_update):
            raise NotImplementedError(
                "refresh_after_position_update=True is not supported with "
                "active_indices_schedule scan mode"
            )

        step = 0
        prepared_state = None
        while step < int(num_steps):
            if profile:
                prepared_state, elapsed_prepare, stage_delta = _profiled_prepare_call(
                    state_curr,
                    prepared_state,
                )
                prepare_seconds += elapsed_prepare
                _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                prepare_calls += 1
                t0 = time.perf_counter()
            else:
                prepared_state = _prepare_or_refresh_state(state_curr, prepared_state)
            _record_shape_signature(prepared_state)
            acc_self_full = _eval_prepared(prepared_state, active_indices=None)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self_full)
                evaluate_seconds += time.perf_counter() - t0
                evaluate_calls += 1
            seg_len = min(int(refresh_every), int(num_steps) - step)
            idx_seg = active_indices_schedule[step : step + seg_len]
            mask_seg = active_mask_schedule[step : step + seg_len]
            if profile:
                t0 = time.perf_counter()
            state_curr, seg_hist = _run_active_segment_scan(
                state_curr,
                acc_self_full,
                idx_seg,
                mask_seg,
                dt_arr,
                add_external=add_external,
                config=config,
                params=params,
            )
            if bool(rematerialize_between_refresh):
                # Rematerialize to a standard dense array layout before the next
                # FMM prepare. This avoids a large one-time prepare penalty after
                # each scan segment.
                state_curr = jnp.asarray(state_curr, dtype=state_curr.dtype)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(state_curr)
                update_seconds += time.perf_counter() - t0
                update_calls += 1
            if return_history:
                history.append(seg_hist)
            step += int(seg_len)

        if return_history:
            return _finalize(jnp.concatenate(history, axis=0))
        return _finalize(state_curr)

    # Fast path: full-particle updates with scan+jit inside each refresh segment.
    if strict_production_lane or (
        active_indices_fn is None and not bool(refresh_after_position_update)
    ):
        step = 0
        prepared_state = None
        while step < int(num_steps):
            if profile:
                prepared_state, elapsed_prepare, stage_delta = _profiled_prepare_call(
                    state_curr,
                    prepared_state,
                )
                prepare_seconds += elapsed_prepare
                _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                prepare_calls += 1
                t0 = time.perf_counter()
            else:
                prepared_state = _prepare_or_refresh_state(state_curr, prepared_state)
            _record_shape_signature(prepared_state)
            acc_self_full = _eval_prepared(prepared_state, active_indices=None)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self_full)
                evaluate_seconds += time.perf_counter() - t0
                evaluate_calls += 1
            seg_len = min(int(refresh_every), int(num_steps) - step)
            if profile:
                t0 = time.perf_counter()
            state_curr, seg_hist = _run_full_segment_scan(
                state_curr,
                acc_self_full,
                dt_arr,
                steps=int(seg_len),
                add_external=add_external,
                config=config,
                params=params,
            )
            if bool(rematerialize_between_refresh):
                # Rematerialize to a standard dense array layout before the next
                # FMM prepare. This avoids a large one-time prepare penalty after
                # each scan segment.
                state_curr = jnp.asarray(state_curr, dtype=state_curr.dtype)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(state_curr)
                update_seconds += time.perf_counter() - t0
                update_calls += 1
            if return_history:
                history.append(seg_hist)
            step += int(seg_len)

        if return_history:
            return _finalize(jnp.concatenate(history, axis=0))
        return _finalize(state_curr)

    # General fallback path for active-index callbacks and/or post-position refresh.
    prepared_state = None
    for step in range(int(num_steps)):
        if step % int(refresh_every) == 0:
            if profile:
                prepared_state, elapsed_prepare, stage_delta = _profiled_prepare_call(
                    state_curr,
                    prepared_state,
                )
                prepare_seconds += elapsed_prepare
                _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                prepare_calls += 1
            else:
                prepared_state = _prepare_or_refresh_state(state_curr, prepared_state)
            _record_shape_signature(prepared_state)

        full_active = active_indices_fn is None
        if full_active:
            active_idx = None
            if prepared_state is None:
                if profile:
                    (
                        prepared_state,
                        elapsed_prepare,
                        stage_delta,
                    ) = _profiled_prepare_call(state_curr, prepared_state)
                    prepare_seconds += elapsed_prepare
                    _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                    prepare_calls += 1
                else:
                    prepared_state = _prepare_or_refresh_state(
                        state_curr,
                        prepared_state,
                    )
                _record_shape_signature(prepared_state)
            if profile:
                t0 = time.perf_counter()
            acc_self = _eval_prepared(prepared_state, active_indices=None)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self)
                evaluate_seconds += time.perf_counter() - t0
                evaluate_calls += 1
            if add_external:
                acc_ext = combined_external_acceleration_vmpa_switch(
                    state_curr,
                    config,
                    params,
                )
                acc_1 = acc_self + acc_ext
            else:
                acc_1 = acc_self

            pos_new = (
                state_curr[:, 0]
                + state_curr[:, 1] * dt_arr
                + 0.5 * acc_1 * (dt_arr**2)
            )
            state_pos = state_curr.at[:, 0].set(pos_new)
        else:
            active_idx = jnp.asarray(
                active_indices_fn(step, state_curr, mass_arr),
                dtype=jnp.int32,
            )
            if prepared_state is None:
                if profile:
                    (
                        prepared_state,
                        elapsed_prepare,
                        stage_delta,
                    ) = _profiled_prepare_call(state_curr, prepared_state)
                    prepare_seconds += elapsed_prepare
                    _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                    prepare_calls += 1
                else:
                    prepared_state = _prepare_or_refresh_state(
                        state_curr,
                        prepared_state,
                    )
                _record_shape_signature(prepared_state)
            if profile:
                t0 = time.perf_counter()
            acc_self = _eval_prepared(prepared_state, active_indices=active_idx)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self)
                evaluate_seconds += time.perf_counter() - t0
                evaluate_calls += 1
            if add_external:
                acc_ext = combined_external_acceleration_vmpa_switch(
                    state_curr,
                    config,
                    params,
                )[active_idx]
                acc_1 = acc_self + acc_ext
            else:
                acc_1 = acc_self

            pos_new_active = (
                state_curr[active_idx, 0]
                + state_curr[active_idx, 1] * dt_arr
                + 0.5 * acc_1 * (dt_arr**2)
            )
            state_pos = state_curr.at[active_idx, 0].set(pos_new_active)

        if bool(refresh_after_position_update):
            if profile:
                prepared_state, elapsed_prepare, stage_delta = _profiled_prepare_call(
                    state_pos,
                    prepared_state,
                )
                prepare_seconds += elapsed_prepare
                _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                prepare_calls += 1
            else:
                prepared_state = _prepare_or_refresh_state(state_pos, prepared_state)
            _record_shape_signature(prepared_state)

        if profile:
            t0 = time.perf_counter()
        if full_active:
            if prepared_state is None:
                if profile:
                    (
                        prepared_state,
                        elapsed_prepare,
                        stage_delta,
                    ) = _profiled_prepare_call(state_pos, prepared_state)
                    prepare_seconds += elapsed_prepare
                    _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                    prepare_calls += 1
                else:
                    prepared_state = _prepare_or_refresh_state(
                        state_pos,
                        prepared_state,
                    )
                _record_shape_signature(prepared_state)
            if profile:
                te = time.perf_counter()
            acc_self_2 = _eval_prepared(prepared_state, active_indices=None)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self_2)
                evaluate_seconds += time.perf_counter() - te
                evaluate_calls += 1
            if add_external:
                acc_ext_2 = combined_external_acceleration_vmpa_switch(
                    state_pos,
                    config,
                    params,
                )
                acc_2 = acc_self_2 + acc_ext_2
            else:
                acc_2 = acc_self_2
            vel_new = state_curr[:, 1] + 0.5 * (acc_1 + acc_2) * dt_arr
            state_curr = state_pos.at[:, 1].set(vel_new)
        else:
            if prepared_state is None:
                if profile:
                    (
                        prepared_state,
                        elapsed_prepare,
                        stage_delta,
                    ) = _profiled_prepare_call(state_pos, prepared_state)
                    prepare_seconds += elapsed_prepare
                    _record_profiled_prepare_elapsed(elapsed_prepare, stage_delta)
                    prepare_calls += 1
                else:
                    prepared_state = _prepare_or_refresh_state(
                        state_pos,
                        prepared_state,
                    )
                _record_shape_signature(prepared_state)
            if profile:
                te = time.perf_counter()
            acc_self_2 = _eval_prepared(prepared_state, active_indices=active_idx)
            if profile:
                if profile_sync:
                    _ = jax.block_until_ready(acc_self_2)
                evaluate_seconds += time.perf_counter() - te
                evaluate_calls += 1
            if add_external:
                acc_ext_2 = combined_external_acceleration_vmpa_switch(
                    state_pos,
                    config,
                    params,
                )[active_idx]
                acc_2 = acc_self_2 + acc_ext_2
            else:
                acc_2 = acc_self_2

            vel_new_active = state_curr[active_idx, 1] + 0.5 * (acc_1 + acc_2) * dt_arr
            state_curr = state_pos.at[active_idx, 1].set(vel_new_active)
        if profile:
            if profile_sync:
                _ = jax.block_until_ready(state_curr)
            update_seconds += time.perf_counter() - t0
            update_calls += 1

        if return_history:
            history.append(state_curr)

    out = jnp.stack(history, axis=0) if return_history else state_curr
    return _finalize(out)


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
    fmm_basis: str = "solidfmm",
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
            state_arr.dtype if fmm_working_dtype is None else jnp.dtype(fmm_working_dtype)
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
        fmm_prepare_stage_memory_split_enabled=(
            fmm_prepare_stage_memory_split_enabled
        ),
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
    fmm_basis: str = "solidfmm",
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


def build_jitted_leapfrog_jaccpot_active(
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
    fmm_basis: str = "solidfmm",
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
    outer_jit: bool = False,
):
    """Return a reusable FMM integrator callable.

    The returned function accepts `(state, mass)` arrays and executes the
    selected FMM integration configuration on jaccpot's internal compiled path.
    Set ``outer_jit=True`` to additionally wrap the full call in ``jax.jit``.
    """

    def _eager(state: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
        return integrate_leapfrog_jaccpot_active(
            state,
            mass,
            config,
            params,
            num_steps=num_steps,
            dt=dt,
            active_indices_fn=active_indices_fn,
            active_indices_schedule=active_indices_schedule,
            active_mask_schedule=active_mask_schedule,
            refresh_every=refresh_every,
            refresh_after_position_update=refresh_after_position_update,
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
            enforce_static_shape_contract=enforce_static_shape_contract,
            static_shape_warmup_prepares=static_shape_warmup_prepares,
            rematerialize_between_refresh=rematerialize_between_refresh,
            return_history=return_history,
        )

    if bool(outer_jit):
        return jax.jit(_eager)
    return _eager
