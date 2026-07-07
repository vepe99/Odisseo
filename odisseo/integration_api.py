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
):
    """Unified integration API across direct and Jaccpot-FMM backends.

    Selector
    --------
    ``config.acceleration_scheme``:
    - direct schemes (`DIRECT_ACC`, `DIRECT_ACC_LAXMAP`, `DIRECT_ACC_MATRIX`,
      `DIRECT_ACC_FOR_LOOP`, `DIRECT_ACC_SHARDING`, `NO_SELF_GRAVITY`)
      route to legacy ``time_integration``.
    - ``FMM_ACC`` routes to the Jaccpot coupler workflow.
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
