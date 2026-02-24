from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

from odisseo.jaccpot_coupling import integrate_leapfrog_jaccpot_active
from odisseo.option_classes import (
    SimulationConfig,
    SimulationParams,
    DIRECT_ACC,
    DIRECT_ACC_LAXMAP,
    DIRECT_ACC_MATRIX,
    DIRECT_ACC_FOR_LOOP,
    DIRECT_ACC_SHARDING,
    NO_SELF_GRAVITY,
    FMM_ACC,
)
from odisseo.time_integration import SnapshotData, time_integration



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
        if not bool(config.fixed_timestep):
            raise NotImplementedError(
                "jaccpot_fmm backend currently requires fixed_timestep=True"
            )

        states_or_final = integrate_leapfrog_jaccpot_active(
            primitive_state,
            mass,
            config,
            params,
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
            fmm_preset=str(config.fmm_preset),
            fmm_basis=str(config.fmm_basis),
            fmm_theta=float(config.fmm_theta),
            fmm_mac_type=str(config.fmm_mac_type),
            fmm_farfield_mode=str(config.fmm_farfield_mode),
            fmm_nearfield_mode=str(config.fmm_nearfield_mode),
            fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
            fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
            return_history=bool(config.return_snapshots),
        )

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
