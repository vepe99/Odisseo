from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.potentials import combined_external_acceleration_vmpa_switch


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
    refresh_every: int = 1,
    refresh_after_position_update: bool = False,
    leaf_size: int = 16,
    max_order: int = 4,
    return_history: bool = False,
) -> jnp.ndarray:
    """Integrate with Jaccpot FMM using optional active-particle substeps.

    Notes
    -----
    - Source tree is refreshed every ``refresh_every`` steps.
    - Between refreshes, accelerations are evaluated for active targets only.
    - External potentials are added in ODISSEO as before.
    """
    from jaccpot import FastMultipoleMethod, OdisseoFMMCoupler

    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    if int(refresh_every) <= 0:
        raise ValueError("refresh_every must be positive")

    state_curr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)

    dt_val = float(params.t_end) / float(num_steps) if dt is None else float(dt)
    dt_arr = jnp.asarray(dt_val, dtype=state_curr.dtype)

    solver = FastMultipoleMethod(
        preset="fast",
        basis="solidfmm",
        G=float(params.G),
        softening=float(config.softening),
        working_dtype=state_curr.dtype,
    )
    coupler = OdisseoFMMCoupler(
        solver=solver,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )

    history = []
    add_external = len(config.external_accelerations) > 0

    for step in range(int(num_steps)):
        if step % int(refresh_every) == 0:
            coupler.prepare(state_curr, mass_arr)

        if active_indices_fn is None:
            active_idx = jnp.arange(state_curr.shape[0], dtype=jnp.int32)
        else:
            active_idx = jnp.asarray(
                active_indices_fn(step, state_curr, mass_arr),
                dtype=jnp.int32,
            )

        acc_self = coupler.accelerations(
            state_curr,
            active_indices=active_idx,
            rebuild_sources=False,
        )
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
            coupler.prepare(state_pos, mass_arr)

        acc_self_2 = coupler.accelerations(
            state_pos,
            active_indices=active_idx,
            rebuild_sources=False,
        )
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

        if return_history:
            history.append(state_curr)

    if return_history:
        return jnp.stack(history, axis=0)
    return state_curr
