from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from odisseo.option_classes import SimulationConfig, SimulationParams
from odisseo.potentials import combined_external_acceleration_vmpa_switch


def _build_fmm_solver(
    *,
    state_dtype,
    config: SimulationConfig,
    params: SimulationParams,
    fmm_preset: str,
    fmm_basis: str,
    fmm_theta: float,
    fmm_mac_type: str,
    fmm_farfield_mode: str,
    fmm_nearfield_mode: str,
    fmm_nearfield_edge_chunk_size: int,
    fmm_tree_leaf_target: int,
    fmm_fixed_order: Optional[int],
    leaf_size: int,
    fmm_jit_tree: Optional[bool],
    fmm_jit_traversal: Optional[bool],
):
    from jaccpot import (
        FMMAdvancedConfig,
        FarFieldConfig,
        FastMultipoleMethod,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset=str(fmm_preset),
        basis=str(fmm_basis),
        theta=float(fmm_theta),
        G=float(params.G),
        softening=float(config.softening),
        working_dtype=state_dtype,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(leaf_target=int(fmm_tree_leaf_target)),
            farfield=FarFieldConfig(mode=str(fmm_farfield_mode)),
            nearfield=NearFieldConfig(
                mode=str(fmm_nearfield_mode),
                edge_chunk_size=int(fmm_nearfield_edge_chunk_size),
            ),
            runtime=RuntimePolicyConfig(
                jit_tree=None if fmm_jit_tree is None else bool(fmm_jit_tree),
                jit_traversal=(
                    None if fmm_jit_traversal is None else bool(fmm_jit_traversal)
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
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
    return_history: bool = False,
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

    dt_val = float(params.t_end) / float(num_steps) if dt is None else float(dt)
    dt_arr = jnp.asarray(dt_val, dtype=state_curr.dtype)

    solver = _build_fmm_solver(
        state_dtype=state_curr.dtype,
        config=config,
        params=params,
        fmm_preset=fmm_preset,
        fmm_basis=fmm_basis,
        fmm_theta=fmm_theta,
        fmm_mac_type=fmm_mac_type,
        fmm_farfield_mode=fmm_farfield_mode,
        fmm_nearfield_mode=fmm_nearfield_mode,
        fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
        fmm_tree_leaf_target=fmm_tree_leaf_target,
        fmm_fixed_order=fmm_fixed_order,
        leaf_size=leaf_size,
        fmm_jit_tree=fmm_jit_tree,
        fmm_jit_traversal=fmm_jit_traversal,
    )
    def _prepare_state(state_in: jnp.ndarray):
        return solver.prepare_state(
            state_in[:, 0, :],
            mass_arr,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )

    def _eval_prepared(prepared_state, active_indices=None):
        return solver.evaluate_prepared_state(
            prepared_state,
            target_indices=active_indices,
            return_potential=False,
        )

    history = []
    add_external = len(config.external_accelerations) > 0

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
        while step < int(num_steps):
            prepared_state = _prepare_state(state_curr)
            acc_self_full = _eval_prepared(prepared_state, active_indices=None)
            seg_len = min(int(refresh_every), int(num_steps) - step)
            idx_seg = active_indices_schedule[step : step + seg_len]
            mask_seg = active_mask_schedule[step : step + seg_len]
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
            if return_history:
                history.append(seg_hist)
            step += int(seg_len)

        if return_history:
            return jnp.concatenate(history, axis=0)
        return state_curr

    # Fast path: full-particle updates with scan+jit inside each refresh segment.
    if active_indices_fn is None and not bool(refresh_after_position_update):
        step = 0
        while step < int(num_steps):
            prepared_state = _prepare_state(state_curr)
            acc_self_full = _eval_prepared(prepared_state, active_indices=None)
            seg_len = min(int(refresh_every), int(num_steps) - step)
            state_curr, seg_hist = _run_full_segment_scan(
                state_curr,
                acc_self_full,
                dt_arr,
                steps=int(seg_len),
                add_external=add_external,
                config=config,
                params=params,
            )
            if return_history:
                history.append(seg_hist)
            step += int(seg_len)

        if return_history:
            return jnp.concatenate(history, axis=0)
        return state_curr

    # General fallback path for active-index callbacks and/or post-position refresh.
    prepared_state = None
    for step in range(int(num_steps)):
        if step % int(refresh_every) == 0:
            prepared_state = _prepare_state(state_curr)

        full_active = active_indices_fn is None
        if full_active:
            active_idx = None
            if prepared_state is None:
                prepared_state = _prepare_state(state_curr)
            acc_self = _eval_prepared(prepared_state, active_indices=None)
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
                prepared_state = _prepare_state(state_curr)
            acc_self = _eval_prepared(prepared_state, active_indices=active_idx)
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
            prepared_state = _prepare_state(state_pos)

        if full_active:
            if prepared_state is None:
                prepared_state = _prepare_state(state_pos)
            acc_self_2 = _eval_prepared(prepared_state, active_indices=None)
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
                prepared_state = _prepare_state(state_pos)
            acc_self_2 = _eval_prepared(prepared_state, active_indices=active_idx)
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
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
) -> jnp.ndarray:
    """Evaluate one FMM acceleration call for an ODISSEO primitive state."""
    state_arr = jnp.asarray(state)
    mass_arr = jnp.asarray(mass)
    solver = _build_fmm_solver(
        state_dtype=state_arr.dtype,
        config=config,
        params=params,
        fmm_preset=fmm_preset,
        fmm_basis=fmm_basis,
        fmm_theta=fmm_theta,
        fmm_mac_type=fmm_mac_type,
        fmm_farfield_mode=fmm_farfield_mode,
        fmm_nearfield_mode=fmm_nearfield_mode,
        fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
        fmm_tree_leaf_target=fmm_tree_leaf_target,
        fmm_fixed_order=fmm_fixed_order,
        leaf_size=leaf_size,
        fmm_jit_tree=fmm_jit_tree,
        fmm_jit_traversal=fmm_jit_traversal,
    )
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
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
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
            fmm_mac_type=fmm_mac_type,
            fmm_farfield_mode=fmm_farfield_mode,
            fmm_nearfield_mode=fmm_nearfield_mode,
            fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
            fmm_tree_leaf_target=fmm_tree_leaf_target,
            fmm_fixed_order=fmm_fixed_order,
            fmm_jit_tree=fmm_jit_tree,
            fmm_jit_traversal=fmm_jit_traversal,
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
    fmm_mac_type: str = "dehnen",
    fmm_farfield_mode: str = "auto",
    fmm_nearfield_mode: str = "auto",
    fmm_nearfield_edge_chunk_size: int = 256,
    fmm_tree_leaf_target: int = 32,
    fmm_fixed_order: Optional[int] = None,
    fmm_jit_tree: Optional[bool] = None,
    fmm_jit_traversal: Optional[bool] = True,
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
            fmm_mac_type=fmm_mac_type,
            fmm_farfield_mode=fmm_farfield_mode,
            fmm_nearfield_mode=fmm_nearfield_mode,
            fmm_nearfield_edge_chunk_size=fmm_nearfield_edge_chunk_size,
            fmm_tree_leaf_target=fmm_tree_leaf_target,
            fmm_fixed_order=fmm_fixed_order,
            fmm_jit_tree=fmm_jit_tree,
            fmm_jit_traversal=fmm_jit_traversal,
            return_history=return_history,
        )

    if bool(outer_jit):
        return jax.jit(_eager)
    return _eager
