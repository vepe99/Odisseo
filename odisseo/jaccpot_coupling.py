from __future__ import annotations

import hashlib
import inspect
import os
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
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


def _contains_jax_tracer(value: Any) -> bool:
    """Return True when any pytree leaf is a JAX tracer."""
    try:
        leaves = jax.tree_util.tree_leaves(value)
    except Exception:
        return False
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


@dataclass(frozen=True)
class JaccpotCoreKernelConfig:
    """Static configuration payload for the shared core-kernel scaffold."""

    mode: str
    leaf_size: int
    max_order: int
    preset: str
    runtime_path: str
    tree_build_mode: str


@dataclass(frozen=True)
class JaccpotCoreKernelOutput:
    """Return payload for the shared core-kernel scaffold."""

    next_state: jnp.ndarray
    acceleration: jnp.ndarray
    prepared_state: Any
    execute_count: int
    prepare_count: int
    refresh_count: int


@dataclass
class AdaptiveCoreRuntimeState:
    """Mutable runtime state for adaptive core-kernel refresh cadence."""

    rhs_calls: int = 0
    prepared_state: Any = None
    last_refresh_rhs_call: int = 0
    last_refresh_positions: Any = None
    refresh_cadence_skips_rhs_calls: int = 0
    refresh_cadence_skips_displacement: int = 0
    refresh_cadence_last_displacement: float = 0.0
    prepared_drop_tracer: int = 0
    prepared_non_large_n_seen: int = 0

    def prepared_input(self, *, enabled: bool) -> Any:
        return self.prepared_state if bool(enabled) else None

    def should_refresh(
        self,
        *,
        enabled: bool,
        prepared_in: Any,
        y_state: jnp.ndarray,
        refresh_rhs_calls: int,
        displacement_threshold: Optional[float],
    ) -> bool:
        if not bool(enabled):
            return True
        if prepared_in is None:
            return True
        if type(prepared_in).__name__ != "LargeNPreparedState":
            self.prepared_non_large_n_seen += 1
            return True

        force_refresh = True
        if int(refresh_rhs_calls) > 1:
            since_refresh = int(self.rhs_calls) - int(self.last_refresh_rhs_call)
            if since_refresh < int(refresh_rhs_calls):
                force_refresh = False
                self.refresh_cadence_skips_rhs_calls += 1
        if displacement_threshold is not None and force_refresh:
            if self.last_refresh_positions is not None:
                disp = jnp.linalg.norm(
                    y_state[:, 0, :] - self.last_refresh_positions, axis=1
                )
                max_disp = float(jnp.max(disp))
                self.refresh_cadence_last_displacement = max_disp
                if max_disp < float(displacement_threshold):
                    force_refresh = False
                    self.refresh_cadence_skips_displacement += 1
        return bool(force_refresh)

    def update_prepared_state(
        self,
        *,
        enabled: bool,
        prepared_out: Any,
        allow_tracer_prepared_cache: bool,
    ) -> None:
        if not bool(enabled):
            self.prepared_state = None
            return
        if prepared_out is None:
            self.prepared_state = None
            return
        if _contains_jax_tracer(prepared_out):
            if bool(allow_tracer_prepared_cache):
                self.prepared_state = prepared_out
            else:
                self.prepared_state = None
                self.prepared_drop_tracer += 1
            return
        self.prepared_state = prepared_out

    def mark_refreshed(self, *, y_state: jnp.ndarray) -> None:
        self.last_refresh_rhs_call = int(self.rhs_calls)
        self.last_refresh_positions = jnp.asarray(y_state[:, 0, :])


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
        # Data-driven cap: jaccpot auto-sizes the static target-block payload to
        # the densest leaf at prepare time. A fixed small cap (previously 32)
        # fails for centrally-concentrated ICs whose inner leaves have very high
        # near-neighbour counts.
        static_target_blocks_cap = "auto"
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
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )
    from yggdrax.interactions import DualTreeTraversalConfig

    # Opt-in fused Pallas near-field/M2L kernels (Ampere+/sm_80+ GPUs). Default
    # off keeps the pure-JAX paths; the solver still falls back automatically on
    # unsupported hardware via jaccpot's `pallas_*_supported()` guards.
    use_pallas = os.environ.get("ODISSEO_FMM_USE_PALLAS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

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
        use_pallas=bool(use_pallas),
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
        fixed_order=(None if fmm_fixed_order is None else int(fmm_fixed_order)),
        # Keep one global leaf-size contract per simulation: tree target and
        # runtime leaf cap are tied to the same value.
        fixed_max_leaf_size=int(leaf_size),
    )


def build_compiled_jaccpot_core_kernel(
    config: SimulationConfig,
    params: SimulationParams,
    *,
    mode: str,
    leaf_size: int = 16,
    max_order: int = 4,
    dt: Optional[float] = None,
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
    fmm_tree_build_mode: str = "static_radix",
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
    """Build a shared jaccpot core-kernel scaffold for fixed/adaptive migration.

    Modes:
    - ``rhs_only``: prepare/evaluate and return acceleration (for diffrax RHS).
    - ``fixed_step_update``: same prepare/evaluate plus one explicit Euler update.

    This is intentionally additive scaffold code and does not replace existing
    production integration paths yet.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"rhs_only", "fixed_step_update"}:
        raise ValueError("mode must be one of {'rhs_only', 'fixed_step_update'}")
    if mode_norm == "fixed_step_update" and dt is None:
        raise ValueError("dt is required for mode='fixed_step_update'")

    core_cfg = JaccpotCoreKernelConfig(
        mode=mode_norm,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        preset=str(fmm_preset),
        runtime_path=str(fmm_runtime_path),
        tree_build_mode=str(fmm_tree_build_mode),
    )

    solver = _build_fmm_solver(
        working_dtype=(
            jnp.dtype(fmm_working_dtype) if fmm_working_dtype is not None else None
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
        fmm_upward_leaf_batch_size=fmm_upward_leaf_batch_size,
    )

    def _eager(
        state: jnp.ndarray,
        mass: jnp.ndarray,
        prepared_state: Optional[Any] = None,
        *,
        refresh_prepared: bool = True,
    ) -> JaccpotCoreKernelOutput:
        state_arr = jnp.asarray(state)
        mass_arr = jnp.asarray(mass)
        if fmm_working_dtype is None:
            # Keep dtype alignment with caller state when dtype was not pinned.
            solver.working_dtype = state_arr.dtype
        refresh_count = 0
        prepare_count = 0
        prepared = None
        refresh_fn = getattr(solver, "refresh_prepared_state", None)
        prepared_type_name = (
            type(prepared_state).__name__ if prepared_state is not None else ""
        )
        if prepared_state is not None and not bool(refresh_prepared):
            prepared = prepared_state
        can_try_refresh = (
            prepared is None
            and prepared_state is not None
            and bool(refresh_prepared)
            and callable(refresh_fn)
            and prepared_type_name == "LargeNPreparedState"
        )
        if can_try_refresh:
            with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
                try:
                    prepared = refresh_fn(
                        prepared_state,
                        state_arr[:, 0, :],
                        mass_arr,
                        leaf_size=int(core_cfg.leaf_size),
                        max_order=int(core_cfg.max_order),
                    )
                    refresh_count = 1
                except (TypeError, NotImplementedError):
                    try:
                        prepared = refresh_fn(
                            state_arr[:, 0, :],
                            mass_arr,
                            prepared_state,
                            leaf_size=int(core_cfg.leaf_size),
                            max_order=int(core_cfg.max_order),
                        )
                        refresh_count = 1
                    except (TypeError, NotImplementedError):
                        prepared = None
        if prepared is None:
            with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
                prepared = solver.prepare_state(
                    state_arr[:, 0, :],
                    mass_arr,
                    leaf_size=int(core_cfg.leaf_size),
                    max_order=int(core_cfg.max_order),
                )
            prepare_count = 1
        acc = solver.evaluate_prepared_state(
            prepared,
            target_indices=None,
            return_potential=False,
        )
        if core_cfg.mode == "fixed_step_update":
            dt_arr = jnp.asarray(float(dt), dtype=state_arr.dtype)
            pos_next = state_arr[:, 0, :] + state_arr[:, 1, :] * dt_arr
            vel_next = state_arr[:, 1, :] + acc * dt_arr
            next_state = jnp.stack((pos_next, vel_next), axis=1)
        else:
            next_state = state_arr
        return JaccpotCoreKernelOutput(
            next_state=next_state,
            acceleration=acc,
            prepared_state=prepared,
            execute_count=1,
            prepare_count=int(prepare_count),
            refresh_count=int(refresh_count),
        )

    return (jax.jit(_eager) if bool(outer_jit) else _eager), core_cfg


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
        ext_full = combined_external_acceleration_vmpa_switch(
            state_curr, config, params
        )
        acc_1 = acc_self_active + ext_full[safe_idx]
    else:
        acc_1 = acc_self_active

    pos_active_new = pos[safe_idx] + vel[safe_idx] * dt_arr + 0.5 * acc_1 * (dt_arr**2)
    pos_new = _scatter_masked_vectors(pos, safe_idx, pos_active_new, active_mask)
    state_pos = state_curr.at[:, 0].set(pos_new)

    if add_external:
        ext_full_2 = combined_external_acceleration_vmpa_switch(
            state_pos, config, params
        )
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


def _default_fused_neighbor_edge_cap(n_particles: int) -> int:
    """Generous up-front neighbor-edge fixed cap for the fused static-radix lane.

    The fused lane sizes the near-field neighbor-edge list to a fixed cap. Its
    N-based bootstrap (~1 edge/particle) underestimates centrally-concentrated
    ICs — a 200k Agama disk has ~4 edges/particle (dense inner leaves are "near"
    almost every other leaf). The cap cannot grow inside the device-resident
    scan (it would break the fixed-shape ``lax.scan`` carry), so it must be set
    before the initial state is built. The neighbor-edge list is just int edge
    ids (~4-8 bytes each), so a generous cap is cheap: the default 16 edges/
    particle (=3.2M / ~26 MB at 200k) covers realistic concentrated disks with
    margin. Tunable via ``ODISSEO_FMM_NEIGHBOR_EDGE_PER_PARTICLE_CAP``; extreme
    ICs can also set ``JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP`` directly.
    """
    try:
        per_particle = int(
            os.environ.get("ODISSEO_FMM_NEIGHBOR_EDGE_PER_PARTICLE_CAP", "16")
        )
    except Exception:
        per_particle = 16
    per_particle = max(1, per_particle)
    return int(per_particle) * int(max(1, n_particles)) + 1


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
    fmm_tree_build_mode: str = "static_radix",
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
    perf_warmup_runs: int = 0,
    perf_measure_runs: int = 1,
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
    profile_sync = str(os.environ.get("ODISSEO_PROFILE_SYNC", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    strict_perf_progress = str(
        os.environ.get("ODISSEO_STRICT_PERF_PROGRESS", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    strict_timing_disable_external = str(
        os.environ.get(
            "ODISSEO_STRICT_DISABLE_EXTERNAL_FOR_TIMING",
            "0",
        )
    ).strip().lower() in {"1", "true", "yes", "on"}
    strict_timing_external_only = str(
        os.environ.get(
            "ODISSEO_STRICT_EXTERNAL_ONLY_FOR_TIMING",
            "0",
        )
    ).strip().lower() in {"1", "true", "yes", "on"}
    if bool(strict_timing_disable_external) and bool(strict_timing_external_only):
        raise ValueError(
            "ODISSEO_STRICT_DISABLE_EXTERNAL_FOR_TIMING and "
            "ODISSEO_STRICT_EXTERNAL_ONLY_FOR_TIMING are mutually exclusive."
        )
    strict_mode_env = (
        str(os.environ.get("JACCPOT_STATIC_STRICT_GPU_MODE", "auto")).strip().lower()
    )
    strict_mode_requested = strict_mode_env in {"on", "auto"}
    strict_production_lane = bool(
        strict_mode_requested
        and str(fmm_preset).strip().lower() == "large_n_gpu"
        and str(fmm_runtime_path).strip().lower() in {"large_n", "auto"}
        and str(fmm_tree_build_mode).strip().lower() == "static_radix"
    )
    if strict_production_lane and int(refresh_every) != 1:
        raise ValueError(
            "strict static-radix production requires refresh_every=1 for "
            "endpoint-correct velocity-Verlet self gravity"
        )
    # Size the fused neighbor-edge fixed cap generously up front (before any
    # prepare caches the env-config), so concentrated ICs fit. It cannot grow
    # inside the device-resident scan, and the edge list is cheap (int ids), so
    # over-provisioning is fine. See _default_fused_neighbor_edge_cap.
    if (
        strict_production_lane
        and os.environ.get("ODISSEO_FMM_NEIGHBOR_EDGE_AUTOSIZE", "1").strip().lower()
        in {"1", "true", "yes", "on"}
        and "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP" not in os.environ
    ):
        os.environ["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = str(
            _default_fused_neighbor_edge_cap(int(state.shape[0]))
        )
    collect_shape_signatures = bool(profile or enforce_static_shape_contract)
    t_total_start = time.perf_counter() if profile else 0.0
    prepare_seconds = 0.0
    evaluate_seconds = 0.0
    update_seconds = 0.0
    strict_runner_wall_seconds = 0.0
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
    refresh_fn = None
    refresh_fn_callable = False
    refresh_fn_signature = None
    refresh_fn_params = {}
    refresh_fn_accepts_var_kw = False
    refresh_call_mode_index: Optional[int] = None
    refresh_call_runner: Optional[Callable[[Any, jnp.ndarray], Any]] = None
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
    shape_signature_unique_post_warmup: set[tuple[tuple[str, tuple[int, ...]], ...]] = (
        set()
    )
    shape_drift_events_post_warmup = 0
    shape_checks_post_warmup = 0
    shape_signature_hashes_post_warmup: list[str] = []
    shape_signature_diff_post_warmup: list[dict[str, Any]] = []
    perf_warmup_runs_i = 0
    perf_measure_runs_i = 1
    perf_warmup_run_seconds: list[float] = []
    perf_measured_run_seconds: list[float] = []
    strict_timing_mode = "full"
    strict_effective_add_external = False
    use_core_scaffold = (
        os.environ.get("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "0").strip() == "1"
    )
    core_scaffold_exec_calls = 0
    core_scaffold_prepare_calls = 0
    core_scaffold_refresh_calls = 0

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
                added.append(
                    {"dtype": key[0], "shape": list(key[1]), "count": int(count)}
                )
            for key, count in (ref_counter - cur_counter).items():
                removed.append(
                    {"dtype": key[0], "shape": list(key[1]), "count": int(count)}
                )
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
        strict_refresh_total_seconds = float(
            runtime_diag.get("refresh_total_seconds", 0.0)
        )
        strict_refresh_component_sum = float(
            runtime_diag.get("refresh_tree_upward_seconds", 0.0)
            + runtime_diag.get("refresh_dual_downward_compute_seconds", 0.0)
            + runtime_diag.get("refresh_nearfield_seconds", 0.0)
            + runtime_diag.get("refresh_dual_artifact_build_seconds", 0.0)
            + runtime_diag.get("refresh_dual_split_shared_far_near_seconds", 0.0)
        )
        strict_refresh_effective_seconds = float(
            strict_refresh_total_seconds
            if strict_refresh_total_seconds > 0.0
            else strict_refresh_component_sum
        )
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
                "strict_production_lane_active": bool(strict_production_lane),
                "strict_runner_wall_seconds": float(strict_runner_wall_seconds),
                "perf_warmup_runs": int(perf_warmup_runs_i),
                "perf_measure_runs": int(perf_measure_runs_i),
                "perf_warmup_run_seconds": list(perf_warmup_run_seconds),
                "perf_measured_run_seconds": list(perf_measured_run_seconds),
                "perf_measured_median_seconds": float(
                    np.median(
                        np.asarray(
                            perf_measured_run_seconds or [strict_runner_wall_seconds],
                            dtype=np.float64,
                        )
                    )
                ),
                "perf_measured_median_step_seconds": float(
                    np.median(
                        np.asarray(
                            perf_measured_run_seconds or [strict_runner_wall_seconds],
                            dtype=np.float64,
                        )
                    )
                    / max(1, int(num_steps))
                ),
                "num_steps": int(num_steps),
                "refresh_every": int(refresh_every),
                "used_external_potential": bool(add_external),
                "strict_effective_external_potential": bool(
                    strict_effective_add_external
                ),
                "strict_timing_mode": str(strict_timing_mode),
                "strict_timing_disable_external": bool(strict_timing_disable_external),
                "strict_timing_external_only": bool(strict_timing_external_only),
                "large_n_eval_diag_mode": str(
                    runtime_diag.get("large_n_eval_diag_mode", "full")
                ),
                "large_n_nearfield_diag_mode": str(
                    runtime_diag.get("large_n_nearfield_diag_mode", "full")
                ),
                "large_n_eval_leaf_nodes_shape": tuple(
                    runtime_diag.get("large_n_eval_leaf_nodes_shape", ())
                ),
                "large_n_eval_local_coefficients_shape": tuple(
                    runtime_diag.get("large_n_eval_local_coefficients_shape", ())
                ),
                "large_n_eval_local_centers_shape": tuple(
                    runtime_diag.get("large_n_eval_local_centers_shape", ())
                ),
                "large_n_eval_active_leaf_count": int(
                    runtime_diag.get("large_n_eval_active_leaf_count", 0)
                ),
                "large_n_eval_max_leaf_size": int(
                    runtime_diag.get("large_n_eval_max_leaf_size", 0)
                ),
                "large_n_eval_leaf_particle_slots": int(
                    runtime_diag.get("large_n_eval_leaf_particle_slots", 0)
                ),
                "large_n_radix_payload_present": bool(
                    runtime_diag.get("large_n_radix_payload_present", False)
                ),
                "large_n_radix_payload_source_particle_shape": tuple(
                    runtime_diag.get("large_n_radix_payload_source_particle_shape", ())
                ),
                "large_n_radix_payload_source_particle_slots": int(
                    runtime_diag.get("large_n_radix_payload_source_particle_slots", 0)
                ),
                "large_n_radix_payload_source_leaf_shape": tuple(
                    runtime_diag.get("large_n_radix_payload_source_leaf_shape", ())
                ),
                "large_n_radix_payload_source_leaf_slots": int(
                    runtime_diag.get("large_n_radix_payload_source_leaf_slots", 0)
                ),
                "large_n_target_block_source_leaf_padded_shape": tuple(
                    runtime_diag.get(
                        "large_n_target_block_source_leaf_padded_shape", ()
                    )
                ),
                "strict_refresh_diag_mode": str(
                    runtime_diag.get("strict_refresh_diag_mode", "full")
                ),
                "strict_refresh_diag_tree_active": bool(
                    runtime_diag.get("strict_refresh_diag_tree_active", True)
                ),
                "strict_refresh_diag_upward_active": bool(
                    runtime_diag.get("strict_refresh_diag_upward_active", True)
                ),
                "strict_refresh_diag_downward_active": bool(
                    runtime_diag.get("strict_refresh_diag_downward_active", True)
                ),
                "strict_refresh_diag_eval_active": bool(
                    runtime_diag.get("strict_refresh_diag_eval_active", True)
                ),
                "strict_refresh_detail_diag_mode": str(
                    runtime_diag.get("strict_refresh_detail_diag_mode", "full")
                ),
                "static_radix_reuse_structures": bool(
                    runtime_diag.get("static_radix_reuse_structures", False)
                ),
                "static_radix_upward_batched": bool(
                    runtime_diag.get("static_radix_upward_batched", False)
                ),
                "static_radix_downward_batched": bool(
                    runtime_diag.get("static_radix_downward_batched", False)
                ),
                "static_radix_compact_pair_reuse_hits": int(
                    runtime_diag.get("static_radix_compact_pair_reuse_hits", 0)
                ),
                "static_radix_compact_pair_reuse_misses": int(
                    runtime_diag.get("static_radix_compact_pair_reuse_misses", 0)
                ),
                "static_radix_tree_leaf_count": int(
                    runtime_diag.get("static_radix_tree_leaf_count", 0)
                ),
                "static_radix_tree_node_count": int(
                    runtime_diag.get("static_radix_tree_node_count", 0)
                ),
                "static_radix_far_pair_count": int(
                    runtime_diag.get("static_radix_far_pair_count", 0)
                ),
                "static_radix_m2l_chunk_count": int(
                    runtime_diag.get("static_radix_m2l_chunk_count", 0)
                ),
                "static_radix_l2l_edge_count": int(
                    runtime_diag.get("static_radix_l2l_edge_count", 0)
                ),
                "used_schedule_scan_mode": bool(active_indices_schedule is not None),
                "used_fast_full_scan_mode": bool(
                    active_indices_fn is None
                    and not bool(refresh_after_position_update)
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
                "runtime_strict_runner_compile_count": int(
                    runtime_diag.get("strict_runner_compile_count", 0)
                ),
                "runtime_strict_runner_execute_count": int(
                    runtime_diag.get("strict_runner_execute_count", 0)
                ),
                "runtime_strict_runner_profile_key_hits": int(
                    runtime_diag.get("strict_runner_profile_key_hits", 0)
                ),
                "runtime_strict_runner_profile_key_misses": int(
                    runtime_diag.get("strict_runner_profile_key_misses", 0)
                ),
                "runtime_strict_runner_fail_fast_reject_count": int(
                    runtime_diag.get("strict_runner_fail_fast_reject_count", 0)
                ),
                "runtime_strict_v2_compile_count": int(
                    runtime_diag.get("strict_v2_compile_count", 0)
                ),
                "runtime_strict_v2_execute_count": int(
                    runtime_diag.get("strict_v2_execute_count", 0)
                ),
                "runtime_strict_v2_profile_key_hits": int(
                    runtime_diag.get("strict_v2_profile_key_hits", 0)
                ),
                "runtime_strict_v2_profile_key_misses": int(
                    runtime_diag.get("strict_v2_profile_key_misses", 0)
                ),
                "runtime_strict_v2_fail_fast_reject_count": int(
                    runtime_diag.get("strict_v2_fail_fast_reject_count", 0)
                ),
                "runtime_strict_fused_mode_active": bool(
                    runtime_diag.get("strict_fused_mode_active", False)
                ),
                "runtime_strict_fused_compile_count": int(
                    runtime_diag.get("strict_fused_compile_count", 0)
                ),
                "runtime_strict_fused_execute_count": int(
                    runtime_diag.get("strict_fused_execute_count", 0)
                ),
                "runtime_strict_fused_profile_key_hits": int(
                    runtime_diag.get("strict_fused_profile_key_hits", 0)
                ),
                "runtime_strict_fused_profile_key_misses": int(
                    runtime_diag.get("strict_fused_profile_key_misses", 0)
                ),
                "runtime_strict_fused_fallback_count": int(
                    runtime_diag.get("strict_fused_fallback_count", 0)
                ),
                "runtime_strict_fused_last_fallback_reason": str(
                    runtime_diag.get("strict_fused_last_fallback_reason", "")
                ),
                "runtime_strict_fused_device_refresh_route_count": int(
                    runtime_diag.get("strict_fused_device_refresh_route_count", 0)
                ),
                "runtime_strict_fused_planner_bypassed_count": int(
                    runtime_diag.get("strict_fused_planner_bypassed_count", 0)
                ),
                "runtime_strict_velocity_verlet_acceleration_carry_active": bool(
                    runtime_diag.get(
                        "strict_velocity_verlet_acceleration_carry_active", False
                    )
                ),
                "runtime_strict_self_force_bootstrap_evaluations": int(
                    runtime_diag.get("strict_self_force_bootstrap_evaluations", 0)
                ),
                "runtime_strict_self_force_initial_full_fmm_evaluations": int(
                    runtime_diag.get(
                        "strict_self_force_initial_full_fmm_evaluations",
                        runtime_diag.get("strict_self_force_bootstrap_evaluations", 0),
                    )
                ),
                "runtime_strict_self_force_endpoint_evaluations": int(
                    runtime_diag.get("strict_self_force_endpoint_evaluations", 0)
                ),
                "runtime_strict_external_bootstrap_evaluations": int(
                    runtime_diag.get("strict_external_bootstrap_evaluations", 0)
                ),
                "runtime_strict_external_endpoint_evaluations": int(
                    runtime_diag.get("strict_external_endpoint_evaluations", 0)
                ),
                "runtime_strict_static_target_block_capacity_ok": bool(
                    runtime_diag.get("strict_static_target_block_capacity_ok", True)
                ),
                "runtime_large_n_radix_fast_occupancy_sort": bool(
                    runtime_diag.get("large_n_radix_fast_occupancy_sort", True)
                ),
                "runtime_large_n_radix_fast_skip_empty_tiles": bool(
                    runtime_diag.get("large_n_radix_fast_skip_empty_tiles", True)
                ),
                "runtime_large_n_eval_diag_mode": str(
                    runtime_diag.get("large_n_eval_diag_mode", "full")
                ),
                "runtime_large_n_nearfield_diag_mode": str(
                    runtime_diag.get("large_n_nearfield_diag_mode", "full")
                ),
                "runtime_large_n_eval_leaf_nodes_shape": tuple(
                    runtime_diag.get("large_n_eval_leaf_nodes_shape", ())
                ),
                "runtime_large_n_eval_local_coefficients_shape": tuple(
                    runtime_diag.get("large_n_eval_local_coefficients_shape", ())
                ),
                "runtime_large_n_eval_local_centers_shape": tuple(
                    runtime_diag.get("large_n_eval_local_centers_shape", ())
                ),
                "runtime_large_n_eval_active_leaf_count": int(
                    runtime_diag.get("large_n_eval_active_leaf_count", 0)
                ),
                "runtime_large_n_eval_max_leaf_size": int(
                    runtime_diag.get("large_n_eval_max_leaf_size", 0)
                ),
                "runtime_large_n_eval_leaf_particle_slots": int(
                    runtime_diag.get("large_n_eval_leaf_particle_slots", 0)
                ),
                "runtime_large_n_radix_payload_present": bool(
                    runtime_diag.get("large_n_radix_payload_present", False)
                ),
                "runtime_large_n_radix_payload_source_particle_shape": tuple(
                    runtime_diag.get("large_n_radix_payload_source_particle_shape", ())
                ),
                "runtime_large_n_radix_payload_source_particle_slots": int(
                    runtime_diag.get("large_n_radix_payload_source_particle_slots", 0)
                ),
                "runtime_large_n_radix_payload_source_leaf_shape": tuple(
                    runtime_diag.get("large_n_radix_payload_source_leaf_shape", ())
                ),
                "runtime_large_n_radix_payload_source_leaf_slots": int(
                    runtime_diag.get("large_n_radix_payload_source_leaf_slots", 0)
                ),
                "runtime_large_n_target_block_source_leaf_padded_shape": tuple(
                    runtime_diag.get(
                        "large_n_target_block_source_leaf_padded_shape", ()
                    )
                ),
                "runtime_strict_refresh_diag_mode": str(
                    runtime_diag.get("strict_refresh_diag_mode", "full")
                ),
                "runtime_strict_refresh_diag_tree_active": bool(
                    runtime_diag.get("strict_refresh_diag_tree_active", True)
                ),
                "runtime_strict_refresh_diag_upward_active": bool(
                    runtime_diag.get("strict_refresh_diag_upward_active", True)
                ),
                "runtime_strict_refresh_diag_downward_active": bool(
                    runtime_diag.get("strict_refresh_diag_downward_active", True)
                ),
                "runtime_strict_refresh_diag_eval_active": bool(
                    runtime_diag.get("strict_refresh_diag_eval_active", True)
                ),
                "runtime_strict_refresh_detail_diag_mode": str(
                    runtime_diag.get("strict_refresh_detail_diag_mode", "full")
                ),
                "runtime_static_radix_reuse_structures": bool(
                    runtime_diag.get("static_radix_reuse_structures", False)
                ),
                "runtime_static_radix_upward_batched": bool(
                    runtime_diag.get("static_radix_upward_batched", False)
                ),
                "runtime_static_radix_downward_batched": bool(
                    runtime_diag.get("static_radix_downward_batched", False)
                ),
                "runtime_static_radix_compact_pair_reuse_hits": int(
                    runtime_diag.get("static_radix_compact_pair_reuse_hits", 0)
                ),
                "runtime_static_radix_compact_pair_reuse_misses": int(
                    runtime_diag.get("static_radix_compact_pair_reuse_misses", 0)
                ),
                "runtime_static_radix_tree_leaf_count": int(
                    runtime_diag.get("static_radix_tree_leaf_count", 0)
                ),
                "runtime_static_radix_tree_node_count": int(
                    runtime_diag.get("static_radix_tree_node_count", 0)
                ),
                "runtime_static_radix_far_pair_count": int(
                    runtime_diag.get("static_radix_far_pair_count", 0)
                ),
                "runtime_static_radix_m2l_chunk_count": int(
                    runtime_diag.get("static_radix_m2l_chunk_count", 0)
                ),
                "runtime_static_radix_l2l_edge_count": int(
                    runtime_diag.get("static_radix_l2l_edge_count", 0)
                ),
                "runtime_strict_fused_fastlane_diag_enabled": bool(
                    runtime_diag.get("strict_fused_fastlane_diag_enabled", False)
                ),
                "runtime_strict_fused_fastlane_attempts": int(
                    runtime_diag.get("strict_fused_fastlane_attempts", 0)
                ),
                "runtime_strict_fused_fastlane_hits": int(
                    runtime_diag.get("strict_fused_fastlane_hits", 0)
                ),
                "runtime_strict_fused_fastlane_misses": int(
                    runtime_diag.get("strict_fused_fastlane_misses", 0)
                ),
                "runtime_strict_fused_fastlane_last_blockers": tuple(
                    str(v)
                    for v in runtime_diag.get(
                        "strict_fused_fastlane_last_blockers", tuple()
                    )
                ),
                "runtime_strict_fused_fastlane_block_counts": {
                    str(k): int(v)
                    for k, v in dict(
                        runtime_diag.get("strict_fused_fastlane_block_counts", {})
                    ).items()
                },
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
                "runtime_strict_unaccounted_seconds": float(
                    max(
                        0.0,
                        float(strict_runner_wall_seconds)
                        - strict_refresh_effective_seconds,
                    )
                ),
                "runtime_strict_refresh_share_of_wall": float(
                    (
                        strict_refresh_effective_seconds
                        / float(strict_runner_wall_seconds)
                    )
                    if float(strict_runner_wall_seconds) > 0.0
                    else 0.0
                ),
                "core_scaffold_enabled": bool(use_core_scaffold),
                "core_scaffold_exec_calls": int(core_scaffold_exec_calls),
                "core_scaffold_prepare_calls": int(core_scaffold_prepare_calls),
                "core_scaffold_refresh_calls": int(core_scaffold_refresh_calls),
            }
        )
        return out_arr

    dt_val = float(params.t_end) / float(num_steps) if dt is None else float(dt)
    dt_arr = jnp.asarray(dt_val, dtype=state_curr.dtype)

    solver = _build_fmm_solver(
        working_dtype=(
            state_curr.dtype
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
        # Adaptive diffrax RHS tracing currently cannot tolerate jaccpot's
        # jitted tree-builder tracer leakage; force eager tree build here.
        fmm_jit_tree=False,
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
            p.kind == inspect.Parameter.VAR_KEYWORD for p in refresh_fn_params.values()
        )
    )

    def _prepare_state(state_in: jnp.ndarray):
        with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
            positions_in = state_in[:, 0, :]
            prepared = solver.prepare_state(
                positions_in,
                mass_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=float(fmm_theta),
                fused_device_mode=True,
            )
            if str(
                os.environ.get(
                    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
                    "1",
                )
            ).strip().lower() in {"1", "true", "yes", "on"}:
                try:
                    prepared = solver.refresh_prepared_state(
                        prepared,
                        positions_in,
                        mass_arr,
                        leaf_size=int(leaf_size),
                        max_order=int(max_order),
                        theta=float(fmm_theta),
                        fused_device_mode=True,
                    )
                except TypeError:
                    pass
            return prepared

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
        nonlocal refresh_call_mode_index
        nonlocal refresh_call_runner

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

        refresh_prepare_attempts += 1
        pos = state_in[:, 0, :]
        leaf = int(leaf_size)
        order = int(max_order)

        if refresh_call_runner is not None:
            try:
                with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
                    out = refresh_call_runner(prev_prepared_state, pos)
                refresh_prepare_successes += 1
                last_prepare_path = "refresh"
                return out
            except TypeError:
                # If signature drift occurs at runtime, re-resolve once.
                refresh_call_mode_index = None
                refresh_call_runner = None
            except (NotImplementedError, RuntimeError, ValueError):
                # Unsupported refresh path for this runtime/profile.
                pass

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
                    if idx == 0:
                        if refresh_fn_accepts_var_kw:
                            refresh_call_runner = (
                                lambda prepared, positions: refresh_fn(
                                    prepared,
                                    positions,
                                    mass_arr,
                                    leaf_size=leaf,
                                    max_order=order,
                                )
                            )
                        else:
                            include_leaf = "leaf_size" in refresh_fn_params
                            include_order = "max_order" in refresh_fn_params
                            if include_leaf and include_order:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        prepared,
                                        positions,
                                        mass_arr,
                                        leaf_size=leaf,
                                        max_order=order,
                                    )
                                )
                            elif include_leaf:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        prepared,
                                        positions,
                                        mass_arr,
                                        leaf_size=leaf,
                                    )
                                )
                            elif include_order:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        prepared,
                                        positions,
                                        mass_arr,
                                        max_order=order,
                                    )
                                )
                            else:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        prepared,
                                        positions,
                                        mass_arr,
                                    )
                                )
                    elif idx == 1:
                        if refresh_fn_accepts_var_kw:
                            refresh_call_runner = (
                                lambda prepared, positions: refresh_fn(
                                    positions,
                                    mass_arr,
                                    prepared,
                                    leaf_size=leaf,
                                    max_order=order,
                                )
                            )
                        else:
                            include_leaf = "leaf_size" in refresh_fn_params
                            include_order = "max_order" in refresh_fn_params
                            if include_leaf and include_order:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        positions,
                                        mass_arr,
                                        prepared,
                                        leaf_size=leaf,
                                        max_order=order,
                                    )
                                )
                            elif include_leaf:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        positions,
                                        mass_arr,
                                        prepared,
                                        leaf_size=leaf,
                                    )
                                )
                            elif include_order:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        positions,
                                        mass_arr,
                                        prepared,
                                        max_order=order,
                                    )
                                )
                            else:
                                refresh_call_runner = (
                                    lambda prepared, positions: refresh_fn(
                                        positions,
                                        mass_arr,
                                        prepared,
                                    )
                                )
                    elif idx == 2:
                        if refresh_fn_accepts_var_kw:
                            refresh_call_runner = (
                                lambda prepared, positions: refresh_fn(
                                    prepared_state=prepared,
                                    positions=positions,
                                    masses=mass_arr,
                                    leaf_size=leaf,
                                    max_order=order,
                                )
                            )
                        else:
                            include_prepared = "prepared_state" in refresh_fn_params
                            include_positions = "positions" in refresh_fn_params
                            include_masses = "masses" in refresh_fn_params
                            include_leaf = "leaf_size" in refresh_fn_params
                            include_order = "max_order" in refresh_fn_params

                            def _run_idx2(prepared, positions):
                                kwargs2 = {}
                                if include_prepared:
                                    kwargs2["prepared_state"] = prepared
                                if include_positions:
                                    kwargs2["positions"] = positions
                                if include_masses:
                                    kwargs2["masses"] = mass_arr
                                if include_leaf:
                                    kwargs2["leaf_size"] = leaf
                                if include_order:
                                    kwargs2["max_order"] = order
                                return refresh_fn(**kwargs2)

                            refresh_call_runner = _run_idx2
                    elif idx == 3:
                        if refresh_fn_accepts_var_kw:
                            refresh_call_runner = (
                                lambda prepared, positions: refresh_fn(
                                    previous_prepared_state=prepared,
                                    positions=positions,
                                    masses=mass_arr,
                                    leaf_size=leaf,
                                    max_order=order,
                                )
                            )
                        else:
                            include_prepared = (
                                "previous_prepared_state" in refresh_fn_params
                            )
                            include_positions = "positions" in refresh_fn_params
                            include_masses = "masses" in refresh_fn_params
                            include_leaf = "leaf_size" in refresh_fn_params
                            include_order = "max_order" in refresh_fn_params

                            def _run_idx3(prepared, positions):
                                kwargs3 = {}
                                if include_prepared:
                                    kwargs3["previous_prepared_state"] = prepared
                                if include_positions:
                                    kwargs3["positions"] = positions
                                if include_masses:
                                    kwargs3["masses"] = mass_arr
                                if include_leaf:
                                    kwargs3["leaf_size"] = leaf
                                if include_order:
                                    kwargs3["max_order"] = order
                                return refresh_fn(**kwargs3)

                            refresh_call_runner = _run_idx3
                    elif idx == 4:
                        refresh_call_runner = lambda prepared, positions: refresh_fn(
                            prepared,
                            positions,
                            mass_arr,
                        )
                    else:
                        refresh_call_runner = lambda prepared, positions: refresh_fn(
                            positions,
                            mass_arr,
                            prepared,
                        )
                refresh_prepare_successes += 1
                last_prepare_path = "refresh"
                return out
            except TypeError:
                continue
            except (NotImplementedError, RuntimeError, ValueError):
                break

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
        return {key: float(runtime_diag.get(key, 0.0)) for key in prepare_stage_keys}

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
            if abs(float(stage_after.get(key, 0.0) - stage_before.get(key, 0.0))) > 0.0
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
                str(key): float(value) for key, value in sorted(stage_delta.items())
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
        if strict_production_lane and not profile:
            positions_in = state_in[:, 0, :]
            prepared_out, acc_out = solver.strict_prepare_refresh_and_evaluate(
                prev_prepared_state,
                positions_in,
                mass_arr,
                leaf_size=int(leaf_size),
                max_order=int(max_order),
                theta=float(fmm_theta),
                jit_traversal=(
                    True if fmm_jit_traversal is None else bool(fmm_jit_traversal)
                ),
            )
            return prepared_out, jnp.asarray(acc_out)
        prepared_out = _prepare_or_refresh_state(state_in, prev_prepared_state)
        acc_out = _eval_prepared(prepared_out, active_indices=None)
        return prepared_out, acc_out

    def _run_nonprofile_full_segmented_loop(
        state_in: jnp.ndarray,
        *,
        num_steps_i: int,
        refresh_every_i: int,
        remat_enabled: bool,
    ) -> jnp.ndarray:
        """Run the non-profile full-particle segmented loop with minimal branching."""
        full_segments = num_steps_i // refresh_every_i
        tail_segment = num_steps_i % refresh_every_i
        prepared_state = None
        state_out = state_in

        if remat_enabled:
            for _ in range(full_segments):
                prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                    state_out, prepared_state
                )
                state_out, _ = _run_full_segment_scan(
                    state_out,
                    acc_self_full,
                    dt_arr,
                    steps=refresh_every_i,
                    add_external=add_external,
                    config=config,
                    params=params,
                )
                state_out = jnp.asarray(state_out, dtype=state_out.dtype)
            if tail_segment > 0:
                prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                    state_out, prepared_state
                )
                state_out, _ = _run_full_segment_scan(
                    state_out,
                    acc_self_full,
                    dt_arr,
                    steps=tail_segment,
                    add_external=add_external,
                    config=config,
                    params=params,
                )
                state_out = jnp.asarray(state_out, dtype=state_out.dtype)
        else:
            for _ in range(full_segments):
                prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                    state_out, prepared_state
                )
                state_out, _ = _run_full_segment_scan(
                    state_out,
                    acc_self_full,
                    dt_arr,
                    steps=refresh_every_i,
                    add_external=add_external,
                    config=config,
                    params=params,
                )
            if tail_segment > 0:
                prepared_state, acc_self_full = _prepare_refresh_and_eval_full(
                    state_out, prepared_state
                )
                state_out, _ = _run_full_segment_scan(
                    state_out,
                    acc_self_full,
                    dt_arr,
                    steps=tail_segment,
                    add_external=add_external,
                    config=config,
                    params=params,
                )
        return state_out

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
        if not bool(return_history):
            # Tight strict lane: avoid generic per-segment profile/history
            # branching and keep host orchestration minimal.
            strict_effective_add_external = bool(
                add_external and not bool(strict_timing_disable_external)
            )
            if bool(strict_timing_external_only):
                strict_effective_add_external = bool(add_external)
            strict_timing_mode = (
                "external_only"
                if bool(strict_timing_external_only)
                else (
                    "jaccpot_self_only"
                    if bool(strict_timing_disable_external)
                    else "full"
                )
            )
            external_acc_fn = None
            if strict_effective_add_external:

                def _strict_external_acceleration(state_in: jnp.ndarray) -> jnp.ndarray:
                    return combined_external_acceleration_vmpa_switch(
                        state_in, config, params
                    )

                external_acc_fn = _strict_external_acceleration

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
            perf_warmup_runs_i = max(0, int(perf_warmup_runs))
            perf_measure_runs_i = max(1, int(perf_measure_runs))
            perf_warmup_run_seconds = []
            perf_measured_run_seconds = []
            state_initial = state_curr
            hist_out = None
            prepared_initial = None
            initial_self_acceleration = None
            if not bool(return_history):
                prepared_initial = _prepare_state(state_initial)
                prepared_initial = jax.block_until_ready(prepared_initial)
                strict_diag_mode = (
                    str(os.environ.get("JACCPOT_STRICT_REFRESH_DIAG_MODE", "full"))
                    .strip()
                    .lower()
                )
                strict_detail_mode = (
                    str(
                        os.environ.get(
                            "JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full"
                        )
                    )
                    .strip()
                    .lower()
                )
                eval_diag_mode = (
                    str(os.environ.get("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "full"))
                    .strip()
                    .lower()
                )
                if (
                    strict_diag_mode != "integrator_only"
                    and strict_detail_mode == "full"
                    and eval_diag_mode != "zero"
                ):
                    initial_self_acceleration = _eval_prepared(
                        prepared_initial, active_indices=None
                    )
                    initial_self_acceleration = jax.block_until_ready(
                        initial_self_acceleration
                    )

            def _run_strict_once(state_in: jnp.ndarray, prepared_state_in):
                strict_kwargs = dict(
                    state=state_in,
                    masses=mass_arr,
                    dt=float(dt_val),
                    num_steps=int(num_steps),
                    refresh_every=int(refresh_every),
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    theta=float(fmm_theta),
                    prepared_state=prepared_state_in,
                    initial_self_acceleration=initial_self_acceleration,
                    jit_traversal=(
                        True if fmm_jit_traversal is None else bool(fmm_jit_traversal)
                    ),
                    add_external=bool(strict_effective_add_external),
                    external_acceleration_fn=external_acc_fn,
                    rematerialize_between_refresh=bool(rematerialize_between_refresh),
                    return_history=bool(return_history),
                )
                # Keep the large-N env overrides (e.g. target-block size, static
                # target-block cap) active while strict_run_v2 traces/compiles the
                # device-resident scan, so the traced refresh resolves the SAME
                # fixed-shape config as the eager prepare. Otherwise the transient
                # overrides expire before compilation and the fused static
                # target-block preflight fails (block_size / cap mismatch).
                with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
                    try:
                        strict_kwargs["return_prepared_state"] = bool(return_history)
                        return solver.strict_run_v2(**strict_kwargs)
                    except TypeError as exc:
                        if "return_prepared_state" not in str(exc):
                            raise
                        strict_kwargs.pop("return_prepared_state", None)
                        return solver.strict_run_v2(**strict_kwargs)

            def _run_external_only_once(state_in: jnp.ndarray):
                zero_self_acc = jnp.zeros_like(state_in[:, 0, :])
                state_out, _ = _run_full_segment_scan(
                    state_in,
                    zero_self_acc,
                    dt_arr,
                    steps=int(num_steps),
                    add_external=bool(add_external),
                    config=config,
                    params=params,
                )
                return state_out

            try:
                for idx in range(perf_warmup_runs_i):
                    if strict_perf_progress:
                        print(
                            f"[odisseo.strict] warmup {idx + 1}/{perf_warmup_runs_i} start",
                            flush=True,
                        )
                    tw = time.perf_counter()
                    if bool(strict_timing_external_only):
                        warm_state = _run_external_only_once(state_initial)
                    else:
                        warm_state, _, _ = _run_strict_once(
                            state_initial,
                            prepared_initial,
                        )
                    _ = jax.block_until_ready(warm_state)
                    elapsed_warm = float(time.perf_counter() - tw)
                    perf_warmup_run_seconds.append(elapsed_warm)
                    if strict_perf_progress:
                        print(
                            f"[odisseo.strict] warmup {idx + 1}/{perf_warmup_runs_i} seconds={elapsed_warm:.6g}",
                            flush=True,
                        )
                for idx in range(perf_measure_runs_i):
                    if strict_perf_progress:
                        print(
                            f"[odisseo.strict] measure {idx + 1}/{perf_measure_runs_i} start",
                            flush=True,
                        )
                    tm = time.perf_counter()
                    if bool(strict_timing_external_only):
                        state_curr = _run_external_only_once(state_initial)
                        hist_out = None
                    else:
                        state_curr, _, hist_out = _run_strict_once(
                            state_initial,
                            prepared_initial,
                        )
                    _ = jax.block_until_ready(
                        hist_out
                        if bool(return_history) and hist_out is not None
                        else state_curr
                    )
                    elapsed_measure = float(time.perf_counter() - tm)
                    perf_measured_run_seconds.append(elapsed_measure)
                    if strict_perf_progress:
                        print(
                            f"[odisseo.strict] measure {idx + 1}/{perf_measure_runs_i} seconds={elapsed_measure:.6g}",
                            flush=True,
                        )
            finally:
                if profile:
                    try:
                        setattr(
                            timing_target, "_refresh_timing_active", old_timing_active
                        )
                    except Exception:
                        pass
            if profile:
                strict_runner_wall_seconds = time.perf_counter() - t0
                update_seconds += strict_runner_wall_seconds
                update_calls += 1
            if bool(return_history) and hist_out is not None:
                return _finalize(hist_out)
            return _finalize(state_curr)

    if (
        bool(use_core_scaffold)
        and active_indices_fn is None
        and active_indices_schedule is None
        and not bool(return_history)
        and not bool(add_external)
        and not bool(refresh_after_position_update)
    ):
        core_kernel, _core_meta = build_compiled_jaccpot_core_kernel(
            config,
            params,
            mode="fixed_step_update",
            dt=float(dt_val),
            leaf_size=int(leaf_size),
            max_order=int(max_order),
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
        prepared_core = None
        for _ in range(int(num_steps)):
            out = core_kernel(state_curr, mass_arr, prepared_core)
            state_curr = jnp.asarray(out.next_state)
            prepared_core = out.prepared_state
            core_scaffold_exec_calls += int(getattr(out, "execute_count", 1))
            core_scaffold_prepare_calls += int(getattr(out, "prepare_count", 0))
            core_scaffold_refresh_calls += int(getattr(out, "refresh_count", 0))
        return _finalize(state_curr)

    if active_indices_schedule is not None:
        active_indices_schedule = jnp.asarray(active_indices_schedule, dtype=jnp.int32)
        if active_indices_schedule.ndim != 2:
            raise ValueError(
                "active_indices_schedule must have shape (num_steps, max_active)"
            )
        if int(active_indices_schedule.shape[0]) != int(num_steps):
            raise ValueError(
                "active_indices_schedule first dimension must equal num_steps"
            )

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
        if not profile and not bool(return_history):
            num_steps_i = int(num_steps)
            refresh_every_i = int(refresh_every)
            remat_enabled = bool(rematerialize_between_refresh)
            state_curr = _run_nonprofile_full_segmented_loop(
                state_curr,
                num_steps_i=num_steps_i,
                refresh_every_i=refresh_every_i,
                remat_enabled=remat_enabled,
            )
            return _finalize(state_curr)

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
                state_curr[:, 0] + state_curr[:, 1] * dt_arr + 0.5 * acc_1 * (dt_arr**2)
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
    use_core_scaffold = (
        os.environ.get("ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD", "0").strip() == "1"
    )
    allow_tracer_prepared_cache = os.environ.get(
        "ODISSEO_FMM_ALLOW_TRACER_PREPARED_CACHE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    adaptive_prepared_cache_mode = (
        str(os.environ.get("ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE", "none"))
        .strip()
        .lower()
    )
    if adaptive_prepared_cache_mode not in {"none", "python"}:
        raise ValueError(
            "ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE must be one of {'none', 'python'}"
        )
    adaptive_python_prepared_cache = adaptive_prepared_cache_mode == "python"
    core_kernel = None
    adaptive_refresh_rhs_calls = max(
        1,
        int(getattr(config, "fmm_adaptive_refresh_rhs_calls", 1)),
    )
    adaptive_refresh_displacement_threshold = getattr(
        config,
        "fmm_adaptive_refresh_displacement_threshold",
        None,
    )
    if adaptive_refresh_displacement_threshold is not None:
        adaptive_refresh_displacement_threshold = float(
            adaptive_refresh_displacement_threshold
        )
    if use_core_scaffold:
        core_kernel, _ = build_compiled_jaccpot_core_kernel(
            config,
            params,
            mode="rhs_only",
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
        "core_exec_calls": 0,
        "core_prepare_calls": 0,
        "core_refresh_calls": 0,
    }
    adaptive_core_state = AdaptiveCoreRuntimeState()

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
        adaptive_core_state.rhs_calls += 1
        if core_kernel is not None:
            prepared_in = adaptive_core_state.prepared_input(
                enabled=adaptive_python_prepared_cache
            )
            force_refresh_prepared = adaptive_core_state.should_refresh(
                enabled=adaptive_python_prepared_cache,
                prepared_in=prepared_in,
                y_state=y,
                refresh_rhs_calls=adaptive_refresh_rhs_calls,
                displacement_threshold=adaptive_refresh_displacement_threshold,
            )
            out = core_kernel(
                y,
                mass_arr,
                prepared_in,
                refresh_prepared=bool(force_refresh_prepared),
            )
            acc_self = out.acceleration
            cache["full_prepare_calls"] += int(out.prepare_count)
            cache["refresh_prepare_calls"] += int(out.refresh_count)
            cache["core_exec_calls"] += int(getattr(out, "execute_count", 1))
            cache["core_prepare_calls"] += int(getattr(out, "prepare_count", 0))
            cache["core_refresh_calls"] += int(getattr(out, "refresh_count", 0))
            prepared_out = getattr(out, "prepared_state", None)
            adaptive_core_state.update_prepared_state(
                enabled=adaptive_python_prepared_cache,
                prepared_out=prepared_out,
                allow_tracer_prepared_cache=allow_tracer_prepared_cache,
            )
            if (
                int(getattr(out, "prepare_count", 0)) > 0
                or int(getattr(out, "refresh_count", 0)) > 0
            ):
                adaptive_core_state.mark_refreshed(y_state=y)
        else:
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
                "adaptive_use_core_kernel_scaffold": bool(use_core_scaffold),
                "adaptive_core_scaffold_exec_calls": int(cache["core_exec_calls"]),
                "adaptive_core_scaffold_prepare_calls": int(
                    cache["core_prepare_calls"]
                ),
                "adaptive_core_scaffold_refresh_calls": int(
                    cache["core_refresh_calls"]
                ),
                "adaptive_core_prepared_drop_tracer": int(
                    adaptive_core_state.prepared_drop_tracer
                ),
                "adaptive_allow_tracer_prepared_cache": bool(
                    allow_tracer_prepared_cache
                ),
                "adaptive_prepared_cache_mode": str(adaptive_prepared_cache_mode),
                "adaptive_python_prepared_cache_enabled": bool(
                    adaptive_python_prepared_cache
                ),
                "adaptive_refresh_rhs_calls_target": int(adaptive_refresh_rhs_calls),
                "adaptive_refresh_displacement_threshold": (
                    None
                    if adaptive_refresh_displacement_threshold is None
                    else float(adaptive_refresh_displacement_threshold)
                ),
                "adaptive_core_refresh_cadence_skips_rhs_calls": int(
                    adaptive_core_state.refresh_cadence_skips_rhs_calls
                ),
                "adaptive_core_refresh_cadence_skips_displacement": int(
                    adaptive_core_state.refresh_cadence_skips_displacement
                ),
                "adaptive_core_refresh_cadence_last_displacement": float(
                    adaptive_core_state.refresh_cadence_last_displacement
                ),
                "adaptive_core_prepared_non_large_n_seen": int(
                    adaptive_core_state.prepared_non_large_n_seen
                ),
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
