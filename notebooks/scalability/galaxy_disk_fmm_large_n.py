"""Galaxy-disk simulation using ODISSEO + jaccpot large-N radix path."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u

from odisseo import construct_initial_state
from odisseo.integration_api import integrate
from odisseo.jaccpot_coupling import (
    _build_fmm_solver,
    _large_n_environment_overrides,
    integrate_leapfrog_jaccpot_active,
)
from odisseo.option_classes import (
    FMM_ACC,
    NFW_POTENTIAL,
    SimulationConfig,
    SimulationParams,
    NFWParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.units import CodeUnits
from odisseo.utils import Angular_momentum, E_kin, center_of_mass

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("perf", "render"), default="perf")
    parser.add_argument("--n-particles", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--t-end-gyr", type=float, default=2.0)
    parser.add_argument("--disk-radius-kpc", type=float, default=12.0)
    parser.add_argument("--disk-height-kpc", type=float, default=0.3)
    parser.add_argument("--disk-mass-msun", type=float, default=6.0e10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="./galaxy_disk_fmm_large_n.npz")
    parser.add_argument(
        "--fmm-refresh-every",
        type=int,
        default=1,
        help="Refresh source-tree/self-field every k steps.",
    )
    parser.add_argument(
        "--fmm-preset",
        type=str,
        default="fast",
        choices=("fast", "balanced", "accurate", "large_n_gpu"),
        help="FMM preset passed to jaccpot/ODISSEO coupling.",
    )
    parser.add_argument(
        "--fmm-runtime-path",
        type=str,
        default="auto",
        choices=("auto", "legacy", "large_n"),
        help="jaccpot runtime path selection.",
    )
    parser.add_argument(
        "--fmm-leaf-size",
        type=int,
        default=256,
        help="Runtime maximum particles per leaf for the large-N FMM path.",
    )
    parser.add_argument(
        "--fmm-tree-leaf-target",
        type=int,
        default=None,
        help="Tree leaf target. Defaults to --fmm-leaf-size.",
    )
    parser.add_argument(
        "--fmm-tree-build-mode",
        type=str,
        default="lbvh",
        choices=("lbvh", "fixed_depth", "static_radix", "adaptive"),
        help="jaccpot tree build mode.",
    )
    parser.add_argument(
        "--fmm-max-order",
        type=int,
        default=4,
        help="FMM expansion order.",
    )
    parser.add_argument(
        "--fmm-nearfield-edge-chunk-size",
        type=int,
        default=256,
        help="Near-field edge chunk size for jaccpot large-N execution.",
    )
    parser.add_argument(
        "--fmm-large-n-target-block-size",
        type=int,
        default=None,
        help="Optional jaccpot large-N target-owned nearfield block size.",
    )
    parser.add_argument(
        "--fmm-large-n-static-target-blocks",
        action="store_true",
        default=None,
        help="Enable fixed-capacity target-block layout for jaccpot large-N.",
    )
    parser.add_argument(
        "--no-fmm-large-n-static-target-blocks",
        dest="fmm_large_n_static_target_blocks",
        action="store_false",
        help="Disable fixed-capacity target-block layout for jaccpot large-N.",
    )
    parser.add_argument(
        "--fmm-large-n-static-target-blocks-max-per-leaf",
        type=int,
        default=None,
        help="Fixed-capacity target-block slots per leaf when static target blocks are enabled.",
    )
    parser.add_argument(
        "--fmm-m2l-chunk-size",
        type=int,
        default=None,
        help="Optional far-field M2L chunk size override.",
    )
    parser.add_argument(
        "--fmm-enforce-static-shape-contract",
        action="store_true",
        help="Fail fast if prepared-state leaf dtype/shape signature drifts across refreshes.",
    )
    parser.add_argument(
        "--fmm-static-shape-warmup-prepares",
        type=int,
        default=0,
        help="Number of pre-run warmup prepare/evaluate calls for compile stabilization.",
    )
    parser.add_argument(
        "--no-fmm-rematerialize-between-refresh",
        action="store_true",
        help="Disable dense rematerialization between refresh segments (experiment only).",
    )
    parser.add_argument(
        "--fmm-prepare-stage-memory-split",
        dest="fmm_prepare_stage_memory_split_enabled",
        action="store_true",
        help="Force jaccpot's lower-peak split prepare-stage build.",
    )
    parser.add_argument(
        "--no-fmm-prepare-stage-memory-split",
        dest="fmm_prepare_stage_memory_split_enabled",
        action="store_false",
        help="Disable jaccpot's split prepare-stage build.",
    )
    parser.set_defaults(fmm_prepare_stage_memory_split_enabled=None)

    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Collect detailed coupling timing breakdown (best in perf mode).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./notebooks/scalability/reports",
        help="Directory to save timing reports (JSON/CSV).",
    )
    parser.add_argument(
        "--conservation-report",
        action="store_true",
        help="Compute and export conservation diagnostics in perf mode.",
    )
    parser.add_argument(
        "--conservation-stride",
        type=int,
        default=1,
        help="Evaluate conservation diagnostics every k-th step when enabled.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Optional regression gate: fail if script runtime exceeds this value.",
    )
    parser.add_argument(
        "--require-static-shape",
        action="store_true",
        help="Optional regression gate: fail if post-warmup prepared-state shapes drift.",
    )
    parser.add_argument(
        "--max-compiled-profile-transitions",
        type=int,
        default=None,
        help="Optional regression gate for jaccpot compiled-profile transitions.",
    )
    parser.add_argument(
        "--max-overflow-reprofiles",
        type=int,
        default=None,
        help="Optional regression gate for large-N overflow profile reprofiles.",
    )
    parser.add_argument(
        "--max-neighbor-edge-reprofiles",
        type=int,
        default=None,
        help="Optional regression gate for large-N neighbor-edge profile reprofiles.",
    )
    parser.add_argument(
        "--min-refresh-prepare-successes",
        type=int,
        default=None,
        help="Optional regression gate for incremental refresh prepare successes.",
    )
    parser.add_argument(
        "--max-abs-de-over-e0",
        type=float,
        default=None,
        help="Optional conservation gate for max |dE/E0|.",
    )
    parser.add_argument(
        "--max-abs-dl-over-l0",
        type=float,
        default=None,
        help="Optional conservation gate for max |dL|/|L0|.",
    )
    parser.add_argument(
        "--max-com-drift",
        type=float,
        default=None,
        help="Optional conservation gate for max center-of-mass drift.",
    )

    # Render / snapshot controls.
    parser.add_argument("--live", action="store_true", help="Play snapshots live after run.")
    parser.add_argument(
        "--movie-path",
        type=str,
        default=None,
        help="Optional output movie path (.mp4 or .gif).",
    )
    parser.add_argument("--movie-fps", type=int, default=24)
    parser.add_argument(
        "--projection",
        type=str,
        default="xy",
        choices=("xy", "xz", "yz"),
        help="Projection used for live/movie rendering.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=1,
        help="Store every k-th integration step in render mode.",
    )
    parser.add_argument(
        "--snapshot-chunk-steps",
        type=int,
        default=20,
        help="Integrate this many steps per chunk in render mode.",
    )
    parser.add_argument(
        "--snapshot-max-particles",
        type=int,
        default=50_000,
        help="Max particles stored/rendered per snapshot (deterministic subsample).",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Include sampled snapshots in output NPZ.",
    )
    parser.add_argument(
        "--snapshot-output",
        type=str,
        default=None,
        help="Optional dedicated snapshot NPZ output path.",
    )
    return parser.parse_args()


def sample_exponential_disk(
    key: jax.Array,
    n_particles: int,
    radial_scale: float,
    vertical_scale: float,
) -> jnp.ndarray:
    """Sample a simple exponential disk in Cartesian coordinates."""
    key_r, key_phi, key_z, key_sign = jax.random.split(key, 4)

    u_r = jax.random.uniform(key_r, shape=(n_particles,), minval=1e-8, maxval=1 - 1e-8)
    radius = -radial_scale * jnp.log1p(-u_r)

    phi = jax.random.uniform(key_phi, shape=(n_particles,), minval=0.0, maxval=2.0 * jnp.pi)

    u_z = jax.random.uniform(key_z, shape=(n_particles,), minval=1e-8, maxval=1 - 1e-8)
    sign = jnp.where(jax.random.uniform(key_sign, shape=(n_particles,)) > 0.5, 1.0, -1.0)
    z = sign * (-vertical_scale * jnp.log1p(-u_z))

    x = radius * jnp.cos(phi)
    y = radius * jnp.sin(phi)
    return jnp.stack((x, y, z), axis=1)


def build_quasi_circular_velocities(
    position: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
) -> jnp.ndarray:
    """Set tangential velocity from external radial acceleration."""
    state0 = construct_initial_state(position, jnp.zeros_like(position))
    acc_ext = combined_external_acceleration_vmpa_switch(state0, config, params)

    x = position[:, 0]
    y = position[:, 1]
    radius = jnp.sqrt(x * x + y * y + 1e-12)
    e_r = jnp.stack((x / radius, y / radius), axis=1)
    a_r = jnp.sum(acc_ext[:, :2] * e_r, axis=1)

    v_c = jnp.sqrt(jnp.maximum(0.0, -radius * a_r))
    e_phi = jnp.stack((-y / radius, x / radius), axis=1)

    vel_xy = e_phi * v_c[:, None]
    return jnp.concatenate((vel_xy, jnp.zeros((position.shape[0], 1))), axis=1)


def _projection_axes(projection: str) -> tuple[int, int, str, str]:
    if projection == "xy":
        return 0, 1, "x [code]", "y [code]"
    if projection == "xz":
        return 0, 2, "x [code]", "z [code]"
    return 1, 2, "y [code]", "z [code]"


def render_positions(
    positions_frames: np.ndarray,
    times: np.ndarray,
    *,
    projection: str,
    live: bool,
    movie_path: str | None,
    movie_fps: int,
) -> None:
    """Render sampled snapshot positions live and/or save as movie."""
    if not live and movie_path is None:
        return

    import matplotlib.pyplot as plt
    from matplotlib import animation

    n_frames = positions_frames.shape[0]
    i0, i1, xlabel, ylabel = _projection_axes(projection)
    x_all = positions_frames[:, :, i0]
    y_all = positions_frames[:, :, i1]

    extent = float(np.percentile(np.abs(np.concatenate((x_all.ravel(), y_all.ravel()))), 99.5))
    extent = max(extent, 1e-6)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)

    scat = ax.scatter(x_all[0], y_all[0], s=0.6, alpha=0.6, linewidths=0)
    title = ax.set_title(f"Galaxy disk evolution: t={times[0]:.3f}")

    def _update(frame: int):
        scat.set_offsets(np.column_stack((x_all[frame], y_all[frame])))
        title.set_text(f"Galaxy disk evolution: t={times[frame]:.3f}")
        return scat, title

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=max(1, int(1000 / max(1, movie_fps))),
        blit=False,
        repeat=False,
    )

    if movie_path is not None:
        movie_path_obj = pathlib.Path(movie_path)
        movie_path_obj.parent.mkdir(parents=True, exist_ok=True)
        suffix = movie_path_obj.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=max(1, movie_fps))
        else:
            writer = animation.FFMpegWriter(fps=max(1, movie_fps), bitrate=1800)
        ani.save(str(movie_path_obj), writer=writer)
        print(f"Saved movie: {movie_path_obj}")

    if live:
        plt.show()
    else:
        plt.close(fig)


def _resolve_fmm_profile(config: SimulationConfig) -> tuple[str, str, jnp.dtype]:
    on_gpu = str(jax.default_backend()).strip().lower() == "gpu"
    large_n = (
        on_gpu
        and int(config.N_particles) >= int(config.fmm_large_n_min_particles)
        and str(config.fmm_preset).strip().lower() == "fast"
    )
    if large_n:
        return "large_n_gpu", "large_n", jnp.float32
    return (
        str(config.fmm_preset),
        str(config.fmm_runtime_path),
        jnp.dtype(jnp.float32),
    )


def _run_perf_mode(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    profile_breakdown: bool,
) -> tuple[np.ndarray, np.ndarray | None, float, dict | None]:
    timing_stats = {} if bool(profile_breakdown) else None
    t0 = time.time()
    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)

    if bool(profile_breakdown):
        final_state = jax.block_until_ready(
            integrate_leapfrog_jaccpot_active(
                state0,
                mass,
                config,
                params,
                num_steps=int(config.num_timesteps),
                refresh_every=int(config.fmm_refresh_every),
                leaf_size=int(config.fmm_leaf_size),
                max_order=int(config.fmm_max_order),
                fmm_preset=fmm_preset,
                fmm_runtime_path=fmm_runtime_path,
                fmm_working_dtype=fmm_dtype,
                fmm_basis=str(config.fmm_basis),
                fmm_theta=float(config.fmm_theta),
                fmm_mac_type=str(config.fmm_mac_type),
                fmm_farfield_mode=str(config.fmm_farfield_mode),
                fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
                fmm_nearfield_mode=str(config.fmm_nearfield_mode),
                fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
                fmm_tree_build_mode=str(config.fmm_tree_build_mode),
                fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
                fmm_fixed_order=config.fmm_fixed_order,
                fmm_jit_tree=config.fmm_jit_tree,
                fmm_jit_traversal=config.fmm_jit_traversal,
                fmm_prepare_stage_memory_split_enabled=(
                    config.fmm_prepare_stage_memory_split_enabled
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
                return_history=False,
                timing_stats=timing_stats,
            )
        )
        history = None
    else:
        final_state = jax.block_until_ready(integrate(state0, mass, config, params))
        history = None

    elapsed = time.time() - t0
    return np.asarray(final_state), history, float(elapsed), timing_stats


def _run_perf_mode_with_history(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    profile_breakdown: bool,
) -> tuple[np.ndarray, np.ndarray, float, dict | None]:
    timing_stats = {} if bool(profile_breakdown) else None
    t0 = time.time()
    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)

    history = jax.block_until_ready(
        integrate_leapfrog_jaccpot_active(
            state0,
            mass,
            config,
            params,
            num_steps=int(config.num_timesteps),
            refresh_every=int(config.fmm_refresh_every),
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
            fmm_preset=fmm_preset,
            fmm_runtime_path=fmm_runtime_path,
            fmm_working_dtype=fmm_dtype,
            fmm_basis=str(config.fmm_basis),
            fmm_theta=float(config.fmm_theta),
            fmm_mac_type=str(config.fmm_mac_type),
            fmm_farfield_mode=str(config.fmm_farfield_mode),
            fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
            fmm_nearfield_mode=str(config.fmm_nearfield_mode),
            fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
            fmm_tree_build_mode=str(config.fmm_tree_build_mode),
            fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
            fmm_fixed_order=config.fmm_fixed_order,
            fmm_jit_tree=config.fmm_jit_tree,
            fmm_jit_traversal=config.fmm_jit_traversal,
            fmm_prepare_stage_memory_split_enabled=(
                config.fmm_prepare_stage_memory_split_enabled
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
            return_history=True,
            timing_stats=timing_stats,
        )
    )
    elapsed = time.time() - t0
    history_np = np.asarray(history)
    return history_np[-1], history_np, float(elapsed), timing_stats


def _select_conservation_rows(states: np.ndarray, step_stride: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(states.shape[0])
    stride = max(1, int(step_stride))
    row_idx = np.arange(0, n_rows, stride, dtype=np.int64)
    if row_idx[-1] != n_rows - 1:
        row_idx = np.concatenate([row_idx, np.asarray([n_rows - 1], dtype=np.int64)])
    return states[row_idx], row_idx


def _compute_conservation_metrics(
    states: np.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    step_stride: int,
) -> dict:
    sampled_states_np, row_idx = _select_conservation_rows(states, step_stride)
    sampled_states = jnp.asarray(sampled_states_np)
    mass_arr = jnp.asarray(mass)

    L_series = jax.vmap(lambda s: jnp.sum(Angular_momentum(s, mass_arr), axis=0))(sampled_states)
    L0_norm = jnp.maximum(jnp.linalg.norm(L_series[0]), 1e-30)
    rel_dL = jnp.linalg.norm(L_series - L_series[0], axis=1) / L0_norm

    com_series = jax.vmap(lambda s: center_of_mass(s, mass_arr))(sampled_states)
    com_drift = jnp.linalg.norm(com_series - com_series[0], axis=1)

    kinetic_series = jax.vmap(lambda s: jnp.sum(E_kin(s, mass_arr)))(sampled_states)

    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)
    solver = _build_fmm_solver(
        working_dtype=jnp.dtype(fmm_dtype),
        config=config,
        params=params,
        fmm_preset=fmm_preset,
        fmm_basis=str(config.fmm_basis),
        fmm_theta=float(config.fmm_theta),
        fmm_runtime_path=fmm_runtime_path,
        fmm_mac_type=str(config.fmm_mac_type),
        fmm_farfield_mode=str(config.fmm_farfield_mode),
        fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
        fmm_nearfield_mode=str(config.fmm_nearfield_mode),
        fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
        fmm_tree_build_mode=str(config.fmm_tree_build_mode),
        fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
        fmm_fixed_order=config.fmm_fixed_order,
        leaf_size=int(config.fmm_leaf_size),
        fmm_jit_tree=config.fmm_jit_tree,
        fmm_jit_traversal=config.fmm_jit_traversal,
        fmm_prepare_stage_memory_split_enabled=(
            config.fmm_prepare_stage_memory_split_enabled
        ),
    )

    potential_values = []
    for i in range(int(sampled_states.shape[0])):
        state_i = sampled_states[i]
        prepared = solver.prepare_state(
            state_i[:, 0, :],
            mass_arr,
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
        )
        _, self_potential = solver.evaluate_prepared_state(
            prepared,
            target_indices=None,
            return_potential=True,
        )
        self_energy = 0.5 * jnp.sum(mass_arr * self_potential)

        ext_energy = jnp.asarray(0.0, dtype=self_energy.dtype)
        if len(config.external_accelerations) > 0:
            _, ext_potential = combined_external_acceleration_vmpa_switch(
                state_i,
                config,
                params,
                return_potential=True,
            )
            ext_energy = jnp.sum(mass_arr * ext_potential)

        potential_values.append(self_energy + ext_energy)

    potential_series = jnp.stack(potential_values)
    total_energy_series = kinetic_series + potential_series
    E0_abs = jnp.maximum(jnp.abs(total_energy_series[0]), 1e-30)
    rel_dE = (total_energy_series - total_energy_series[0]) / E0_abs

    step_idx = row_idx.astype(np.int64)

    return {
        "conservation_sample_stride": int(max(1, step_stride)),
        "conservation_num_samples": int(sampled_states.shape[0]),
        "sampled_step_indices": step_idx.tolist(),
        "max_abs_dE_over_E0": float(jnp.max(jnp.abs(rel_dE))),
        "final_abs_dE_over_E0": float(jnp.abs(rel_dE[-1])),
        "max_abs_dL_over_L0": float(jnp.max(jnp.abs(rel_dL))),
        "final_abs_dL_over_L0": float(jnp.abs(rel_dL[-1])),
        "max_com_drift": float(jnp.max(com_drift)),
        "final_com_drift": float(com_drift[-1]),
    }


def _write_conservation_report(
    *,
    report_dir: str,
    conservation_stats: dict,
) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = pathlib.Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"galaxy_disk_conservation_{stamp}.json"
    csv_path = out_dir / f"galaxy_disk_conservation_{stamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(conservation_stats, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in conservation_stats.items():
            if isinstance(v, (list, tuple)):
                writer.writerow({"metric": str(k), "value": json.dumps(v)})
            else:
                writer.writerow({"metric": str(k), "value": v})

    return json_path, csv_path


def _run_render_mode(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    snapshot_stride: int,
    snapshot_chunk_steps: int,
    snapshot_max_particles: int,
    profile_breakdown: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict | None]:
    """Chunked integration for efficient snapshot collection.

    Returns
    -------
    final_state, sampled_positions, sampled_times, sample_indices, elapsed_seconds, timing_stats
    """
    n = int(config.N_particles)
    stride = max(1, int(snapshot_stride))
    chunk_steps = max(1, int(snapshot_chunk_steps))
    sample_cap = max(1, int(snapshot_max_particles))
    step_particles = max(1, n // sample_cap)
    sample_indices_np = np.arange(0, n, step_particles, dtype=np.int64)
    sample_indices = jnp.asarray(sample_indices_np, dtype=jnp.int32)

    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)

    state_curr = state0
    frames = []
    frame_times = []

    # Include initial frame.
    frames.append(np.asarray(state_curr[sample_indices, 0, :], dtype=np.float32))
    frame_times.append(0.0)

    timing_stats = {} if bool(profile_breakdown) else None
    cumulative = {
        "total_seconds": 0.0,
        "prepare_seconds": 0.0,
        "evaluate_seconds": 0.0,
        "update_seconds": 0.0,
        "prepare_calls": 0,
        "evaluate_calls": 0,
        "update_calls": 0,
    }

    t0 = time.time()
    num_steps = int(config.num_timesteps)
    t_end = float(params.t_end)

    done = 0
    while done < num_steps:
        seg = min(chunk_steps, num_steps - done)
        seg_stats = {} if bool(profile_breakdown) else None
        hist = jax.block_until_ready(
            integrate_leapfrog_jaccpot_active(
                state_curr,
                mass,
                config,
                params,
                num_steps=int(seg),
                refresh_every=int(config.fmm_refresh_every),
                leaf_size=int(config.fmm_leaf_size),
                max_order=int(config.fmm_max_order),
                fmm_preset=fmm_preset,
                fmm_runtime_path=fmm_runtime_path,
                fmm_working_dtype=fmm_dtype,
                fmm_basis=str(config.fmm_basis),
                fmm_theta=float(config.fmm_theta),
                fmm_mac_type=str(config.fmm_mac_type),
                fmm_farfield_mode=str(config.fmm_farfield_mode),
                fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
                fmm_nearfield_mode=str(config.fmm_nearfield_mode),
                fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
                fmm_tree_build_mode=str(config.fmm_tree_build_mode),
                fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
                fmm_fixed_order=config.fmm_fixed_order,
                fmm_jit_tree=config.fmm_jit_tree,
                fmm_jit_traversal=config.fmm_jit_traversal,
                fmm_prepare_stage_memory_split_enabled=(
                    config.fmm_prepare_stage_memory_split_enabled
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
                return_history=True,
                timing_stats=seg_stats,
            )
        )

        if bool(profile_breakdown) and seg_stats is not None:
            for k in cumulative:
                cumulative[k] += float(seg_stats.get(k, 0.0))

        # Pull only sampled positions for this chunk to host.
        pos_chunk = np.asarray(hist[:, sample_indices, 0, :], dtype=np.float32)

        for local in range(seg):
            global_step = done + local + 1
            if (global_step % stride) == 0 or global_step == num_steps:
                frames.append(pos_chunk[local])
                frame_times.append((global_step / num_steps) * t_end)

        state_curr = hist[-1]
        done += seg

    elapsed = time.time() - t0

    if bool(profile_breakdown):
        timing_stats = dict(cumulative)
        timing_stats.update(
            {
                "num_steps": num_steps,
                "refresh_every": int(config.fmm_refresh_every),
                "used_external_potential": bool(len(config.external_accelerations) > 0),
                "chunk_steps": int(chunk_steps),
                "snapshot_stride": int(stride),
                "sampled_particles": int(sample_indices_np.shape[0]),
            }
        )

    return (
        np.asarray(state_curr),
        np.asarray(frames, dtype=np.float32),
        np.asarray(frame_times, dtype=np.float64),
        sample_indices_np,
        float(elapsed),
        timing_stats,
    )


def _write_timing_report(
    *,
    report_dir: str,
    timing_stats: dict,
) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = pathlib.Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"galaxy_disk_profile_{stamp}.json"
    csv_path = out_dir / f"galaxy_disk_profile_{stamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(timing_stats, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in timing_stats.items():
            writer.writerow({"metric": str(k), "value": v})

    return json_path, csv_path


def _check_timing_gates(args: argparse.Namespace, timing_stats: dict) -> None:
    if args.max_runtime_seconds is not None and float(
        timing_stats["script_runtime_seconds"]
    ) > float(args.max_runtime_seconds):
        raise RuntimeError(
            "Runtime regression: "
            f"script_runtime_seconds={timing_stats['script_runtime_seconds']:.6g} "
            f"> threshold={float(args.max_runtime_seconds):.6g}"
        )
    if bool(args.require_static_shape) and not bool(
        timing_stats.get("shape_signature_stable_post_warmup", False)
    ):
        raise RuntimeError(
            "Static-shape regression: post-warmup prepared-state shapes drifted "
            f"({timing_stats.get('shape_signature_drift_events_post_warmup', 'unknown')} events)."
        )

    gate_specs = (
        (
            args.max_compiled_profile_transitions,
            "runtime_compiled_profile_transitions",
            "Compiled-profile transition regression",
        ),
        (
            args.max_overflow_reprofiles,
            "runtime_large_n_overflow_profile_reprofiles",
            "Overflow profile reprofile regression",
        ),
        (
            args.max_neighbor_edge_reprofiles,
            "runtime_large_n_neighbor_edges_profile_reprofiles",
            "Neighbor-edge profile reprofile regression",
        ),
    )
    for threshold, key, label in gate_specs:
        if threshold is None:
            continue
        value = int(timing_stats.get(key, 0))
        if value > int(threshold):
            raise RuntimeError(
                f"{label}: {key}={value} > threshold={int(threshold)}"
            )

    if args.min_refresh_prepare_successes is not None:
        value = int(timing_stats.get("refresh_prepare_successes", 0))
        threshold = int(args.min_refresh_prepare_successes)
        if value < threshold:
            raise RuntimeError(
                "Incremental refresh regression: "
                f"refresh_prepare_successes={value} < threshold={threshold}"
            )


def main() -> None:
    args = parse_args()

    code_units = CodeUnits(10.0 * u.kpc, 1.0e10 * u.Msun, G=1.0, unit_time=(1.0 * u.Gyr))
    n_particles = int(args.n_particles)

    rd = (args.disk_radius_kpc * u.kpc).to(code_units.code_length).value
    zd = (args.disk_height_kpc * u.kpc).to(code_units.code_length).value
    total_mass = (args.disk_mass_msun * u.Msun).to(code_units.code_mass).value
    t_end = (args.t_end_gyr * u.Gyr).to(code_units.code_time).value

    # Isolated live-disk realization in an external NFW halo.
    config = SimulationConfig(
        N_particles=n_particles,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=int(args.num_steps),
        return_snapshots=False,
        external_accelerations=(NFW_POTENTIAL,),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        fmm_preset=str(args.fmm_preset),
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=True,
        fmm_runtime_path=str(args.fmm_runtime_path),
        fmm_refresh_every=int(args.fmm_refresh_every),
        fmm_leaf_size=int(args.fmm_leaf_size),
        fmm_tree_build_mode=str(args.fmm_tree_build_mode),
        fmm_tree_leaf_target=(
            int(args.fmm_leaf_size)
            if args.fmm_tree_build_mode == "static_radix"
            or args.fmm_tree_leaf_target is None
            else int(args.fmm_tree_leaf_target)
        ),
        fmm_max_order=int(args.fmm_max_order),
        fmm_m2l_chunk_size=(
            None if args.fmm_m2l_chunk_size is None else int(args.fmm_m2l_chunk_size)
        ),
        fmm_nearfield_mode="bucketed",
        fmm_nearfield_edge_chunk_size=int(args.fmm_nearfield_edge_chunk_size),
        fmm_large_n_target_block_size=(
            None
            if args.fmm_large_n_target_block_size is None
            else int(args.fmm_large_n_target_block_size)
        ),
        fmm_large_n_static_target_blocks=args.fmm_large_n_static_target_blocks,
        fmm_large_n_static_target_blocks_max_per_leaf=(
            None
            if args.fmm_large_n_static_target_blocks_max_per_leaf is None
            else int(args.fmm_large_n_static_target_blocks_max_per_leaf)
        ),
        fmm_jit_tree=True,
        fmm_jit_traversal=True,
        fmm_prepare_stage_memory_split_enabled=args.fmm_prepare_stage_memory_split_enabled,
        fmm_enforce_static_shape_contract=bool(args.fmm_enforce_static_shape_contract),
        fmm_static_shape_warmup_prepares=int(args.fmm_static_shape_warmup_prepares),
        fmm_rematerialize_between_refresh=not bool(
            args.no_fmm_rematerialize_between_refresh
        ),
    )

    params = SimulationParams(
        G=1.0,
        t_end=t_end,
        NFW_params=NFWParams(
            Mvir=(1.0e12 * u.Msun).to(code_units.code_mass).value,
            r_s=(20.0 * u.kpc).to(code_units.code_length).value,
        ),
    )

    key = jax.random.PRNGKey(int(args.seed))
    pos = sample_exponential_disk(key, n_particles, rd, zd)
    vel = build_quasi_circular_velocities(pos, config, params)

    mass = jnp.full((n_particles,), total_mass / n_particles, dtype=jnp.float32)
    state0 = construct_initial_state(pos.astype(jnp.float32), vel.astype(jnp.float32))

    sampled_positions = None
    sampled_times = None
    sample_indices = None
    history = None

    if args.mode == "perf":
        if args.live or args.movie_path:
            print("[info] perf mode ignores --live/--movie-path. Use --mode render for visualization.")
        if bool(args.conservation_report):
            final_state, history, elapsed, timing_stats = _run_perf_mode_with_history(
                state0,
                mass,
                config,
                params,
                profile_breakdown=bool(args.profile_breakdown),
            )
        else:
            final_state, history, elapsed, timing_stats = _run_perf_mode(
                state0,
                mass,
                config,
                params,
                profile_breakdown=bool(args.profile_breakdown),
            )
    else:
        (
            final_state,
            sampled_positions,
            sampled_times,
            sample_indices,
            elapsed,
            timing_stats,
        ) = _run_render_mode(
            state0,
            mass,
            config,
            params,
            snapshot_stride=int(args.snapshot_stride),
            snapshot_chunk_steps=int(args.snapshot_chunk_steps),
            snapshot_max_particles=int(args.snapshot_max_particles),
            profile_breakdown=bool(args.profile_breakdown),
        )

        render_positions(
            sampled_positions,
            sampled_times,
            projection=str(args.projection),
            live=bool(args.live),
            movie_path=args.movie_path,
            movie_fps=int(args.movie_fps),
        )

    payload = {
        "final_state": np.asarray(final_state),
        "mass": np.asarray(mass),
        "runtime_seconds": np.asarray(elapsed),
        "n_particles": np.asarray(n_particles),
        "num_steps": np.asarray(int(args.num_steps)),
        "mode": np.asarray(args.mode),
    }

    if bool(args.save_snapshots) and sampled_positions is not None:
        payload["snapshot_positions"] = sampled_positions
        payload["snapshot_times"] = sampled_times
        payload["snapshot_particle_indices"] = sample_indices

    np.savez(args.output, **payload)
    print(f"Saved {args.output}")
    print(f"Runtime: {elapsed:.3f} s")

    if bool(args.save_snapshots) and sampled_positions is not None and args.snapshot_output is not None:
        np.savez_compressed(
            args.snapshot_output,
            snapshot_positions=sampled_positions,
            snapshot_times=sampled_times,
            snapshot_particle_indices=sample_indices,
        )
        print(f"Saved snapshots: {args.snapshot_output}")

    if bool(args.profile_breakdown) and timing_stats is not None:
        timing_stats = dict(timing_stats)
        timing_stats.update(
            {
                "script_runtime_seconds": float(elapsed),
                "n_particles": int(n_particles),
                "num_steps": int(args.num_steps),
                "mode": str(args.mode),
                "fmm_preset_requested": str(args.fmm_preset),
                "fmm_runtime_path_requested": str(args.fmm_runtime_path),
                "fmm_refresh_every_requested": int(args.fmm_refresh_every),
                "fmm_leaf_size_requested": int(args.fmm_leaf_size),
                "fmm_tree_build_mode_requested": str(args.fmm_tree_build_mode),
                "fmm_tree_leaf_target_requested": (
                    int(args.fmm_leaf_size)
                    if args.fmm_tree_leaf_target is None
                    else int(args.fmm_tree_leaf_target)
                ),
                "fmm_max_order_requested": int(args.fmm_max_order),
                "fmm_m2l_chunk_size_requested": (
                    None
                    if args.fmm_m2l_chunk_size is None
                    else int(args.fmm_m2l_chunk_size)
                ),
                "fmm_nearfield_edge_chunk_size_requested": int(
                    args.fmm_nearfield_edge_chunk_size
                ),
                "fmm_large_n_target_block_size_requested": (
                    None
                    if args.fmm_large_n_target_block_size is None
                    else int(args.fmm_large_n_target_block_size)
                ),
                "fmm_large_n_static_target_blocks_requested": (
                    args.fmm_large_n_static_target_blocks
                ),
                "fmm_large_n_static_target_blocks_max_per_leaf_requested": (
                    None
                    if args.fmm_large_n_static_target_blocks_max_per_leaf is None
                    else int(args.fmm_large_n_static_target_blocks_max_per_leaf)
                ),
                "fmm_large_n_effective_environment_overrides": (
                    _large_n_environment_overrides(
                        config,
                        fmm_preset=str(args.fmm_preset),
                    )
                ),
                "fmm_prepare_stage_memory_split_enabled": (
                    config.fmm_prepare_stage_memory_split_enabled
                ),
                "fmm_enforce_static_shape_contract": bool(
                    args.fmm_enforce_static_shape_contract
                ),
                "fmm_static_shape_warmup_prepares": int(
                    args.fmm_static_shape_warmup_prepares
                ),
                "fmm_rematerialize_between_refresh": bool(
                    not args.no_fmm_rematerialize_between_refresh
                ),
                "conservation_report_enabled": bool(args.conservation_report),
                "used_visualization": bool(args.mode == "render" and (args.live or args.movie_path is not None)),
                "output_file": str(args.output),
            }
        )
        _check_timing_gates(args, timing_stats)
        json_path, csv_path = _write_timing_report(
            report_dir=str(args.report_dir),
            timing_stats=timing_stats,
        )
        print(f"Saved timing report JSON: {json_path}")
        print(f"Saved timing report CSV : {csv_path}")

    if bool(args.mode == "perf" and args.conservation_report):
        if history is None:
            raise RuntimeError("conservation report requires integration history in perf mode")

        states = np.concatenate((np.asarray(state0)[None, ...], np.asarray(history)), axis=0)
        conservation_stats = _compute_conservation_metrics(
            states,
            mass,
            config,
            params,
            step_stride=int(args.conservation_stride),
        )
        conservation_stats.update(
            {
                "script_runtime_seconds": float(elapsed),
                "n_particles": int(n_particles),
                "num_steps": int(args.num_steps),
                "fmm_preset_requested": str(args.fmm_preset),
                "fmm_runtime_path_requested": str(args.fmm_runtime_path),
                "fmm_refresh_every_requested": int(args.fmm_refresh_every),
                "fmm_leaf_size_requested": int(args.fmm_leaf_size),
                "fmm_tree_build_mode_requested": str(args.fmm_tree_build_mode),
                "fmm_tree_leaf_target_requested": (
                    int(args.fmm_leaf_size)
                    if args.fmm_tree_leaf_target is None
                    else int(args.fmm_tree_leaf_target)
                ),
                "fmm_max_order_requested": int(args.fmm_max_order),
                "fmm_m2l_chunk_size_requested": (
                    None
                    if args.fmm_m2l_chunk_size is None
                    else int(args.fmm_m2l_chunk_size)
                ),
                "fmm_nearfield_edge_chunk_size_requested": int(
                    args.fmm_nearfield_edge_chunk_size
                ),
                "fmm_large_n_target_block_size_requested": (
                    None
                    if args.fmm_large_n_target_block_size is None
                    else int(args.fmm_large_n_target_block_size)
                ),
                "fmm_large_n_static_target_blocks_requested": (
                    args.fmm_large_n_static_target_blocks
                ),
                "fmm_large_n_static_target_blocks_max_per_leaf_requested": (
                    None
                    if args.fmm_large_n_static_target_blocks_max_per_leaf is None
                    else int(args.fmm_large_n_static_target_blocks_max_per_leaf)
                ),
                "fmm_large_n_effective_environment_overrides": (
                    _large_n_environment_overrides(
                        config,
                        fmm_preset=str(args.fmm_preset),
                    )
                ),
                "fmm_prepare_stage_memory_split_enabled": (
                    config.fmm_prepare_stage_memory_split_enabled
                ),
                "fmm_enforce_static_shape_contract": bool(
                    args.fmm_enforce_static_shape_contract
                ),
                "fmm_static_shape_warmup_prepares": int(
                    args.fmm_static_shape_warmup_prepares
                ),
                "fmm_rematerialize_between_refresh": bool(
                    not args.no_fmm_rematerialize_between_refresh
                ),
                "output_file": str(args.output),
            }
        )

        if args.max_abs_de_over_e0 is not None and float(
            conservation_stats["max_abs_dE_over_E0"]
        ) > float(args.max_abs_de_over_e0):
            raise RuntimeError(
                "Conservation regression: "
                f"max_abs_dE_over_E0={conservation_stats['max_abs_dE_over_E0']:.6g} "
                f"> threshold={float(args.max_abs_de_over_e0):.6g}"
            )
        if args.max_abs_dl_over_l0 is not None and float(
            conservation_stats["max_abs_dL_over_L0"]
        ) > float(args.max_abs_dl_over_l0):
            raise RuntimeError(
                "Conservation regression: "
                f"max_abs_dL_over_L0={conservation_stats['max_abs_dL_over_L0']:.6g} "
                f"> threshold={float(args.max_abs_dl_over_l0):.6g}"
            )
        if args.max_com_drift is not None and float(
            conservation_stats["max_com_drift"]
        ) > float(args.max_com_drift):
            raise RuntimeError(
                "Conservation regression: "
                f"max_com_drift={conservation_stats['max_com_drift']:.6g} "
                f"> threshold={float(args.max_com_drift):.6g}"
            )
        c_json_path, c_csv_path = _write_conservation_report(
            report_dir=str(args.report_dir),
            conservation_stats=conservation_stats,
        )
        print(f"Saved conservation report JSON: {c_json_path}")
        print(f"Saved conservation report CSV : {c_csv_path}")

    if args.max_runtime_seconds is not None and float(elapsed) > float(
        args.max_runtime_seconds
    ):
        raise RuntimeError(
            f"Runtime regression: elapsed={float(elapsed):.6g}s "
            f"> threshold={float(args.max_runtime_seconds):.6g}s"
        )


if __name__ == "__main__":
    main()
