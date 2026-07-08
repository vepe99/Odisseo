"""Galaxy-disk simulation using ODISSEO + jaccpot large-N radix path."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import pathlib
import os
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
    _temporary_large_n_environment,
    integrate_diffrax_jaccpot_active,
    integrate_leapfrog_jaccpot_active,
)
from odisseo.render_callback import (
    FrameSink,
    PositionSink,
    make_density_step_callback,
    make_position_step_callback,
)
from odisseo.option_classes import (
    FMM_ACC,
    NFW_POTENTIAL,
    THICK_MN3_DISK,
    THIN_MN3_DISK,
    SimulationConfig,
    SimulationParams,
    NFWParams,
    ThickMN3DiskParams,
    ThinMN3DiskParams,
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
        "--state-dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Dtype used for ODISSEO state/mass arrays. FMM may still use its configured working dtype.",
    )
    parser.add_argument(
        "--initial-accel-report",
        action="store_true",
        help="Write sampled initial acceleration diagnostics before integration.",
    )
    parser.add_argument(
        "--initial-accel-sample-targets",
        type=int,
        default=512,
        help="Number of deterministic particles used for initial acceleration diagnostics.",
    )
    parser.add_argument(
        "--initial-accel-direct-source-chunk",
        type=int,
        default=8192,
        help="Source chunk size for sampled direct-sum initial acceleration diagnostics.",
    )
    parser.add_argument(
        "--ic-velocity-potential",
        type=str,
        default="nfw",
        choices=("nfw", "nfw_analytic_disk"),
        help=(
            "Potential used only to initialize quasi-circular velocities. "
            "The integration still uses the runtime external potential plus live self-gravity."
        ),
    )
    parser.add_argument(
        "--ic-analytic-disk-mass-factor",
        type=float,
        default=1.0,
        help="Analytic disk mass used for IC velocities as a factor of the live disk mass.",
    )
    parser.add_argument(
        "--ic-thick-disk-mass-fraction",
        type=float,
        default=0.0,
        help="Fraction of the analytic IC disk mass assigned to the thick MN3 component.",
    )
    parser.add_argument(
        "--ic-thin-disk-radius-kpc",
        type=float,
        default=None,
        help="Thin MN3 radial scale for IC velocities. Defaults to --disk-radius-kpc.",
    )
    parser.add_argument(
        "--ic-thin-disk-height-kpc",
        type=float,
        default=None,
        help="Thin MN3 vertical scale for IC velocities. Defaults to --disk-height-kpc.",
    )
    parser.add_argument(
        "--ic-thick-disk-radius-kpc",
        type=float,
        default=None,
        help="Thick MN3 radial scale for IC velocities. Defaults to --disk-radius-kpc.",
    )
    parser.add_argument(
        "--ic-thick-disk-height-kpc",
        type=float,
        default=None,
        help="Thick MN3 vertical scale for IC velocities. Defaults to 3.333 * --disk-height-kpc.",
    )
    parser.add_argument(
        "--ic-source",
        type=str,
        default="generate",
        choices=("generate", "load"),
        help="Generate ICs from seed/potential or load fixed ICs from --ic-input-path.",
    )
    parser.add_argument(
        "--ic-input-path",
        type=str,
        default=None,
        help="Input NPZ path used when --ic-source=load.",
    )
    parser.add_argument(
        "--ic-output-path",
        type=str,
        default=None,
        help="Optional NPZ path to persist generated ICs for later reuse.",
    )
    parser.add_argument(
        "--ic-require-runtime-potential-match",
        action="store_true",
        help="Require IC velocity potential to match runtime external potential setup (default on).",
    )
    parser.add_argument(
        "--no-ic-require-runtime-potential-match",
        dest="ic_require_runtime_potential_match",
        action="store_false",
        help="Allow IC velocity potential to differ from runtime external potential.",
    )
    parser.set_defaults(ic_require_runtime_potential_match=True)
    parser.add_argument(
        "--fmm-refresh-every",
        type=int,
        default=1,
        help="Refresh source-tree/self-field every k steps.",
    )
    parser.add_argument(
        "--adaptive-timestep",
        action="store_true",
        help="Use adaptive diffrax timesteps for FMM integration (fixed timestep remains default).",
    )
    parser.add_argument(
        "--fmm-adaptive-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for adaptive diffrax FMM mode.",
    )
    parser.add_argument(
        "--fmm-adaptive-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for adaptive diffrax FMM mode.",
    )
    parser.add_argument(
        "--fmm-adaptive-max-dt",
        type=float,
        default=None,
        help="Optional maximum dt (code time) for adaptive diffrax FMM mode.",
    )
    parser.add_argument(
        "--fmm-adaptive-min-dt",
        type=float,
        default=None,
        help="Optional minimum dt (code time) for adaptive diffrax FMM mode.",
    )
    parser.add_argument(
        "--fmm-adaptive-refresh-rhs-calls",
        type=int,
        default=1,
        help="Adaptive cadence: refresh prepared state every k RHS calls (python cache mode).",
    )
    parser.add_argument(
        "--fmm-adaptive-refresh-displacement-threshold",
        type=float,
        default=None,
        help="Adaptive cadence: refresh when max displacement since last refresh reaches this threshold.",
    )
    parser.add_argument(
        "--adaptive-prepared-cache-mode",
        type=str,
        default="none",
        choices=("none", "python"),
        help="Adaptive prepared-state cache mode for diffrax core scaffold.",
    )
    parser.add_argument(
        "--fmm-preset",
        type=str,
        default="large_n_gpu",
        choices=("fast", "balanced", "accurate", "large_n_gpu"),
        help="FMM preset passed to jaccpot/ODISSEO coupling.",
    )
    parser.add_argument(
        "--fmm-runtime-path",
        type=str,
        default="large_n",
        choices=("auto", "legacy", "large_n"),
        help="jaccpot runtime path selection.",
    )
    parser.add_argument(
        "--fmm-theta",
        type=float,
        default=0.8,
        help="FMM MAC opening angle theta.",
    )
    parser.add_argument(
        "--fmm-mac-type",
        type=str,
        default="dehnen",
        choices=("geometric", "dehnen", "dehnen_error"),
        help="FMM multipole acceptance criterion.",
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
        default="static_radix",
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
        "--no-fmm-large-n-environment-overrides",
        action="store_true",
        help="Disable ODISSEO's temporary JACCPOT_LARGE_N_* environment overrides.",
    )
    parser.add_argument(
        "--fmm-m2l-chunk-size",
        type=int,
        default=None,
        help="Optional far-field M2L chunk size override.",
    )
    parser.add_argument(
        "--fmm-max-pair-queue",
        type=int,
        default=None,
        help="Optional explicit jaccpot traversal pair-queue capacity.",
    )
    parser.add_argument(
        "--fmm-pair-process-block",
        type=int,
        default=None,
        help="Optional jaccpot traversal pair processing block size.",
    )
    parser.add_argument(
        "--fmm-max-interactions-per-node",
        type=int,
        default=None,
        help="Optional jaccpot traversal interaction-list capacity per node.",
    )
    parser.add_argument(
        "--fmm-max-neighbors-per-leaf",
        type=int,
        default=None,
        help="Optional jaccpot traversal neighbor-list capacity per leaf.",
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
        "--perf-warmup-runs",
        type=int,
        default=0,
        help="Number of full perf runs to execute before measured timing (compile/warmup excluded).",
    )
    parser.add_argument(
        "--perf-measure-runs",
        type=int,
        default=1,
        help="Number of measured full perf runs; report the median as runtime_seconds.",
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
        "--fmm-upward-leaf-batch-size",
        type=int,
        default=None,
        help="Optional jaccpot upward sweep leaf batch size override.",
    )

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
        "--require-finite-final-state",
        action="store_true",
        help="Optional regression gate: fail if final_state contains NaN or Inf values.",
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
        "--min-adaptive-cadence-skips-rhs-calls",
        type=int,
        default=None,
        help="Optional regression gate for adaptive cadence RHS-call skip count.",
    )
    parser.add_argument(
        "--min-adaptive-cadence-skips-displacement",
        type=int,
        default=None,
        help="Optional regression gate for adaptive cadence displacement skip count.",
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

    # Render controls (mode=render uses the fused callback path).
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
        help="Projection used for movie rendering.",
    )
    parser.add_argument(
        "--render-resolution",
        type=int,
        default=900,
        help="Square pixel resolution for density-rendered movie frames.",
    )
    parser.add_argument(
        "--render-stride",
        type=int,
        default=10,
        help="mode=render: emit one frame every N steps from inside the "
        "fused device-resident scan (via jax.debug.callback).",
    )
    parser.add_argument(
        "--render-snapshot-output",
        type=str,
        default=None,
        help="mode=render: optional NPZ to save the streamed subsampled particle "
        "positions (snapshot_positions [T,Ns,3], snapshot_times [T]) for analysis "
        "/ ring scoring (e.g. the AGAMA IC sweep).",
    )
    parser.add_argument(
        "--render-cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap used by the density renderer.",
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


def build_ic_velocity_potential(
    args: argparse.Namespace,
    code_units: CodeUnits,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    total_mass: float,
) -> tuple[SimulationConfig, SimulationParams, dict[str, object]]:
    """Return the potential used only for quasi-circular IC velocities."""
    mode = str(args.ic_velocity_potential).strip().lower()
    if mode == "nfw":
        return config, params, {
            "ic_velocity_potential": mode,
            "ic_uses_analytic_disk": False,
        }

    analytic_mass = float(total_mass) * max(0.0, float(args.ic_analytic_disk_mass_factor))
    thick_fraction = float(np.clip(float(args.ic_thick_disk_mass_fraction), 0.0, 1.0))
    thin_mass = analytic_mass * (1.0 - thick_fraction)
    thick_mass = analytic_mass * thick_fraction

    thin_radius = (
        float(args.disk_radius_kpc)
        if args.ic_thin_disk_radius_kpc is None
        else float(args.ic_thin_disk_radius_kpc)
    )
    thin_height = (
        float(args.disk_height_kpc)
        if args.ic_thin_disk_height_kpc is None
        else float(args.ic_thin_disk_height_kpc)
    )
    thick_radius = (
        float(args.disk_radius_kpc)
        if args.ic_thick_disk_radius_kpc is None
        else float(args.ic_thick_disk_radius_kpc)
    )
    thick_height = (
        3.3333333333333335 * float(args.disk_height_kpc)
        if args.ic_thick_disk_height_kpc is None
        else float(args.ic_thick_disk_height_kpc)
    )

    external = [NFW_POTENTIAL, THIN_MN3_DISK]
    if thick_mass > 0.0:
        external.append(THICK_MN3_DISK)

    ic_config = config._replace(external_accelerations=tuple(external))
    ic_params = params._replace(
        ThinMN3Disk_params=ThinMN3DiskParams(
            M=thin_mass,
            hr=(thin_radius * u.kpc).to(code_units.code_length).value,
            hz=(thin_height * u.kpc).to(code_units.code_length).value,
        ),
        ThickMN3Disk_params=ThickMN3DiskParams(
            M=thick_mass,
            hr=(thick_radius * u.kpc).to(code_units.code_length).value,
            hz=(thick_height * u.kpc).to(code_units.code_length).value,
        ),
    )
    return ic_config, ic_params, {
        "ic_velocity_potential": mode,
        "ic_uses_analytic_disk": True,
        "ic_analytic_disk_mass_factor": float(args.ic_analytic_disk_mass_factor),
        "ic_thick_disk_mass_fraction": thick_fraction,
        "ic_thin_disk_mass_code": float(thin_mass),
        "ic_thick_disk_mass_code": float(thick_mass),
        "ic_thin_disk_radius_kpc": thin_radius,
        "ic_thin_disk_height_kpc": thin_height,
        "ic_thick_disk_radius_kpc": thick_radius,
        "ic_thick_disk_height_kpc": thick_height,
    }


def _save_ic_file(
    path: str,
    *,
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    seed: int,
    n_particles: int,
    ic_velocity_metadata: dict[str, object],
) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state0": np.asarray(state0),
        "mass": np.asarray(mass),
        "seed": np.asarray(int(seed)),
        "n_particles": np.asarray(int(n_particles)),
        "state_dtype": np.asarray(str(state0.dtype)),
        "mass_dtype": np.asarray(str(mass.dtype)),
        "ic_velocity_potential": np.asarray(str(ic_velocity_metadata["ic_velocity_potential"])),
        "ic_uses_analytic_disk": np.asarray(bool(ic_velocity_metadata["ic_uses_analytic_disk"])),
    }
    np.savez_compressed(out, **payload)
    print(f"Saved IC file: {out}")


def _load_ic_file(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _validate_ic_source_config(args: argparse.Namespace) -> None:
    if str(args.ic_source) == "load" and not args.ic_input_path:
        raise ValueError("--ic-input-path is required when --ic-source=load")


def _summarize_finite(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"finite_count": 0}
    return {
        "finite_count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "p999": float(np.percentile(finite, 99.9)),
        "max": float(np.max(finite)),
    }


def _direct_self_acceleration_targets(
    positions: jnp.ndarray,
    mass: jnp.ndarray,
    target_indices: np.ndarray,
    *,
    softening: float,
    source_chunk: int,
) -> np.ndarray:
    target_idx = jnp.asarray(target_indices, dtype=jnp.int32)
    targets = positions[target_idx]
    out = jnp.zeros_like(targets)
    softening_sq = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    n = int(positions.shape[0])
    chunk = max(1, int(source_chunk))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        src = positions[start:stop]
        src_mass = mass[start:stop]
        diff = targets[:, None, :] - src[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
        inv_dist3 = jnp.reciprocal(dist_sq * jnp.sqrt(dist_sq))
        src_idx = jnp.arange(start, stop, dtype=jnp.int32)
        self_mask = src_idx[None, :] == target_idx[:, None]
        inv_dist3 = jnp.where(self_mask, 0.0, inv_dist3)
        out = out - jnp.sum(
            diff * inv_dist3[:, :, None] * src_mass[None, :, None],
            axis=1,
        )
        out = jax.block_until_ready(out)
    return np.asarray(out)


def _compute_initial_acceleration_diagnostics(
    *,
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    args: argparse.Namespace,
) -> dict[str, object]:
    n = int(state0.shape[0])
    n_targets = min(n, max(1, int(args.initial_accel_sample_targets)))
    stride = max(1, n // n_targets)
    target_indices = np.arange(0, n, stride, dtype=np.int32)[:n_targets]
    target_idx_jax = jnp.asarray(target_indices, dtype=jnp.int32)

    pos = state0[:, 0, :]
    vel = state0[:, 1, :]
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
        fmm_max_pair_queue=config.fmm_max_pair_queue,
        fmm_pair_process_block=config.fmm_pair_process_block,
        fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=(
            config.fmm_prepare_stage_memory_split_enabled
        ),
        fmm_upward_leaf_batch_size=config.fmm_upward_leaf_batch_size,
    )
    with _temporary_large_n_environment(config, fmm_preset=fmm_preset):
        prepared = solver.prepare_state(
            pos,
            mass,
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
        )
    acc_fmm_full = solver.evaluate_prepared_state(
        prepared,
        target_indices=None,
        return_potential=False,
    )
    acc_fmm = np.asarray(jax.block_until_ready(acc_fmm_full[target_idx_jax]))
    acc_direct = _direct_self_acceleration_targets(
        pos,
        mass,
        target_indices,
        softening=float(config.softening),
        source_chunk=int(args.initial_accel_direct_source_chunk),
    )
    acc_ext = np.asarray(
        combined_external_acceleration_vmpa_switch(state0, config, params)
    )[target_indices]

    pos_t = np.asarray(pos)[target_indices]
    vel_t = np.asarray(vel)[target_indices]
    radius = np.linalg.norm(pos_t[:, :2], axis=1)
    e_r = pos_t[:, :2] / np.maximum(radius[:, None], 1.0e-30)
    centripetal = np.sum(vel_t[:, :2] * vel_t[:, :2], axis=1) / np.maximum(radius, 1.0e-30)
    radial_direct_total = np.sum((acc_direct + acc_ext)[:, :2] * e_r, axis=1)
    radial_fmm_total = np.sum((acc_fmm + acc_ext)[:, :2] * e_r, axis=1)

    err = np.linalg.norm(acc_fmm - acc_direct, axis=1)
    direct_norm = np.linalg.norm(acc_direct, axis=1)
    rel_err = err / np.maximum(direct_norm, 1.0e-30)
    worst = np.argsort(rel_err)[-10:][::-1]

    return {
        "sampled_target_count": int(target_indices.shape[0]),
        "sampled_target_stride": int(stride),
        "state_dtype": str(state0.dtype),
        "mass_dtype": str(mass.dtype),
        "fmm_working_dtype": str(jnp.dtype(fmm_dtype)),
        "fmm_preset_resolved": str(fmm_preset),
        "fmm_runtime_path_resolved": str(fmm_runtime_path),
        "softening_code": float(config.softening),
        "radius_code": _summarize_finite(radius),
        "external_acc_norm": _summarize_finite(np.linalg.norm(acc_ext, axis=1)),
        "direct_self_acc_norm": _summarize_finite(direct_norm),
        "fmm_self_acc_norm": _summarize_finite(np.linalg.norm(acc_fmm, axis=1)),
        "fmm_vs_direct_abs_err": _summarize_finite(err),
        "fmm_vs_direct_rel_err": _summarize_finite(rel_err),
        "circular_residual_direct_self_abs": _summarize_finite(
            np.abs(centripetal + radial_direct_total)
        ),
        "circular_residual_fmm_self_abs": _summarize_finite(
            np.abs(centripetal + radial_fmm_total)
        ),
        "worst_fmm_vs_direct_rel_err": [
            {
                "target_index": int(target_indices[i]),
                "radius_code": float(radius[i]),
                "rel_err": float(rel_err[i]),
                "abs_err": float(err[i]),
                "direct_acc_norm": float(direct_norm[i]),
                "fmm_acc_norm": float(np.linalg.norm(acc_fmm[i])),
                "external_acc_norm": float(np.linalg.norm(acc_ext[i])),
            }
            for i in worst
        ],
    }


def _write_initial_acceleration_report(
    *,
    report_dir: str,
    accel_stats: dict,
) -> pathlib.Path:
    out_dir = pathlib.Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"galaxy_disk_initial_acceleration_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(accel_stats, f, indent=2)
    return json_path


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


def _run_perf_mode_once(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    profile_breakdown: bool,
    warmup_runs: int = 0,
    measure_runs: int = 1,
) -> tuple[jnp.ndarray, np.ndarray | None, dict | None]:
    timing_stats = {} if bool(profile_breakdown) else None
    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)

    if bool(profile_breakdown):
        if bool(config.fixed_timestep):
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
                    fmm_max_pair_queue=config.fmm_max_pair_queue,
                    fmm_pair_process_block=config.fmm_pair_process_block,
                    fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
                    fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
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
                    perf_warmup_runs=int(warmup_runs),
                    perf_measure_runs=int(measure_runs),
                    timing_stats=timing_stats,
                )
            )
        else:
            final_state = jax.block_until_ready(
                integrate_diffrax_jaccpot_active(
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
                    fmm_max_pair_queue=config.fmm_max_pair_queue,
                    fmm_pair_process_block=config.fmm_pair_process_block,
                    fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
                    fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
                    fmm_prepare_stage_memory_split_enabled=(
                        config.fmm_prepare_stage_memory_split_enabled
                    ),
                    rematerialize_between_refresh=bool(
                        config.fmm_rematerialize_between_refresh
                    ),
                    return_history=False,
                    timing_stats=timing_stats,
                )
            )
    else:
        final_state = jax.block_until_ready(integrate(state0, mass, config, params))

    return final_state, None, timing_stats


def _run_perf_mode(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    profile_breakdown: bool,
    warmup_runs: int = 0,
    measure_runs: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, float, dict | None]:
    t0 = time.perf_counter()
    final_state_jax, history, timing_stats = _run_perf_mode_once(
        state0,
        mass,
        config,
        params,
        profile_breakdown=bool(profile_breakdown),
        warmup_runs=max(0, int(warmup_runs)),
        measure_runs=max(1, int(measure_runs)),
    )
    total_elapsed = float(time.perf_counter() - t0)
    elapsed = total_elapsed
    if timing_stats is not None and "perf_measured_median_seconds" in timing_stats:
        elapsed = float(timing_stats["perf_measured_median_seconds"])
        timing_stats["perf_total_wall_seconds_including_warmup"] = float(total_elapsed)
    return np.asarray(final_state_jax), history, float(elapsed), timing_stats


_PROJECTION_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_AXIS_NAMES = ("x", "y", "z")


def _run_render_mode(
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
    *,
    res: int,
    stride: int,
    projection: str,
    cmap: str,
    out_path: str,
    fps: int,
    snapshot_output: str | None = None,
    length_kpc: float = 10.0,
    time_per_step_gyr: float | None = None,
) -> tuple[np.ndarray, float, int]:
    """Render on the high-performance fused lane via a jax.debug.callback hook.

    Runs the fused device-resident velocity-Verlet scan (``return_history=False``)
    and streams frames out with ``step_callback``: the 2D density projection is
    computed on-device and only that small grid crosses to the host every
    ``stride`` steps (fire-and-forget), so the GPU is not stalled. Frames are
    encoded to a movie after ``block_until_ready`` flushes the callbacks.
    """
    fmm_preset, fmm_runtime_path, fmm_dtype = _resolve_fmm_profile(config)
    axes = _PROJECTION_AXES.get(str(projection).lower(), (0, 1))
    pos0 = state0[:, 0, :]
    bmin = np.asarray(jax.device_get(jnp.min(pos0, axis=0)), dtype=np.float64)
    bmax = np.asarray(jax.device_get(jnp.max(pos0, axis=0)), dtype=np.float64)
    # pad by 2% so edge particles aren't clipped as the system evolves
    pad = 0.02 * (bmax - bmin + 1e-6)
    sink = FrameSink()
    density_cb = make_density_step_callback(
        sink, bmin - pad, bmax + pad, res=int(res), axes=axes
    )
    # Also stream subsampled particle positions for a low-noise (particle-based)
    # ring metric -- a sparse density grid is shot-noise-limited. One combined
    # fire-and-forget callback keeps it single-hook and minimal-sync.
    n_all = int(pos0.shape[0])
    n_samp = min(n_all, 20000)
    sample_idx = np.linspace(0, n_all - 1, n_samp, dtype=np.int64)
    psink = PositionSink()
    position_cb = make_position_step_callback(psink, sample_idx)

    def step_cb(step_index, state):
        density_cb(step_index, state)
        position_cb(step_index, state)

    t0 = time.perf_counter()
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
            fmm_max_pair_queue=config.fmm_max_pair_queue,
            fmm_pair_process_block=config.fmm_pair_process_block,
            fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
            fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
            fmm_prepare_stage_memory_split_enabled=(
                config.fmm_prepare_stage_memory_split_enabled
            ),
            rematerialize_between_refresh=bool(
                config.fmm_rematerialize_between_refresh
            ),
            return_history=False,
            step_callback=step_cb,
            step_callback_stride=int(stride),
        )
    )
    elapsed = float(time.perf_counter() - t0)
    n_frames = len(sink.frames)
    if len(psink.positions) >= 2:
        ring = psink.ring_metric()
        print(
            f"[ring] rms_start={ring['ring_rms_start']:.4f} "
            f"rms_end={ring['ring_rms_end']:.4f} "
            f"growth={ring['ring_rms_end'] / max(ring['ring_rms_start'], 1e-9):.2f}x "
            f"r99_growth={ring['r99_growth']:.3f}"
        )
    if snapshot_output and len(psink.positions) >= 1:
        steps_arr, pos_arr = psink.stack()  # pos_arr: [T, Ns, 3]
        dt_gyr = (
            float(params.t_end) / float(config.num_timesteps)
            if int(config.num_timesteps) > 0
            else 1.0
        )
        np.savez_compressed(
            snapshot_output,
            snapshot_positions=pos_arr.astype(np.float32),
            snapshot_times=(steps_arr.astype(np.float64) * dt_gyr),
        )
    if n_frames:
        # Scientific frames: kpc axes, colorbar, per-frame time stamp.
        ax0, ax1 = axes
        extent_kpc = [
            float((bmin[ax0] - pad[ax0]) * length_kpc),
            float((bmax[ax0] + pad[ax0]) * length_kpc),
            float((bmin[ax1] - pad[ax1]) * length_kpc),
            float((bmax[ax1] + pad[ax1]) * length_kpc),
        ]
        dt_step_gyr = time_per_step_gyr
        if dt_step_gyr is None:
            dt_step_gyr = (
                float(params.t_end) / float(config.num_timesteps)
                if int(config.num_timesteps) > 0
                else None
            )
        sink.encode(
            out_path,
            fps=int(fps),
            cmap=str(cmap),
            extent=extent_kpc,
            xlabel=f"{_AXIS_NAMES[ax0]} [kpc]",
            ylabel=f"{_AXIS_NAMES[ax1]} [kpc]",
            dt_time=dt_step_gyr,
            time_label="Gyr",
        )
    return np.asarray(final_state), elapsed, n_frames


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

    if bool(config.fixed_timestep):
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
                fmm_max_pair_queue=config.fmm_max_pair_queue,
                fmm_pair_process_block=config.fmm_pair_process_block,
                fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
                fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
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
    else:
        history = jax.block_until_ready(
            integrate_diffrax_jaccpot_active(
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
                fmm_max_pair_queue=config.fmm_max_pair_queue,
                fmm_pair_process_block=config.fmm_pair_process_block,
                fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
                fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
                fmm_prepare_stage_memory_split_enabled=(
                    config.fmm_prepare_stage_memory_split_enabled
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
        fmm_max_pair_queue=config.fmm_max_pair_queue,
        fmm_pair_process_block=config.fmm_pair_process_block,
        fmm_max_interactions_per_node=config.fmm_max_interactions_per_node,
        fmm_max_neighbors_per_leaf=config.fmm_max_neighbors_per_leaf,
        fmm_prepare_stage_memory_split_enabled=(
            config.fmm_prepare_stage_memory_split_enabled
        ),
        fmm_upward_leaf_batch_size=config.fmm_upward_leaf_batch_size,
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


def _finite_norm_stats(prefix: str, values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {
            f"{prefix}_finite_count": 0,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
        }
    norms = np.linalg.norm(arr, axis=-1)
    finite = np.isfinite(norms)
    finite_norms = norms[finite]
    if finite_norms.size == 0:
        return {
            f"{prefix}_finite_count": 0,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_finite_count": int(finite_norms.size),
        f"{prefix}_p50": float(np.percentile(finite_norms, 50.0)),
        f"{prefix}_p90": float(np.percentile(finite_norms, 90.0)),
        f"{prefix}_p99": float(np.percentile(finite_norms, 99.0)),
        f"{prefix}_max": float(np.max(finite_norms)),
    }


def _final_state_finite_stats(final_state: np.ndarray) -> dict[str, object]:
    arr = np.asarray(final_state)
    finite = np.isfinite(arr)
    stats: dict[str, object] = {
        "final_state_shape": list(arr.shape),
        "final_state_element_count": int(arr.size),
        "final_state_finite_count": int(np.count_nonzero(finite)),
        "final_state_nan_count": int(np.count_nonzero(np.isnan(arr))),
        "final_state_inf_count": int(np.count_nonzero(np.isinf(arr))),
        "final_state_all_finite": bool(np.all(finite)),
    }
    if arr.ndim >= 3 and arr.shape[1] >= 2:
        stats.update(_finite_norm_stats("final_state_position_norm", arr[:, 0, :]))
        stats.update(_finite_norm_stats("final_state_velocity_norm", arr[:, 1, :]))
    return stats


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
    if bool(getattr(args, "require_finite_final_state", False)) and not bool(
        timing_stats.get("final_state_all_finite", False)
    ):
        raise RuntimeError(
            "Final-state regression: final_state contains non-finite values "
            f"(finite={timing_stats.get('final_state_finite_count', 'unknown')}/"
            f"{timing_stats.get('final_state_element_count', 'unknown')}, "
            f"nan={timing_stats.get('final_state_nan_count', 'unknown')}, "
            f"inf={timing_stats.get('final_state_inf_count', 'unknown')})."
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

    if args.min_adaptive_cadence_skips_rhs_calls is not None:
        value = int(timing_stats.get("adaptive_core_refresh_cadence_skips_rhs_calls", 0))
        threshold = int(args.min_adaptive_cadence_skips_rhs_calls)
        if value < threshold:
            raise RuntimeError(
                "Adaptive cadence regression (rhs-calls gate): "
                f"adaptive_core_refresh_cadence_skips_rhs_calls={value} < threshold={threshold}"
            )

    if args.min_adaptive_cadence_skips_displacement is not None:
        value = int(timing_stats.get("adaptive_core_refresh_cadence_skips_displacement", 0))
        threshold = int(args.min_adaptive_cadence_skips_displacement)
        if value < threshold:
            raise RuntimeError(
                "Adaptive cadence regression (displacement gate): "
                f"adaptive_core_refresh_cadence_skips_displacement={value} < threshold={threshold}"
            )


def main() -> None:
    args = parse_args()
    _validate_ic_source_config(args)
    os.environ["ODISSEO_FMM_ADAPTIVE_PREPARED_CACHE_MODE"] = str(
        args.adaptive_prepared_cache_mode
    )

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
        fixed_timestep=not bool(args.adaptive_timestep),
        num_timesteps=int(args.num_steps),
        return_snapshots=False,
        external_accelerations=(NFW_POTENTIAL,),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        fmm_preset=str(args.fmm_preset),
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=True,
        fmm_runtime_path=str(args.fmm_runtime_path),
        fmm_theta=float(args.fmm_theta),
        fmm_mac_type=str(args.fmm_mac_type),
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
        fmm_large_n_environment_overrides_enabled=not bool(
            args.no_fmm_large_n_environment_overrides
        ),
        fmm_jit_tree=True,
        fmm_jit_traversal=True,
        fmm_max_pair_queue=(
            None if args.fmm_max_pair_queue is None else int(args.fmm_max_pair_queue)
        ),
        fmm_pair_process_block=(
            None
            if args.fmm_pair_process_block is None
            else int(args.fmm_pair_process_block)
        ),
        fmm_max_interactions_per_node=(
            None
            if args.fmm_max_interactions_per_node is None
            else int(args.fmm_max_interactions_per_node)
        ),
        fmm_max_neighbors_per_leaf=(
            None
            if args.fmm_max_neighbors_per_leaf is None
            else int(args.fmm_max_neighbors_per_leaf)
        ),
        fmm_prepare_stage_memory_split_enabled=args.fmm_prepare_stage_memory_split_enabled,
        fmm_upward_leaf_batch_size=(
            None
            if args.fmm_upward_leaf_batch_size is None
            else int(args.fmm_upward_leaf_batch_size)
        ),
        fmm_enforce_static_shape_contract=bool(args.fmm_enforce_static_shape_contract),
        fmm_static_shape_warmup_prepares=int(args.fmm_static_shape_warmup_prepares),
        fmm_rematerialize_between_refresh=not bool(
            args.no_fmm_rematerialize_between_refresh
        ),
        fmm_adaptive_rtol=float(args.fmm_adaptive_rtol),
        fmm_adaptive_atol=float(args.fmm_adaptive_atol),
        fmm_adaptive_max_dt=(
            None if args.fmm_adaptive_max_dt is None else float(args.fmm_adaptive_max_dt)
        ),
        fmm_adaptive_min_dt=(
            None if args.fmm_adaptive_min_dt is None else float(args.fmm_adaptive_min_dt)
        ),
        fmm_adaptive_refresh_rhs_calls=max(1, int(args.fmm_adaptive_refresh_rhs_calls)),
        fmm_adaptive_refresh_displacement_threshold=(
            None
            if args.fmm_adaptive_refresh_displacement_threshold is None
            else float(args.fmm_adaptive_refresh_displacement_threshold)
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

    state_dtype = jnp.float64 if str(args.state_dtype) == "float64" else jnp.float32
    mass = jnp.full((n_particles,), total_mass / n_particles, dtype=state_dtype)

    if bool(args.ic_require_runtime_potential_match) and str(args.ic_velocity_potential) != "nfw":
        raise ValueError(
            "Runtime-consistent IC policy is enabled; use --ic-velocity-potential nfw "
            "or pass --no-ic-require-runtime-potential-match."
        )

    if str(args.ic_source) == "load":
        ic_raw = _load_ic_file(str(args.ic_input_path))
        if int(np.asarray(ic_raw["n_particles"])) != n_particles:
            raise ValueError(
                f"IC file n_particles={int(np.asarray(ic_raw['n_particles']))} "
                f"does not match requested n_particles={n_particles}"
            )
        state0 = jnp.asarray(ic_raw["state0"], dtype=state_dtype)
        mass = jnp.asarray(ic_raw["mass"], dtype=state_dtype)
        ic_velocity_potential = str(np.asarray(ic_raw["ic_velocity_potential"]).item())
        if bool(args.ic_require_runtime_potential_match) and ic_velocity_potential != "nfw":
            raise ValueError(
                "Loaded IC file was not generated with runtime-consistent 'nfw' IC potential."
            )
        ic_velocity_metadata = {
            "ic_velocity_potential": ic_velocity_potential,
            "ic_uses_analytic_disk": bool(np.asarray(ic_raw["ic_uses_analytic_disk"]).item()),
            "ic_source": "load",
        }
    else:
        key = jax.random.PRNGKey(int(args.seed))
        pos = sample_exponential_disk(key, n_particles, rd, zd)
        ic_config, ic_params, ic_velocity_metadata = build_ic_velocity_potential(
            args,
            code_units,
            config,
            params,
            total_mass=total_mass,
        )
        vel = build_quasi_circular_velocities(pos, ic_config, ic_params)
        state0 = construct_initial_state(pos.astype(state_dtype), vel.astype(state_dtype))
        ic_velocity_metadata["ic_source"] = "generate"
        if args.ic_output_path is not None:
            _save_ic_file(
                str(args.ic_output_path),
                state0=state0,
                mass=mass,
                seed=int(args.seed),
                n_particles=n_particles,
                ic_velocity_metadata=ic_velocity_metadata,
            )

    history = None
    initial_accel_report_path = None

    if bool(args.initial_accel_report):
        accel_stats = _compute_initial_acceleration_diagnostics(
            state0=state0,
            mass=mass,
            config=config,
            params=params,
            args=args,
        )
        accel_stats.update(ic_velocity_metadata)
        initial_accel_report_path = _write_initial_acceleration_report(
            report_dir=str(args.report_dir),
            accel_stats=accel_stats,
        )
        print(f"Saved initial acceleration report JSON: {initial_accel_report_path}")

    if args.mode == "render":
        out_path = str(args.movie_path or "./galaxy_render.gif")
        final_state, elapsed, n_frames = _run_render_mode(
            state0,
            mass,
            config,
            params,
            res=int(args.render_resolution),
            stride=int(args.render_stride),
            projection=str(args.projection),
            cmap=str(args.render_cmap),
            out_path=out_path,
            fps=int(args.movie_fps),
            snapshot_output=args.render_snapshot_output,
            length_kpc=float((1.0 * code_units.code_length).to(u.kpc).value),
            time_per_step_gyr=(
                float(args.t_end_gyr) / float(args.num_steps)
                if int(args.num_steps) > 0
                else None
            ),
        )
        print(
            f"Runtime: {elapsed:.3f} s | frames: {n_frames} "
            f"(every {int(args.render_stride)} steps) | movie: {out_path}"
        )
        return

    # Only perf mode reaches here (mode=render returns above via the fused
    # callback render path).
    if args.movie_path:
        print("[info] perf mode ignores --movie-path; use --mode render to render a movie.")
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
            warmup_runs=int(args.perf_warmup_runs),
            measure_runs=int(args.perf_measure_runs),
        )

    final_state_np = np.asarray(final_state)
    dt_code = float(t_end) / float(int(args.num_steps)) if int(args.num_steps) > 0 else float("nan")
    dt_gyr = float(args.t_end_gyr) / float(int(args.num_steps)) if int(args.num_steps) > 0 else float("nan")
    final_state_finite_stats = _final_state_finite_stats(final_state_np)

    payload = {
        "final_state": final_state_np,
        "mass": np.asarray(mass),
        "runtime_seconds": np.asarray(elapsed),
        "n_particles": np.asarray(n_particles),
        "num_steps": np.asarray(int(args.num_steps)),
        "t_end_gyr": np.asarray(float(args.t_end_gyr)),
        "t_end_code": np.asarray(float(t_end)),
        "dt_gyr": np.asarray(float(dt_gyr)),
        "dt_code": np.asarray(float(dt_code)),
        "mode": np.asarray(args.mode),
        "ic_velocity_potential": np.asarray(str(ic_velocity_metadata["ic_velocity_potential"])),
        "ic_source": np.asarray(str(ic_velocity_metadata.get("ic_source", "generate"))),
        "state_dtype": np.asarray(str(state0.dtype)),
        "mass_dtype": np.asarray(str(mass.dtype)),
    }

    np.savez(args.output, **payload)
    print(f"Saved {args.output}")
    print(f"Runtime: {elapsed:.3f} s")

    timing_stats = {} if timing_stats is None else dict(timing_stats)
    large_n_eval_diag_mode = str(
        os.environ.get("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "full")
    ).strip().lower()
    if large_n_eval_diag_mode not in {
        "full",
        "near_only",
        "far_only",
        "local_only",
        "near_zero",
        "far_zero",
        "permutation_only",
        "zero",
    }:
        large_n_eval_diag_mode = "full"
    large_n_nearfield_diag_mode = str(
        os.environ.get("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", "full")
    ).strip().lower()
    if large_n_nearfield_diag_mode not in {
        "full",
        "self_only",
        "pairs_only",
        "overflow_only",
        "zero",
    }:
        large_n_nearfield_diag_mode = "full"
    strict_refresh_diag_mode = str(
        os.environ.get("JACCPOT_STRICT_REFRESH_DIAG_MODE", "full")
    ).strip().lower()
    if strict_refresh_diag_mode not in {
        "full",
        "tree_only",
        "upward_only",
        "downward_only",
        "eval_only",
        "integrator_only",
    }:
        strict_refresh_diag_mode = "full"
    strict_refresh_detail_diag_mode = str(
        os.environ.get("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", "full")
    ).strip().lower()
    if strict_refresh_detail_diag_mode not in {
        "full",
        "tree_sort_only",
        "tree_metadata_only",
        "p2m_only",
        "m2m_only",
        "m2l_only",
        "l2l_only",
        "downward_artifacts_only",
    }:
        strict_refresh_detail_diag_mode = "full"
    static_radix_reuse_structures = str(
        os.environ.get("JACCPOT_STATIC_RADIX_REUSE_STRUCTURES", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    static_radix_upward_batched = str(
        os.environ.get("JACCPOT_STATIC_RADIX_UPWARD_BATCHED", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    static_radix_downward_batched = str(
        os.environ.get("JACCPOT_STATIC_RADIX_DOWNWARD_BATCHED", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if strict_refresh_diag_mode == "integrator_only":
        strict_refresh_diag_flags = (False, False, False, False)
    elif strict_refresh_diag_mode == "eval_only":
        strict_refresh_diag_flags = (False, False, False, True)
    elif strict_refresh_diag_mode == "tree_only":
        strict_refresh_diag_flags = (True, False, False, False)
    elif strict_refresh_diag_mode == "upward_only":
        strict_refresh_diag_flags = (True, True, False, False)
    elif strict_refresh_diag_mode == "downward_only":
        strict_refresh_diag_flags = (True, True, True, False)
    else:
        strict_refresh_diag_flags = (True, True, True, True)
    (
        strict_refresh_diag_tree_active,
        strict_refresh_diag_upward_active,
        strict_refresh_diag_downward_active,
        strict_refresh_diag_eval_active,
    ) = strict_refresh_diag_flags
    timing_stats.update(
        {
            "script_runtime_seconds": float(elapsed),
            "n_particles": int(n_particles),
            "num_steps": int(args.num_steps),
            "t_end_gyr": float(args.t_end_gyr),
            "t_end_code": float(t_end),
            "dt_gyr": float(dt_gyr),
            "dt_code": float(dt_code),
            "perf_warmup_runs_requested": int(args.perf_warmup_runs),
            "perf_measure_runs_requested": int(args.perf_measure_runs),
            "mode": str(args.mode),
            "large_n_eval_diag_mode": str(large_n_eval_diag_mode),
            "runtime_large_n_eval_diag_mode": str(large_n_eval_diag_mode),
            "large_n_nearfield_diag_mode": str(large_n_nearfield_diag_mode),
            "runtime_large_n_nearfield_diag_mode": str(large_n_nearfield_diag_mode),
            "strict_refresh_diag_mode": str(strict_refresh_diag_mode),
            "strict_refresh_diag_tree_active": bool(strict_refresh_diag_tree_active),
            "strict_refresh_diag_upward_active": bool(strict_refresh_diag_upward_active),
            "strict_refresh_diag_downward_active": bool(strict_refresh_diag_downward_active),
            "strict_refresh_diag_eval_active": bool(strict_refresh_diag_eval_active),
            "runtime_strict_refresh_diag_mode": str(strict_refresh_diag_mode),
            "runtime_strict_refresh_diag_tree_active": bool(
                strict_refresh_diag_tree_active
            ),
            "runtime_strict_refresh_diag_upward_active": bool(
                strict_refresh_diag_upward_active
            ),
            "runtime_strict_refresh_diag_downward_active": bool(
                strict_refresh_diag_downward_active
            ),
            "runtime_strict_refresh_diag_eval_active": bool(
                strict_refresh_diag_eval_active
            ),
            "strict_refresh_detail_diag_mode": str(strict_refresh_detail_diag_mode),
            "runtime_strict_refresh_detail_diag_mode": str(
                strict_refresh_detail_diag_mode
            ),
            "runtime_static_radix_reuse_structures": bool(
                static_radix_reuse_structures
            ),
            "runtime_static_radix_upward_batched": bool(
                static_radix_upward_batched
            ),
            "runtime_static_radix_downward_batched": bool(
                static_radix_downward_batched
            ),
            "fmm_integration_timestep_mode": (
                "fixed" if bool(config.fixed_timestep) else "adaptive_diffrax"
            ),
            "state_dtype_requested": str(args.state_dtype),
            "state_dtype": str(state0.dtype),
            "mass_dtype": str(mass.dtype),
            "initial_accel_report_enabled": bool(args.initial_accel_report),
            "initial_accel_report_path": (
                None if initial_accel_report_path is None else str(initial_accel_report_path)
            ),
            "disk_mass_msun": float(args.disk_mass_msun),
            "disk_mass_code": float(total_mass),
            "disk_radius_kpc": float(args.disk_radius_kpc),
            "disk_radius_code": float(rd),
            "disk_height_kpc": float(args.disk_height_kpc),
            "disk_height_code": float(zd),
            "runtime_external_potentials": "NFW_POTENTIAL",
            "runtime_nfw_mvir_msun": 1.0e12,
            "runtime_nfw_mvir_code": float(params.NFW_params.Mvir),
            "runtime_nfw_rs_kpc": 20.0,
            "runtime_nfw_rs_code": float(params.NFW_params.r_s),
            "fmm_preset_requested": str(args.fmm_preset),
            "fmm_runtime_path_requested": str(args.fmm_runtime_path),
            "fmm_theta_requested": float(args.fmm_theta),
            "fmm_mac_type_requested": str(args.fmm_mac_type),
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
            "fmm_max_pair_queue_requested": (
                None
                if args.fmm_max_pair_queue is None
                else int(args.fmm_max_pair_queue)
            ),
            "fmm_pair_process_block_requested": (
                None
                if args.fmm_pair_process_block is None
                else int(args.fmm_pair_process_block)
            ),
            "fmm_max_interactions_per_node_requested": (
                None
                if args.fmm_max_interactions_per_node is None
                else int(args.fmm_max_interactions_per_node)
            ),
            "fmm_max_neighbors_per_leaf_requested": (
                None
                if args.fmm_max_neighbors_per_leaf is None
                else int(args.fmm_max_neighbors_per_leaf)
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
            "fmm_large_n_environment_overrides_enabled": bool(
                not args.no_fmm_large_n_environment_overrides
            ),
            "fmm_prepare_stage_memory_split_enabled": (
                config.fmm_prepare_stage_memory_split_enabled
            ),
            "fmm_upward_leaf_batch_size": config.fmm_upward_leaf_batch_size,
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
            "render_resolution": int(args.render_resolution),
            "render_cmap": str(args.render_cmap),
            "movie_path": None if args.movie_path is None else str(args.movie_path),
            "output_file": str(args.output),
        }
    )
    timing_stats.update(final_state_finite_stats)
    timing_stats.update(ic_velocity_metadata)
    if timing_stats is not None:
        if bool(args.profile_breakdown):
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
                "fmm_integration_timestep_mode": (
                    "fixed" if bool(config.fixed_timestep) else "adaptive_diffrax"
                ),
                "state_dtype_requested": str(args.state_dtype),
                "state_dtype": str(state0.dtype),
                "mass_dtype": str(mass.dtype),
                "initial_accel_report_enabled": bool(args.initial_accel_report),
                "initial_accel_report_path": (
                    None
                    if initial_accel_report_path is None
                    else str(initial_accel_report_path)
                ),
                "disk_mass_msun": float(args.disk_mass_msun),
                "disk_mass_code": float(total_mass),
                "disk_radius_kpc": float(args.disk_radius_kpc),
                "disk_radius_code": float(rd),
                "disk_height_kpc": float(args.disk_height_kpc),
                "disk_height_code": float(zd),
                "runtime_external_potentials": "NFW_POTENTIAL",
                "runtime_nfw_mvir_msun": 1.0e12,
                "runtime_nfw_mvir_code": float(params.NFW_params.Mvir),
                "runtime_nfw_rs_kpc": 20.0,
                "runtime_nfw_rs_code": float(params.NFW_params.r_s),
                "fmm_preset_requested": str(args.fmm_preset),
                "fmm_runtime_path_requested": str(args.fmm_runtime_path),
                "fmm_theta_requested": float(args.fmm_theta),
                "fmm_mac_type_requested": str(args.fmm_mac_type),
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
                "fmm_max_pair_queue_requested": (
                    None
                    if args.fmm_max_pair_queue is None
                    else int(args.fmm_max_pair_queue)
                ),
                "fmm_pair_process_block_requested": (
                    None
                    if args.fmm_pair_process_block is None
                    else int(args.fmm_pair_process_block)
                ),
                "fmm_max_interactions_per_node_requested": (
                    None
                    if args.fmm_max_interactions_per_node is None
                    else int(args.fmm_max_interactions_per_node)
                ),
                "fmm_max_neighbors_per_leaf_requested": (
                    None
                    if args.fmm_max_neighbors_per_leaf is None
                    else int(args.fmm_max_neighbors_per_leaf)
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
                "fmm_large_n_environment_overrides_enabled": bool(
                    not args.no_fmm_large_n_environment_overrides
                ),
                "fmm_prepare_stage_memory_split_enabled": (
                    config.fmm_prepare_stage_memory_split_enabled
                ),
                "fmm_upward_leaf_batch_size": config.fmm_upward_leaf_batch_size,
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
        conservation_stats.update(ic_velocity_metadata)

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
