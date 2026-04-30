"""Radix fast-lane investigation harness for ODISSEO + jaccpot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
from jaccpot import (
    FMMAdvancedConfig,
    FarFieldConfig,
    FastMultipoleMethod,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)

from odisseo import construct_initial_state
from odisseo.integration_api import _resolve_fmm_runtime_profile
from odisseo.jaccpot_coupling import _build_fmm_solver, _run_full_segment_scan
from odisseo.option_classes import FMM_ACC, NFW_POTENTIAL, NFWParams, SimulationConfig, SimulationParams
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.units import CodeUnits

jax.config.update("jax_enable_x64", True)


def sample_exponential_disk(
    key: jax.Array,
    n_particles: int,
    radial_scale: float,
    vertical_scale: float,
) -> jnp.ndarray:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--fmm-refresh-every", type=int, default=4)
    parser.add_argument("--fmm-preset", type=str, default="large_n_gpu")
    parser.add_argument("--fmm-runtime-path", type=str, default="large_n")
    parser.add_argument("--fmm-working-dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--fmm-jit-tree", action="store_true")
    parser.add_argument("--fmm-jit-traversal", action="store_true")
    parser.add_argument(
        "--fmm-tree-build-mode",
        type=str,
        default="static_radix",
        choices=("lbvh", "fixed_depth", "static_radix", "adaptive"),
    )
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--fmm-nearfield-edge-chunk-size", type=int, default=256)
    parser.add_argument(
        "--large-n-target-block-size",
        type=int,
        default=None,
        help=(
            "Set JACCPOT_LARGE_N_TARGET_BLOCK_SIZE for this run. "
            "Leave unset to preserve the current environment/default."
        ),
    )
    parser.add_argument(
        "--large-n-static-target-blocks",
        action="store_true",
        help="Enable JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS for this run.",
    )
    parser.add_argument(
        "--large-n-static-target-blocks-max-per-leaf",
        type=int,
        default=None,
        help=(
            "Set JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF for this run. "
            "Leave unset to preserve the current environment/default."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-end-gyr", type=float, default=2.0)
    parser.add_argument("--disk-radius-kpc", type=float, default=12.0)
    parser.add_argument("--disk-height-kpc", type=float, default=0.3)
    parser.add_argument("--disk-mass-msun", type=float, default=6.0e10)
    parser.add_argument("--segments-to-benchmark", type=int, default=5)
    parser.add_argument(
        "--cold-start-order",
        type=str,
        default="direct_first",
        choices=("direct_first", "coupler_first"),
    )
    parser.add_argument(
        "--skip-steady-state-warmup",
        action="store_true",
        help="Skip per-state warmup pass before timed rows.",
    )
    parser.add_argument("--assert-fast-lane", action="store_true")
    parser.add_argument("--report-dir", type=str, default="./notebooks/scalability/reports")
    parser.add_argument("--report-stem", type=str, default="radix_fastlane")
    return parser.parse_args()


def _apply_large_n_env_overrides(args: argparse.Namespace) -> None:
    if args.large_n_target_block_size is not None:
        os.environ["JACCPOT_LARGE_N_TARGET_BLOCK_SIZE"] = str(
            int(args.large_n_target_block_size)
        )
    if bool(args.large_n_static_target_blocks):
        os.environ["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS"] = "1"
    if args.large_n_static_target_blocks_max_per_leaf is not None:
        os.environ["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] = str(
            int(args.large_n_static_target_blocks_max_per_leaf)
        )


def _large_n_env_snapshot() -> dict[str, str | None]:
    keys = (
        "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE",
        "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL",
        "JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL",
    )
    return {key: os.environ.get(key) for key in keys}


def _dtype_from_name(name: str):
    return jnp.float32 if str(name).strip().lower() == "float32" else jnp.float64


def _build_simulation(args: argparse.Namespace):
    code_units = CodeUnits(10.0 * u.kpc, 1.0e10 * u.Msun, G=1.0, unit_time=(1.0 * u.Gyr))
    n_particles = int(args.n_particles)

    rd = (args.disk_radius_kpc * u.kpc).to(code_units.code_length).value
    zd = (args.disk_height_kpc * u.kpc).to(code_units.code_length).value
    total_mass = (args.disk_mass_msun * u.Msun).to(code_units.code_mass).value
    t_end = (args.t_end_gyr * u.Gyr).to(code_units.code_time).value

    config = SimulationConfig(
        N_particles=n_particles,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=int(args.num_steps),
        return_snapshots=False,
        external_accelerations=(NFW_POTENTIAL,),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        fmm_preset=str(args.fmm_preset),
        fmm_auto_large_n_profile=False,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=False,
        fmm_runtime_path=str(args.fmm_runtime_path),
        fmm_refresh_every=int(args.fmm_refresh_every),
        fmm_leaf_size=int(args.leaf_size),
        fmm_tree_leaf_target=int(args.leaf_size),
        fmm_tree_build_mode=str(args.fmm_tree_build_mode),
        fmm_nearfield_mode="bucketed",
        fmm_nearfield_edge_chunk_size=int(args.fmm_nearfield_edge_chunk_size),
        fmm_jit_tree=bool(args.fmm_jit_tree),
        fmm_jit_traversal=bool(args.fmm_jit_traversal),
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
    state_dtype = _dtype_from_name(args.fmm_working_dtype)
    mass_dtype = state_dtype
    mass = jnp.full((n_particles,), total_mass / n_particles, dtype=mass_dtype)
    state0 = construct_initial_state(pos.astype(state_dtype), vel.astype(state_dtype))
    return state0, mass, config, params


def _make_coupler_solver(args: argparse.Namespace, state: jnp.ndarray, config: SimulationConfig, params: SimulationParams):
    return _build_fmm_solver(
        working_dtype=state.dtype,
        config=config,
        params=params,
        fmm_preset=str(args.fmm_preset),
        fmm_basis=str(config.fmm_basis),
        fmm_theta=float(config.fmm_theta),
        fmm_runtime_path=str(args.fmm_runtime_path),
        fmm_mac_type=str(config.fmm_mac_type),
        fmm_farfield_mode=str(config.fmm_farfield_mode),
        fmm_m2l_chunk_size=config.fmm_m2l_chunk_size,
        fmm_nearfield_mode=str(config.fmm_nearfield_mode),
        fmm_nearfield_edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
        fmm_tree_build_mode=str(config.fmm_tree_build_mode),
        fmm_tree_leaf_target=int(config.fmm_tree_leaf_target),
        fmm_fixed_order=config.fmm_fixed_order,
        leaf_size=int(args.leaf_size),
        fmm_jit_tree=bool(args.fmm_jit_tree),
        fmm_jit_traversal=bool(args.fmm_jit_traversal),
        fmm_prepare_stage_memory_split_enabled=(
            config.fmm_prepare_stage_memory_split_enabled
        ),
    )


def _make_direct_solver(args: argparse.Namespace, state: jnp.ndarray, config: SimulationConfig, params: SimulationParams):
    # Standalone constructor matching the ODISSEO coupler's static-radix knobs.
    return FastMultipoleMethod(
        preset=str(args.fmm_preset),
        runtime_path=str(args.fmm_runtime_path),
        theta=float(config.fmm_theta),
        G=float(params.G),
        softening=float(config.softening),
        working_dtype=state.dtype,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode=str(args.fmm_tree_build_mode),
                leaf_target=int(args.leaf_size),
            ),
            farfield=FarFieldConfig(
                mode=str(config.fmm_farfield_mode),
                m2l_chunk_size=config.fmm_m2l_chunk_size,
            ),
            nearfield=NearFieldConfig(
                mode=str(config.fmm_nearfield_mode),
                edge_chunk_size=int(config.fmm_nearfield_edge_chunk_size),
            ),
            runtime=RuntimePolicyConfig(
                jit_tree=bool(args.fmm_jit_tree),
                jit_traversal=bool(args.fmm_jit_traversal),
            ),
            mac_type=str(config.fmm_mac_type),
        ),
        fixed_max_leaf_size=int(args.leaf_size),
    )


_PREPARE_STAGE_KEYS = (
    "refresh_input_seconds",
    "refresh_tree_upward_seconds",
    "refresh_dual_downward_seconds",
    "refresh_nearfield_seconds",
    "refresh_dual_setup_seconds",
    "refresh_dual_artifact_build_seconds",
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
)


def _runtime_diagnostics(solver) -> dict:
    getter = getattr(solver, "get_runtime_diagnostics", None)
    if not callable(getter):
        return {}
    try:
        return dict(getter())
    except Exception:
        return {}


def _runtime_backend(solver):
    return getattr(solver, "_impl", solver)


def _diagnostic_delta(before: dict, after: dict, keys: tuple[str, ...]) -> dict:
    delta = {}
    for key in keys:
        try:
            delta[key] = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        except Exception:
            delta[key] = 0.0
    return delta


def _time_prepare_eval(
    solver,
    state: jnp.ndarray,
    mass: jnp.ndarray,
    leaf_size: int,
    max_order: int,
    *,
    collect_prepare_stage_diagnostics: bool = False,
) -> tuple[float, float, dict]:
    diag_before = {}
    previous_timing_active = False
    runtime_backend = _runtime_backend(solver)
    if bool(collect_prepare_stage_diagnostics):
        diag_before = _runtime_diagnostics(solver)
        previous_timing_active = bool(
            getattr(runtime_backend, "_refresh_timing_active", False)
        )
        setattr(runtime_backend, "_refresh_timing_active", True)

    t0 = time.perf_counter()
    try:
        prepared = solver.prepare_state(
            state[:, 0, :],
            mass,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
        )
        jax.block_until_ready(prepared)
    finally:
        if bool(collect_prepare_stage_diagnostics):
            setattr(runtime_backend, "_refresh_timing_active", previous_timing_active)
    t1 = time.perf_counter()

    prepare_stage_diagnostics = {}
    if bool(collect_prepare_stage_diagnostics):
        diag_after_prepare = _runtime_diagnostics(solver)
        stage_delta = _diagnostic_delta(
            diag_before,
            diag_after_prepare,
            _PREPARE_STAGE_KEYS,
        )
        top_stage_keys = (
            "refresh_input_seconds",
            "refresh_tree_upward_seconds",
            "refresh_dual_downward_seconds",
            "refresh_nearfield_seconds",
        )
        top_stage_sum = sum(float(stage_delta.get(key, 0.0)) for key in top_stage_keys)
        prepare_stage_diagnostics = {
            "prepare_stage_delta": stage_delta,
            "prepare_stage_top_level_sum_seconds": float(top_stage_sum),
            "prepare_stage_unaccounted_seconds": float((t1 - t0) - top_stage_sum),
        }

    acc = solver.evaluate_prepared_state(
        prepared,
        target_indices=None,
        return_potential=False,
    )
    jax.block_until_ready(acc)
    t2 = time.perf_counter()
    return float(t1 - t0), float(t2 - t1), prepare_stage_diagnostics


def _generate_segment_end_states(
    args: argparse.Namespace,
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
) -> list[jnp.ndarray]:
    solver = _make_coupler_solver(args, state0, config, params)
    dt_arr = jnp.asarray(float(params.t_end) / float(config.num_timesteps), dtype=state0.dtype)
    steps_per_segment = max(1, int(args.fmm_refresh_every))
    num_segments = max(1, int(args.segments_to_benchmark))

    states = []
    state = state0
    for _ in range(num_segments):
        prepared = solver.prepare_state(
            state[:, 0, :],
            mass,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
        )
        acc = solver.evaluate_prepared_state(prepared, target_indices=None, return_potential=False)
        state, _ = _run_full_segment_scan(
            state,
            acc,
            dt_arr,
            steps=int(steps_per_segment),
            add_external=True,
            config=config,
            params=params,
        )
        state = jnp.asarray(state, dtype=state.dtype)
        states.append(state)
    return states


def _benchmark_solver_on_states(
    solver_name: str,
    solver,
    states: list[jnp.ndarray],
    *,
    leaf_size: int,
    max_order: int,
    mass: jnp.ndarray,
    steady_state_warmup: bool,
) -> list[dict]:
    if bool(steady_state_warmup):
        for state in states:
            _time_prepare_eval(
                solver,
                state,
                mass,
                leaf_size=leaf_size,
                max_order=max_order,
            )

    rows = []
    for idx, state in enumerate(states):
        prep_s, eval_s, _ = _time_prepare_eval(
            solver,
            state,
            mass,
            leaf_size=leaf_size,
            max_order=max_order,
        )
        rows.append(
            {
                "solver_kind": solver_name,
                "state_index": int(idx),
                "state_dtype": str(state.dtype),
                "prepare_seconds": float(prep_s),
                "evaluate_seconds": float(eval_s),
                "total_seconds": float(prep_s + eval_s),
            }
        )
    return rows


def _cold_warmup(
    solver,
    state0: jnp.ndarray,
    mass: jnp.ndarray,
    *,
    leaf_size: int,
    max_order: int,
) -> dict:
    prep_s, eval_s, prepare_diagnostics = _time_prepare_eval(
        solver,
        state0,
        mass,
        leaf_size=leaf_size,
        max_order=max_order,
        collect_prepare_stage_diagnostics=True,
    )
    return {
        "prepare_seconds": float(prep_s),
        "evaluate_seconds": float(eval_s),
        "total_seconds": float(prep_s + eval_s),
        "prepare_stage_diagnostics": prepare_diagnostics,
    }


def _assert_fast_lane(args: argparse.Namespace, state0: jnp.ndarray, config: SimulationConfig):
    eff_preset, eff_runtime_path, eff_dtype = _resolve_fmm_runtime_profile(state0, config)
    if bool(args.assert_fast_lane):
        if str(eff_preset) != "large_n_gpu":
            raise RuntimeError(f"Expected effective preset=large_n_gpu, got {eff_preset}")
        if str(eff_runtime_path) != "large_n":
            raise RuntimeError(f"Expected effective runtime_path=large_n, got {eff_runtime_path}")
        if str(jnp.dtype(eff_dtype)) != "float32":
            raise RuntimeError(f"Expected effective dtype=float32, got {jnp.dtype(eff_dtype)}")
    return {
        "effective_preset_from_integration_api": str(eff_preset),
        "effective_runtime_path_from_integration_api": str(eff_runtime_path),
        "effective_dtype_from_integration_api": str(jnp.dtype(eff_dtype)),
    }


def _write_reports(report_dir: str, report_stem: str, payload: dict, rows: list[dict]) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir = pathlib.Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{report_stem}_{stamp}.json"
    csv_path = out_dir / f"{report_stem}_{stamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "solver_kind",
        "state_index",
        "state_dtype",
        "prepare_seconds",
        "evaluate_seconds",
        "total_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    args = parse_args()
    _apply_large_n_env_overrides(args)
    state0, mass, config, params = _build_simulation(args)
    effective_cfg = _assert_fast_lane(args, state0, config)

    solver_direct = _make_direct_solver(args, state0, config, params)
    solver_coupler = _make_coupler_solver(args, state0, config, params)

    cold_solver_order = (
        ("odisseo_coupler_builder", solver_coupler),
        ("direct_jaccpot", solver_direct),
    )
    if str(args.cold_start_order) == "direct_first":
        cold_solver_order = tuple(reversed(cold_solver_order))

    cold = {}
    for solver_name, solver in cold_solver_order:
        cold[solver_name] = _cold_warmup(
            solver,
            state0,
            mass,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
        )

    states_for_benchmark = [state0] + _generate_segment_end_states(args, state0, mass, config, params)

    rows = []
    rows.extend(
        _benchmark_solver_on_states(
            "direct_jaccpot",
            solver_direct,
            states_for_benchmark,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
            mass=mass,
            steady_state_warmup=not bool(args.skip_steady_state_warmup),
        )
    )
    rows.extend(
        _benchmark_solver_on_states(
            "odisseo_coupler_builder",
            solver_coupler,
            states_for_benchmark,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
            mass=mass,
            steady_state_warmup=not bool(args.skip_steady_state_warmup),
        )
    )

    by_solver = {}
    for row in rows:
        key = row["solver_kind"]
        if key not in by_solver:
            by_solver[key] = {"prepare_seconds": 0.0, "evaluate_seconds": 0.0, "total_seconds": 0.0, "num_states": 0}
        by_solver[key]["prepare_seconds"] += float(row["prepare_seconds"])
        by_solver[key]["evaluate_seconds"] += float(row["evaluate_seconds"])
        by_solver[key]["total_seconds"] += float(row["total_seconds"])
        by_solver[key]["num_states"] += 1

    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "args": vars(args),
        "large_n_environment": _large_n_env_snapshot(),
        "effective_runtime_config": effective_cfg,
        "state0_dtype": str(state0.dtype),
        "mass_dtype": str(mass.dtype),
        "cold_start_single_call": cold,
        "cold_start_order": str(args.cold_start_order),
        "cold_start_measured_before_state_generation": True,
        "num_states_benchmarked_per_solver": int(len(states_for_benchmark)),
        "summary_by_solver": by_solver,
    }

    json_path, csv_path = _write_reports(
        report_dir=str(args.report_dir),
        report_stem=str(args.report_stem),
        payload=payload,
        rows=rows,
    )

    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV report : {csv_path}")
    for solver_name, summary in by_solver.items():
        print(
            f"{solver_name}: prepare={summary['prepare_seconds']:.3f}s "
            f"evaluate={summary['evaluate_seconds']:.3f}s total={summary['total_seconds']:.3f}s "
            f"over {summary['num_states']} states"
        )


if __name__ == "__main__":
    main()
