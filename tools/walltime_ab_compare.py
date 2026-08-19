#!/usr/bin/env python3
"""Wall-time-only strict A/B comparison runner.

Purpose: compare end-to-end throughput without internal timing instrumentation.
This is the canonical perf oracle for the strict static-radix 200k lane.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import select
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_FIXED_POLICY_ENV: dict[str, str] = {
    "JACCPOT_STATIC_RUNTIME_FIXED_SIZING": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP": "1",
    "JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "131072",
    "YGGDRAX_STATIC_RADIX_REUSE_NODE_RANGES": "1",
    "JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS": "1",
    "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE": "16",
    "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE": "4",
}
DEFAULT_FUSED_PROFILE_SET = "100000,200000,400000"

STATIC_RADIX_ENV_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "JACCPOT_STATIC_STRICT_GPU_MODE",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK",
    "JACCPOT_STATIC_STRICT_FUSED_MODE",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_STATIC_RUNTIME_FIXED_SIZING",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK",
    "JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP",
    "JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "YGGDRAX_STATIC_RADIX_REUSE_NODE_RANGES",
    "JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN",
    "JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE",
    "JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED",
    "JACCPOT_LARGE_N_RADIX_FAST_OCCUPANCY_SORT",
    "JACCPOT_LARGE_N_RADIX_FAST_SKIP_EMPTY_TILES",
    "JACCPOT_LARGE_N_RADIX_FAST_COMPONENTWISE_PAIRS",
    "JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE",
    "JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE",
    "JACCPOT_STRICT_REFRESH_DIAG_MODE",
    "JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE",
    "JACCPOT_LARGE_N_EVAL_DIAG_MODE",
    "JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE",
    "JACCPOT_STATIC_RADIX_REUSE_STRUCTURES",
    "JACCPOT_STATIC_RADIX_UPWARD_BATCHED",
    "JACCPOT_STATIC_RADIX_DOWNWARD_BATCHED",
    "JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "ODISSEO_STRICT_DISABLE_EXTERNAL_FOR_TIMING",
    "ODISSEO_STRICT_EXTERNAL_ONLY_FOR_TIMING",
)

DEFAULT_ODISSEO_IC_ROOT = Path(
    os.environ.get(
        "ODISSEO_IC_ROOT",
        "/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache",
    )
).expanduser()
DEFAULT_ODISSEO_IC_FILENAME = "odisseo_fixed_agama_ic_200k.npz"


def _status(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _select_single_gpu(
    *,
    require_autocvd: bool,
    autocvd_timeout_seconds: int | None,
) -> str | None:
    try:
        from autocvd import autocvd
    except Exception as exc:
        if require_autocvd:
            raise RuntimeError("autocvd is required but could not be imported") from exc
        _status(f"autocvd unavailable; keeping CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
        return os.environ.get("CUDA_VISIBLE_DEVICES")

    _status("Selecting one free GPU with autocvd")
    autocvd(num_gpus=1, timeout=autocvd_timeout_seconds)
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    _status(f"Using CUDA_VISIBLE_DEVICES={selected!r}")
    return selected




def _default_ic_path() -> Path:
    return DEFAULT_ODISSEO_IC_ROOT / DEFAULT_ODISSEO_IC_FILENAME


def _ic_regeneration_command(*, ic_path: Path, args: argparse.Namespace) -> str:
    return (
        "micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py "
        f"--output {ic_path} "
        f"--n-particles {int(args.n_particles)} "
        f"--state-dtype {str(args.state_dtype)} "
        "--seed 7"
    )


def _resolve_ic_input_path(args: argparse.Namespace) -> Path:
    ic_path = Path(args.ic_input_path) if args.ic_input_path is not None else _default_ic_path()
    ic_path = ic_path.expanduser()
    if not ic_path.exists():
        cmd = _ic_regeneration_command(ic_path=ic_path, args=args)
        raise FileNotFoundError(
            "IC input file not found: "
            f"{ic_path}\n"
            "Generate canonical IC with:\n"
            f"  {cmd}"
        )
    return ic_path

def _run_streaming(
    cmd: list[str],
    *,
    env: dict[str, str],
    status_label: str,
    status_interval: float,
) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None

    lines: list[str] = []
    last_status = time.perf_counter()
    interval = max(1.0, float(status_interval))
    while proc.poll() is None:
        timeout = max(0.1, interval - (time.perf_counter() - last_status))
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if ready:
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                lines.append(line.rstrip("\n"))
            continue

        now = time.perf_counter()
        if now - last_status >= interval:
            _status(f"{status_label} still running")
            last_status = now

    remainder = proc.stdout.read()
    if remainder:
        print(remainder, end="", flush=True)
        lines.extend(remainder.splitlines())
    return int(proc.wait()), "\n".join(lines)


def _latest_profile_json(report_dir: Path) -> Path:
    candidates = sorted(report_dir.glob("galaxy_disk_profile_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No profile json found in {report_dir}")
    return candidates[-1]


def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "script_runtime_seconds",
        "t_end_gyr",
        "t_end_code",
        "dt_gyr",
        "dt_code",
        "final_state_all_finite",
        "final_state_element_count",
        "final_state_finite_count",
        "final_state_nan_count",
        "final_state_inf_count",
        "final_state_position_norm_p50",
        "final_state_position_norm_p90",
        "final_state_position_norm_p99",
        "final_state_position_norm_max",
        "final_state_velocity_norm_p50",
        "final_state_velocity_norm_p90",
        "final_state_velocity_norm_p99",
        "final_state_velocity_norm_max",
        "runtime_strict_fused_mode_active",
        "runtime_strict_fused_fallback_count",
        "runtime_strict_fused_last_fallback_reason",
        "runtime_strict_fused_compile_count",
        "runtime_strict_fused_execute_count",
        "runtime_strict_fused_device_refresh_route_count",
        "runtime_strict_fused_planner_bypassed_count",
        "runtime_strict_velocity_verlet_acceleration_carry_active",
        "runtime_strict_self_force_bootstrap_evaluations",
        "runtime_strict_self_force_initial_full_fmm_evaluations",
        "runtime_strict_self_force_endpoint_evaluations",
        "runtime_strict_external_bootstrap_evaluations",
        "runtime_strict_external_endpoint_evaluations",
        "runtime_strict_static_target_block_capacity_ok",
        "runtime_large_n_radix_fast_occupancy_sort",
        "runtime_large_n_radix_fast_skip_empty_tiles",
        "large_n_eval_diag_mode",
        "runtime_large_n_eval_diag_mode",
        "large_n_nearfield_diag_mode",
        "runtime_large_n_nearfield_diag_mode",
        "large_n_eval_leaf_nodes_shape",
        "large_n_eval_local_coefficients_shape",
        "large_n_eval_local_centers_shape",
        "large_n_eval_active_leaf_count",
        "large_n_eval_max_leaf_size",
        "large_n_eval_leaf_particle_slots",
        "large_n_radix_payload_present",
        "large_n_radix_payload_source_particle_shape",
        "large_n_radix_payload_source_particle_slots",
        "large_n_radix_payload_source_leaf_shape",
        "large_n_radix_payload_source_leaf_slots",
        "large_n_target_block_source_leaf_padded_shape",
        "runtime_large_n_eval_leaf_nodes_shape",
        "runtime_large_n_eval_local_coefficients_shape",
        "runtime_large_n_eval_local_centers_shape",
        "runtime_large_n_eval_active_leaf_count",
        "runtime_large_n_eval_max_leaf_size",
        "runtime_large_n_eval_leaf_particle_slots",
        "runtime_large_n_radix_payload_present",
        "runtime_large_n_radix_payload_source_particle_shape",
        "runtime_large_n_radix_payload_source_particle_slots",
        "runtime_large_n_radix_payload_source_leaf_shape",
        "runtime_large_n_radix_payload_source_leaf_slots",
        "runtime_large_n_target_block_source_leaf_padded_shape",
        "strict_refresh_diag_mode",
        "strict_refresh_diag_tree_active",
        "strict_refresh_diag_upward_active",
        "strict_refresh_diag_downward_active",
        "strict_refresh_diag_eval_active",
        "runtime_strict_refresh_diag_mode",
        "strict_refresh_detail_diag_mode",
        "runtime_strict_refresh_detail_diag_mode",
        "runtime_strict_refresh_diag_tree_active",
        "runtime_strict_refresh_diag_upward_active",
        "runtime_strict_refresh_diag_downward_active",
        "runtime_strict_refresh_diag_eval_active",
        "runtime_static_radix_reuse_structures",
        "runtime_static_radix_upward_batched",
        "runtime_static_radix_downward_batched",
        "runtime_static_radix_tree_leaf_count",
        "runtime_static_radix_tree_node_count",
        "runtime_static_radix_far_pair_count",
        "runtime_static_radix_compact_pair_reuse_hits",
        "runtime_static_radix_compact_pair_reuse_misses",
        "runtime_static_radix_m2l_chunk_count",
        "runtime_static_radix_l2l_edge_count",
        "strict_timing_mode",
        "strict_effective_external_potential",
        "strict_timing_disable_external",
        "strict_timing_external_only",
        "perf_warmup_runs",
        "perf_measure_runs",
        "perf_warmup_run_seconds",
        "perf_measured_run_seconds",
        "perf_measured_median_seconds",
        "perf_measured_median_step_seconds",
        "runtime_large_n_overflow_profile_reprofiles",
        "runtime_large_n_neighbor_edges_profile_reprofiles",
    )
    return {key: profile.get(key) for key in keys if key in profile}




def _state_finite_digest(npz_path: str | Path) -> dict[str, Any]:
    with np.load(npz_path) as data:
        state = np.asarray(data["final_state"])
        digest: dict[str, Any] = {
            "shape": list(state.shape),
            "element_count": int(state.size),
            "finite_count": int(np.count_nonzero(np.isfinite(state))),
            "nan_count": int(np.count_nonzero(np.isnan(state))),
            "inf_count": int(np.count_nonzero(np.isinf(state))),
            "all_finite": bool(np.all(np.isfinite(state))),
        }
        for key in ("t_end_gyr", "t_end_code", "dt_gyr", "dt_code"):
            if key in data:
                digest[key] = float(np.asarray(data[key]))
        if state.ndim >= 3 and state.shape[1] >= 2:
            for label, values in (
                ("position_norm", state[:, 0, :]),
                ("velocity_norm", state[:, 1, :]),
            ):
                norms = np.linalg.norm(values, axis=-1)
                finite_norms = norms[np.isfinite(norms)]
                digest[f"{label}_finite_count"] = int(finite_norms.size)
                if finite_norms.size:
                    digest[f"{label}_p50"] = float(np.percentile(finite_norms, 50.0))
                    digest[f"{label}_p90"] = float(np.percentile(finite_norms, 90.0))
                    digest[f"{label}_p99"] = float(np.percentile(finite_norms, 99.0))
                    digest[f"{label}_max"] = float(np.max(finite_norms))
                else:
                    digest[f"{label}_p50"] = None
                    digest[f"{label}_p90"] = None
                    digest[f"{label}_p99"] = None
                    digest[f"{label}_max"] = None
        return digest

def run_case(case_name: str, out_dir: Path, env: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "out.npz"

    report_dir = out_dir / "reports"

    cmd = [
        "micromamba",
        "run",
        "-n",
        "odisseo",
        "python",
        "notebooks/scalability/galaxy_disk_fmm_large_n.py",
        "--mode",
        "perf",
        "--n-particles",
        str(args.n_particles),
        "--num-steps",
        str(args.num_steps),
        "--state-dtype",
        args.state_dtype,
        "--fmm-preset",
        "large_n_gpu",
        "--fmm-runtime-path",
        "large_n",
        "--fmm-tree-build-mode",
        "static_radix",
        "--fmm-leaf-size",
        str(args.leaf_size),
        "--fmm-refresh-every",
        str(args.refresh_every),
        "--no-fmm-large-n-environment-overrides",
        "--ic-source",
        "load",
        "--ic-input-path",
        str(args.ic_input_path),
        "--output",
        str(out_npz),
        "--perf-warmup-runs",
        str(args.perf_warmup_runs),
        "--perf-measure-runs",
        str(args.perf_measure_runs),
    ]
    if args.t_end_gyr is not None:
        cmd.extend(["--t-end-gyr", str(args.t_end_gyr)])
    if bool(args.profile_breakdown):
        cmd.extend(["--profile-breakdown", "--report-dir", str(report_dir)])

    _status(
        f"Starting {case_name}: n={args.n_particles}, steps={args.num_steps}, "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r}"
    )
    t0 = time.perf_counter()
    returncode, output = _run_streaming(
        cmd,
        env=env,
        status_label=case_name,
        status_interval=max(5.0, float(args.status_interval_seconds)),
    )
    wall = time.perf_counter() - t0

    if returncode != 0:
        raise RuntimeError(
            f"Case '{case_name}' failed (exit {returncode})\n"
            f"OUTPUT:\n{output}"
        )

    result = {
        "case": case_name,
        "wall_seconds": float(wall),
        "output_npz": str(out_npz),
    }
    result["final_state_digest"] = _state_finite_digest(out_npz)
    if bool(args.include_stdout_tail):
        result["stdout_tail"] = "\n".join(output.strip().splitlines()[-8:])
    if bool(args.profile_breakdown):
        profile_path = _latest_profile_json(report_dir)
        with profile_path.open() as f:
            profile = json.load(f)
        result["profile_path"] = str(profile_path)
        result["profile_digest"] = _profile_digest(profile)
    _status(f"Finished {case_name}: wall_seconds={wall:.3f}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-input-path", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--n-particles", type=int, default=200000)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument(
        "--t-end-gyr",
        type=float,
        default=None,
        help="Optional physical integration time passed through to the simulator. Defaults to simulator setting.",
    )
    p.add_argument("--state-dtype", type=str, default="float32")
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--refresh-every", type=int, default=1)
    p.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        help="Baseline env key=value (repeatable)",
    )
    p.add_argument(
        "--variant-env",
        action="append",
        default=[],
        help="Variant-only env key=value (repeatable)",
    )
    p.add_argument(
        "--include-stdout-tail",
        action="store_true",
        help="Include final stdout lines in JSON output (off by default).",
    )
    p.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Collect diagnostic profile reports for short validation runs (off for canonical walltime).",
    )
    p.add_argument(
        "--no-autocvd",
        action="store_true",
        help="Do not call autocvd before running cases.",
    )
    p.add_argument(
        "--require-autocvd",
        action="store_true",
        help="Fail if autocvd cannot select one GPU.",
    )
    p.add_argument(
        "--autocvd-timeout-seconds",
        type=int,
        default=None,
        help="Maximum seconds to wait for one free GPU when using autocvd.",
    )
    p.add_argument(
        "--status-interval-seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval while each subprocess is running.",
    )
    p.add_argument(
        "--perf-warmup-runs",
        type=int,
        default=1,
        help="Full in-process perf runs to exclude before measured timing.",
    )
    p.add_argument(
        "--perf-measure-runs",
        type=int,
        default=3,
        help="Measured in-process perf runs; simulator reports median runtime.",
    )
    p.add_argument(
        "--fixed-policy",
        action="store_true",
        help="Enable static fixed production policy env for baseline and variant.",
    )
    p.add_argument(
        "--fixed-overflow-cap",
        type=int,
        default=None,
        help="Optional fixed overflow profile cap when --fixed-policy is enabled.",
    )
    p.add_argument(
        "--fixed-neighbor-cap",
        type=int,
        default=None,
        help="Required fixed neighbor-edge profile cap when --fixed-policy is enabled.",
    )
    p.add_argument(
        "--require-finite-final-state",
        action="store_true",
        help="Fail if either case writes NaN or Inf values in final_state.",
    )
    args = p.parse_args()
    args.ic_input_path = _resolve_ic_input_path(args)

    selected_cuda = None
    if not bool(args.no_autocvd):
        selected_cuda = _select_single_gpu(
            require_autocvd=bool(args.require_autocvd),
            autocvd_timeout_seconds=args.autocvd_timeout_seconds,
        )

    base_env = os.environ.copy()
    base_env.update(
        {
            "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
            "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
            "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK": "0",
        }
    )
    if selected_cuda is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = str(selected_cuda)

    if bool(args.fixed_policy):
        _status("Applying fixed-policy runtime env for baseline/variant")
        base_env.update(DEFAULT_FIXED_POLICY_ENV)
        base_env["JACCPOT_STATIC_STRICT_FUSED_MODE"] = "on"
        base_env["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = DEFAULT_FUSED_PROFILE_SET
        if args.fixed_neighbor_cap is None:
            raise ValueError(
                "--fixed-policy requires explicit --fixed-neighbor-cap "
                "(user-selected global static cap)."
            )
        if int(args.fixed_neighbor_cap) <= 0:
            raise ValueError("--fixed-neighbor-cap must be > 0 when --fixed-policy is enabled.")
        if args.fixed_overflow_cap is not None:
            base_env["JACCPOT_LARGE_N_OVERFLOW_PROFILE_FIXED_CAP"] = str(
                int(args.fixed_overflow_cap)
            )
        base_env["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = str(
            int(args.fixed_neighbor_cap)
        )

    baseline_env = dict(base_env)
    var_env = dict(base_env)
    for item in args.baseline_env:
        if "=" not in item:
            raise ValueError(f"Invalid --baseline-env: {item}")
        k, v = item.split("=", 1)
        baseline_env[k] = v
    if bool(args.fixed_policy):
        var_env["JACCPOT_STATIC_STRICT_FUSED_MODE"] = "off"
    for item in args.variant_env:
        if "=" not in item:
            raise ValueError(f"Invalid --variant-env: {item}")
        k, v = item.split("=", 1)
        var_env[k] = v

    out_root = args.out_root
    baseline = run_case("baseline", out_root / "baseline", baseline_env, args)
    variant = run_case("variant", out_root / "variant", var_env, args)

    if bool(args.require_finite_final_state):
        bad_cases = [
            result["case"]
            for result in (baseline, variant)
            if not bool(result.get("final_state_digest", {}).get("all_finite", False))
        ]
        if bad_cases:
            raise RuntimeError(
                "Non-finite final_state detected for case(s): "
                f"{bad_cases}; see final_state_digest in the case summaries."
            )

    delta = variant["wall_seconds"] - baseline["wall_seconds"]
    baseline_state = np.asarray(np.load(baseline["output_npz"])["final_state"])
    variant_state = np.asarray(np.load(variant["output_npz"])["final_state"])
    state_delta = baseline_state - variant_state
    state_delta_finite = np.isfinite(state_delta)
    states_finite = bool(
        baseline.get("final_state_digest", {}).get("all_finite", False)
        and variant.get("final_state_digest", {}).get("all_finite", False)
    )
    if states_finite:
        trajectory_delta = {
            "states_all_finite": True,
            "delta_all_finite": bool(np.all(state_delta_finite)),
            "max_abs": float(np.max(np.abs(state_delta))),
            "rms": float(np.sqrt(np.mean(np.square(state_delta)))),
            "relative_l2": float(
                np.linalg.norm(state_delta.reshape(-1))
                / max(float(np.linalg.norm(variant_state.reshape(-1))), 1.0e-30)
            ),
            "position_max_abs": float(np.max(np.abs(state_delta[:, 0, :]))),
            "position_relative_l2": float(
                np.linalg.norm(state_delta[:, 0, :].reshape(-1))
                / max(
                    float(np.linalg.norm(variant_state[:, 0, :].reshape(-1))),
                    1.0e-30,
                )
            ),
            "velocity_max_abs": float(np.max(np.abs(state_delta[:, 1, :]))),
            "velocity_relative_l2": float(
                np.linalg.norm(state_delta[:, 1, :].reshape(-1))
                / max(
                    float(np.linalg.norm(variant_state[:, 1, :].reshape(-1))),
                    1.0e-30,
                )
            ),
        }
    else:
        trajectory_delta = {
            "states_all_finite": False,
            "delta_all_finite": bool(np.all(state_delta_finite)),
            "finite_delta_count": int(np.count_nonzero(state_delta_finite)),
            "nan_delta_count": int(np.count_nonzero(np.isnan(state_delta))),
            "inf_delta_count": int(np.count_nonzero(np.isinf(state_delta))),
            "max_abs": None,
            "rms": None,
            "relative_l2": None,
            "position_max_abs": None,
            "position_relative_l2": None,
            "velocity_max_abs": None,
            "velocity_relative_l2": None,
        }
    frozen_baseline = {
        "n_particles": int(args.n_particles),
        "num_steps": int(args.num_steps),
        "t_end_gyr": (None if args.t_end_gyr is None else float(args.t_end_gyr)),
        "state_dtype": str(args.state_dtype),
        "fmm_preset": "large_n_gpu",
        "fmm_runtime_path": "large_n",
        "fmm_tree_build_mode": "static_radix",
        "fmm_leaf_size": int(args.leaf_size),
        "fmm_refresh_every": int(args.refresh_every),
        "ic_source": "load",
        "ic_input_path": str(args.ic_input_path),
        "fixed_policy": {
            "enabled": bool(args.fixed_policy),
            "fixed_neighbor_cap": (
                int(args.fixed_neighbor_cap) if args.fixed_neighbor_cap is not None else None
            ),
            "fixed_overflow_cap": (
                int(args.fixed_overflow_cap) if args.fixed_overflow_cap is not None else None
            ),
            "requires_explicit_neighbor_cap": True,
        },
        "env": {key: baseline_env.get(key) for key in STATIC_RADIX_ENV_KEYS},
    }
    summary = {
        "frozen_baseline": frozen_baseline,
        "baseline": baseline,
        "variant": variant,
        "trajectory_delta_baseline_minus_variant": trajectory_delta,
        "delta_variant_minus_baseline_seconds": float(delta),
    }

    summary_path = out_root / "walltime_ab_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    _status(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
