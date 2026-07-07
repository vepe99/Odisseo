#!/usr/bin/env python3
"""Forensic fused-path audit runner with Nsight Systems integration.

This tool executes fused-on vs fused-off A/B cases for strict static-radix
large-N, captures timing diagnostics and optional Nsight Systems traces, and
emits consolidated bottleneck summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RUN_CLASS_DEFAULTS: dict[str, dict[str, Any]] = {
    "S1": {
        "n_particles": 200_000,
        "num_steps": 2,
        "state_dtype": "float32",
        "leaf_size": 256,
        "refresh_every": 1,
    },
    "S2": {
        "n_particles": 200_000,
        "num_steps": 8,
        "state_dtype": "float32",
        "leaf_size": 256,
        "refresh_every": 1,
    },
    "S3": {
        "n_particles": 200_000,
        "num_steps": 20,
        "state_dtype": "float32",
        "leaf_size": 256,
        "refresh_every": 1,
    },
}

DEFAULT_FUSED_PROFILE_SET = "100000,200000,400000"
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
DEFAULT_ODISSEO_IC_ROOT = Path(
    os.environ.get(
        "ODISSEO_IC_ROOT",
        "/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache",
    )
).expanduser()
DEFAULT_ODISSEO_IC_FILENAME = "odisseo_fixed_agama_ic_200k.npz"

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

TIMING_BUCKET_MAP: dict[str, str] = {
    "runtime_refresh_input_seconds": "host staging + input normalization",
    "runtime_refresh_tree_upward_seconds": "tree build + upward sweep kernels",
    "runtime_refresh_dual_downward_seconds": "dual traversal + downward stages",
    "runtime_refresh_nearfield_seconds": "nearfield evaluation",
    "runtime_refresh_compile_or_sync_suspect_seconds": "sync/dispatch/compile overhead",
    "runtime_refresh_nearfield_neighbor_padding_seconds": "nearfield carry padding",
    "runtime_refresh_nearfield_state_pack_seconds": "nearfield state packing",
    "runtime_refresh_nearfield_radix_payload_seconds": "radix payload rebuild",
}


@dataclass
class CaseConfig:
    n_particles: int
    num_steps: int
    state_dtype: str
    leaf_size: int
    refresh_every: int


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
        _status(
            "autocvd unavailable; keeping "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
        return os.environ.get("CUDA_VISIBLE_DEVICES")

    _status("Selecting one free GPU with autocvd")
    autocvd(num_gpus=1, timeout=autocvd_timeout_seconds)
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    _status(f"Using CUDA_VISIBLE_DEVICES={selected!r}")
    return selected


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
    cands = sorted(
        report_dir.glob("galaxy_disk_profile_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise FileNotFoundError(f"No profile json found in {report_dir}")
    return cands[-1]


def _parse_env_items(items: Iterable[str], *, arg_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {arg_name}: {item}")
        key, val = item.split("=", 1)
        out[key] = val
    return out


def _resolve_case_config(args: argparse.Namespace) -> CaseConfig:
    defaults: dict[str, Any] = {}
    if args.run_class is not None:
        defaults.update(RUN_CLASS_DEFAULTS[str(args.run_class)])
    n_particles = int(
        args.n_particles if args.n_particles is not None else defaults.get("n_particles", 200_000)
    )
    num_steps = int(
        args.num_steps if args.num_steps is not None else defaults.get("num_steps", 20)
    )
    state_dtype = str(
        args.state_dtype if args.state_dtype is not None else defaults.get("state_dtype", "float32")
    )
    leaf_size = int(
        args.leaf_size if args.leaf_size is not None else defaults.get("leaf_size", 256)
    )
    refresh_every = int(
        args.refresh_every if args.refresh_every is not None else defaults.get("refresh_every", 1)
    )
    return CaseConfig(
        n_particles=n_particles,
        num_steps=num_steps,
        state_dtype=state_dtype,
        leaf_size=leaf_size,
        refresh_every=refresh_every,
    )




def _default_ic_path() -> Path:
    return DEFAULT_ODISSEO_IC_ROOT / DEFAULT_ODISSEO_IC_FILENAME


def _ic_regeneration_command(*, ic_path: Path, cfg: CaseConfig) -> str:
    return (
        "micromamba run -n odisseo python tools/agama_generate_scm_disk_ic.py "
        f"--output {ic_path} "
        f"--n-particles {int(cfg.n_particles)} "
        f"--state-dtype {str(cfg.state_dtype)} "
        "--seed 7"
    )


def _resolve_ic_input_path(args: argparse.Namespace, *, cfg: CaseConfig) -> Path:
    ic_path = Path(args.ic_input_path) if args.ic_input_path is not None else _default_ic_path()
    ic_path = ic_path.expanduser()
    if not ic_path.exists():
        cmd = _ic_regeneration_command(ic_path=ic_path, cfg=cfg)
        raise FileNotFoundError(
            "IC input file not found: "
            f"{ic_path}\n"
            "Generate canonical IC with:\n"
            f"  {cmd}"
        )
    return ic_path

def _git_pointer(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        sha = None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--short"],
                text=True,
            ).strip()
        )
    except Exception:
        dirty = None
    return {"repo_root": str(repo_root), "sha": sha, "dirty": dirty}


def _normalize_ns(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _find_csv_with_suffix(root: Path, suffix: str) -> Path | None:
    cands = sorted(root.glob(f"*{suffix}.csv"))
    return cands[-1] if cands else None


def _csv_has_data_rows(path: Path | None) -> bool:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            return next(reader, None) is not None
    except Exception:
        return False


def _norm_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _row_lookup(row: dict[str, str]) -> dict[str, str]:
    return {_norm_col_name(k): v for k, v in row.items()}


def _row_get(
    row: dict[str, str],
    *,
    exact: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> str | None:
    lookup = _row_lookup(row)
    for key in exact:
        val = lookup.get(_norm_col_name(key))
        if val not in (None, ""):
            return val
    if contains:
        for col, val in lookup.items():
            if val in (None, ""):
                continue
            if all(tok in col for tok in contains):
                return val
    return None


def _compute_interval_metrics(intervals: list[tuple[float, float]]) -> dict[str, Any]:
    if not intervals:
        return {
            "gpu_active_percent": None,
            "gpu_busy_ms": 0.0,
            "gpu_span_ms": 0.0,
            "host_idle_gap_mean_ms": None,
            "host_idle_gap_p95_ms": None,
            "host_idle_gap_max_ms": None,
            "kernel_count": 0,
        }
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for start_ns, end_ns in intervals:
        if not merged or start_ns > merged[-1][1]:
            merged.append((start_ns, end_ns))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ns))

    gaps_ms: list[float] = []
    for i in range(1, len(merged)):
        gap_ns = max(0.0, merged[i][0] - merged[i - 1][1])
        gaps_ms.append(gap_ns / 1e6)

    busy_ns = sum(max(0.0, e - s) for s, e in merged)
    span_ns = max(0.0, merged[-1][1] - merged[0][0])
    active_pct = (100.0 * busy_ns / span_ns) if span_ns > 0 else None
    p95_gap = statistics.quantiles(gaps_ms, n=100)[94] if len(gaps_ms) >= 2 else (
        gaps_ms[0] if len(gaps_ms) == 1 else None
    )
    return {
        "gpu_active_percent": active_pct,
        "gpu_busy_ms": busy_ns / 1e6,
        "gpu_span_ms": span_ns / 1e6,
        "host_idle_gap_mean_ms": (statistics.fmean(gaps_ms) if gaps_ms else None),
        "host_idle_gap_p95_ms": p95_gap,
        "host_idle_gap_max_ms": (max(gaps_ms) if gaps_ms else None),
        "kernel_count": len(intervals),
    }


def _parse_nsys_metrics(
    *,
    nsys_dir: Path,
    script_runtime_seconds: float | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    kernel_intervals: list[tuple[float, float]] = []

    kern_trace = _find_csv_with_suffix(nsys_dir, "cuda_kern_exec_trace")
    gpu_trace = _find_csv_with_suffix(nsys_dir, "cuda_gpu_trace")
    api_sum = _find_csv_with_suffix(nsys_dir, "cuda_api_sum")

    trace_source = kern_trace if _csv_has_data_rows(kern_trace) else gpu_trace
    if trace_source is not None:
        rows = _read_csv_rows(trace_source)
        for row in rows:
            name_val = str(
                _row_get(
                    row,
                    exact=("Name", "Kernel Name", "API Function"),
                )
                or ""
            ).lower()
            if "memcpy" in name_val:
                continue
            start_ns = _normalize_ns(
                _row_get(
                    row,
                    exact=("Start", "Kernel Start (ns)", "API Start (ns)", "Start (ns)"),
                    contains=("start",),
                )
            )
            dur_ns = _normalize_ns(
                _row_get(
                    row,
                    exact=("Duration", "Kernel Dur (ns)", "API Dur (ns)", "Duration (ns)"),
                    contains=("dur",),
                )
            )
            if start_ns is None or dur_ns is None:
                continue
            if dur_ns <= 0:
                continue
            kernel_intervals.append((start_ns, start_ns + dur_ns))

    interval_metrics = _compute_interval_metrics(kernel_intervals)
    metrics.update(interval_metrics)
    if script_runtime_seconds and script_runtime_seconds > 0:
        metrics["kernel_launch_density_per_s"] = float(interval_metrics["kernel_count"]) / float(
            script_runtime_seconds
        )
        if kernel_intervals:
            last_kernel_end = max(end for _, end in kernel_intervals)
            tail_start = last_kernel_end - float(script_runtime_seconds) * 1.0e9
            tail_intervals = [
                (max(start, tail_start), min(end, last_kernel_end))
                for start, end in kernel_intervals
                if end > tail_start and start < last_kernel_end
            ]
            tail_intervals = [(start, end) for start, end in tail_intervals if end > start]
            tail_metrics = _compute_interval_metrics(tail_intervals)
            metrics["measured_tail_gpu_active_percent"] = tail_metrics.get(
                "gpu_active_percent"
            )
            metrics["measured_tail_gpu_busy_ms"] = tail_metrics.get("gpu_busy_ms")
            metrics["measured_tail_gpu_span_ms"] = tail_metrics.get("gpu_span_ms")
            metrics["measured_tail_host_idle_gap_p95_ms"] = tail_metrics.get(
                "host_idle_gap_p95_ms"
            )
            metrics["measured_tail_host_idle_gap_max_ms"] = tail_metrics.get(
                "host_idle_gap_max_ms"
            )
            metrics["measured_tail_kernel_count"] = tail_metrics.get("kernel_count")
    else:
        metrics["kernel_launch_density_per_s"] = None

    h2d_calls = 0
    d2h_calls = 0
    sync_hotspots: list[dict[str, Any]] = []
    if api_sum is not None:
        rows = _read_csv_rows(api_sum)
        for row in rows:
            name = str(_row_get(row, exact=("Name",)) or "")
            lname = name.lower()
            calls = int(
                float(
                    _row_get(
                        row,
                        exact=("Calls", "Num Calls"),
                        contains=("calls",),
                    )
                    or 0
                )
            )
            total_time = _normalize_ns(
                _row_get(
                    row,
                    exact=("Total Time (ns)", "Total Time"),
                    contains=("totaltime",),
                )
            )
            avg_time = _normalize_ns(
                _row_get(
                    row,
                    exact=("Average", "Avg", "Avg (ns)"),
                    contains=("avg",),
                )
            )
            if "memcpyh to d" in lname or "memcpy hto d" in lname:
                h2d_calls += calls
            if "memcpyd to h" in lname or "memcpy dto h" in lname:
                d2h_calls += calls
            if "synchronize" in lname or "streamwaitevent" in lname:
                sync_hotspots.append(
                    {
                        "api": name,
                        "calls": calls,
                        "total_time_ns": total_time,
                        "avg_time_ns": avg_time,
                    }
                )
    metrics["h2d_transfer_calls"] = h2d_calls
    metrics["d2h_transfer_calls"] = d2h_calls
    metrics["sync_hotspots"] = sorted(
        sync_hotspots,
        key=lambda x: float(x.get("total_time_ns") or 0.0),
        reverse=True,
    )[:10]
    return metrics


def _run_nsys_capture(
    *,
    nsys_bin: str,
    nsys_trace: str,
    nsys_cuda_graph_trace: str,
    nsys_dir: Path,
    case_name: str,
    env: dict[str, str],
    app_cmd: list[str],
    status_interval_seconds: float,
) -> tuple[int, str, Path | None]:
    nsys_dir.mkdir(parents=True, exist_ok=True)
    rep_base = nsys_dir / case_name
    cmd = [
        nsys_bin,
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--trace",
        nsys_trace,
    ]
    if str(nsys_cuda_graph_trace).strip().lower() not in {"", "none", "off", "0"}:
        cmd.extend(["--cuda-graph-trace", str(nsys_cuda_graph_trace)])
    cmd.extend([
        "-o",
        str(rep_base),
    ])
    cmd += app_cmd
    returncode, output = _run_streaming(
        cmd,
        env=env,
        status_label=f"{case_name}/nsys",
        status_interval=status_interval_seconds,
    )
    rep_paths = [
        rep_base.with_suffix(".nsys-rep"),
        rep_base.with_suffix(".qdrep"),
    ]
    rep_path = next((p for p in rep_paths if p.exists()), None)
    return returncode, output, rep_path


def _run_nsys_stats(
    *,
    nsys_bin: str,
    rep_path: Path,
    nsys_dir: Path,
    case_name: str,
) -> tuple[bool, str]:
    stats_prefix = nsys_dir / f"{case_name}_stats"
    cmd = [
        nsys_bin,
        "stats",
        "--format",
        "csv",
        "--output",
        str(stats_prefix),
        "--report",
        "cuda_kern_exec_trace",
        "--report",
        "cuda_gpu_trace",
        "--report",
        "cuda_api_sum",
        str(rep_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def _build_stage_to_timeline_mapping() -> dict[str, dict[str, Any]]:
    rows = []
    for key, meaning in TIMING_BUCKET_MAP.items():
        rows.append(
            {
                "timing_bucket_key": key,
                "timeline_region": meaning,
                "attribution_status": "mapped",
            }
        )
    return {"rows": rows}


PROFILE_DIGEST_KEYS: tuple[str, ...] = (
    "script_runtime_seconds",
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
    "runtime_large_n_overflow_profile_reprofiles",
    "runtime_large_n_neighbor_edges_profile_reprofiles",
)


def _profile_digest(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: profile.get(key) for key in PROFILE_DIGEST_KEYS if key in profile}


def _build_case_command(cfg: CaseConfig, ic_input_path: Path, out_npz: Path, report_dir: Path) -> list[str]:
    return [
        "micromamba",
        "run",
        "-n",
        "odisseo",
        "python",
        "notebooks/scalability/galaxy_disk_fmm_large_n.py",
        "--mode",
        "perf",
        "--n-particles",
        str(cfg.n_particles),
        "--num-steps",
        str(cfg.num_steps),
        "--state-dtype",
        cfg.state_dtype,
        "--fmm-preset",
        "large_n_gpu",
        "--fmm-runtime-path",
        "large_n",
        "--fmm-tree-build-mode",
        "static_radix",
        "--fmm-leaf-size",
        str(cfg.leaf_size),
        "--fmm-refresh-every",
        str(cfg.refresh_every),
        "--no-fmm-large-n-environment-overrides",
        "--profile-breakdown",
        "--report-dir",
        str(report_dir),
        "--ic-source",
        "load",
        "--ic-input-path",
        str(ic_input_path),
        "--output",
        str(out_npz),
        "--perf-warmup-runs",
        str(int(getattr(_build_case_command, "perf_warmup_runs", 1))),
        "--perf-measure-runs",
        str(int(getattr(_build_case_command, "perf_measure_runs", 1))),
    ]


def run_case(
    case_name: str,
    *,
    out_dir: Path,
    env: dict[str, str],
    cfg: CaseConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "out.npz"
    report_dir = out_dir / "reports"
    nsys_dir = out_dir / "nsys"

    app_cmd = _build_case_command(cfg, args.ic_input_path, out_npz, report_dir)
    _status(
        f"Starting {case_name}: n={cfg.n_particles}, steps={cfg.num_steps}, "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')!r}"
    )

    t0 = time.perf_counter()
    if bool(args.nsys_capture):
        returncode, output, rep_path = _run_nsys_capture(
            nsys_bin=args.nsys_bin,
            nsys_trace=args.nsys_trace,
            nsys_cuda_graph_trace=args.nsys_cuda_graph_trace,
            nsys_dir=nsys_dir,
            case_name=case_name,
            env=env,
            app_cmd=app_cmd,
            status_interval_seconds=max(5.0, float(args.status_interval_seconds)),
        )
    else:
        returncode, output = _run_streaming(
            app_cmd,
            env=env,
            status_label=case_name,
            status_interval=max(5.0, float(args.status_interval_seconds)),
        )
        rep_path = None
    wall = time.perf_counter() - t0

    if returncode != 0:
        raise RuntimeError(
            f"Case '{case_name}' failed (exit {returncode})\n"
            f"OUTPUT:\n{output}"
        )

    profile_path = _latest_profile_json(report_dir)
    with profile_path.open() as f:
        timing_profile = json.load(f)

    nsys_stats_ok = False
    nsys_stats_log = ""
    nsys_metrics: dict[str, Any] = {}
    if rep_path is not None:
        nsys_stats_ok, nsys_stats_log = _run_nsys_stats(
            nsys_bin=args.nsys_bin,
            rep_path=rep_path,
            nsys_dir=nsys_dir,
            case_name=case_name,
        )
        nsys_metrics = _parse_nsys_metrics(
            nsys_dir=nsys_dir,
            script_runtime_seconds=float(timing_profile.get("script_runtime_seconds", 0.0)),
        )

    result: dict[str, Any] = {
        "case": case_name,
        "wall_seconds": float(wall),
        "output_npz": str(out_npz),
        "profile_path": str(profile_path),
        "profile_digest": _profile_digest(timing_profile),
        "timing_profile": timing_profile,
        "nsys_report": (str(rep_path) if rep_path is not None else None),
        "nsys_stats_ok": bool(nsys_stats_ok),
        "nsys_metrics": nsys_metrics,
    }
    if bool(args.include_stdout_tail):
        result["stdout_tail"] = "\n".join(output.strip().splitlines()[-8:])
    if bool(args.emit_metadata):
        meta = {
            "case": case_name,
            "command": app_cmd,
            "effective_env_subset": {key: env.get(key) for key in STATIC_RADIX_ENV_KEYS},
            "nsys_stats_log": nsys_stats_log,
        }
        meta_path = out_dir / "case_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        result["case_metadata_path"] = str(meta_path)

    _status(f"Finished {case_name}: wall_seconds={wall:.3f}")
    return result


def _delta_dict(a: dict[str, Any], b: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = float(bv) - float(av)
        else:
            out[key] = None
    return out


def _build_bottleneck_table(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
) -> list[dict[str, Any]]:
    b_profile = baseline.get("timing_profile", {})
    v_profile = variant.get("timing_profile", {})
    rows: list[dict[str, Any]] = []
    for key, desc in TIMING_BUCKET_MAP.items():
        b_val = float(b_profile.get(key, 0.0) or 0.0)
        v_val = float(v_profile.get(key, 0.0) or 0.0)
        rows.append(
            {
                "bucket": key,
                "timeline_region": desc,
                "baseline_seconds": b_val,
                "variant_seconds": v_val,
                "delta_variant_minus_baseline_seconds": v_val - b_val,
            }
        )
    rows.sort(key=lambda r: abs(float(r["delta_variant_minus_baseline_seconds"])), reverse=True)
    return rows


def _build_gate_flags(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    cfg: CaseConfig,
    run_class: str | None,
    gpu_active_threshold: float,
    max_idle_gap_ms_threshold: float,
) -> dict[str, Any]:
    b_profile = baseline.get("profile_digest", {})
    b_nsys = baseline.get("nsys_metrics", {})
    v_time = float(variant.get("wall_seconds", 0.0) or 0.0)
    b_time = float(baseline.get("wall_seconds", 0.0) or 0.0)
    fused_active = bool(b_profile.get("runtime_strict_fused_mode_active", False))
    fallback_count = int(b_profile.get("runtime_strict_fused_fallback_count", 0) or 0)
    gpu_active = b_nsys.get("gpu_active_percent")
    idle_gap_p95 = b_nsys.get("host_idle_gap_p95_ms")

    return {
        "flag_fused_active_but_slower": bool(fused_active and v_time < b_time),
        "flag_fallback_present": bool(fallback_count > 0),
        "flag_gpu_idle_above_threshold": bool(
            idle_gap_p95 is not None and float(idle_gap_p95) > float(max_idle_gap_ms_threshold)
        ),
        "flag_gpu_active_below_threshold": bool(
            gpu_active is not None and float(gpu_active) < float(gpu_active_threshold)
        ),
        "flag_s2_requires_fused_win": bool(
            str(run_class or "") == "S2" and not (b_time < v_time)
        ),
        "gate_context": {
            "run_class": run_class,
            "n_particles": cfg.n_particles,
            "num_steps": cfg.num_steps,
            "gpu_active_threshold_percent": float(gpu_active_threshold),
            "max_idle_gap_ms_threshold": float(max_idle_gap_ms_threshold),
        },
    }


def _render_markdown_report(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    v = summary["variant"]
    gates = summary["gate_flags"]
    rows = summary["bottleneck_table"]
    lines = [
        "# Fused Audit Report",
        "",
        f"- Audit tag: `{summary['audit']['audit_tag']}`",
        f"- Run class: `{summary['audit']['run_class']}`",
        f"- Lane: `n={summary['frozen_lane']['n_particles']}, steps={summary['frozen_lane']['num_steps']}, leaf={summary['frozen_lane']['fmm_leaf_size']}, refresh={summary['frozen_lane']['fmm_refresh_every']}`",
        "",
        "## A/B Outcome",
        "",
        f"- Baseline (fused on) wall seconds: `{b['wall_seconds']:.6f}`",
        f"- Variant (fused off) wall seconds: `{v['wall_seconds']:.6f}`",
        f"- Delta variant-baseline seconds: `{summary['delta_variant_minus_baseline_seconds']:.6f}`",
        "",
        "## Gate Flags",
        "",
    ]
    for key, val in gates.items():
        if key == "gate_context":
            continue
        lines.append(f"- {key}: `{val}`")
    lines += [
        "",
        "## Bottleneck Table",
        "",
        "| Bucket | Timeline Region | Baseline s | Variant s | Delta v-b s |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows[:16]:
        lines.append(
            f"| {row['bucket']} | {row['timeline_region']} | "
            f"{float(row['baseline_seconds']):.6f} | {float(row['variant_seconds']):.6f} | "
            f"{float(row['delta_variant_minus_baseline_seconds']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _resolve_out_root(args: argparse.Namespace, *, run_class: str | None) -> Path:
    if args.out_root is not None:
        return args.out_root
    if not bool(args.audit_mode):
        raise ValueError("--out-root is required when --audit-mode is not set.")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = str(args.audit_tag).strip() or "fused_audit"
    rc = run_class or "manual"
    return args.audit_root / stamp / f"{tag}_{rc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-input-path", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--n-particles", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--state-dtype", type=str, default=None)
    p.add_argument("--leaf-size", type=int, default=None)
    p.add_argument("--refresh-every", type=int, default=None)
    p.add_argument("--baseline-env", action="append", default=[])
    p.add_argument("--variant-env", action="append", default=[])
    p.add_argument("--include-stdout-tail", action="store_true")
    p.add_argument("--no-autocvd", action="store_true")
    p.add_argument("--require-autocvd", action="store_true")
    p.add_argument(
        "--autocvd-timeout-seconds",
        type=int,
        default=None,
        help="Maximum seconds to wait for one free GPU when using autocvd.",
    )
    p.add_argument("--status-interval-seconds", type=float, default=30.0)
    p.add_argument("--perf-warmup-runs", type=int, default=1)
    p.add_argument("--perf-measure-runs", type=int, default=1)
    p.add_argument("--audit-mode", action="store_true")
    p.add_argument("--nsys-capture", action="store_true")
    p.add_argument("--nsys-bin", type=str, default="nsys")
    p.add_argument("--nsys-trace", type=str, default="cuda,nvtx")
    p.add_argument("--nsys-cuda-graph-trace", type=str, default="node")
    p.add_argument("--audit-tag", type=str, default="fused_forensic")
    p.add_argument("--run-class", choices=sorted(RUN_CLASS_DEFAULTS.keys()), default=None)
    p.add_argument("--emit-metadata", action="store_true")
    p.add_argument("--audit-root", type=Path, default=Path("/tmp/odisseo_fused_audit"))
    p.add_argument("--gpu-active-threshold", type=float, default=60.0)
    p.add_argument("--max-idle-gap-ms-threshold", type=float, default=5.0)
    p.add_argument(
        "--fixed-policy",
        action="store_true",
        help="Enable static fixed production policy env for both baseline and variant.",
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
    args = p.parse_args()

    cfg = _resolve_case_config(args)
    _build_case_command.perf_warmup_runs = max(0, int(args.perf_warmup_runs))
    _build_case_command.perf_measure_runs = max(1, int(args.perf_measure_runs))
    ic_input_path = _resolve_ic_input_path(args, cfg=cfg)
    args.ic_input_path = ic_input_path
    out_root = _resolve_out_root(args, run_class=args.run_class)
    out_root.mkdir(parents=True, exist_ok=True)

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
            "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": DEFAULT_FUSED_PROFILE_SET,
            "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
        }
    )
    if selected_cuda is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = str(selected_cuda)

    if bool(args.fixed_policy):
        _status("Applying fixed-policy runtime env for baseline/variant")
        base_env.update(DEFAULT_FIXED_POLICY_ENV)
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

    base_env.update(_parse_env_items(args.baseline_env, arg_name="--baseline-env"))
    var_env = dict(base_env)
    var_env["JACCPOT_STATIC_STRICT_FUSED_MODE"] = "off"
    var_env.update(_parse_env_items(args.variant_env, arg_name="--variant-env"))

    if bool(args.emit_metadata):
        env_snapshot = {
            "base_env": {key: base_env.get(key) for key in STATIC_RADIX_ENV_KEYS}
            | {"ODISSEO_IC_ROOT": str(DEFAULT_ODISSEO_IC_ROOT)},
            "variant_overrides": {key: var_env.get(key) for key in STATIC_RADIX_ENV_KEYS}
            | {"ODISSEO_IC_ROOT": str(DEFAULT_ODISSEO_IC_ROOT)},
            "git": {
                "odisseo": _git_pointer(Path("/export/home/tbuck/Odisseo")),
                "jaccpot": _git_pointer(Path("/export/home/tbuck/jaccpot")),
            },
            "timing_bucket_map": _build_stage_to_timeline_mapping(),
        }
        with (out_root / "environment_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(env_snapshot, f, indent=2)

    baseline = run_case(
        "baseline",
        out_dir=out_root / "baseline",
        env=base_env,
        cfg=cfg,
        args=args,
    )
    variant = run_case(
        "variant",
        out_dir=out_root / "variant",
        env=var_env,
        cfg=cfg,
        args=args,
    )

    nsys_metric_keys = (
        "gpu_active_percent",
        "gpu_busy_ms",
        "gpu_span_ms",
        "host_idle_gap_mean_ms",
        "host_idle_gap_p95_ms",
        "host_idle_gap_max_ms",
        "kernel_count",
        "measured_tail_gpu_active_percent",
        "measured_tail_host_idle_gap_p95_ms",
        "measured_tail_host_idle_gap_max_ms",
        "measured_tail_kernel_count",
        "kernel_launch_density_per_s",
        "h2d_transfer_calls",
        "d2h_transfer_calls",
    )
    nsys_deltas = _delta_dict(
        baseline.get("nsys_metrics", {}),
        variant.get("nsys_metrics", {}),
        nsys_metric_keys,
    )

    timing_key_set = list(TIMING_BUCKET_MAP.keys())
    timing_deltas = _delta_dict(
        baseline.get("timing_profile", {}),
        variant.get("timing_profile", {}),
        timing_key_set,
    )

    gate_flags = _build_gate_flags(
        baseline=baseline,
        variant=variant,
        cfg=cfg,
        run_class=args.run_class,
        gpu_active_threshold=float(args.gpu_active_threshold),
        max_idle_gap_ms_threshold=float(args.max_idle_gap_ms_threshold),
    )
    bottleneck_table = _build_bottleneck_table(
        baseline=baseline,
        variant=variant,
    )

    summary = {
        "audit": {
            "audit_mode": bool(args.audit_mode),
            "nsys_capture": bool(args.nsys_capture),
            "nsys_cuda_graph_trace": str(args.nsys_cuda_graph_trace),
            "perf_warmup_runs": int(args.perf_warmup_runs),
            "perf_measure_runs": int(args.perf_measure_runs),
            "audit_tag": str(args.audit_tag),
            "run_class": args.run_class,
            "out_root": str(out_root),
        },
        "frozen_lane": {
            "n_particles": cfg.n_particles,
            "num_steps": cfg.num_steps,
            "state_dtype": cfg.state_dtype,
            "fmm_preset": "large_n_gpu",
            "fmm_runtime_path": "large_n",
            "fmm_tree_build_mode": "static_radix",
            "fmm_leaf_size": cfg.leaf_size,
            "fmm_refresh_every": cfg.refresh_every,
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
            "env": {key: base_env.get(key) for key in STATIC_RADIX_ENV_KEYS},
        },
        "baseline": baseline,
        "variant": variant,
        "delta_variant_minus_baseline_seconds": float(variant["wall_seconds"] - baseline["wall_seconds"]),
        "delta_nsys_variant_minus_baseline": nsys_deltas,
        "delta_timing_variant_minus_baseline": timing_deltas,
        "bottleneck_table": bottleneck_table,
        "gate_flags": gate_flags,
    }

    summary_path = out_root / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = out_root / "audit_report.md"
    md_path.write_text(_render_markdown_report(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    _status(f"Saved summary: {summary_path}")
    _status(f"Saved markdown report: {md_path}")


if __name__ == "__main__":
    main()
