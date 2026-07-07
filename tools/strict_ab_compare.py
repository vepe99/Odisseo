#!/usr/bin/env python3
"""Run deterministic strict static-radix A/B diagnostic comparisons.

Note: this tool enables --profile-breakdown and is for diagnostics only.
Use tools/walltime_ab_compare.py as the canonical throughput oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_ODISSEO_IC_ROOT = Path(
    os.environ.get(
        "ODISSEO_IC_ROOT",
        "/export/home/tbuck/Odisseo/notebooks/scalability/ic_cache",
    )
).expanduser()
DEFAULT_ODISSEO_IC_FILENAME = "odisseo_fixed_agama_ic_200k.npz"




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

def _latest_profile_json(report_dir: Path) -> Path:
    cands = sorted(report_dir.glob("galaxy_disk_profile_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No profile json found in {report_dir}")
    return cands[-1]


def _run_case(case_name: str, report_dir: Path, output_npz: Path, env: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

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
        "--profile-breakdown",
        "--report-dir",
        str(report_dir),
        "--output",
        str(output_npz),
    ]

    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Case '{case_name}' failed with code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    profile_path = _latest_profile_json(report_dir)
    with profile_path.open() as f:
        profile = json.load(f)

    return {
        "case": case_name,
        "profile_path": str(profile_path),
        "output_npz": str(output_npz),
        "total_seconds": float(profile.get("total_seconds", 0.0)),
        "strict_runner_wall_seconds": float(profile.get("strict_runner_wall_seconds", 0.0)),
        "runtime_refresh_tree_upward_seconds": float(profile.get("runtime_refresh_tree_upward_seconds", 0.0)),
        "runtime_refresh_dual_downward_compute_seconds": float(profile.get("runtime_refresh_dual_downward_compute_seconds", 0.0)),
        "runtime_refresh_nearfield_seconds": float(profile.get("runtime_refresh_nearfield_seconds", 0.0)),
        "runtime_refresh_total_seconds": float(profile.get("runtime_refresh_total_seconds", 0.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-input-path", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--n-particles", type=int, default=200000)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--state-dtype", type=str, default="float32")
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--refresh-every", type=int, default=1)
    p.add_argument(
        "--variant-env",
        action="append",
        default=[],
        help="Extra env for variant case in KEY=VALUE form (repeatable).",
    )
    args = p.parse_args()
    args.ic_input_path = _resolve_ic_input_path(args)

    env_base = os.environ.copy()
    env_base.update(
        {
            "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
            "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
        }
    )
    env_variant = dict(env_base)
    for kv in args.variant_env:
        if "=" not in kv:
            raise ValueError(f"Invalid --variant-env '{kv}', expected KEY=VALUE")
        k, v = kv.split("=", 1)
        env_variant[k] = v

    out_root = args.out_root
    baseline = _run_case(
        "baseline",
        report_dir=out_root / "baseline" / "reports",
        output_npz=out_root / "baseline" / "out.npz",
        env=env_base,
        args=args,
    )
    variant = _run_case(
        "variant",
        report_dir=out_root / "variant" / "reports",
        output_npz=out_root / "variant" / "out.npz",
        env=env_variant,
        args=args,
    )

    delta = {
        "total_seconds": variant["total_seconds"] - baseline["total_seconds"],
        "strict_runner_wall_seconds": variant["strict_runner_wall_seconds"] - baseline["strict_runner_wall_seconds"],
        "runtime_refresh_tree_upward_seconds": variant["runtime_refresh_tree_upward_seconds"] - baseline["runtime_refresh_tree_upward_seconds"],
        "runtime_refresh_dual_downward_compute_seconds": variant["runtime_refresh_dual_downward_compute_seconds"] - baseline["runtime_refresh_dual_downward_compute_seconds"],
        "runtime_refresh_nearfield_seconds": variant["runtime_refresh_nearfield_seconds"] - baseline["runtime_refresh_nearfield_seconds"],
        "runtime_refresh_total_seconds": variant["runtime_refresh_total_seconds"] - baseline["runtime_refresh_total_seconds"],
    }

    summary = {
        "baseline": baseline,
        "variant": variant,
        "delta_variant_minus_baseline": delta,
    }
    summary_path = out_root / "ab_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
