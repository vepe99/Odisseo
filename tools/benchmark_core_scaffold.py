#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp

from odisseo.jaccpot_coupling import (
    integrate_diffrax_jaccpot_active,
    integrate_leapfrog_jaccpot_active,
)
from odisseo.option_classes import DOPRI5, FMM_ACC, SimulationConfig, SimulationParams


@dataclass
class BenchRow:
    integration_mode: str
    scaffold_enabled: bool
    elapsed_s: float
    out_shape: tuple[int, ...]
    adaptive_total_steps: int | None
    adaptive_full_prepare_calls: int | None
    adaptive_refresh_prepare_calls: int | None
    core_scaffold_exec_calls: int | None
    core_scaffold_prepare_calls: int | None
    core_scaffold_refresh_calls: int | None


def _run_case(*, adaptive: bool, scaffold: bool, args: argparse.Namespace) -> BenchRow:
    os.environ["ODISSEO_FMM_USE_CORE_KERNEL_SCAFFOLD"] = "1" if scaffold else "0"
    os.environ["JACCPOT_STATIC_STRICT_GPU_MODE"] = args.strict_mode
    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform

    key = jax.random.PRNGKey(args.seed)
    pos = args.init_sigma * jax.random.normal(
        key, (args.n_particles, 3), dtype=jnp.float32
    )
    vel = jnp.zeros((args.n_particles, 3), dtype=jnp.float32)
    state = jnp.stack((pos, vel), axis=1)
    mass = jnp.ones((args.n_particles,), dtype=jnp.float32) / float(args.n_particles)

    cfg = SimulationConfig(
        N_particles=args.n_particles,
        num_timesteps=args.num_steps,
        fixed_timestep=(not adaptive),
        diffrax_solver=DOPRI5,
        acceleration_scheme=FMM_ACC,
        softening=args.softening,
        fmm_tree_build_mode=args.tree_build_mode,
        fmm_preset=args.fmm_preset,
        fmm_runtime_path=args.fmm_runtime_path,
        fmm_leaf_size=args.leaf_size,
        fmm_adaptive_rtol=args.adaptive_rtol,
        fmm_adaptive_atol=args.adaptive_atol,
        fmm_adaptive_max_dt=args.adaptive_max_dt,
        fmm_adaptive_min_dt=args.adaptive_min_dt,
    )
    params = SimulationParams(G=args.G, t_end=args.t_end)
    timing: dict = {}
    t0 = time.perf_counter()
    if adaptive:
        out = integrate_diffrax_jaccpot_active(
            state,
            mass,
            cfg,
            params,
            num_steps=args.num_steps,
            leaf_size=args.leaf_size,
            max_order=args.max_order,
            fmm_preset=args.fmm_preset,
            fmm_runtime_path=args.fmm_runtime_path,
            fmm_tree_build_mode=args.tree_build_mode,
            timing_stats=timing,
        )
    else:
        out = integrate_leapfrog_jaccpot_active(
            state,
            mass,
            cfg,
            params,
            num_steps=args.num_steps,
            leaf_size=args.leaf_size,
            max_order=args.max_order,
            fmm_preset=args.fmm_preset,
            fmm_runtime_path=args.fmm_runtime_path,
            fmm_tree_build_mode=args.tree_build_mode,
            timing_stats=timing,
        )
    elapsed = time.perf_counter() - t0
    return BenchRow(
        integration_mode=("adaptive" if adaptive else "fixed"),
        scaffold_enabled=scaffold,
        elapsed_s=float(elapsed),
        out_shape=tuple(int(v) for v in out.shape),
        adaptive_total_steps=(
            int(timing["adaptive_total_steps"])
            if "adaptive_total_steps" in timing
            else None
        ),
        adaptive_full_prepare_calls=(
            int(timing["adaptive_full_prepare_calls"])
            if "adaptive_full_prepare_calls" in timing
            else None
        ),
        adaptive_refresh_prepare_calls=(
            int(timing["adaptive_refresh_prepare_calls"])
            if "adaptive_refresh_prepare_calls" in timing
            else None
        ),
        core_scaffold_exec_calls=(
            int(
                timing.get(
                    "core_scaffold_exec_calls",
                    timing.get("adaptive_core_scaffold_exec_calls", 0),
                )
            )
            if (
                "core_scaffold_exec_calls" in timing
                or "adaptive_core_scaffold_exec_calls" in timing
            )
            else None
        ),
        core_scaffold_prepare_calls=(
            int(
                timing.get(
                    "core_scaffold_prepare_calls",
                    timing.get("adaptive_core_scaffold_prepare_calls", 0),
                )
            )
            if (
                "core_scaffold_prepare_calls" in timing
                or "adaptive_core_scaffold_prepare_calls" in timing
            )
            else None
        ),
        core_scaffold_refresh_calls=(
            int(
                timing.get(
                    "core_scaffold_refresh_calls",
                    timing.get("adaptive_core_scaffold_refresh_calls", 0),
                )
            )
            if (
                "core_scaffold_refresh_calls" in timing
                or "adaptive_core_scaffold_refresh_calls" in timing
            )
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark shared core-kernel scaffold on/off for fixed/adaptive."
    )
    parser.add_argument("--n-particles", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--t-end", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-sigma", type=float, default=0.1)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=1e-6)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--fmm-preset", type=str, default="large_n_gpu")
    parser.add_argument("--fmm-runtime-path", type=str, default="large_n")
    parser.add_argument("--tree-build-mode", type=str, default="static_radix")
    parser.add_argument("--strict-mode", type=str, default="off")
    parser.add_argument("--adaptive-rtol", type=float, default=5e-1)
    parser.add_argument("--adaptive-atol", type=float, default=1e-2)
    parser.add_argument("--adaptive-max-dt", type=float, default=1e-2)
    parser.add_argument("--adaptive-min-dt", type=float, default=1e-10)
    parser.add_argument("--platform", type=str, default="", help="e.g. cpu or gpu")
    parser.add_argument("--mode", choices=("fixed", "adaptive", "both"), default="both")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    rows: list[BenchRow] = []
    modes = ["fixed", "adaptive"] if args.mode == "both" else [args.mode]
    for mode in modes:
        adaptive = mode == "adaptive"
        rows.append(_run_case(adaptive=adaptive, scaffold=False, args=args))
        rows.append(_run_case(adaptive=adaptive, scaffold=True, args=args))

    payload = [asdict(r) for r in rows]
    print(json.dumps(payload, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
