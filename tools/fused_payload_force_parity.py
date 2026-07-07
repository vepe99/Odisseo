#!/usr/bin/env python3
"""Direct force parity gate for the strict fused static-radix payload route."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for root in (REPO_ROOT.parent / "jaccpot", REPO_ROOT.parent / "yggdrax"):
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

from odisseo.jaccpot_coupling import _build_fmm_solver, _temporary_large_n_environment
from odisseo.option_classes import (
    FMM_ACC,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)
from odisseo.units import CodeUnits


DEFAULT_IC = (
    REPO_ROOT
    / "notebooks"
    / "scalability"
    / "ic_cache"
    / "odisseo_fixed_agama_ic_200k.npz"
)

FIXED_POLICY_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_PACK": "0",
    "JACCPOT_STATIC_RUNTIME_FIXED_SIZING": "1",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": "100000,200000,400000",
    "JACCPOT_STATIC_STRICT_FUSED_COMPILED_SEGMENT_LOOP": "1",
    "JACCPOT_STATIC_STRICT_FUSED_JIT_REFRESH_EVAL": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN": "1",
}


def _status(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _select_single_gpu(*, require_autocvd: bool, timeout: int | None) -> str | None:
    try:
        from autocvd import autocvd
    except Exception as exc:
        if require_autocvd:
            raise RuntimeError("autocvd is required but could not be imported") from exc
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    _status("Selecting one free GPU with autocvd")
    autocvd(num_gpus=1, timeout=timeout)
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    _status(f"Using CUDA_VISIBLE_DEVICES={selected!r}")
    return selected


@contextmanager
def temporary_env(assignments: dict[str, str]):
    old = {key: os.environ.get(key) for key in assignments}
    try:
        for key, value in assignments.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_config(n_particles: int, state_dtype: str) -> tuple[SimulationConfig, SimulationParams]:
    code_units = CodeUnits(10.0 * u.kpc, 1.0e10 * u.Msun, G=1.0, unit_time=1.0 * u.Gyr)
    config = SimulationConfig(
        N_particles=int(n_particles),
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=1,
        external_accelerations=(NFW_POTENTIAL,),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        fmm_preset="large_n_gpu",
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=True,
        fmm_runtime_path="large_n",
        fmm_theta=0.8,
        fmm_mac_type="dehnen",
        fmm_refresh_every=1,
        fmm_leaf_size=256,
        fmm_tree_build_mode="static_radix",
        fmm_tree_leaf_target=256,
        fmm_max_order=4,
        fmm_nearfield_mode="bucketed",
        fmm_nearfield_edge_chunk_size=256,
        fmm_large_n_environment_overrides_enabled=False,
        fmm_jit_tree=True,
        fmm_jit_traversal=True,
        fmm_max_pair_queue=524288,
        fmm_pair_process_block=256,
        fmm_max_interactions_per_node=16384,
        fmm_max_neighbors_per_leaf=8192,
    )
    params = SimulationParams(
        G=1.0,
        t_end=1.0,
        NFW_params=NFWParams(
            Mvir=(1.0e12 * u.Msun).to(code_units.code_mass).value,
            r_s=(20.0 * u.kpc).to(code_units.code_length).value,
        ),
    )
    return config, params


def _build_solver(config: SimulationConfig, params: SimulationParams):
    return _build_fmm_solver(
        working_dtype=jnp.float32,
        config=config,
        params=params,
        fmm_preset="large_n_gpu",
        fmm_basis=str(config.fmm_basis),
        fmm_theta=float(config.fmm_theta),
        fmm_runtime_path=str(config.fmm_runtime_path),
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
        fmm_prepare_stage_memory_split_enabled=config.fmm_prepare_stage_memory_split_enabled,
        fmm_upward_leaf_batch_size=config.fmm_upward_leaf_batch_size,
    )


def _strip_strict_fused_payload(prepared: Any) -> Any:
    return replace(
        prepared,
        nearfield_target_block_source_leaf_ids_padded=None,
        nearfield_target_block_valid_mask_padded=None,
        radix_fast_payload=None,
    )


def _accel_for_mode(
    *,
    mode: str,
    state: jnp.ndarray,
    masses: jnp.ndarray,
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload_enabled = mode == "payload"
    env = {
        **FIXED_POLICY_ENV,
        "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1" if payload_enabled else "0",
    }
    with temporary_env(env), _temporary_large_n_environment(config, fmm_preset="large_n_gpu"):
        solver = _build_solver(config, params)
        prepared = solver.prepare_state(
            state[:, 0, :],
            masses,
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
            theta=float(config.fmm_theta),
            fused_device_mode=True,
        )
        prepared = jax.block_until_ready(prepared)
        if payload_enabled:
            prepared = solver.refresh_prepared_state(
                prepared,
                state[:, 0, :],
                masses,
                leaf_size=int(config.fmm_leaf_size),
                max_order=int(config.fmm_max_order),
                theta=float(config.fmm_theta),
                fused_device_mode=True,
            )
            prepared = jax.block_until_ready(prepared)
        else:
            prepared = _strip_strict_fused_payload(prepared)
        accel = solver.evaluate_prepared_state(
            prepared,
            target_indices=None,
            return_potential=False,
        )
        accel = np.asarray(jax.block_until_ready(accel))
        runtime_diag = solver.get_runtime_diagnostics()
    return accel, runtime_diag


def _summarize_diff(baseline: np.ndarray, payload: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(payload - baseline)
    diff_norm = np.linalg.norm(diff, axis=1)
    base_norm = np.linalg.norm(baseline, axis=1)
    rel = diff_norm / np.maximum(base_norm, 1.0e-12)
    l2_rel = float(np.linalg.norm(diff) / max(np.linalg.norm(baseline), 1.0e-12))
    worst_idx = np.argsort(diff_norm)[-10:][::-1]
    return {
        "max_abs_component": float(np.max(np.abs(diff))),
        "rms_component": float(np.sqrt(np.mean(diff * diff))),
        "l2_relative": l2_rel,
        "particle_diff_norm_percentiles": {
            str(q): float(np.percentile(diff_norm, q))
            for q in (50, 90, 99, 99.9, 100)
        },
        "particle_relative_norm_percentiles": {
            str(q): float(np.percentile(rel, q))
            for q in (50, 90, 99, 99.9, 100)
        },
        "worst_particles": [
            {
                "index": int(i),
                "diff_norm": float(diff_norm[i]),
                "baseline_norm": float(base_norm[i]),
                "relative_norm": float(rel[i]),
                "baseline": [float(x) for x in baseline[i]],
                "payload": [float(x) for x in payload[i]],
            }
            for i in worst_idx
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-input-path", type=pathlib.Path, default=DEFAULT_IC)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--require-autocvd", action="store_true")
    parser.add_argument("--no-autocvd", action="store_true")
    parser.add_argument("--autocvd-timeout-seconds", type=int, default=180)
    parser.add_argument("--fixed-neighbor-cap", type=int, default=1048576)
    parser.add_argument("--static-target-blocks-max-per-leaf", type=int, default=32)
    parser.add_argument("--max-l2-relative", type=float, default=1.0e-4)
    parser.add_argument("--max-abs-component", type=float, default=5.0e-3)
    args = parser.parse_args()

    if not args.no_autocvd:
        _select_single_gpu(
            require_autocvd=bool(args.require_autocvd),
            timeout=args.autocvd_timeout_seconds,
        )

    fixed_env = {
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": str(int(args.fixed_neighbor_cap)),
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": str(
            int(args.static_target_blocks_max_per_leaf)
        ),
    }
    with temporary_env(fixed_env):
        raw = np.load(args.ic_input_path, allow_pickle=False)
        state = jnp.asarray(raw["state0"], dtype=jnp.float32)
        masses = jnp.asarray(raw["mass"], dtype=jnp.float32)
        config, params = _build_config(int(state.shape[0]), "float32")

        _status("Computing baseline strict-fused acceleration")
        baseline, baseline_diag = _accel_for_mode(
            mode="baseline",
            state=state,
            masses=masses,
            config=config,
            params=params,
        )
        _status("Computing payload strict-fused acceleration")
        payload, payload_diag = _accel_for_mode(
            mode="payload",
            state=state,
            masses=masses,
            config=config,
            params=params,
        )

    summary = {
        "ic_input_path": str(args.ic_input_path),
        "n_particles": int(state.shape[0]),
        "fixed_neighbor_cap": int(args.fixed_neighbor_cap),
        "static_target_blocks_max_per_leaf": int(args.static_target_blocks_max_per_leaf),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thresholds": {
            "max_l2_relative": float(args.max_l2_relative),
            "max_abs_component": float(args.max_abs_component),
        },
        "diff": _summarize_diff(baseline, payload),
        "baseline_diag": baseline_diag,
        "payload_diag": payload_diag,
    }
    summary["passed"] = bool(
        summary["diff"]["l2_relative"] <= float(args.max_l2_relative)
        and summary["diff"]["max_abs_component"] <= float(args.max_abs_component)
    )

    text = json.dumps(summary, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        _status(f"Saved parity report: {args.out}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
