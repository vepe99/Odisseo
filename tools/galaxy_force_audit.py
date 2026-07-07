from __future__ import annotations

import argparse
import os
import pathlib
import sys
from contextlib import contextmanager

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy import units as u

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
JACCPOT_ROOT = REPO_ROOT.parent / "jaccpot"
YGGDRAX_ROOT = REPO_ROOT.parent / "yggdrax"
for path in (JACCPOT_ROOT, YGGDRAX_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from notebooks.scalability.galaxy_disk_fmm_large_n import (  # noqa: E402
    build_ic_velocity_potential,
    build_quasi_circular_velocities,
    sample_exponential_disk,
)
from odisseo import construct_initial_state  # noqa: E402
from odisseo.jaccpot_coupling import (  # noqa: E402
    _build_fmm_solver,
    _temporary_large_n_environment,
)
from odisseo.option_classes import (  # noqa: E402
    FMM_ACC,
    NFW_POTENTIAL,
    NFWParams,
    SimulationConfig,
    SimulationParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch  # noqa: E402
from odisseo.units import CodeUnits  # noqa: E402


@contextmanager
def temporary_env(assignments: dict[str, str]):
    old = {key: os.environ.get(key) for key in assignments}
    try:
        for key, value in assignments.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def direct_targets(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    target_indices: np.ndarray,
    *,
    softening: float,
    chunk: int,
) -> np.ndarray:
    targets = positions[jnp.asarray(target_indices, dtype=jnp.int32)]
    out = jnp.zeros_like(targets)
    soft2 = jnp.asarray(float(softening) ** 2, dtype=positions.dtype)
    n = int(positions.shape[0])
    target_idx_jax = jnp.asarray(target_indices, dtype=jnp.int32)
    for start in range(0, n, int(chunk)):
        stop = min(n, start + int(chunk))
        src = positions[start:stop]
        m = masses[start:stop]
        diff = targets[:, None, :] - src[None, :, :]
        r2 = jnp.sum(diff * diff, axis=-1) + soft2
        inv_r3 = jnp.reciprocal(r2 * jnp.sqrt(r2))
        global_idx = jnp.arange(start, stop, dtype=jnp.int32)
        self_mask = global_idx[None, :] == target_idx_jax[:, None]
        inv_r3 = jnp.where(self_mask, 0.0, inv_r3)
        out = out - jnp.sum(diff * inv_r3[:, :, None] * m[None, :, None], axis=1)
        out = jax.block_until_ready(out)
    return np.asarray(out)


def summarize_compare(name: str, target_ids: np.ndarray, fmm: np.ndarray, direct: np.ndarray):
    err = np.linalg.norm(fmm - direct, axis=1)
    ref = np.linalg.norm(direct, axis=1)
    rel = err / np.maximum(ref, 1.0e-12)
    print(
        name,
        "n",
        int(target_ids.size),
        "rel_p50_p90_p99_max",
        *[float(x) for x in np.percentile(rel, [50, 90, 99])],
        float(np.max(rel)),
    )
    worst = np.argsort(rel)[-20:][::-1]
    print(name, "worst target rel abs_err direct_norm fmm_norm fmm direct")
    for idx in worst:
        print(
            int(target_ids[idx]),
            float(rel[idx]),
            float(err[idx]),
            float(ref[idx]),
            float(np.linalg.norm(fmm[idx])),
            [float(x) for x in fmm[idx]],
            [float(x) for x in direct[idx]],
        )


def build_case(args: argparse.Namespace):
    code_units = CodeUnits(10.0 * u.kpc, 1.0e10 * u.Msun, G=1.0, unit_time=1.0 * u.Gyr)
    rd = (12.0 * u.kpc).to(code_units.code_length).value
    zd = (0.3 * u.kpc).to(code_units.code_length).value
    total_mass = (6.0e10 * u.Msun).to(code_units.code_mass).value
    config = SimulationConfig(
        N_particles=int(args.n_particles),
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=int(args.num_steps),
        external_accelerations=(NFW_POTENTIAL,),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        fmm_preset="large_n_gpu",
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=True,
        fmm_runtime_path="large_n",
        fmm_theta=float(args.theta),
        fmm_mac_type="dehnen",
        fmm_refresh_every=1,
        fmm_leaf_size=int(args.leaf_size),
        fmm_tree_build_mode="static_radix",
        fmm_tree_leaf_target=int(args.leaf_size),
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
        t_end=(float(args.t_end_gyr) * u.Gyr).to(code_units.code_time).value,
        NFW_params=NFWParams(
            Mvir=(1.0e12 * u.Msun).to(code_units.code_mass).value,
            r_s=(20.0 * u.kpc).to(code_units.code_length).value,
        ),
    )
    state_dtype = jnp.float64 if args.state_dtype == "float64" else jnp.float32
    pos = sample_exponential_disk(jax.random.PRNGKey(int(args.seed)), int(args.n_particles), rd, zd).astype(state_dtype)
    dummy = argparse.Namespace(
        ic_velocity_potential="nfw_analytic_disk",
        ic_analytic_disk_mass_factor=1.0,
        ic_thick_disk_mass_fraction=0.0,
        disk_radius_kpc=12.0,
        disk_height_kpc=0.3,
        ic_thin_disk_radius_kpc=None,
        ic_thin_disk_height_kpc=None,
        ic_thick_disk_radius_kpc=None,
        ic_thick_disk_height_kpc=None,
    )
    ic_config, ic_params, _ = build_ic_velocity_potential(dummy, code_units, config, params, total_mass=total_mass)
    vel = build_quasi_circular_velocities(pos, ic_config, ic_params).astype(state_dtype)
    state = construct_initial_state(pos, vel)
    masses = jnp.full((int(args.n_particles),), total_mass / int(args.n_particles), dtype=state_dtype)
    return config, params, state, masses


def fmm_accel(solver, config, state, masses):
    with _temporary_large_n_environment(config, fmm_preset="large_n_gpu"):
        prepared = solver.prepare_state(
            state[:, 0, :],
            masses,
            leaf_size=int(config.fmm_leaf_size),
            max_order=int(config.fmm_max_order),
        )
    prepared = jax.block_until_ready(prepared)
    acc = solver.evaluate_prepared_state(prepared, target_indices=None, return_potential=False)
    return np.asarray(jax.block_until_ready(acc)), prepared


def build_solver(config, params):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-particles", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--t-end-gyr", type=float, default=2.0)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--leaf-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--state-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--direct-source-chunk", type=int, default=8192)
    args = parser.parse_args()

    config, params, state, masses = build_case(args)
    solver = build_solver(config, params)

    base_targets = np.arange(0, int(args.n_particles), max(1, int(args.n_particles) // int(args.sample_count)), dtype=np.int32)[: int(args.sample_count)]
    special = np.asarray([3219, 146081, 94916, 158040, 182412], dtype=np.int32)
    target_ids = np.unique(np.concatenate([base_targets, special]))

    print("target_count", int(target_ids.size), "special", special.tolist())
    acc0, prepared0 = fmm_accel(solver, config, state, masses)
    print(
        "prepared0",
        {
            "overflow": int(getattr(prepared0, "nearfield_target_block_overflow_active_blocks", -1)),
            "block_padded_shape": tuple(np.asarray(getattr(prepared0, "nearfield_target_block_source_leaf_ids_padded")).shape),
            "radix_source_particle_shape": tuple(np.asarray(getattr(prepared0.radix_fast_payload, "source_particle_ids")).shape),
        },
    )
    direct0 = direct_targets(
        state[:, 0, :],
        masses,
        target_ids,
        softening=float(config.softening),
        chunk=int(args.direct_source_chunk),
    )
    summarize_compare("t0_self", target_ids, acc0[target_ids], direct0)

    ext0 = combined_external_acceleration_vmpa_switch(state, config, params)
    dt = jnp.asarray(float(params.t_end) / float(config.num_timesteps), dtype=state.dtype)
    acc_total0 = jnp.asarray(acc0, dtype=state.dtype) + ext0
    pos1 = state[:, 0, :] + state[:, 1, :] * dt + 0.5 * acc_total0 * (dt**2)
    state_pos = state.at[:, 0, :].set(pos1)
    ext1 = combined_external_acceleration_vmpa_switch(state_pos, config, params)
    vel1 = state[:, 1, :] + 0.5 * (acc_total0 + jnp.asarray(acc0, dtype=state.dtype) + ext1) * dt
    state1 = state_pos.at[:, 1, :].set(vel1)
    state1 = jax.block_until_ready(state1)

    acc1, prepared1 = fmm_accel(solver, config, state1, masses)
    print(
        "prepared1",
        {
            "overflow": int(getattr(prepared1, "nearfield_target_block_overflow_active_blocks", -1)),
            "block_padded_shape": tuple(np.asarray(getattr(prepared1, "nearfield_target_block_source_leaf_ids_padded")).shape),
            "radix_source_particle_shape": tuple(np.asarray(getattr(prepared1.radix_fast_payload, "source_particle_ids")).shape),
        },
    )
    direct1 = direct_targets(
        state1[:, 0, :],
        masses,
        target_ids,
        softening=float(config.softening),
        chunk=int(args.direct_source_chunk),
    )
    summarize_compare("t1_self", target_ids, acc1[target_ids], direct1)

    fresh_solver = build_solver(config, params)
    acc1_fresh, prepared1_fresh = fmm_accel(fresh_solver, config, state1, masses)
    print(
        "prepared1_fresh",
        {
            "overflow": int(getattr(prepared1_fresh, "nearfield_target_block_overflow_active_blocks", -1)),
            "block_padded_shape": tuple(np.asarray(getattr(prepared1_fresh, "nearfield_target_block_source_leaf_ids_padded")).shape),
            "radix_source_particle_shape": tuple(np.asarray(getattr(prepared1_fresh.radix_fast_payload, "source_particle_ids")).shape),
        },
    )
    summarize_compare("t1_self_fresh_solver", target_ids, acc1_fresh[target_ids], direct1)

    r0 = np.linalg.norm(np.asarray(state[:, 0, :])[:, :2], axis=1)
    r1 = np.linalg.norm(np.asarray(state1[:, 0, :])[:, :2], axis=1)
    dr = r1 - r0
    worst = np.argsort(dr)[-20:][::-1]
    print("step1_largest_dr id r0 r1 dr acc_self_norm ext_norm")
    ext0_np = np.asarray(ext0)
    for idx in worst:
        print(
            int(idx),
            float(r0[idx]),
            float(r1[idx]),
            float(dr[idx]),
            float(np.linalg.norm(acc0[idx])),
            float(np.linalg.norm(ext0_np[idx])),
        )


if __name__ == "__main__":
    main()
