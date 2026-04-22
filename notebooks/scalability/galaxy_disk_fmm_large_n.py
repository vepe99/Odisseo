"""Galaxy-disk simulation using ODISSEO + jaccpot large-N radix path."""

from __future__ import annotations

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u

from odisseo import construct_initial_state
from odisseo.integration_api import integrate
from odisseo.option_classes import (
    FMM_ACC,
    THICK_MN3_DISK,
    THIN_MN3_DISK,
    SimulationConfig,
    SimulationParams,
    ThickMN3DiskParams,
    ThinMN3DiskParams,
)
from odisseo.potentials import combined_external_acceleration_vmpa_switch
from odisseo.units import CodeUnits

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, default=200_000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-snapshots", type=int, default=200)
    parser.add_argument("--t-end-gyr", type=float, default=2.0)
    parser.add_argument("--disk-radius-kpc", type=float, default=12.0)
    parser.add_argument("--disk-height-kpc", type=float, default=0.3)
    parser.add_argument("--disk-mass-msun", type=float, default=6.0e10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="./galaxy_disk_fmm_large_n.npz")

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
        "--max-render-particles",
        type=int,
        default=50_000,
        help="Max particles to render per frame (downsampled deterministically).",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Include rendered snapshots in npz output.",
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


def render_simulation(
    states: np.ndarray,
    times: np.ndarray,
    *,
    projection: str,
    max_render_particles: int,
    live: bool,
    movie_path: str | None,
    movie_fps: int,
) -> None:
    """Render snapshots live and/or save as movie."""
    if not live and movie_path is None:
        return

    import matplotlib.pyplot as plt
    from matplotlib import animation

    n_frames, n_particles = states.shape[0], states.shape[1]
    step = max(1, n_particles // max(1, int(max_render_particles)))
    states_ds = states[:, ::step, 0, :]

    i0, i1, xlabel, ylabel = _projection_axes(projection)
    x_all = states_ds[:, :, i0]
    y_all = states_ds[:, :, i1]

    extent = float(np.percentile(np.abs(np.concatenate((x_all.ravel(), y_all.ravel()))), 99.5))
    extent = max(extent, 1e-6)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)

    scat = ax.scatter(x_all[0], y_all[0], s=0.5, alpha=0.6, linewidths=0)
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


def main() -> None:
    args = parse_args()

    code_units = CodeUnits(10.0 * u.kpc, 1.0e10 * u.Msun, G=1.0, unit_time=(1.0 * u.Gyr))
    n_particles = int(args.n_particles)

    rd = (args.disk_radius_kpc * u.kpc).to(code_units.code_length).value
    zd = (args.disk_height_kpc * u.kpc).to(code_units.code_length).value
    total_mass = (args.disk_mass_msun * u.Msun).to(code_units.code_mass).value
    t_end = (args.t_end_gyr * u.Gyr).to(code_units.code_time).value

    want_visual = bool(args.live) or (args.movie_path is not None)

    config = SimulationConfig(
        N_particles=n_particles,
        acceleration_scheme=FMM_ACC,
        fixed_timestep=True,
        num_timesteps=int(args.num_steps),
        return_snapshots=want_visual,
        num_snapshots=max(2, int(args.num_snapshots)),
        external_accelerations=(THIN_MN3_DISK, THICK_MN3_DISK),
        softening=(0.02 * u.kpc).to(code_units.code_length).value,
        # Integrated large-N jaccpot radix profile selector.
        fmm_preset="fast",
        fmm_auto_large_n_profile=True,
        fmm_large_n_min_particles=100_000,
        fmm_large_n_force_fp32=True,
        fmm_runtime_path="auto",
        fmm_refresh_every=1,
        fmm_leaf_size=64,
        fmm_tree_leaf_target=64,
        fmm_nearfield_mode="bucketed",
        fmm_nearfield_edge_chunk_size=256,
        fmm_jit_tree=True,
        fmm_jit_traversal=True,
    )

    params = SimulationParams(
        G=1.0,
        t_end=t_end,
        ThinMN3Disk_params=ThinMN3DiskParams(
            M=total_mass,
            hr=(3.0 * u.kpc).to(code_units.code_length).value,
            hz=(0.3 * u.kpc).to(code_units.code_length).value,
        ),
        ThickMN3Disk_params=ThickMN3DiskParams(
            M=0.2 * total_mass,
            hr=(4.0 * u.kpc).to(code_units.code_length).value,
            hz=(1.0 * u.kpc).to(code_units.code_length).value,
        ),
    )

    key = jax.random.PRNGKey(int(args.seed))
    pos = sample_exponential_disk(key, n_particles, rd, zd)
    vel = build_quasi_circular_velocities(pos, config, params)

    mass = jnp.full((n_particles,), total_mass / n_particles, dtype=jnp.float64)
    state0 = construct_initial_state(pos.astype(jnp.float64), vel.astype(jnp.float64))

    t0 = time.time()
    sim_out = jax.block_until_ready(integrate(state0, mass, config, params))
    elapsed = time.time() - t0

    if want_visual:
        states = np.asarray(sim_out.states)
        times = np.asarray(sim_out.times)
        final_state = states[-1]
        render_simulation(
            states,
            times,
            projection=str(args.projection),
            max_render_particles=int(args.max_render_particles),
            live=bool(args.live),
            movie_path=args.movie_path,
            movie_fps=int(args.movie_fps),
        )
    else:
        states = None
        times = None
        final_state = np.asarray(sim_out)

    payload = {
        "final_state": np.asarray(final_state),
        "mass": np.asarray(mass),
        "runtime_seconds": np.asarray(elapsed),
        "n_particles": np.asarray(n_particles),
        "num_steps": np.asarray(int(args.num_steps)),
    }
    if bool(args.save_snapshots) and states is not None:
        payload["snapshot_states"] = states
        payload["snapshot_times"] = times

    np.savez(args.output, **payload)
    print(f"Saved {args.output}")
    print(f"Runtime: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
