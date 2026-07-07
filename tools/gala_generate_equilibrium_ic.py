#!/usr/bin/env python3
"""Generate reusable galaxy ICs with gala and save Odisseo-compatible NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import gala.potential as gp
import gala.units as gu
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, required=True, help="Output IC NPZ path.")
    p.add_argument("--n-particles", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--state-dtype", type=str, default="float32", choices=("float32", "float64"))
    p.add_argument("--disk-mass-msun", type=float, default=6.0e10)
    p.add_argument("--disk-radius-kpc", type=float, default=12.0)
    p.add_argument("--disk-height-kpc", type=float, default=0.3)
    p.add_argument("--halo-mvir-msun", type=float, default=1.0e12)
    p.add_argument("--halo-rs-kpc", type=float, default=20.0)
    p.add_argument(
        "--disk-potential-mass-factor",
        type=float,
        default=1.0,
        help="Mass factor for the analytic disk potential used in IC velocity setup.",
    )
    p.add_argument(
        "--sigma-r-kms",
        type=float,
        default=0.0,
        help="Optional Gaussian radial velocity dispersion [km/s].",
    )
    p.add_argument(
        "--sigma-z-kms",
        type=float,
        default=0.0,
        help="Optional Gaussian vertical velocity dispersion [km/s].",
    )
    return p.parse_args()


def sample_exponential_disk(rng: np.random.Generator, n: int, rd_kpc: float, zd_kpc: float) -> np.ndarray:
    u_r = rng.uniform(1e-8, 1.0 - 1e-8, size=n)
    radius = -rd_kpc * np.log1p(-u_r)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u_z = rng.uniform(1e-8, 1.0 - 1e-8, size=n)
    sign = np.where(rng.uniform(size=n) > 0.5, 1.0, -1.0)
    z = sign * (-zd_kpc * np.log1p(-u_z))
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    return np.stack((x, y, z), axis=1)


def code_velocity_from_kms(v_kms: np.ndarray) -> np.ndarray:
    # Odisseo code units: L0=10 kpc, M0=1e10 Msun, G=1 => T0≈14.91085 Myr.
    # Therefore V0=L0/T0≈655.812 km/s.
    v0_kms = 655.812
    return v_kms / v0_kms


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n = int(args.n_particles)
    pos_kpc = sample_exponential_disk(
        rng,
        n=n,
        rd_kpc=float(args.disk_radius_kpc),
        zd_kpc=float(args.disk_height_kpc),
    )

    units = gu.galactic
    halo = gp.NFWPotential(
        m=float(args.halo_mvir_msun) * u.Msun,
        r_s=float(args.halo_rs_kpc) * u.kpc,
        units=units,
    )
    disk = gp.MiyamotoNagaiPotential(
        m=float(args.disk_mass_msun) * float(args.disk_potential_mass_factor) * u.Msun,
        a=float(args.disk_radius_kpc) * u.kpc,
        b=max(1e-4, float(args.disk_height_kpc)) * u.kpc,
        units=units,
    )
    pot = halo + disk

    x = pos_kpc[:, 0]
    y = pos_kpc[:, 1]
    radius = np.sqrt(x * x + y * y + 1e-12)

    q = np.vstack((radius, np.zeros_like(radius), np.zeros_like(radius))) * u.kpc
    v_circ = np.asarray(pot.circular_velocity(q).to_value(u.km / u.s), dtype=np.float64)

    ephi = np.stack((-y / radius, x / radius), axis=1)
    vel_xy_kms = ephi * v_circ[:, None]

    sigma_r = max(0.0, float(args.sigma_r_kms))
    sigma_z = max(0.0, float(args.sigma_z_kms))
    if sigma_r > 0.0:
        er = np.stack((x / radius, y / radius), axis=1)
        vr = rng.normal(0.0, sigma_r, size=n)
        vel_xy_kms += er * vr[:, None]
    vz_kms = rng.normal(0.0, sigma_z, size=n) if sigma_z > 0.0 else np.zeros(n, dtype=np.float64)

    pos_code = pos_kpc / 10.0
    vel_code = np.column_stack((code_velocity_from_kms(vel_xy_kms[:, 0]), code_velocity_from_kms(vel_xy_kms[:, 1]), code_velocity_from_kms(vz_kms)))

    state_dtype = np.float64 if str(args.state_dtype) == "float64" else np.float32
    state0 = np.stack((pos_code, vel_code), axis=1).astype(state_dtype, copy=False)
    total_mass_code = float(args.disk_mass_msun) / 1.0e10
    mass = np.full((n,), total_mass_code / float(n), dtype=state_dtype)

    np.savez_compressed(
        out,
        state0=state0,
        mass=mass,
        seed=np.asarray(int(args.seed)),
        n_particles=np.asarray(n),
        state_dtype=np.asarray(str(state0.dtype)),
        mass_dtype=np.asarray(str(mass.dtype)),
        ic_velocity_potential=np.asarray("gala_nfw_plus_mn"),
        ic_uses_analytic_disk=np.asarray(True),
        ic_source=np.asarray("gala"),
        sigma_r_kms=np.asarray(sigma_r),
        sigma_z_kms=np.asarray(sigma_z),
    )
    print(f"Saved gala IC file: {out}")


if __name__ == "__main__":
    main()

