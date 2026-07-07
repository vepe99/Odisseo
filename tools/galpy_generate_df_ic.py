#!/usr/bin/env python3
"""Generate Odisseo-compatible ICs from a galpy Dehnen DF disk model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from galpy.df import dehnendf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--n-particles", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--state-dtype", type=str, default="float32", choices=("float32", "float64"))
    p.add_argument("--disk-mass-msun", type=float, default=6.0e10)
    p.add_argument("--disk-radius-kpc", type=float, default=12.0)
    p.add_argument("--disk-height-kpc", type=float, default=0.3)
    p.add_argument("--sigma-r-over-vc", type=float, default=0.16)
    p.add_argument("--sigma-z-kms", type=float, default=6.0)
    p.add_argument(
        "--use-corrected-df",
        action="store_true",
        help="Enable galpy's corrected Dehnen DF table build (slower; may fail on some setups).",
    )
    p.add_argument("--ro-kpc", type=float, default=8.0, help="galpy R normalization [kpc]")
    p.add_argument("--vo-kms", type=float, default=220.0, help="galpy V normalization [km/s]")
    return p.parse_args()


def code_velocity_from_kms(v_kms: np.ndarray) -> np.ndarray:
    v0_kms = 655.812  # Odisseo code unit velocity (L0=10kpc, M0=1e10Msun, G=1)
    return v_kms / v0_kms


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(args.seed))

    rd_kpc = float(args.disk_radius_kpc)
    ro_kpc = float(args.ro_kpc)
    vo_kms = float(args.vo_kms)
    hr = rd_kpc / ro_kpc
    sr = float(args.sigma_r_over_vc)

    df = dehnendf(
        profileParams=(hr, 1.0, sr),
        beta=0.0,
        correct=bool(args.use_corrected_df),
        ro=ro_kpc,
        vo=vo_kms,
    )
    n = int(args.n_particles)
    # Sample positions explicitly; then sample velocities from the DF at each R.
    u_r = np.random.uniform(1e-8, 1.0 - 1e-8, size=n)
    R = -rd_kpc * np.log1p(-u_r)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    vR = np.empty(n, dtype=np.float64)
    vT = np.empty(n, dtype=np.float64)
    # galpy expects scalar R; batch by radial bins to avoid per-particle calls.
    n_bins = 256
    r_edges = np.quantile(R, np.linspace(0.0, 1.0, n_bins + 1))
    r_edges[0] = min(r_edges[0], 0.0)
    r_edges[-1] = max(r_edges[-1], np.max(R) + 1e-9)
    bin_ids = np.digitize(R, r_edges[1:-1], right=False)
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        r_mid = 0.5 * (r_edges[b] + r_edges[b + 1])
        vrvt = np.asarray(df.sampleVRVT(R=r_mid / ro_kpc, n=int(idx.size), target=True), dtype=np.float64)
        vR[idx] = vrvt[:, 0]
        vT[idx] = vrvt[:, 1]

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    vx = vR * np.cos(phi) - vT * np.sin(phi)
    vy = vR * np.sin(phi) + vT * np.cos(phi)

    # Vertical structure: exponential z and Gaussian vz.
    u = np.random.uniform(1e-8, 1.0 - 1e-8, size=n)
    sign = np.where(np.random.uniform(size=n) > 0.5, 1.0, -1.0)
    z = sign * (-float(args.disk_height_kpc) * np.log1p(-u))
    vz = np.random.normal(0.0, max(0.0, float(args.sigma_z_kms)), size=n)

    pos_code = np.column_stack((x, y, z)) / 10.0
    vel_code = np.column_stack(
        (
            code_velocity_from_kms(vx),
            code_velocity_from_kms(vy),
            code_velocity_from_kms(vz),
        )
    )

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
        ic_velocity_potential=np.asarray("galpy_dehnen_df"),
        ic_uses_analytic_disk=np.asarray(True),
        ic_source=np.asarray("galpy_df"),
        sigma_r_over_vc=np.asarray(float(args.sigma_r_over_vc)),
        sigma_z_kms=np.asarray(float(args.sigma_z_kms)),
        ro_kpc=np.asarray(ro_kpc),
        vo_kms=np.asarray(vo_kms),
    )
    print(f"Saved galpy DF IC file: {out}")


if __name__ == "__main__":
    main()
