#!/usr/bin/env python3
"""Generate Odisseo-compatible equilibrium-like disk ICs with AGAMA."""

from __future__ import annotations

import argparse
from pathlib import Path

import agama
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--n-particles", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--state-dtype", type=str, default="float32", choices=("float32", "float64"))

    # Odisseo code units: L0=10kpc, M0=1e10Msun, G=1.
    p.add_argument("--halo-mass-code", type=float, default=100.0)
    p.add_argument("--halo-rs-code", type=float, default=2.0)
    p.add_argument("--disk-mass-code", type=float, default=6.0)
    p.add_argument("--disk-radial-scale-code", type=float, default=0.24)
    p.add_argument("--disk-height-code", type=float, default=0.03)

    # Quasi-isothermal DF parameters in code units (MW-like defaults from AGAMA examples).
    p.add_argument("--sigma-r0-code", type=float, default=0.102)
    p.add_argument("--sigma-z0-code", type=float, default=0.068)
    p.add_argument("--sigma-r-scale-code", type=float, default=1.0)
    p.add_argument("--sigma-z-scale-code", type=float, default=1.0)
    p.add_argument("--df-rdisk-code", type=float, default=0.24)
    p.add_argument("--df-hdisk-code", type=float, default=0.03)
    p.add_argument("--sigma-min-frac", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "[DEPRECATED] tools/agama_generate_equilibrium_ic.py is legacy. "
        "Prefer tools/agama_generate_scm_disk_ic.py for production IC generation."
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Make AGAMA deterministic for sampling.
    np.random.seed(int(args.seed))

    # Ensure AGAMA uses the same (dimensionless) unit system as Odisseo code units.
    agama.setUnits(length=1, mass=1, velocity=1)

    # Runtime-like axisymmetric potential: NFW halo + Miyamoto-Nagai disk.
    pot = agama.Potential(
        dict(type="NFW", mass=float(args.halo_mass_code), scaleradius=float(args.halo_rs_code)),
        dict(
            type="MiyamotoNagai",
            mass=float(args.disk_mass_code),
            scaleradius=float(args.disk_radial_scale_code),
            scaleheight=max(1e-5, float(args.disk_height_code)),
        ),
    )

    # Quasi-isothermal disk DF in the same potential (tutorial-style),
    # then rescale Sigma0 so DF total mass matches target disk mass.
    df_params = dict(
        type="QuasiIsothermal",
        potential=pot,
        Sigma0=1.0,
        Rdisk=float(args.df_rdisk_code),
        sigmar0=float(args.sigma_r0_code),
        Rsigmar=float(args.sigma_r_scale_code),
        sigmamin=max(1e-6, float(args.sigma_r0_code) * float(args.sigma_min_frac)),
    )
    # Different AGAMA builds may accept slightly different capitalization for
    # vertical-structure keys; try robustly.
    hd = max(1e-5, float(args.df_hdisk_code))
    sz0 = float(args.sigma_z0_code)
    rsz = float(args.sigma_z_scale_code)
    trials = [
        dict(df_params, Hdisk=hd, sigmaz0=sz0, Rsigmaz=rsz),
        dict(df_params, hdisk=hd, sigmaz0=sz0, Rsigmaz=rsz),
        dict(df_params, Hdisk=hd, sigmaz0=sz0, rsigmaz=rsz),
        dict(df_params, hdisk=hd, sigmaz0=sz0, rsigmaz=rsz),
        dict(df_params, Hdisk=hd),
        dict(df_params, hdisk=hd),
        dict(df_params, sigmaz0=sz0, Rsigmaz=rsz),
        dict(df_params, sigmaz0=sz0, rsigmaz=rsz),
    ]
    df_tmp = None
    used = None
    last_err = None
    for trial in trials:
        try:
            df_tmp = agama.DistributionFunction(**trial)
            used = trial
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if df_tmp is None:
        raise RuntimeError(f"Failed to construct AGAMA QuasiIsothermal DF: {last_err}")
    m_tmp = float(df_tmp.totalMass())
    if not np.isfinite(m_tmp) or m_tmp <= 0.0:
        raise RuntimeError(f"AGAMA DF totalMass invalid: {m_tmp}")
    used["Sigma0"] *= float(args.disk_mass_code) / m_tmp
    df = agama.DistributionFunction(**used)

    gm = agama.GalaxyModel(pot, df)
    xv, m = gm.sample(int(args.n_particles))

    state_dtype = np.float64 if str(args.state_dtype) == "float64" else np.float32
    xv = np.asarray(xv, dtype=state_dtype)
    m = np.asarray(m, dtype=state_dtype)
    msum = float(np.sum(m))
    if msum <= 0.0:
        raise RuntimeError("AGAMA returned non-positive total sample mass")
    m = m * (float(args.disk_mass_code) / msum)
    state0 = np.stack((xv[:, :3], xv[:, 3:6]), axis=1)

    np.savez_compressed(
        out,
        state0=state0,
        mass=m,
        seed=np.asarray(int(args.seed)),
        n_particles=np.asarray(int(args.n_particles)),
        state_dtype=np.asarray(str(state0.dtype)),
        mass_dtype=np.asarray(str(m.dtype)),
        ic_velocity_potential=np.asarray("agama_quasiisothermal"),
        ic_uses_analytic_disk=np.asarray(True),
        ic_source=np.asarray("agama"),
        halo_mass_code=np.asarray(float(args.halo_mass_code)),
        halo_rs_code=np.asarray(float(args.halo_rs_code)),
        disk_mass_code=np.asarray(float(args.disk_mass_code)),
        disk_radial_scale_code=np.asarray(float(args.disk_radial_scale_code)),
        disk_height_code=np.asarray(float(args.disk_height_code)),
        sigma_r0_code=np.asarray(float(args.sigma_r0_code)),
        sigma_z0_code=np.asarray(float(args.sigma_z0_code)),
    )
    print(f"Saved AGAMA IC file: {out}")


if __name__ == "__main__":
    main()
