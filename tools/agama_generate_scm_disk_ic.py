#!/usr/bin/env python3
"""Generate rotating disk ICs from an AGAMA self-consistent model (SCM-style).

Strict mode is the default:
- no fallback sampling if SCM iteration fails;
- fail fast if the sampled ICs do not pass minimum rotation diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import agama
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--n-particles", type=int, default=200000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--state-dtype", type=str, default="float32", choices=("float32", "float64"))

    # Runtime-matched code units: length=1 (10 kpc), mass=1 (1e10 Msun), G=1.
    p.add_argument("--disk-mass-code", type=float, default=6.0)
    p.add_argument("--halo-mvir-code", type=float, default=100.0)
    p.add_argument("--halo-rs-code", type=float, default=2.0)
    p.add_argument("--rdisk-code", type=float, default=0.24)
    p.add_argument("--hdisk-code", type=float, default=0.03)
    p.add_argument("--qmin", type=float, default=1.6)
    p.add_argument("--rsigmar-code", type=float, default=0.48)
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--disk-df-type", type=str, default="exponential", choices=("exponential", "quasiisothermal"))
    p.add_argument("--jr0-code", type=float, default=None)
    p.add_argument("--jz0-code", type=float, default=None)
    p.add_argument("--jphi0-code", type=float, default=None)
    p.add_argument("--require-scm-convergence", action="store_true", default=True)
    p.add_argument("--allow-scm-fallback", action="store_true", default=False)
    p.add_argument("--min-prograde-frac", type=float, default=0.90)
    p.add_argument("--min-median-vphi-code", type=float, default=0.25)
    return p.parse_args()


def _make_prograde_exponential_df(*, mass: float, jr0: float, jz0: float, jphi0: float) -> object:
    # AGAMA's MW example uses a callable DF with explicit prograde-only support (Jphi > 0).
    add_j_den = 0.05 * jphi0
    add_j_vel = 0.25 * jphi0
    pr = 0.0
    pz = 0.0

    def df(j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=np.float64)
        jp = np.maximum(0.0, j[:, 2])
        jvel = jp + add_j_vel
        jden = jp + add_j_den
        xr = (jvel / jphi0) ** pr / jr0
        xz = (jvel / jphi0) ** pz / jz0
        fr = xr * np.exp(-xr * j[:, 0])
        fz = xz * np.exp(-xz * j[:, 1])
        fp = np.abs(j[:, 2]) * np.exp(-jden / jphi0) / (jphi0**2)
        return np.where(j[:, 2] > 0.0, fr * fz * fp, 0.0)

    norm = float(mass) / float(agama.DistributionFunction(df).totalMass())

    def normalized_df(j: np.ndarray) -> np.ndarray:
        return norm * df(j)

    return normalized_df


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Keep AGAMA in default dimensionless units so gravity uses G=1, matching ODISSEO.

    rdisk = float(args.rdisk_code)
    hdisk = float(args.hdisk_code)
    disk_mass_code = float(args.disk_mass_code)
    halo_mvir_code = float(args.halo_mvir_code)
    halo_rs_code = float(args.halo_rs_code)

    diskparams = dict(type="Disk", scaleRadius=rdisk, scaleHeight=-hdisk, surfaceDensity=1.0)
    m0disk = float(agama.Density(diskparams).totalMass())
    if not np.isfinite(m0disk) or m0disk <= 0.0:
        raise RuntimeError("Invalid unscaled AGAMA disk mass.")
    diskparams["surfaceDensity"] *= float(disk_mass_code) / m0disk
    pottotal = agama.Potential(
        dict(type="NFW", mass=halo_mvir_code, scaleradius=halo_rs_code),
        diskparams,
    )

    if str(args.disk_df_type).lower() == "quasiisothermal":
        # Set disk radial dispersion from Toomre-Q target.
        sigmar0 = 1.0
        rsigmar = float(args.rsigmar_code)

        def toomre_q(r: np.ndarray) -> np.ndarray:
            force, deriv = pottotal.eval(np.column_stack([r, r * 0, r * 0]), acc=True, der=True)
            kappa = (-deriv.T[0] - 3 * force.T[0] / r) ** 0.5
            sigma = diskparams["surfaceDensity"] * np.exp(-r / diskparams["scaleRadius"])
            sigmar = sigmar0 * np.exp(-r / rsigmar)
            return sigmar * kappa / np.maximum(sigma * 3.36, 1e-12)

        rr = np.logspace(-2, 1.5, 301)
        sigmar0 *= float(args.qmin) / np.min(toomre_q(rr))

        rdisk_factor = 1.0 - 0.025 * float(args.qmin)
        dfdiskparams = dict(
            type="QuasiIsothermal",
            potential=pottotal,
            Sigma0=diskparams["surfaceDensity"],
            Rdisk=rdisk * rdisk_factor,
            Hdisk=hdisk,
            sigmar0=sigmar0,
            Rsigmar=rsigmar,
            sigmamin=max(1e-6, sigmar0 * 0.01),
        )
        dfdiskparams["Sigma0"] *= agama.Density(diskparams).totalMass() / agama.DistributionFunction(**dfdiskparams).totalMass()
        dfdisk = agama.DistributionFunction(**dfdiskparams)
    else:
        # Prograde-only exponential DF following AGAMA's MW example pattern.
        v_circ_ref = float(np.sqrt(max(1e-12, -rdisk * pottotal.force(rdisk, 0.0, 0.0)[0])))
        jphi0 = float(args.jphi0_code) if args.jphi0_code is not None else max(1e-4, rdisk * v_circ_ref)
        jr0 = float(args.jr0_code) if args.jr0_code is not None else 0.18 * jphi0
        jz0 = float(args.jz0_code) if args.jz0_code is not None else 0.06 * jphi0
        dfdisk = _make_prograde_exponential_df(
            mass=float(agama.Density(diskparams).totalMass()),
            jr0=jr0,
            jz0=jz0,
            jphi0=jphi0,
        )

    dfhalo = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=pottotal,
        density=agama.Density(type="NFW", mass=halo_mvir_code, scaleradius=halo_rs_code),
        beta0=0.0,
        rotfrac=0.0,
    )

    use_potential = pottotal
    use_af = None
    scm_failed: Exception | None = None
    if int(args.iterations) > 0:
        model = agama.SelfConsistentModel(
            rminSph=0.02,
            rmaxSph=max(30.0, 15 * halo_rs_code),
            sizeRadialSph=36,
            lmaxAngularSph=6,
            RminCyl=0.02,
            RmaxCyl=max(4.0, 10 * rdisk),
            sizeRadialCyl=24,
            zminCyl=max(1e-3, 0.2 * hdisk),
            zmaxCyl=max(2.0, 10 * rdisk),
            sizeVerticalCyl=20,
        )
        model.potential = pottotal
        model.components.append(
            agama.Component(
                df=dfdisk,
                disklike=True,
                rminCyl=0.02,
                rmaxCyl=max(3.0, 8 * rdisk),
                sizeRadialCyl=20,
                zminCyl=max(1e-3, 0.3 * hdisk),
                zmaxCyl=max(1.0, 10 * hdisk),
                sizeVerticalCyl=14,
            )
        )
        model.components.append(
            agama.Component(
                df=dfhalo,
                disklike=False,
                rminSph=0.02,
                rmaxSph=max(30.0, 15 * halo_rs_code),
                sizeRadialSph=24,
                lmaxAngularSph=6,
            )
        )
        try:
            for _ in range(int(args.iterations)):
                model.iterate()
            use_potential = model.potential
            use_af = model.af
        except Exception as exc:  # noqa: BLE001
            scm_failed = exc

    strict_convergence = bool(args.require_scm_convergence) and not bool(args.allow_scm_fallback)
    if scm_failed is not None and strict_convergence:
        raise RuntimeError(
            "SCM iteration failed in strict mode. "
            "No fallback sampling is permitted; tune IC/DF parameters and rerun."
        ) from scm_failed
    if scm_failed is not None:
        print(f"[warn] SCM iteration failed; using initial potential because --allow-scm-fallback is set: {scm_failed}")

    if use_af is None:
        model_disk = agama.GalaxyModel(potential=use_potential, df=dfdisk)
    else:
        model_disk = agama.GalaxyModel(potential=use_potential, df=dfdisk, af=use_af)
    xv, m = model_disk.sample(int(args.n_particles))
    xv = np.asarray(xv)
    m = np.asarray(m)
    m = m * (float(disk_mass_code) / max(float(np.sum(m)), 1e-12))

    dtype = np.float64 if str(args.state_dtype) == "float64" else np.float32
    state0 = np.stack((xv[:, :3], xv[:, 3:6]), axis=1).astype(dtype, copy=False)
    mass = m.astype(dtype, copy=False)

    # Strict acceptance gate: reject ICs without clear net prograde rotation.
    pos = state0[:, 0, :].astype(np.float64)
    vel = state0[:, 1, :].astype(np.float64)
    rxy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) + 1e-12
    vphi = (-pos[:, 1] * vel[:, 0] + pos[:, 0] * vel[:, 1]) / rxy
    prograde_frac = float(np.mean(vphi > 0.0))
    median_vphi = float(np.median(vphi))
    if prograde_frac < float(args.min_prograde_frac) or median_vphi < float(args.min_median_vphi_code):
        raise RuntimeError(
            "Generated ICs failed rotation acceptance gate: "
            f"prograde_frac={prograde_frac:.4f} (min={float(args.min_prograde_frac):.4f}), "
            f"median_vphi={median_vphi:.4f} (min={float(args.min_median_vphi_code):.4f})."
        )

    np.savez_compressed(
        out,
        state0=state0,
        mass=mass,
        seed=np.asarray(int(args.seed)),
        n_particles=np.asarray(int(args.n_particles)),
        state_dtype=np.asarray(str(state0.dtype)),
        mass_dtype=np.asarray(str(mass.dtype)),
        ic_source=np.asarray("agama_scm"),
        ic_velocity_potential=np.asarray("nfw"),
        ic_uses_analytic_disk=np.asarray(True),
        halo_mass_code=np.asarray(float(halo_mvir_code)),
        halo_rs_code=np.asarray(float(halo_rs_code)),
        disk_mass_code=np.asarray(float(disk_mass_code)),
        disk_radial_scale_code=np.asarray(rdisk),
        disk_height_code=np.asarray(hdisk),
        rdisk_code=np.asarray(rdisk),
        hdisk_code=np.asarray(hdisk),
        qmin=np.asarray(float(args.qmin)),
        runtime_potential_match=np.asarray(True),
        ic_prograde_fraction=np.asarray(prograde_frac),
        ic_median_vphi_code=np.asarray(median_vphi),
        scm_converged=np.asarray(scm_failed is None),
        scm_iterations=np.asarray(int(args.iterations)),
        strict_convergence=np.asarray(strict_convergence),
    )
    print(f"Saved AGAMA SCM IC file: {out}")


if __name__ == "__main__":
    main()
