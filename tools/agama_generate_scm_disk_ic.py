#!/usr/bin/env python3
"""Generate rotating disk (+ optional bulge) ICs from an AGAMA self-consistent model.

Strict mode is the default:
- no fallback sampling if SCM iteration fails;
- fail fast if the sampled ICs do not pass minimum rotation diagnostics.

The NFW halo stays ANALYTIC and is not sampled -- the output carries
``halo_mass_code``/``halo_rs_code`` so the rollout can add that term per particle.
The bulge, when requested, IS sampled and therefore self-gravitating, which is the
whole point of asking for one: a live bulge changes the central potential the disc
sees, an analytic one does not.

Two things the bulge forces that a disc-only IC never had to think about:

**The rotation gate must look at the DISC alone.** A pressure-supported bulge has
``prograde_frac`` near 0.5 by construction, so a gate over the concatenated
population reads a perfectly good IC as a failure -- and, worse, could pass a disc
that had stopped rotating if the bulge fraction happened to compensate.

**The output is SHUFFLED.** Downstream the mesh rollout trims N to a multiple of
``ndev * leaf_size`` by taking a PREFIX. Concatenated disc-then-bulge, that prefix
trim deletes bulge particles only -- silently changing the mass ratio that was
asked for. Use ``--quantum`` to emit an exactly-divisible N and skip the trim
entirely; the shuffle is the belt to that braces.
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
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--disk-df-type", type=str, default="exponential", choices=("exponential", "quasiisothermal"))
    p.add_argument("--jr0-code", type=float, default=None)
    p.add_argument("--jz0-code", type=float, default=None)
    p.add_argument("--jphi0-code", type=float, default=None)
    # Relative warmth knobs for the exponential DF (radial/vertical action scale
    # as a fraction of jphi0). Larger jr0 -> hotter disk -> higher Toomre Q ->
    # suppresses ring/axisymmetric instabilities (Q ~ sqrt(jr0)). Used only when
    # the absolute --jr0-code/--jz0-code are not given.
    p.add_argument("--jr0-factor", type=float, default=0.38)
    p.add_argument("--jz0-factor", type=float, default=0.10)
    # Bulge (sampled, self-gravitating). 0.0 disables it and reproduces the
    # disc-only IC exactly -- the shuffle is skipped in that case too.
    p.add_argument("--bulge-mass-code", type=float, default=0.0,
                   help="Bulge mass in code units; 0 = no bulge (disc-only, as before).")
    p.add_argument("--bulge-scale-code", type=float, default=0.08,
                   help="Bulge Dehnen scale radius a. gamma=1 gives Hernquist, whose "
                        "half-mass radius is a*(1+sqrt(2)) = 2.414a.")
    p.add_argument("--bulge-gamma", type=float, default=1.0,
                   help="Dehnen inner slope. 1.0 = Hernquist (classical bulge).")
    p.add_argument("--bulge-beta0", type=float, default=0.0,
                   help="Velocity anisotropy of the bulge QuasiSpherical DF.")
    p.add_argument("--bulge-rotfrac", type=float, default=0.0,
                   help="Fraction of the bulge DF put into net rotation (0 = pressure "
                        "supported).")
    # Emit an N the mesh rollout can take whole: it requires N = ndev * k * leaf_size
    # so that cap == count and no device is padded.
    # Both DFs have infinite support, and at these particle counts the extreme tail
    # gets populated: a Hernquist bulge has M(>r) ~ 2a/r, so the expected largest
    # radius among N particles is ~2aN -- 5.6e5 code units at a=0.08 and N=3.5e6,
    # measured. That single particle sets the tree's bounding box, which collapses
    # Morton resolution for every real particle. Clip it.
    p.add_argument("--rmax-code", type=float, default=20.0,
                   help="Discard sampled particles beyond this spherical radius and "
                        "resample to hit the requested N exactly. 20 = 10x the halo "
                        "scale radius, and clips 0.8%% of a Hernquist bulge's mass. "
                        "0 disables (keeps the raw infinite tail).")
    p.add_argument("--quantum", type=int, default=0,
                   help="Round the total particle count DOWN to a multiple of this "
                        "(set it to ndev*leaf_size). 0 = emit --n-particles exactly.")
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


def _sample_truncated(
    model: object, n: int, rmax: float, label: str
) -> "np.ndarray":
    """Sample exactly ``n`` particles with spherical radius <= ``rmax``.

    Rejection with top-up rather than one oversampled draw, because the acceptance
    fraction is not known in advance: it depends on the DF, the converged potential
    and ``rmax`` together. Truncating the profile is the POINT here, not a side
    effect -- the discarded particles are the ones whose only physical role would be
    to blow up the tree's bounding box.

    Returns POSITIONS AND VELOCITIES ONLY. AGAMA's ``sample(k)`` spreads the DF's
    whole mass over the ``k`` particles it was asked for, so a large first draw and a
    small top-up draw come back with per-particle masses differing by the ratio of
    their sizes. Concatenating those and rescaling the total silently produces a
    two-population IC -- measured: 1.17e-04 against 6.09e-05 in the same component.
    The caller assigns one uniform mass instead, which is what a single draw returns
    anyway (verified below), and is the discretisation we want: unequal masses would
    make the heavy species sink by dynamical friction.
    """
    if rmax <= 0.0:
        xv, _m = model.sample(n)  # type: ignore[attr-defined]
        return np.asarray(xv, dtype=np.float64)

    keep_xv: list = []
    have = 0
    request = n
    tries = 0
    while have < n:
        tries += 1
        if tries > 20:
            raise RuntimeError(
                f"{label}: could not reach {n} particles within rmax={rmax} after 20 "
                f"draws (have {have}); rmax is probably too small for this profile."
            )
        xv, m = model.sample(int(request))  # type: ignore[attr-defined]
        xv = np.asarray(xv, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        if tries == 1 and m.size > 1:
            # The uniform-mass assumption the caller relies on. If AGAMA ever starts
            # returning a mass spectrum from a single draw, this must be revisited
            # rather than quietly averaged away.
            spread = float(m.max() / max(m.min(), 1e-300))
            if not (spread < 1.0 + 1e-6):
                raise RuntimeError(
                    f"{label}: AGAMA returned non-uniform particle masses from one "
                    f"draw (max/min = {spread:.6g}); the uniform-mass assignment in "
                    f"the caller is no longer valid."
                )
        r = np.sqrt((xv[:, :3] ** 2).sum(axis=1))
        ok = r <= rmax
        keep_xv.append(xv[ok])
        have += int(ok.sum())
        frac = max(float(ok.mean()), 1e-3)
        if tries == 1:
            print(
                f"[info] {label}: {100.0 * (1.0 - frac):.3f} % of the draw exceeded "
                f"rmax={rmax:g} and was discarded"
            )
        # Ask for the shortfall plus 5 % headroom so this normally ends in two draws.
        request = int(np.ceil((n - have) / frac * 1.05)) + 1
    return np.concatenate(keep_xv, axis=0)[:n]


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

    # The bulge enters the TOTAL potential before anything else is derived from it.
    # Every downstream quantity -- the Toomre-Q normalisation, jphi0 from v_circ, the
    # QuasiSpherical halo DF, and the SCM iteration itself -- reads `pottotal`, so a
    # bulge added later would leave the disc DF tuned for a galaxy that does not exist.
    bulge_mass_code = float(args.bulge_mass_code)
    bulge_scale = float(args.bulge_scale_code)
    bulge_gamma = float(args.bulge_gamma)
    want_bulge = bulge_mass_code > 0.0
    if want_bulge and not (bulge_scale > 0.0):
        raise RuntimeError(f"--bulge-scale-code must be > 0, got {bulge_scale!r}")
    bulgeparams = (
        dict(type="Dehnen", mass=bulge_mass_code, scaleRadius=bulge_scale, gamma=bulge_gamma)
        if want_bulge
        else None
    )

    pot_components = [dict(type="NFW", mass=halo_mvir_code, scaleradius=halo_rs_code), diskparams]
    if bulgeparams is not None:
        pot_components.append(bulgeparams)
    pottotal = agama.Potential(*pot_components)

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
        jr0 = float(args.jr0_code) if args.jr0_code is not None else float(args.jr0_factor) * jphi0
        jz0 = float(args.jz0_code) if args.jz0_code is not None else float(args.jz0_factor) * jphi0
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

    # The bulge DF is QuasiSpherical against its OWN density in the TOTAL potential --
    # the same construction as the halo. Solving for f(E,L) in the full potential is
    # what makes the sampled bulge an equilibrium of the galaxy it is actually in,
    # rather than of an isolated Hernquist sphere that would then relax on contact.
    dfbulge = (
        agama.DistributionFunction(
            type="QuasiSpherical",
            potential=pottotal,
            density=agama.Density(**bulgeparams),
            beta0=float(args.bulge_beta0),
            rotfrac=float(args.bulge_rotfrac),
        )
        if bulgeparams is not None
        else None
    )

    use_potential = pottotal
    use_af = None
    scm_failed: Exception | None = None
    if int(args.iterations) > 0:
        model = agama.SelfConsistentModel(
            # The global spherical grid has to resolve the innermost component, which
            # with a bulge is the bulge and not the halo.
            rminSph=min(0.02, max(1e-4, 0.05 * bulge_scale)) if want_bulge else 0.02,
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
        if dfbulge is not None:
            # The bulge grid must reach INSIDE the disc's rminCyl=0.02: a Hernquist
            # cusp puts a real mass fraction within 0.2a, and a component whose grid
            # starts outside its own core iterates to a hollowed-out centre. Hence
            # rminSph tied to the bulge scale, not to the shared 0.02.
            model.components.append(
                agama.Component(
                    df=dfbulge,
                    disklike=False,
                    rminSph=max(1e-4, 0.05 * bulge_scale),
                    rmaxSph=max(5.0, 50 * bulge_scale),
                    sizeRadialSph=30,
                    lmaxAngularSph=6,
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

    # ---- particle budget -------------------------------------------------------
    # Split N between the components at EQUAL PARTICLE MASS. Unequal masses would
    # make the heavier species sink by dynamical friction against the lighter one --
    # a numerical secular effect that looks exactly like the bulge growth this run
    # is meant to measure.
    n_total = int(args.n_particles)
    quantum = int(args.quantum)
    if quantum > 0:
        n_rounded = (n_total // quantum) * quantum
        if n_rounded <= 0:
            raise RuntimeError(
                f"--quantum {quantum} exceeds --n-particles {n_total}; nothing to emit."
            )
        if n_rounded != n_total:
            print(f"[info] rounding N {n_total} -> {n_rounded} (multiple of {quantum})")
        n_total = n_rounded

    total_mass_code = disk_mass_code + (bulge_mass_code if want_bulge else 0.0)
    if want_bulge:
        n_disk = int(round(n_total * disk_mass_code / total_mass_code))
        n_disk = min(max(n_disk, 1), n_total - 1)
        n_bulge = n_total - n_disk
    else:
        n_disk, n_bulge = n_total, 0

    # ---- sample ----------------------------------------------------------------
    af_kw = {} if use_af is None else dict(af=use_af)
    rmax_code = float(args.rmax_code)
    model_disk = agama.GalaxyModel(potential=use_potential, df=dfdisk, **af_kw)
    xv_d = _sample_truncated(model_disk, n_disk, rmax_code, "disc")
    # One uniform mass per component, carrying that component's whole target mass --
    # which also puts the clipped tail's mass back into the particles that remain.
    # The counts were split BY MASS, so m_disc == m_bulge to rounding.
    m_d = np.full(n_disk, disk_mass_code / n_disk, dtype=np.float64)

    if n_bulge > 0:
        model_bulge = agama.GalaxyModel(potential=use_potential, df=dfbulge, **af_kw)
        xv_b = _sample_truncated(model_bulge, n_bulge, rmax_code, "bulge")
        m_b = np.full(n_bulge, bulge_mass_code / n_bulge, dtype=np.float64)
        xv = np.concatenate((xv_d, xv_b), axis=0)
        m = np.concatenate((m_d, m_b), axis=0)
        component = np.concatenate(
            (np.zeros(n_disk, np.int8), np.ones(n_bulge, np.int8)), axis=0
        )
    else:
        xv, m, component = xv_d, m_d, np.zeros(n_disk, np.int8)

    # ---- strict acceptance gate, on the DISC ONLY -------------------------------
    # A pressure-supported bulge sits at prograde_frac ~ 0.5 and median vphi ~ 0, so
    # gating the concatenated population measures the mass ratio, not the disc's
    # rotation. Evaluate BEFORE the shuffle, while the component slices are still
    # contiguous -- and gate on the component that is supposed to rotate.
    def _rotation_diagnostics(sl: slice) -> tuple[float, float]:
        pos_ = xv[sl, :3]
        vel_ = xv[sl, 3:6]
        rxy_ = np.sqrt(pos_[:, 0] ** 2 + pos_[:, 1] ** 2) + 1e-12
        vphi_ = (-pos_[:, 1] * vel_[:, 0] + pos_[:, 0] * vel_[:, 1]) / rxy_
        return float(np.mean(vphi_ > 0.0)), float(np.median(vphi_))

    prograde_frac, median_vphi = _rotation_diagnostics(slice(0, n_disk))
    if prograde_frac < float(args.min_prograde_frac) or median_vphi < float(args.min_median_vphi_code):
        raise RuntimeError(
            "Generated DISC ICs failed rotation acceptance gate: "
            f"prograde_frac={prograde_frac:.4f} (min={float(args.min_prograde_frac):.4f}), "
            f"median_vphi={median_vphi:.4f} (min={float(args.min_median_vphi_code):.4f})."
        )
    if n_bulge > 0:
        bulge_prograde_frac, bulge_median_vphi = _rotation_diagnostics(
            slice(n_disk, n_total)
        )
        print(
            f"[info] disc  n={n_disk:,} prograde={prograde_frac:.4f} "
            f"median_vphi={median_vphi:.4f}"
        )
        print(
            f"[info] bulge n={n_bulge:,} prograde={bulge_prograde_frac:.4f} "
            f"median_vphi={bulge_median_vphi:.4f}"
        )
    else:
        bulge_prograde_frac, bulge_median_vphi = float("nan"), float("nan")

    # ---- shuffle ---------------------------------------------------------------
    # See the module docstring: the rollout trims N with a PREFIX, which on a
    # disc-then-bulge concatenation deletes bulge particles only. Shuffle so any
    # prefix is a fair sample of both. Skipped with no bulge so the disc-only IC is
    # unchanged from the runs already on record.
    if n_bulge > 0:
        perm = np.random.default_rng(int(args.seed)).permutation(n_total)
        xv = xv[perm]
        m = m[perm]
        component = component[perm]

    dtype = np.float64 if str(args.state_dtype) == "float64" else np.float32
    state0 = np.stack((xv[:, :3], xv[:, 3:6]), axis=1).astype(dtype, copy=False)
    mass = m.astype(dtype, copy=False)

    np.savez_compressed(
        out,
        state0=state0,
        mass=mass,
        # 0 = disc, 1 = bulge, aligned with the SHUFFLED rows. Kept so a diagnostic
        # can separate the components after the fact; nothing in the rollout needs it.
        component=component,
        n_disk=np.asarray(int(n_disk)),
        n_bulge=np.asarray(int(n_bulge)),
        ic_has_bulge=np.asarray(bool(n_bulge > 0)),
        ic_shuffled=np.asarray(bool(n_bulge > 0)),
        bulge_mass_code=np.asarray(float(bulge_mass_code if want_bulge else 0.0)),
        bulge_scale_code=np.asarray(float(bulge_scale if want_bulge else 0.0)),
        bulge_gamma=np.asarray(float(bulge_gamma if want_bulge else 0.0)),
        bulge_beta0=np.asarray(float(args.bulge_beta0)),
        bulge_rotfrac=np.asarray(float(args.bulge_rotfrac)),
        bulge_prograde_fraction=np.asarray(float(bulge_prograde_frac)),
        bulge_median_vphi_code=np.asarray(float(bulge_median_vphi)),
        rmax_code=np.asarray(float(args.rmax_code)),
        total_baryon_mass_code=np.asarray(float(total_mass_code)),
        particle_mass_code=np.asarray(float(total_mass_code / max(n_total, 1))),
        seed=np.asarray(int(args.seed)),
        n_particles=np.asarray(int(n_total)),
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
    print(
        f"  N={n_total:,} (disc {n_disk:,} + bulge {n_bulge:,}), "
        f"M_disc={disk_mass_code:g} M_bulge={bulge_mass_code if want_bulge else 0.0:g}, "
        f"m_particle={total_mass_code / max(n_total, 1):.6e}"
    )


if __name__ == "__main__":
    main()
