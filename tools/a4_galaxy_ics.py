#!/usr/bin/env python3
"""Galaxy initial conditions for jaccpot's accuracy suite (multi-node plan A4).

The models of the 2026-10-31 production runs (code units: 10 kpc, 1e10 Msun, G = 1;
the production disc 6 / Rd 0.24 / h 0.03 in a LIVE NFW halo 100 / rs 2 cut off by
exp(-(r/rcut)^2), self-consistent with agama's SCM):

``galaxy``  the disc + Hernquist bulge (1.2 / 0.08) galaxy with the production action
            DF (Q ~ 2: a stable, featureless disc). The disc-heating problem and the
            merger's progenitors. (``/export/scratch/tbuck/merger-2026-10-31/gen_galaxy.py``)
``bar``     the bar-unstable disc: no bulge, a QuasiIsothermal disc at Toomre Q 1.2 at
            2 Rd. (``/export/scratch/tbuck/bar-2026-10-31/gen_disc.py``)
``merger``  two ``galaxy`` realisations (seeds ``s`` and ``s + 1``) on a parabolic orbit
            from 100 kpc, nominal pericentre 4 kpc, spins tilted (30, 0) and (60, 150) deg
            -- the 2026-10-31 merger (``assemble.py``, ``orbit.py``).

Output: jaccpot ``bench/accuracy/ics.py``'s ``.npz`` format -- ``state0`` (N, 2, 3)
float32, ``mass`` (N,) float32, ``component`` (N,) int8, ``component_names``, ``meta``
(JSON). Every component block is shuffled, so a per-component prefix is a fair
subsample (the accuracy suite's nested N ladders).

agama is seeded (``agama.setRandomSeed``) as well as numpy: before 2026-11-01 the bar
generator seeded numpy only, and two "seeds" drew the same particles.

Needs agama (the odisseo env: ``micromamba run -n odisseo python tools/a4_galaxy_ics.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import agama
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agama_generate_scm_disk_ic import (  # noqa: E402
    _make_prograde_exponential_df,
    _sample_truncated,
)

DISK = dict(mass=6.0, rdisk=0.24, hdisk=0.03)
HALO = dict(mass=100.0, rs=2.0, rcut=10.0)


def _halo_params(halo_mass: float, rs: float, rcut: float) -> dict:
    return dict(
        type="Spheroid", densityNorm=halo_mass / (4 * np.pi * rs**3), scaleRadius=rs,
        alpha=1.0, beta=3.0, gamma=1.0, outerCutoffRadius=rcut, cutoffStrength=2.0,
    )


def _disk_params(mass: float, rdisk: float, hdisk: float) -> dict:
    p = dict(type="Disk", scaleRadius=rdisk, scaleHeight=-hdisk, surfaceDensity=1.0)
    p["surfaceDensity"] *= mass / agama.Density(p).totalMass()
    return p


def _scm(pottotal, comps_spec, a) -> object:
    rmax_sph = max(40.0, 3 * a.halo_rcut)
    model = agama.SelfConsistentModel(
        rminSph=1e-3, rmaxSph=rmax_sph, sizeRadialSph=40, lmaxAngularSph=6,
        RminCyl=0.02, RmaxCyl=max(4.0, 10 * a.rdisk), sizeRadialCyl=24,
        zminCyl=max(1e-3, 0.2 * a.hdisk), zmaxCyl=max(2.0, 10 * a.rdisk), sizeVerticalCyl=20,
    )
    model.potential = pottotal
    for kind, df in comps_spec:
        if kind == "disc":
            model.components.append(agama.Component(
                df=df, disklike=True, rminCyl=0.02, rmaxCyl=max(3.0, 8 * a.rdisk), sizeRadialCyl=20,
                zminCyl=max(1e-3, 0.3 * a.hdisk), zmaxCyl=max(1.0, 10 * a.hdisk), sizeVerticalCyl=14,
            ))
        elif kind == "bulge":
            model.components.append(agama.Component(
                df=df, disklike=False, rminSph=max(1e-4, 0.05 * a.bulge_scale),
                rmaxSph=max(5.0, 50 * a.bulge_scale), sizeRadialSph=30, lmaxAngularSph=6,
            ))
        else:
            model.components.append(agama.Component(
                df=df, disklike=False, rminSph=0.02, rmaxSph=rmax_sph, sizeRadialSph=30, lmaxAngularSph=6,
            ))
    t0 = time.time()
    for it in range(a.iterations):
        model.iterate()
        print(f"[scm] iteration {it + 1}/{a.iterations} ({time.time() - t0:.0f} s)", flush=True)
    return model


def sample_model(a: argparse.Namespace, kind: str, seed: int) -> dict:
    """One galaxy (``kind`` ``galaxy`` or ``bar``) in the suite's format.

    Parameters
    ----------
    a : argparse.Namespace
        Model and budget parameters.
    kind : str
        ``galaxy`` or ``bar``.
    seed : int
        numpy and agama seed.

    Returns
    -------
    dict
        Arrays and ``meta`` (dict).
    """
    np.random.seed(seed)
    agama.setRandomSeed(seed)
    agama.setNumThreads(0)
    diskparams = _disk_params(a.disk_mass, a.rdisk, a.hdisk)
    haloparams = _halo_params(a.halo_mass, a.halo_rs, a.halo_rcut)
    halo_den = agama.Density(**haloparams)
    halo_total = float(halo_den.totalMass())
    has_bulge = a.bulge_mass > 0
    pots = [haloparams, diskparams]
    if has_bulge:
        bulgeparams = dict(type="Dehnen", mass=a.bulge_mass, scaleRadius=a.bulge_scale, gamma=1.0)
        pots.append(bulgeparams)
    pottotal = agama.Potential(*pots)
    if kind == "galaxy":
        v_circ = float(np.sqrt(-a.rdisk * pottotal.force(a.rdisk, 0.0, 0.0)[0]))
        jphi0 = a.rdisk * v_circ
        dfdisk = _make_prograde_exponential_df(
            mass=a.disk_mass, jr0=a.jr0_factor * jphi0, jz0=a.jz0_factor * jphi0, jphi0=jphi0
        )
    else:
        sigma0 = a.disk_mass / (2 * np.pi * a.rdisk**2)
        r2, h = 2 * a.rdisk, 1e-4
        vc2 = lambda r: -r * pottotal.force(r, 0.0, 0.0)[0]  # noqa: E731
        om2 = vc2(r2) / r2**2
        dom2 = (vc2(r2 + h) / (r2 + h) ** 2 - vc2(r2 - h) / (r2 - h) ** 2) / (2 * h)
        kappa = np.sqrt(4 * om2 + r2 * dom2)
        sig_r2 = a.q_target * 3.36 * sigma0 * np.exp(-2.0) / kappa
        dfdisk = agama.DistributionFunction(
            type="QuasiIsothermal", Sigma0=sigma0, Rdisk=a.rdisk, Hdisk=a.hdisk,
            sigmar0=sig_r2 * np.e, Rsigmar=2 * a.rdisk, potential=pottotal,
        )
    spec = [("disc", dfdisk)]
    budget = [(dfdisk, a.disk_mass, "disc")]
    if has_bulge:
        dfbulge = agama.DistributionFunction(
            type="QuasiSpherical", potential=pottotal, density=agama.Density(**bulgeparams),
            beta0=0.0, rotfrac=0.0,
        )
        spec.append(("bulge", dfbulge))
        budget.append((dfbulge, a.bulge_mass, "bulge"))
    dfhalo = agama.DistributionFunction(
        type="QuasiSpherical", potential=pottotal, density=halo_den, beta0=0.0, rotfrac=0.0
    )
    spec.append(("halo", dfhalo))
    budget.append((dfhalo, halo_total, "halo"))
    model = _scm(pottotal, spec, a)

    n = a.n_particles
    m = (a.disk_mass + a.bulge_mass + halo_total / a.halo_mass_factor) / n
    counts = [int(round(b[1] / m)) for b in budget[:-1]]
    counts.append(n - sum(counts))
    rng = np.random.default_rng(seed)
    xvs, ms, cs = [], [], []
    for k, ((df, mass_tot, label), cnt) in enumerate(zip(budget, counts)):
        gm = agama.GalaxyModel(potential=model.potential, df=df, af=model.af)
        xv = _sample_truncated(gm, cnt, a.rmax, label)[rng.permutation(cnt)]
        xvs.append(xv.astype(np.float32))
        ms.append(np.full(cnt, mass_tot / cnt, np.float32))
        cs.append(np.full(cnt, k, np.int8))
        print(f"[sample] {label} {cnt:,}", flush=True)
    xv = np.concatenate(xvs)
    meta = dict(kind=kind, seed=seed, counts=counts, halo_total_mass=halo_total,
                units="10 kpc, 1e10 Msun, G = 1; time unit 149.1 Myr",
                params={k: v for k, v in vars(a).items() if k not in ("func",)})
    return dict(
        state0=np.stack((xv[:, :3], xv[:, 3:6]), axis=1), mass=np.concatenate(ms),
        component=np.concatenate(cs), component_names=np.array([b[2] for b in budget]), meta=meta,
    )


def _spin_rotation(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    rz = np.array([[np.cos(p), -np.sin(p), 0], [np.sin(p), np.cos(p), 0], [0, 0, 1]])
    return rz @ ry


def _relative_orbit(m_tot: float, r0: float, rp: float) -> tuple:
    """Point-mass parabola: relative position/velocity of B w.r.t. A, L along +z."""
    v = np.sqrt(2 * m_tot / r0)
    vt = np.sqrt(2 * m_tot * rp) / r0
    return np.array([r0, 0.0, 0.0]), np.array([-np.sqrt(v * v - vt * vt), vt, 0.0])


def assemble(gal_a: dict, gal_b: dict, a: argparse.Namespace) -> dict:
    """Two progenitors on the parabolic orbit, in the system's rest frame.

    Parameters
    ----------
    gal_a : dict
        Progenitor A.
    gal_b : dict
        Progenitor B.
    a : argparse.Namespace
        ``r0``, ``rp``, ``spin_a``, ``spin_b``.

    Returns
    -------
    dict
        The merger IC (components ``A_disc, A_bulge, A_halo, B_disc, ...``).
    """
    m_gal = float(np.sum(gal_a["mass"].astype(np.float64)))
    d, w = _relative_orbit(2 * m_gal, a.r0, a.rp)
    states, masses, comps, names = [], [], [], []
    for k, (tag, g, spin, sign) in enumerate((("A", gal_a, a.spin_a, -1.0), ("B", gal_b, a.spin_b, 1.0))):
        st = g["state0"].astype(np.float64)
        m = g["mass"].astype(np.float64)
        st[:, 0] -= (m[:, None] * st[:, 0]).sum(0) / m.sum()
        st[:, 1] -= (m[:, None] * st[:, 1]).sum(0) / m.sum()
        rot = _spin_rotation(*spin)
        st[:, 0] = st[:, 0] @ rot.T + sign * 0.5 * d
        st[:, 1] = st[:, 1] @ rot.T + sign * 0.5 * w
        states.append(st)
        masses.append(m)
        comps.append(g["component"].astype(np.int8) + 3 * k)
        names += [f"{tag}_{nm}" for nm in g["component_names"]]
    state = np.concatenate(states)
    mass = np.concatenate(masses)
    for k in range(2):
        state[:, k] -= (mass[:, None] * state[:, k]).sum(0) / mass.sum()
    meta = dict(kind="merger", r0=a.r0, rp_nominal=a.rp, spin_a=list(a.spin_a), spin_b=list(a.spin_b),
                galaxy_mass=m_gal, d0=d.tolist(), w0=w.tolist(), seeds=[gal_a["meta"]["seed"], gal_b["meta"]["seed"]],
                units=gal_a["meta"]["units"], progenitor=gal_a["meta"]["params"])
    return dict(state0=state.astype(np.float32), mass=mass.astype(np.float32), component=np.concatenate(comps),
                component_names=np.array(names), meta=meta)


def save(path: str, ic: dict) -> None:
    """Write ``ic`` (``meta`` as a JSON string).

    Parameters
    ----------
    path : str
        Target ``.npz``.
    ic : dict
        Arrays and ``meta``.
    """
    out = dict(ic)
    out["meta"] = json.dumps(ic["meta"])
    np.savez(path, **out)
    print(f"[done] {path} N={ic['mass'].shape[0]:,}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("galaxy", "bar", "merger"))
    p.add_argument("--output", required=True)
    p.add_argument("--n-particles", type=int, required=True, help="per galaxy (merger: each progenitor)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--disk-mass", type=float, default=DISK["mass"])
    p.add_argument("--rdisk", type=float, default=DISK["rdisk"])
    p.add_argument("--hdisk", type=float, default=DISK["hdisk"])
    p.add_argument("--jr0-factor", type=float, default=0.38, help="galaxy: the production action DF")
    p.add_argument("--jz0-factor", type=float, default=0.10)
    p.add_argument("--q-target", type=float, default=1.2, help="bar: Toomre Q at 2 Rd")
    p.add_argument("--bulge-mass", type=float, default=None, help="default 1.2 (galaxy, merger), 0 (bar)")
    p.add_argument("--bulge-scale", type=float, default=0.08)
    p.add_argument("--halo-mass", type=float, default=HALO["mass"])
    p.add_argument("--halo-rs", type=float, default=HALO["rs"])
    p.add_argument("--halo-rcut", type=float, default=HALO["rcut"])
    p.add_argument("--halo-mass-factor", type=float, default=None, help="default 5 (galaxy, merger), 2 (bar)")
    p.add_argument("--rmax", type=float, default=25.0)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--r0", type=float, default=10.0)
    p.add_argument("--rp", type=float, default=0.4)
    p.add_argument("--spin-a", type=float, nargs=2, default=(30.0, 0.0))
    p.add_argument("--spin-b", type=float, nargs=2, default=(60.0, 150.0))
    a = p.parse_args(argv)
    if a.bulge_mass is None:
        a.bulge_mass = 0.0 if a.kind == "bar" else 1.2
    if a.halo_mass_factor is None:
        a.halo_mass_factor = 2.0 if a.kind == "bar" else 5.0
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.kind == "merger":
        gal_a = sample_model(a, "galaxy", a.seed)
        gal_b = sample_model(a, "galaxy", a.seed + 1)
        save(a.output, assemble(gal_a, gal_b, a))
    else:
        save(a.output, sample_model(a, a.kind, a.seed))


if __name__ == "__main__":
    main()
