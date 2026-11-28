"""Shared helpers of the GC-in-MW demo scripts (notebooks/gc_in_mw/).

Code units throughout: 10 kpc, 1e10 Msun, G = 1 (odisseo.globular_clusters.GALAXY_UNITS;
time unit 149.1 Myr, velocity unit 65.58 km/s).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

TIME_MYR = 149.1  # code time unit [Myr] (10 kpc, 1e10 Msun, G = 1)
KPC = 10.0  # code length [kpc]
KMS = 65.58  # code velocity [km/s]


def runs_dir(*parts: str) -> str:
    """``$ODISSEO_GC_RUNS`` (default ``notebooks/gc_in_mw/runs``) joined with ``parts`` (created)."""
    base = os.environ.get("ODISSEO_GC_RUNS", os.path.join(HERE, "runs"))
    path = os.path.join(base, *parts)
    os.makedirs(path if not os.path.splitext(path)[1] else os.path.dirname(path), exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


# ---------------------------------------------------------------------------
# The live galaxy.
# ---------------------------------------------------------------------------


def load_galaxy(path: str, n: int | None = None, milky_way_sense: bool = True):
    """A galaxy IC of ``tools/a4_galaxy_ics.py``; ``n``: a fair per-component subsample.

    Every component block of the file is shuffled, so its first rows are a fair
    subsample; each component keeps its share of ``n`` and its total mass. The galaxy
    is then put at rest at the origin (its centre of mass), and with
    ``milky_way_sense`` turned by 180 degrees about x if its disc (component 0)
    rotates with L_z > 0: the Milky Way rotates with L_z < 0 in astropy's
    Galactocentric frame (the Sun moves along +y at x < 0), and the GCs' observed
    orbits must keep their sense relative to the disc. The model is symmetric under
    that rotation; the a4 'galaxy' disc has L_z > 0.

    Returns
    -------
    (np.ndarray, np.ndarray, dict)
        ``[n, 2, 3]`` float64 state, ``[n]`` masses, the file's ``meta`` (plus
        ``flipped`` and ``com_removed``).
    """
    d = np.load(path, allow_pickle=False)
    state, mass, comp = d["state0"].astype(np.float64), d["mass"].astype(np.float64), d["component"]
    meta = json.loads(str(d["meta"]))
    if n is not None and n < len(mass):
        keep_s, keep_m, keep_c = [], [], []
        ids = np.unique(comp)
        counts = [int(np.sum(comp == c)) for c in ids]
        take = [max(1, int(round(n * c / len(mass)))) for c in counts]
        take[-1] = n - sum(take[:-1])
        for c, t in zip(ids, take):
            rows = np.flatnonzero(comp == c)[:t]
            keep_s.append(state[rows])
            keep_m.append(mass[rows] * (mass[comp == c].sum() / mass[rows].sum()))
            keep_c.append(comp[rows])
        state, mass, comp = np.concatenate(keep_s), np.concatenate(keep_m), np.concatenate(keep_c)
    com = (mass[:, None, None] * state).sum(0) / mass.sum()
    state = state - com[None]
    meta["com_removed"] = com.tolist()
    disc = comp == 0
    lz = np.sum(mass[disc] * (state[disc, 0, 0] * state[disc, 1, 1] - state[disc, 0, 1] * state[disc, 1, 0]))
    meta["flipped"] = bool(milky_way_sense and lz > 0)
    if meta["flipped"]:
        state[:, :, 1:] *= -1.0  # (x, y, z) -> (x, -y, -z): a rotation, L_z -> -L_z
    return state, mass, meta


# ---------------------------------------------------------------------------
# The static Odisseo potential (NFW halo + MN3 exponential disc + Hernquist bulge).
# ---------------------------------------------------------------------------

STATIC_NOMINAL = dict(halo_M=100.0, halo_rs=2.0, disc_M=6.0, disc_hr=0.24, disc_hz=0.03, bulge_M=1.2, bulge_a=0.08)
"""The live model's nominal components (tools/a4_galaxy_ics.py 'galaxy', code units)."""


def static_config_params(p: dict, **config_kw):
    """Odisseo ``(SimulationConfig, SimulationParams)`` for the static MW of parameters ``p``.

    NFW (``NFWParams(Mvir=halo_M, r_s=halo_rs)``: Phi = -G M ln(1 + r/rs) / r, the agama
    halo's normalisation before its cutoff), the sech^2 MN3 approximation of the
    exponential disc (``ThinMN3DiskParams``, Smith et al. 2015), and a Hernquist bulge
    as ``TwoPowerTriaxialParams(alpha=1, beta=4, rho=M / (2 pi a^3))``.
    """
    from odisseo.option_classes import (
        NFW_POTENTIAL,
        THIN_MN3_DISK,
        TWO_POWER_TRIAXIAL,
        NFWParams,
        SimulationConfig,
        SimulationParams,
        ThinMN3DiskParams,
        TwoPowerTriaxialParams,
    )

    config = SimulationConfig(
        external_accelerations=(NFW_POTENTIAL, THIN_MN3_DISK, TWO_POWER_TRIAXIAL), sech2_z=True, **config_kw
    )
    params = SimulationParams(
        G=1.0,
        NFW_params=NFWParams(Mvir=float(p["halo_M"]), r_s=float(p["halo_rs"])),
        ThinMN3Disk_params=ThinMN3DiskParams(M=float(p["disc_M"]), hr=float(p["disc_hr"]), hz=float(p["disc_hz"])),
        TwoPowerTriaxial_params=TwoPowerTriaxialParams(
            rho=float(p["bulge_M"]) / (2 * np.pi * float(p["bulge_a"]) ** 3), a=float(p["bulge_a"]),
            b=1.0, c=1.0, alpha=1.0, beta=4.0,
        ),
    )
    return config, params


def static_accel(p: dict):
    """A numpy field ``x [K, 3] -> a [K, 3]`` of the static MW (Odisseo's own potentials)."""
    import jax
    import jax.numpy as jnp

    from odisseo.potentials import combined_external_acceleration_vmpa_switch

    config, params = static_config_params(p)
    if config not in _STATIC_FIELD:  # one program per configuration; the parameters are traced

        @jax.jit
        def f(x, prm):
            state = jnp.stack([x, jnp.zeros_like(x)], 1)
            return combined_external_acceleration_vmpa_switch(state, config, prm)

        _STATIC_FIELD[config] = f
    f = _STATIC_FIELD[config]
    return lambda x: np.asarray(f(jnp.asarray(np.atleast_2d(x), jnp.float64), params))


_STATIC_FIELD: dict = {}


# ---------------------------------------------------------------------------
# Streams.
# ---------------------------------------------------------------------------


def orbit_frame(x0, v0):
    """Orthonormal ``(e1, e2, n)``: ``e1`` along the progenitor's position, ``n`` its angular momentum."""
    n = np.cross(x0, v0)
    n /= np.linalg.norm(n)
    e1 = x0 / np.linalg.norm(x0)
    return e1, np.cross(n, e1), n


def stream_coordinates(x, x0, v0):
    """Stream stars in the progenitor's orbital frame.

    Returns ``(phi1, phi2, r)``: ``phi1`` [deg] the angle along the orbit from the
    progenitor (positive ahead), ``phi2`` [deg] the angle out of its orbital plane, ``r``
    the Galactocentric radius (code units).
    """
    e1, e2, n = orbit_frame(np.asarray(x0, float), np.asarray(v0, float))
    x = np.asarray(x, float)
    r = np.linalg.norm(x, axis=-1)
    phi1 = np.degrees(np.arctan2(x @ e2, x @ e1))
    phi2 = np.degrees(np.arcsin(np.clip((x @ n) / r, -1, 1)))
    return phi1, phi2, r


def stream_track(phi1, phi2, r, bins, min_count: int = 10):
    """Median ``phi2`` and ``r`` (and counts) in ``phi1`` bins; ``nan`` where fewer than ``min_count``."""
    idx = np.digitize(phi1, bins) - 1
    nb = len(bins) - 1
    med2, medr, cnt = np.full(nb, np.nan), np.full(nb, np.nan), np.zeros(nb, int)
    for b in range(nb):
        sel = idx == b
        cnt[b] = int(sel.sum())
        if cnt[b] >= min_count:
            med2[b], medr[b] = np.median(phi2[sel]), np.median(r[sel])
    return med2, medr, cnt
