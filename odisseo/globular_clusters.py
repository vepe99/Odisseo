"""Milky Way globular clusters for Odisseo: catalogue, orbits, realisations, the hybrid scheme.

The Odisseo side of the GC-in-MW science layer (nornax ``docs/hybrid_hermite.md``
section 12 item 4). It is deliberately thin: the physics -- the clusters' Hermite
integration, the slow force, stripping, ledgers, checkpoints -- lives in nornax
(``ClusterHybridHermite``, ``ClustersRunnerScheme``) and jaccpot (``ProductionRunner``).
This module turns catalogue rows into the ``(xc, vc, mc)`` cluster block those take,
in Odisseo code units, and builds the scheme with documented defaults.

THE CATALOGUE. Holger Baumgardt's Galactic globular cluster database
(https://people.smp.uq.edu.au/HolgerBaumgardt/globular/): the orbit table
(positions, distances, proper motions, line-of-sight velocities; Vasiliev & Baumgardt
2021, Baumgardt & Vasiliev 2021) and the structural table (masses, radii; Baumgardt &
Hilker 2018 and later). ``tools/fetch_baumgardt.py`` downloads both and records the
URL, date and sha256; :func:`load_baumgardt` parses them. Cite those papers when you
use it.

THE FRAME. Observed (ICRS) -> Galactocentric with astropy. The default is astropy's
``Galactocentric()`` frame as configured in the running astropy (``"v4.0"``
parameters since astropy 4.0: R0 = 8.122 kpc, z_sun = 20.8 pc, v_sun = (12.9, 245.6,
7.78) km/s); :func:`galactocentric_frame` records the values it used.
:data:`BAUMGARDT_FRAME` reproduces the catalogue's own Galactocentric columns (R0 =
8.178 kpc, z_sun = 0, v_sun = (11.1, 252.24, 7.25) km/s; inferred from the table by
least squares, residuals at its rounding). Axes: astropy's right-handed frame (x from
the Sun towards the Galactic centre, y along Galactic rotation, z to the NGP); the
catalogue's X points from the centre TO the Sun, so X = -x, U = -v_x.

UNITS. :data:`GALAXY_UNITS`: 10 kpc, 1e10 Msun, G = 1 (the live galaxy models of
``tools/a4_galaxy_ics.py``; time unit 149.1 Myr, velocity unit 65.6 km/s).

THE REALISATIONS. ``"king"``: a King (1966) model with W0 from the catalogue's
concentration ``c = log10(r_t / r_c)`` (its Spitzer core radius is King's radius
r0), scaled to the catalogue's 3D half-mass radius and mass; ``"plummer"``:
:func:`odisseo.initial_condition.Plummer_sphere` with a = r_h / 1.305. Each cluster is
shifted to its exact centre (mass-weighted) before it is put on its orbit.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from odisseo.units import CodeUnits

__all__ = [
    "BAUMGARDT_BASE_URL",
    "BAUMGARDT_FILES",
    "BAUMGARDT_FRAME",
    "GALAXY_UNITS",
    "GCRecord",
    "KingModel",
    "galactocentric_frame",
    "parse_baumgardt_orbits",
    "parse_baumgardt_structure",
    "load_baumgardt",
    "normalize_name",
    "observed_to_galactocentric",
    "to_code_units",
    "king_w0_from_concentration",
    "sample_king",
    "sample_plummer",
    "cluster_table",
    "clusters_from_catalogue",
    "default_cluster_softening",
    "default_reserve",
    "crossing_times",
    "default_n_sub",
    "integrate_orbit",
    "solver_settings",
    "build_solvers",
    "clusters_scheme",
    "repark_empty_slots",
    "install_empty_slot_repark",
    "patch_jaccpot_csr_restore_preflight",
]

BAUMGARDT_BASE_URL = "https://people.smp.uq.edu.au/HolgerBaumgardt/globular/"
"""The database's home page."""

BAUMGARDT_FILES = {"orbits": "orbits_table.txt", "structure": "combined_table.txt"}
"""The two plain-text tables :func:`load_baumgardt` reads (relative to the base URL)."""

BAUMGARDT_FRAME = dict(
    galcen_distance=8.178,  # kpc
    z_sun=0.0,  # pc
    galcen_v_sun=(11.1, 252.24, 7.25),  # km/s
    roll=0.0,  # deg
)
"""Galactocentric parameters that reproduce the catalogue's X, Y, Z, U, V, W.

Inferred from the orbit table by least squares (2026-11-24, 165 clusters): positions
to 0.01 kpc and velocities to 0.1 km/s (95th percentile; the table's rounding).
"""


def _galaxy_units() -> CodeUnits:
    import astropy.units as u

    return CodeUnits(10.0 * u.kpc, 1e10 * u.Msun, G=1.0)


GALAXY_UNITS = _galaxy_units()
"""10 kpc, 1e10 Msun, G = 1: the units of ``tools/a4_galaxy_ics.py``'s galaxies."""

_ALIASES = {
    "47_TUC": "NGC_104",
    "OMEGA_CEN": "NGC_5139",
    "M2": "NGC_7089",
    "M3": "NGC_5272",
    "M4": "NGC_6121",
    "M5": "NGC_5904",
    "M10": "NGC_6254",
    "M12": "NGC_6218",
    "M13": "NGC_6205",
    "M15": "NGC_7078",
    "M22": "NGC_6656",
    "M53": "NGC_5024",
    "M55": "NGC_6809",
    "M68": "NGC_4590",
    "M92": "NGC_6341",
}


def normalize_name(name: str) -> str:
    """The catalogue's spelling of a cluster name.

    ``"Pal 5"``, ``"pal_5"`` and ``"PAL5"`` -> ``"PAL_5"``; ``"NGC 104"`` and
    ``"47 Tuc"`` -> ``"NGC_104"``; a few Messier names (``"M68"``) map to their NGC
    number. The comparison is case-insensitive (the key is upper case).

    Parameters
    ----------
    name : str
        A cluster name.

    Returns
    -------
    str
        The key :func:`load_baumgardt` uses.
    """
    key = "_".join(str(name).replace("-", "_").split()).upper()
    if key in _ALIASES:
        return _ALIASES[key]
    for prefix in ("NGC", "PAL", "IC", "TERZAN", "TON", "AM", "E", "BH", "ARP", "FSR", "HP"):
        if key.startswith(prefix) and len(key) > len(prefix) and key[len(prefix)].isdigit():
            return f"{prefix}_{key[len(prefix):]}"
    return key


@dataclasses.dataclass(frozen=True)
class GCRecord:
    """One cluster of the catalogue (the columns this module uses).

    Attributes
    ----------
    name : str
        The catalogue's name (e.g. ``"Pal_5"``).
    ra, dec : float
        ICRS position [deg].
    distance : float
        Heliocentric distance [kpc].
    rv : float
        Line-of-sight velocity [km/s].
    pmra_cosdec, pmdec : float
        Proper motion [mas/yr].
    mass : float
        Present-day mass [Msun] (``nan`` if the structural table lacks the cluster).
    r_half : float
        3D half-mass radius [pc].
    r_core : float
        Core radius (Spitzer 1987; King's r0) [pc].
    r_tidal : float
        Tidal radius (Webb et al. 2013 eq 8) [pc].
    published : Optional[dict]
        The catalogue's own Galactocentric ``X, Y, Z`` [kpc] and ``U, V, W`` [km/s].
    """

    name: str
    ra: float
    dec: float
    distance: float
    rv: float
    pmra_cosdec: float
    pmdec: float
    mass: float = float("nan")
    r_half: float = float("nan")
    r_core: float = float("nan")
    r_tidal: float = float("nan")
    published: Optional[dict] = None

    @property
    def concentration(self) -> float:
        """King concentration ``c = log10(r_tidal / r_core)`` (``nan`` if missing)."""
        if not (self.r_core > 0 and self.r_tidal > 0):
            return float("nan")
        return float(math.log10(self.r_tidal / self.r_core))


def _rows(text: str) -> list:
    return [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def parse_baumgardt_orbits(text: str) -> Dict[str, dict]:
    """Parse the orbit table (``orbits_table.txt``).

    Columns (1-based, the table's header): 1 name, 2 RA, 3 DEC, 6 R_sun, 9 <RV>,
    11 mu_alpha cos(delta), 13 mu_delta, 16/18/20 X/Y/Z, 22/24/26 U/V/W.

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    dict
        ``{key: row}`` with ``key`` from :func:`normalize_name`.

    Raises
    ------
    ValueError
        If a row does not have the table's 31 columns.
    """
    out = {}
    for r in _rows(text):
        if len(r) != 31:
            raise ValueError(f"orbit table: expected 31 columns, got {len(r)} in {r[:2]}")
        f = [float(x) for x in r[1:]]
        out[normalize_name(r[0])] = dict(
            name=r[0], ra=f[0], dec=f[1], distance=f[4], rv=f[7], pmra_cosdec=f[9], pmdec=f[11],
            published=dict(X=f[14], Y=f[16], Z=f[18], U=f[20], V=f[22], W=f[24]),
        )
    return out


def parse_baumgardt_structure(text: str) -> Dict[str, dict]:
    """Parse the structural table (``combined_table.txt``).

    Columns used (1-based): 1 name, 10 mass [Msun], 16 r_c, 18 r_h,m (3D), 19 r_t
    [pc]. Trailing columns may be missing (the rotation columns of a few rows).

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    dict
        ``{key: dict(mass, r_core, r_half, r_tidal)}``.

    Raises
    ------
    ValueError
        If a row is too short to hold the columns used.
    """
    out = {}
    for r in _rows(text):
        if len(r) < 19:
            raise ValueError(f"structure table: row too short ({len(r)} columns): {r[:2]}")
        out[normalize_name(r[0])] = dict(
            mass=float(r[9]), r_core=float(r[15]), r_half=float(r[17]), r_tidal=float(r[18])
        )
    return out


def load_baumgardt(
    orbits: Optional[str] = None, structure: Optional[str] = None, directory: Optional[str] = None
) -> Dict[str, GCRecord]:
    """The catalogue as ``{key: GCRecord}`` (orbit table joined with the structural one).

    Parameters
    ----------
    orbits, structure : Optional[str]
        Paths to the two tables. Default: :data:`BAUMGARDT_FILES` in ``directory``.
    directory : Optional[str]
        Where :func:`tools.fetch_baumgardt` put them (default: ``$ODISSEO_GC_DATA``).

    Returns
    -------
    dict
        Every cluster of the orbit table; the structural columns are ``nan`` where
        the structural table lacks it.

    Raises
    ------
    FileNotFoundError
        If no path is given and ``$ODISSEO_GC_DATA`` is unset.
    """
    if orbits is None or structure is None:
        directory = directory or os.environ.get("ODISSEO_GC_DATA")
        if not directory:
            raise FileNotFoundError(
                "pass orbits=/structure= or directory= (or set ODISSEO_GC_DATA); fetch the "
                "tables with tools/fetch_baumgardt.py"
            )
        orbits = orbits or os.path.join(directory, BAUMGARDT_FILES["orbits"])
        structure = structure or os.path.join(directory, BAUMGARDT_FILES["structure"])
    with open(orbits) as fh:
        orb = parse_baumgardt_orbits(fh.read())
    with open(structure) as fh:
        stru = parse_baumgardt_structure(fh.read())
    return {k: GCRecord(**v, **stru.get(k, {})) for k, v in orb.items()}


def _select(catalogue: Mapping[str, GCRecord], names: Iterable[str]) -> list:
    out = []
    for n in names:
        key = normalize_name(n)
        if key not in catalogue:
            raise KeyError(f"{n!r} ({key}) is not in the catalogue")
        out.append(catalogue[key])
    return out


# ---------------------------------------------------------------------------
# Frames and units.
# ---------------------------------------------------------------------------


def galactocentric_frame(**overrides: Any) -> Any:
    """astropy's ``Galactocentric`` frame, defaults from the running astropy.

    Parameters
    ----------
    **overrides : Any
        ``galcen_distance`` [kpc], ``z_sun`` [pc], ``galcen_v_sun`` [km/s, 3-tuple],
        ``roll`` [deg] as plain numbers (or astropy quantities); e.g.
        ``**BAUMGARDT_FRAME``.

    Returns
    -------
    astropy.coordinates.Galactocentric
        The frame.
    """
    import astropy.units as u
    from astropy.coordinates import CartesianDifferential, Galactocentric

    kw = {}
    for key, unit in (("galcen_distance", u.kpc), ("z_sun", u.pc), ("roll", u.deg)):
        if key in overrides and overrides[key] is not None:
            kw[key] = u.Quantity(overrides[key], unit)
    if overrides.get("galcen_v_sun") is not None:
        kw["galcen_v_sun"] = CartesianDifferential(u.Quantity(overrides["galcen_v_sun"], u.km / u.s))
    return Galactocentric(**kw)


def frame_parameters(frame: Any) -> dict:
    """The frame's parameters as plain numbers (for records and metadata)."""
    import astropy
    import astropy.units as u

    v = frame.galcen_v_sun
    v = [getattr(v, a, None) if hasattr(v, a) else getattr(v, b) for a, b in (("d_x", "x"), ("d_y", "y"), ("d_z", "z"))]
    return dict(
        astropy=astropy.__version__,
        galcen_distance_kpc=float(frame.galcen_distance.to_value(u.kpc)),
        z_sun_pc=float(frame.z_sun.to_value(u.pc)),
        galcen_v_sun_kms=[float(c.to_value(u.km / u.s)) for c in v],
        roll_deg=float(frame.roll.to_value(u.deg)),
    )


def observed_to_galactocentric(records: Sequence[GCRecord], frame: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Galactocentric positions [kpc] and velocities [km/s] of catalogue clusters.

    Parameters
    ----------
    records : sequence of GCRecord
        The clusters.
    frame : Any
        A ``Galactocentric`` frame (:func:`galactocentric_frame`); default astropy's.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``[K, 3]`` positions and ``[K, 3]`` velocities (astropy's axes).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    frame = galactocentric_frame() if frame is None else frame
    c = SkyCoord(
        ra=np.array([r.ra for r in records]) * u.deg,
        dec=np.array([r.dec for r in records]) * u.deg,
        distance=np.array([r.distance for r in records]) * u.kpc,
        pm_ra_cosdec=np.array([r.pmra_cosdec for r in records]) * u.mas / u.yr,
        pm_dec=np.array([r.pmdec for r in records]) * u.mas / u.yr,
        radial_velocity=np.array([r.rv for r in records]) * u.km / u.s,
        frame="icrs",
    )
    g = c.transform_to(frame)
    pos = np.stack([g.x.to_value(u.kpc), g.y.to_value(u.kpc), g.z.to_value(u.kpc)], -1)
    vel = np.stack([g.v_x.to_value(u.km / u.s), g.v_y.to_value(u.km / u.s), g.v_z.to_value(u.km / u.s)], -1)
    return pos, vel


def to_code_units(pos_kpc: Any, vel_kms: Any, units: CodeUnits = GALAXY_UNITS) -> Tuple[np.ndarray, np.ndarray]:
    """kpc, km/s -> code length, code velocity."""
    import astropy.units as u

    return (np.asarray(pos_kpc, float) * u.kpc.to(units.code_length),
            np.asarray(vel_kms, float) * (u.km / u.s).to(units.code_velocity))


def _mass_code(msun: float, units: CodeUnits) -> float:
    import astropy.units as u

    return float(msun * u.Msun.to(units.code_mass))


def _pc_code(pc: float, units: CodeUnits) -> float:
    import astropy.units as u

    return float(pc * u.pc.to(units.code_length))


# ---------------------------------------------------------------------------
# King (1966) models.
# ---------------------------------------------------------------------------


def _king_rho(psi: Any) -> Any:
    """King's density as a function of the relative potential (sigma = 1), unnormalised."""
    from scipy.special import erf

    psi = np.maximum(np.asarray(psi, float), 0.0)
    return np.exp(psi) * erf(np.sqrt(psi)) - np.sqrt(4.0 * psi / np.pi) * (1.0 + 2.0 * psi / 3.0)


class KingModel:
    """A King (1966) model in units G = sigma = r0 = 1 (r0: King's radius).

    ``rho(psi) = rho0 [e^psi erf(sqrt psi) - sqrt(4 psi / pi) (1 + 2 psi / 3)] / rho~(W0)``
    with ``4 pi G rho0 r0^2 = 9 sigma^2``, so Poisson's equation is
    ``psi'' + 2 psi' / r = -9 rho~(psi) / rho~(W0)``, ``psi(0) = W0``, ``psi'(0) = 0``,
    integrated outwards to the tidal radius ``psi(r_t) = 0``. ``M(r) = -r^2 psi'(r)``.

    Parameters
    ----------
    w0 : float
        Central dimensionless potential (0.1 - 16).
    n_grid : int
        Radial grid points.
    """

    def __init__(self, w0: float, n_grid: int = 2000):
        from scipy.integrate import solve_ivp

        if not 0.05 <= w0 <= 18.0:
            raise ValueError(f"King W0 must be within [0.05, 18], got {w0}")
        self.w0 = float(w0)
        norm = float(_king_rho(self.w0))

        def rhs(r, y):
            return [y[1], -9.0 * float(_king_rho(y[0])) / norm - 2.0 * y[1] / r]

        def edge(r, y):
            return y[0]

        edge.terminal, edge.direction = True, -1
        r0 = 1e-6
        y0 = [self.w0 - 1.5 * r0**2, -3.0 * r0]
        sol = solve_ivp(rhs, (r0, 1e4), y0, events=edge, rtol=1e-10, atol=1e-12, dense_output=True,
                        method="DOP853")
        if not sol.t_events[0].size:
            raise RuntimeError(f"King W0={w0}: no tidal radius found")
        self.r_t = float(sol.t_events[0][0])
        r = np.concatenate([[0.0], np.geomspace(r0, self.r_t, n_grid - 1)])
        y = sol.sol(np.maximum(r, r0))
        self.r = r
        self.psi = np.maximum(y[0], 0.0)
        self.psi[0], self.psi[-1] = self.w0, 0.0
        self.mass_r = np.maximum(-(r**2) * y[1], 0.0)
        self.mass_r[0] = 0.0
        self.mass = float(self.mass_r[-1])
        self.rho0 = 9.0 / (4.0 * np.pi)
        self.density = self.rho0 * _king_rho(self.psi) / norm
        self.r_half = float(np.interp(0.5 * self.mass, self.mass_r, self.r))

    @property
    def concentration(self) -> float:
        """``log10(r_t / r0)``."""
        return float(np.log10(self.r_t))

    def sample(self, n: int, rng: np.random.Generator, n_v: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """``n`` particles (positions, velocities) in the model's units (total mass ``self.mass``).

        Radii by the inverse of ``M(r)``; speeds from ``p(v) ~ v^2 (e^(psi - v^2/2) - 1)``
        on ``[0, sqrt(2 psi)]`` by a per-particle tabulated inverse CDF; isotropic.
        """
        u = rng.uniform(0.0, 1.0, n)
        rr = np.interp(u * self.mass, self.mass_r, self.r)
        psi = np.interp(rr, self.r, self.psi)
        s = np.linspace(0.0, 1.0, n_v)
        speeds = np.empty(n)
        for lo in range(0, n, 4096):
            p = psi[lo:lo + 4096, None]
            pdf = s[None, :] ** 2 * np.expm1(p * (1.0 - s[None, :] ** 2))
            cdf = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]), 1)], 1)
            cdf /= np.where(cdf[:, -1:] > 0, cdf[:, -1:], 1.0)
            uu = rng.uniform(0.0, 1.0, p.shape[0])
            j = np.clip((cdf < uu[:, None]).sum(1), 1, n_v - 1)
            rows = np.arange(p.shape[0])
            c0, c1 = cdf[rows, j - 1], cdf[rows, j]
            frac = np.where(c1 > c0, (uu - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
            speeds[lo:lo + 4096] = (s[j - 1] + frac * (s[j] - s[j - 1])) * np.sqrt(2.0 * p[:, 0])
        return rr[:, None] * _unit_vectors(n, rng), speeds[:, None] * _unit_vectors(n, rng)


def _unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def king_w0_from_concentration(c: float, lo: float = 0.1, hi: float = 16.0) -> float:
    """W0 of the King model with ``log10(r_t / r0) = c`` (clipped to ``[c(lo), c(hi)]``)."""
    from scipy.optimize import brentq

    c_lo, c_hi = KingModel(lo, 400).concentration, KingModel(hi, 400).concentration
    if not np.isfinite(c):
        raise ValueError("concentration is not finite")
    if c <= c_lo:
        return lo
    if c >= c_hi:
        return hi
    return float(brentq(lambda w: KingModel(w, 400).concentration - c, lo, hi, xtol=1e-6))


def _centred(x: np.ndarray, v: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = m / m.sum()
    return x - (w[:, None] * x).sum(0), v - (w[:, None] * v).sum(0)


def sample_king(n: int, *, w0: float, mass: float, r_half: float, G: float = 1.0,
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A King cluster of ``n`` equal-mass stars, centred (exact centre of mass and velocity).

    Parameters
    ----------
    n : int
        Stars.
    w0 : float
        King W0.
    mass : float
        Total mass (any units consistent with ``G``).
    r_half : float
        3D half-mass radius (same length units).
    G : float
        Gravitational constant.
    rng : numpy Generator
        Random numbers (default: a fresh ``default_rng(0)``).

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        ``[n, 3]`` positions, ``[n, 3]`` velocities, ``[n]`` masses.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    model = KingModel(w0)
    x, v = model.sample(n, rng)
    length = r_half / model.r_half
    vel = math.sqrt(G * (mass / model.mass) / length)
    m = np.full(n, mass / n)
    x, v = _centred(x * length, v * vel, m)
    return x, v, m


def sample_plummer(n: int, *, mass: float, r_half: float, G: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Plummer cluster via :func:`odisseo.initial_condition.Plummer_sphere` (a = r_h / 1.305), centred.

    Same signature and returns as :func:`sample_king`.
    """
    import jax

    from odisseo.initial_condition import Plummer_sphere
    from odisseo.option_classes import PlummerParams, SimulationConfig, SimulationParams

    rng = np.random.default_rng(0) if rng is None else rng
    a = r_half / (1.0 / math.sqrt(2.0 ** (2.0 / 3.0) - 1.0))  # r_h = 1.3048 a
    config = SimulationConfig(N_particles=int(n))
    params = SimulationParams(G=float(G), Plummer_params=PlummerParams(a=float(a), Mtot=float(mass)))
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    x, v, _ = Plummer_sphere(key, config, params)
    m = np.full(n, mass / n)
    x, v = _centred(np.asarray(x, float), np.asarray(v, float), m)
    return x, v, m


# ---------------------------------------------------------------------------
# The API.
# ---------------------------------------------------------------------------


def cluster_table(names: Sequence[str], catalogue: Mapping[str, GCRecord], *, frame: Any = None,
                  units: CodeUnits = GALAXY_UNITS, w0_default: float = 5.0) -> list:
    """Per-cluster parameters in code units: centre, mass, half-mass radius, W0.

    Parameters
    ----------
    names : sequence of str
        Cluster names (:func:`normalize_name`).
    catalogue : mapping
        :func:`load_baumgardt`'s result.
    frame : Any
        Galactocentric frame (default astropy's).
    units : CodeUnits
        Code units.
    w0_default : float
        W0 where the catalogue has no concentration.

    Returns
    -------
    list of dict
        ``name, x [3], v [3], mass, r_half, w0, concentration`` (code units) plus
        ``mass_msun, r_half_pc``.

    Raises
    ------
    ValueError
        If a cluster has no mass or half-mass radius in the catalogue.
    """
    recs = _select(catalogue, names)
    for r in recs:
        if not (r.mass > 0 and r.r_half > 0):
            raise ValueError(f"{r.name}: no mass / half-mass radius in the structural table")
    pos, vel = observed_to_galactocentric(recs, frame)
    xs, vs = to_code_units(pos, vel, units)
    out = []
    for k, r in enumerate(recs):
        c = r.concentration
        out.append(dict(
            name=r.name, x=xs[k], v=vs[k], mass=_mass_code(r.mass, units), r_half=_pc_code(r.r_half, units),
            w0=king_w0_from_concentration(c) if np.isfinite(c) else float(w0_default), concentration=c,
            mass_msun=r.mass, r_half_pc=r.r_half,
        ))
    return out


def clusters_from_catalogue(
    names: Sequence[str],
    M: int = 1000,
    *,
    model: str = "king",
    catalogue: Optional[Mapping[str, GCRecord]] = None,
    frame: Any = None,
    units: CodeUnits = GALAXY_UNITS,
    centres: Optional[Tuple[Any, Any]] = None,
    seed: int = 0,
    G: float = 1.0,
    w0_default: float = 5.0,
    return_table: bool = False,
) -> Any:
    """The clusters' ``(xc, vc, mc)`` block for ``ClustersRunnerScheme``, code units.

    Parameters
    ----------
    names : sequence of str
        Cluster names, e.g. ``["Pal 5", "NGC 5466"]``.
    M : int
        Stars per cluster (equal masses within a cluster).
    model : str
        ``"king"`` (W0 from the concentration) or ``"plummer"``.
    catalogue : mapping
        :func:`load_baumgardt`'s result (default: ``load_baumgardt()``).
    frame : Any
        Galactocentric frame (:func:`galactocentric_frame`; default astropy's).
    units : CodeUnits
        Code units (default :data:`GALAXY_UNITS`).
    centres : Optional[tuple]
        ``([K, 3], [K, 3])`` code-unit centres to put the clusters at instead of
        today's (e.g. an orbit integrated back, :func:`integrate_orbit`).
    seed : int
        Random seed (cluster ``k`` uses ``default_rng([seed, k])``).
    G : float
        Gravitational constant in code units.
    w0_default : float
        W0 where the catalogue has no concentration.
    return_table : bool
        Also return :func:`cluster_table`'s rows.

    Returns
    -------
    tuple
        ``xc [K, M, 3]``, ``vc [K, M, 3]``, ``mc [K, M]`` (float64 numpy), and the table
        with ``return_table=True``.

    Raises
    ------
    ValueError
        For an unknown ``model``.
    """
    if model not in ("king", "plummer"):
        raise ValueError(f"model must be 'king' or 'plummer', got {model!r}")
    catalogue = load_baumgardt() if catalogue is None else catalogue
    table = cluster_table(names, catalogue, frame=frame, units=units, w0_default=w0_default)
    K = len(table)
    xc, vc, mc = np.zeros((K, M, 3)), np.zeros((K, M, 3)), np.zeros((K, M))
    for k, row in enumerate(table):
        rng = np.random.default_rng([int(seed), k])
        if model == "king":
            x, v, m = sample_king(M, w0=row["w0"], mass=row["mass"], r_half=row["r_half"], G=G, rng=rng)
        else:
            x, v, m = sample_plummer(M, mass=row["mass"], r_half=row["r_half"], G=G, rng=rng)
        cx, cv = (row["x"], row["v"]) if centres is None else (np.asarray(centres[0])[k], np.asarray(centres[1])[k])
        xc[k], vc[k], mc[k] = x + cx, v + cv, m
    return (xc, vc, mc, table) if return_table else (xc, vc, mc)


def default_cluster_softening(table: Sequence[dict], fraction: float = 1.0 / 20.0) -> float:
    """``eps_c``: ``fraction`` of the smallest Plummer-equivalent scale ``r_h / 1.305``.

    HH sessions 1-3 used ``r_c / 20`` for Plummer clusters of scale ``r_c``; this is the
    same rule with the scale read from the half-mass radius.
    """
    return float(fraction * min(row["r_half"] for row in table) / 1.3048)


def crossing_times(table: Sequence[dict], G: float = 1.0) -> np.ndarray:
    """Each cluster's half-mass crossing time ``sqrt(r_h^3 / (G M))`` (code units)."""
    return np.array([math.sqrt(row["r_half"] ** 3 / (G * row["mass"])) for row in table])


def default_n_sub(dt: float, table: Sequence[dict], per_crossing: int = 20, minimum: int = 8,
                  G: float = 1.0) -> int:
    """Fixed internal substeps: ``per_crossing`` steps per crossing time of the MOST compact cluster.

    One ``n_sub`` serves every cluster (adaptive per-cluster steps are nornax's Aarseth
    work, ``docs/hybrid_hermite.md`` section 12 item 3), so the most compact cluster sets
    it: Pal 5 alone (t_cr ~ 0.12 code units) needs ``minimum`` at ``dt = 1e-3``; the
    catalogue's 165 clusters (t_cr down to ~6e-4) need ~35.
    """
    return int(max(minimum, math.ceil(per_crossing * dt / float(crossing_times(table, G).min()))))


def default_reserve(K: int, M: int) -> int:
    """Free galaxy rows for stripped stars: ``K M / 8`` rounded up to 1024 (>= 1024).

    A full reserve is grown at an edge (``StrippingPolicy.grow``), at the cost of one
    re-plan and a recompilation; a run that will dissolve its clusters should pass
    ``K * M``.
    """
    return int(max(1024, -(-(K * M // 8) // 1024) * 1024))


def integrate_orbit(x0: Any, v0: Any, accel: Any, t: float, dt: float, *, n_out: int = 0) -> Any:
    """Leapfrog (KDK) test-particle orbits in a static field ``accel(x [K, 3]) -> [K, 3]``.

    A negative ``t`` integrates backwards (``dt`` is taken with the sign of ``t``).

    Parameters
    ----------
    x0, v0 : array
        ``[K, 3]`` start.
    accel : callable
        The field (numpy in, numpy out).
    t : float
        Span.
    dt : float
        Step (magnitude).
    n_out : int
        Also return ``n_out + 1`` evenly spaced samples (``[n_out + 1, K, 2, 3]``).

    Returns
    -------
    tuple
        ``(x, v)`` at ``t``, plus the samples when ``n_out > 0``.
    """
    n = max(1, int(round(abs(t) / dt)))
    h = t / n
    x, v = np.array(x0, float), np.array(v0, float)
    every = max(1, n // n_out) if n_out else 0
    track = [np.stack([x, v], -2)] if n_out else None
    a = accel(x)
    for i in range(n):
        v = v + 0.5 * h * a
        x = x + h * v
        a = accel(x)
        v = v + 0.5 * h * a
        if n_out and (i + 1) % every == 0:
            track.append(np.stack([x, v], -2))
    return (x, v, np.array(track)) if n_out else (x, v)


# ---------------------------------------------------------------------------
# Solvers and the scheme (nornax / jaccpot do the work).
# ---------------------------------------------------------------------------


def solver_settings(backend: Optional[str] = None, **overrides: Any) -> Any:
    """jaccpot ``GalaxySettings`` for a GC run.

    On a GPU jaccpot's ``GALAXY_SETTINGS`` (leaf 64, p6, ``dehnen_error`` at 1e-4,
    fp32: the hybrid runs' galaxy). On CPU the geometric Dehnen MAC at theta 0.6 with
    no cell-level floor (``dehnen_error``'s force scale needs the Pallas near field,
    which the CPU route only interprets: too slow for a demo; HH3a's CPU gates did
    the same).
    """
    import jax
    from jaccpot.production import GALAXY_SETTINGS

    backend = backend or jax.default_backend()
    s = GALAXY_SETTINGS
    if backend == "cpu":
        s = dataclasses.replace(s, mac_type="dehnen", adaptive_eps=None, theta=0.6, cell_min_level=0)
    return dataclasses.replace(s, **overrides) if overrides else s


def build_solvers(eps_g: float, eps_c: float, *, G: float = 1.0, backend: Optional[str] = None,
                  cluster_theta: float = 0.5, cluster_leaf_headroom: float = 2.0) -> Tuple[Any, Any, Any]:
    """The galaxy solver, the clusters' star solver and the runner's settings.

    The galaxy: :func:`odisseo.runner_coupling.build_galaxy_solver` with
    :func:`solver_settings` (Odisseo's runner lane). The star provider (back-reaction
    and coupling): the same builder, geometric Dehnen MAC at ``cluster_theta`` (its
    rebuild carries no force scale; nornax section 10), softening ``eps_c``, leaf headroom ``cluster_leaf_headroom`` (2: the star tree spreads as stars strip).

    Returns
    -------
    tuple
        ``(galaxy_solver, cluster_solver, settings)``.
    """
    from odisseo.runner_coupling import build_galaxy_solver

    import jax

    backend = backend or jax.default_backend()
    settings = solver_settings(backend)
    # use_pallas as Odisseo's runner lane (runner_coupling._runner_use_pallas): on CPU only under
    # dehnen_error (interpret mode), never for the geometric MAC
    on_cpu = backend == "cpu"
    gal = build_galaxy_solver(settings, G=G, softening=eps_g,
                              use_pallas=(settings.mac_type == "dehnen_error") if on_cpu else
                              ((settings.mac_type == "dehnen_error") or None))
    # the star tree's leaves get more headroom than the galaxy's (1.15): a stripped cluster spreads, its
    # tree's needs creep up, and every capacity re-plan recompiles the whole program set (~20 min on a
    # loaded host: 1e6/1e7 + Pal 5, 2026-11-25)
    cs = dataclasses.replace(settings, mac_type="dehnen", adaptive_eps=None, theta=float(cluster_theta),
                             leaf_headroom=float(cluster_leaf_headroom))
    clu = build_galaxy_solver(cs, G=G, softening=eps_c, use_pallas=False if on_cpu else None)
    return gal, clu, settings


def clusters_scheme(
    clusters: Tuple[Any, Any, Any],
    *,
    eps_c: float,
    cluster_solver: Any,
    G: float = 1.0,
    n_sub: int = 16,
    reserve: Optional[int] = None,
    policy: Any = "default",
    cross: str = "fmm",
    coupling: str = "fmm",
    cluster_leaf_size: int = 64,
    cluster_batch: Optional[int] = 1,
    block_size: Optional[int] = 1024,
    repark_empty: bool = True,
    **scheme_kwargs: Any,
) -> Any:
    """``nornax.adapters.jaccpot_runner.ClustersRunnerScheme`` with documented defaults.

    - ``ClusterHybridHermite(n_sub=16, order=4)`` (fixed internal steps ``dt / n_sub``),
      ``MembershipSplit(DirectSumGravity(G, eps_c, block_size=1024))``,
      ``cluster_batch=1`` (one cluster's pair tensor at a time): HH3a/HH3b's 1e8 run;
    - ``StrippingPolicy()`` (``"default"``): nornax's defaults -- strip radius 2 r_J,
      tidal fit, Omega about the galaxy's centre, dwell 1, dissolution below 8 members
      (``docs/hybrid_hermite.md`` section 12 item 6: unchanged here); ``None``: no
      stripping;
    - ``reserve``: :func:`default_reserve`;
    - ``options=dict(cross="fmm", coupling="fmm", cluster_leaf_size=64)``;
    - ``repark_empty=True``: every edge ends with :func:`repark_empty_slots` (an Odisseo
      workaround: without it the star tree re-plans at every edge of a moving cluster).

    Parameters
    ----------
    clusters : tuple
        ``(xc, vc, mc)`` ``[K, M]`` (:func:`clusters_from_catalogue`).
    eps_c : float
        The clusters' internal softening (:func:`default_cluster_softening`).
    cluster_solver : Any
        The star provider (:func:`build_solvers`).
    G : float
        Gravitational constant.
    n_sub, reserve, policy, cross, coupling, cluster_leaf_size, cluster_batch, block_size, repark_empty
        See above.
    **scheme_kwargs : Any
        More ``ClustersRunnerScheme`` fields (``max_retries``, ``measure_jumps``, ...).

    Returns
    -------
    ClustersRunnerScheme
        The scheme for ``ProductionRunner(scheme=...)``.
    """
    import jax.numpy as jnp
    from nornax.adapters.jaccpot_runner import ClustersRunnerScheme
    from nornax.forces.direct import DirectSumGravity
    from nornax.hybrid import MembershipSplit
    from nornax.hybrid_clusters import ClusterHybridHermite, StrippingPolicy

    xc, vc, mc = clusters
    K, M = np.shape(mc)
    scheme = ClusterHybridHermite(
        n_sub=int(n_sub),
        order=4,
        split=MembershipSplit(DirectSumGravity(G=float(G), softening=float(eps_c), block_size=block_size)),
        cluster_batch=cluster_batch,
    )
    out = ClustersRunnerScheme(
        scheme,
        (jnp.asarray(xc), jnp.asarray(vc), jnp.asarray(mc)),
        reserve=default_reserve(K, M) if reserve is None else int(reserve),
        policy=StrippingPolicy() if policy == "default" else policy,
        cluster_solver=cluster_solver,
        options=dict(cross=cross, coupling=coupling, cluster_leaf_size=int(cluster_leaf_size)),
        **scheme_kwargs,
    )
    if repark_empty:
        install_empty_slot_repark(out)
    return out


def repark_empty_slots(state: Any) -> Any:
    """Move every cluster's EMPTY slots (stripped stars) inside its member radius, co-moving.

    nornax keeps a stripped star's slot where the star left, at rest, and builds the
    clusters' star tree over all ``[K, M]`` slots: as the cluster moves on (~1 kpc per
    50-step segment for Pal 5) that tree's box grows, and the star tree re-plans --
    and recompiles -- at EVERY edge (~22 min per segment at 1e7 + 4096 on a shared
    A100, 2026-11-25). This puts each empty slot at ``X + d min(1, R / |d|)`` (``X``,
    ``V``: the filled slots' centre of mass and velocity, ``R`` their farthest member,
    ``d`` the slot's offset from ``X``) with velocity ``V``, so the empty slots stay
    within the cluster's own box. An empty slot is massless: the internal force,
    the stripping policy, the back-reaction groups (filled slots only), the energy
    and the momenta do not see it; it only shapes the star tree's box. Deterministic
    from the state, so restarts stay bitwise. Dissolved clusters (no mass) are left
    alone. A request to nornax is recorded (HH4b REPORT): exclude empty slots from the
    star tree's box, or park them co-moving there.

    Parameters
    ----------
    state : ClusterHybridState
        The scheme's state.

    Returns
    -------
    ClusterHybridState
        The state with the empty slots moved.
    """
    import jax.numpy as jnp

    cl = state.clusters
    x, v, m = cl.positions, cl.velocities, cl.masses
    filled = m > 0
    mk = jnp.sum(jnp.where(filled, m, 0.0), axis=1)
    live = mk > 0
    safe = jnp.where(live, mk, 1.0)[:, None]
    X = jnp.sum(jnp.where(filled[..., None], m[..., None] * x, 0.0), axis=1) / safe
    V = jnp.sum(jnp.where(filled[..., None], m[..., None] * v, 0.0), axis=1) / safe
    d = x - X[:, None, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    R = jnp.max(jnp.where(filled, r, 0.0), axis=1)[:, None]
    shrink = jnp.where(r > R, R / jnp.where(r > 0, r, 1.0), 1.0)
    move = (~filled) & live[:, None]
    x_new = jnp.where(move[..., None], X[:, None, :] + d * shrink[..., None], x).astype(x.dtype)
    v_new = jnp.where(move[..., None], jnp.broadcast_to(V[:, None, :], v.shape), v).astype(v.dtype)
    return state._replace(clusters=cl._replace(positions=x_new, velocities=v_new))


def install_empty_slot_repark(scheme: Any) -> Any:
    """Repark empty slots (:func:`repark_empty_slots`) right after every transfer and at every edge's end.

    The transfer's re-park matters most: nornax's stripping edge refreshes the slow
    force right after the transfer, and a star tree that still holds the just-emptied
    slots out in the tails overflowed its capacities there ('refresh', 'star tree':
    a full re-plan and recompilation per burst of stripping). The empty slots are
    massless, so the transfer's ledger jump (computed before) is unchanged. In place;
    returns ``scheme``.
    """
    import dataclasses as _dc

    real_edge = scheme.edge
    real_progs = getattr(scheme, "_edge_progs", None)

    def edge(carry: Any, **kw: Any) -> tuple:
        carry, info = real_edge(carry, **kw)
        return _dc.replace(carry, state=repark_empty_slots(carry.state)), info

    object.__setattr__(scheme, "edge", edge)
    if real_progs is not None:

        class _Progs:
            def __init__(self, progs: Any) -> None:
                self._progs = progs

            def __getattr__(self, name: str) -> Any:
                return getattr(self._progs, name)

            def transfer(self, state: Any, *args: Any, **kw: Any) -> Any:
                out = self._progs.transfer(state, *args, **kw)
                return (repark_empty_slots(out[0]),) + tuple(out[1:])

        object.__setattr__(scheme, "_edge_progs", lambda slow: _Progs(real_progs(slow)))
    return scheme


def patch_jaccpot_csr_restore_preflight() -> bool:
    """Let a restored large-N GPU run pass jaccpot's static target-block preflight on the CSR lane.

    jaccpot's fused lane records ``(num_leaves, block_size, cap)`` on the solver at an
    eager prepare and refuses a traced refresh whose key differs
    (``_large_n_pipeline.prepare_large_n_state``). After ``ProductionRunner.run(
    restart_from=...)`` the force is re-prepared at the restart's positions and then
    runs under the WRITER's pinned trace state (``jaccpot.carried``), whose leaf count
    differs: the first segment raises "fused payload static target-block cap was not
    preflighted ..." (1e7 galaxy + Pal 5, 2026-11-25; the CPU route at 2e4 restores
    fine). On the CSR near-field lane the cap is forced to 1 and the target-block
    rectangle is a one-block placeholder that is never read (jaccpot's own comment), so
    the key's leaf count does not matter there. This patch makes the check accept a key
    that differs only in ``num_leaves`` when both caps are 1; every other mismatch still
    raises. Only for continuing runs; a request to jaccpot is recorded (HH4b REPORT):
    carry the preflight key in the pinned trace state. Turning the fused payload off
    instead (``JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED=0``) breaks ``dehnen_error``
    ("return_force_scale is wired on the radix fast-lane acceleration route only").

    Returns
    -------
    bool
        Whether the patch was installed (``False``: no such check in this jaccpot, or
        already patched).
    """
    import builtins

    try:
        import jaccpot.runtime._large_n_pipeline as lnp
    except ImportError:
        return False
    attr = builtins.getattr(lnp, "_STATIC_TARGET_BLOCKS_PREFLIGHT_ATTR", None)
    if attr is None or builtins.getattr(lnp, "_odisseo_csr_preflight_patch", False):
        return False

    class _CsrKey(tuple):
        def __eq__(self, other: Any) -> bool:
            if tuple.__eq__(self, other) is True:
                return True
            try:
                return len(other) == 3 and self[1] == other[1] and self[2] == 1 and other[2] == 1
            except TypeError:
                return False

        __hash__ = tuple.__hash__

    def _getattr(obj: Any, name: str, *default: Any) -> Any:
        value = builtins.getattr(obj, name, *default)
        if name == attr and isinstance(value, tuple) and len(value) == 3:
            return _CsrKey(value)
        return value

    lnp.getattr = _getattr  # module-global lookup shadows the builtin in that module only
    lnp._odisseo_csr_preflight_patch = True
    return True
