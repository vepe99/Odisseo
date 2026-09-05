"""Reproducible initial conditions for the multi-GPU FMM / tree benchmarks.

All generators are seeded and return ``(positions[N,3] float32, masses[N] float32)``.
The controlled distributions (uniform, Plummer) drive the accuracy sweeps; the
separated-cluster IC is the far-field stress test; the cached disk IC is the
"realistic" galaxy used by the standing Bonsai comparison.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

DISK_IC_TIPSY = "/export/home/tbuck/Odisseo/benchmark_a100/bonsai_reference/disk_ic.tipsy"


def uniform_box(n: int, *, seed: int = 0, box: float = 1.0, total_mass: float = 1.0):
    """Uniform random points in ``[-box, box]^3`` with equal masses."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-box, box, size=(n, 3)).astype(np.float32)
    mass = np.full(n, total_mass / n, np.float32)
    return pos, mass


def plummer_sphere(n: int, *, seed: int = 0, a: float = 1.0, total_mass: float = 1.0):
    """Sample a Plummer sphere of scale radius ``a`` with equal-mass particles.

    Radii via the inverse enclosed-mass CDF ``r = a / sqrt(X^{-2/3} - 1)``,
    directions isotropic on the sphere.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = a / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    # isotropic directions
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    sin_t = np.sqrt(1.0 - mu * mu)
    pos = np.stack(
        [r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * mu], axis=1
    ).astype(np.float32)
    mass = np.full(n, total_mass / n, np.float32)
    return pos, mass


def separated_clusters(ndev: int, per: int, *, seed: int = 4, sep: float = 6.0):
    """``ndev`` spatially separated uniform cubes (one per Morton domain).

    This is the far-field stress test: cross-domain pairs are genuinely
    far-field, so the coarse M2L path is exercised (adjacent/overlapping domains
    would resolve everything as near-field).
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0, 0, 0], [sep, 0, 0], [0, sep, 0], [0, 0, sep]], dtype=np.float32
    )[:ndev]
    pos = np.concatenate(
        [centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return pos, mass


def _read_tipsy_standard(path: str):
    """Read a standard-format tipsy binary (Bonsai's dump layout).

    Header: ``double time; int nbodies; int ndim; int nsph; int ndark; int nstar``
    (+ 4 bytes pad -> 32-byte header).  Dark particles: ``float mass; float pos[3];
    float vel[3]; float eps; float phi`` (9 floats).  Star particles append two
    extra floats (metals, tform) -> 11 floats.  We read mass + positions only.
    """
    data = Path(path).read_bytes()
    time_, nbodies, ndim, nsph, ndark, nstar = struct.unpack_from("<diiiii", data, 0)
    off = 32  # 8 (double) + 20 (5 ints) + 4 (pad)
    pos = np.empty((nbodies, 3), np.float32)
    mass = np.empty(nbodies, np.float32)
    i = 0
    # gas (nsph): mass,pos[3],vel[3],rho,temp,hsmooth,metals,phi = 12 floats
    for _ in range(nsph):
        vals = struct.unpack_from("<12f", data, off)
        mass[i] = vals[0]
        pos[i] = vals[1:4]
        off += 48
        i += 1
    for _ in range(ndark):  # dark: 9 floats
        vals = struct.unpack_from("<9f", data, off)
        mass[i] = vals[0]
        pos[i] = vals[1:4]
        off += 36
        i += 1
    for _ in range(nstar):  # star: 11 floats
        vals = struct.unpack_from("<11f", data, off)
        mass[i] = vals[0]
        pos[i] = vals[1:4]
        off += 44
        i += 1
    return pos, mass, dict(time=time_, nbodies=nbodies, nsph=nsph, ndark=ndark, nstar=nstar)


def disk_ic(path: str = DISK_IC_TIPSY, *, subsample: int | None = None, seed: int = 0):
    """Load the cached 200k galaxy-disk IC used by the Bonsai benchmark.

    ``subsample`` (if given) randomly draws that many particles (mass rescaled so
    the total is preserved) so the O(N^2) direct reference stays affordable.
    """
    pos, mass, hdr = _read_tipsy_standard(path)
    if subsample is not None and subsample < pos.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pos.shape[0], size=subsample, replace=False)
        frac = pos.shape[0] / subsample
        pos = pos[idx]
        mass = (mass[idx] * frac).astype(np.float32)
    return pos, mass


IC_GENERATORS = {
    "uniform": uniform_box,
    "plummer": plummer_sphere,
    "clusters": separated_clusters,
    "disk": disk_ic,
}
