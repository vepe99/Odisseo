#!/usr/bin/env python3
"""The live galaxy's own field along the orbit vs its agama potential and the static fit (numpy + agama, CPU).

The N-body realisation (gcmw.load_galaxy: recentred, turned to the Milky Way's sense) is summed directly
(Plummer-softened at --eps, chunked) at points along the live-model point orbit of prepare.py. Its
force is compared with the agama self-consistent potential the ICs were drawn from and with the static
Odisseo fit. Run in the odisseo env (agama).

usage: python field_check.py --galaxy mw_1000000.npz --potential mw_1000000.pot.ini --prepare runs/prepare_1gyr
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcmw  # noqa: E402


def direct_accel(pts, x, m, eps, chunk=200000):
    a = np.zeros_like(pts)
    for lo in range(0, len(m), chunk):
        d = x[None, lo:lo + chunk, :] - pts[:, None, :]
        r2 = np.sum(d * d, -1) + eps * eps
        a += np.sum(m[None, lo:lo + chunk, None] * d / r2[..., None] ** 1.5, 1)
    return a


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--galaxy", required=True)
    p.add_argument("--potential", required=True)
    p.add_argument("--prepare", required=True)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--points", type=int, default=60)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import agama

    os.makedirs(a.out, exist_ok=True)
    pz = np.load(os.path.join(a.prepare, "prepare.npz"))
    with open(os.path.join(a.prepare, "prepare.json")) as fh:
        prep = json.load(fh)
    track = pz["fwd_live"][:, 0, 0]  # the live-model point orbit, forward
    idx = np.linspace(0, len(track) - 1, a.points).astype(int)
    pts = track[idx]
    state, mass, meta = gcmw.load_galaxy(a.galaxy)
    a_nb = direct_accel(pts, state[:, 0], mass, a.eps)
    pot = agama.Potential(a.potential)
    a_ag = np.asarray(pot.force(pts))
    a_st = gcmw.static_accel(prep["fit_params"])(pts)
    n = np.linalg.norm(a_ag, axis=1)
    rel = lambda u, v: np.linalg.norm(u - v, axis=1) / n  # noqa: E731
    radial = lambda u: -np.sum(u * pts, 1) / np.linalg.norm(pts, axis=1)  # noqa: E731
    out = dict(
        n_points=len(pts), eps=a.eps, galaxy_flipped=meta.get("flipped"),
        nbody_vs_agama=dict(median=float(np.median(rel(a_nb, a_ag))), max=float(rel(a_nb, a_ag).max())),
        static_vs_agama=dict(median=float(np.median(rel(a_st, a_ag))), max=float(rel(a_st, a_ag).max())),
        nbody_vs_static=dict(median=float(np.median(rel(a_nb, a_st))), max=float(rel(a_nb, a_st).max())),
        radial_ratio_nbody_over_agama=dict(median=float(np.median(radial(a_nb) / radial(a_ag)))),
        radial_ratio_static_over_agama=dict(median=float(np.median(radial(a_st) / radial(a_ag)))),
        mass_total=float(mass.sum()),
    )
    gcmw.write_json(os.path.join(a.out, "field_check.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
