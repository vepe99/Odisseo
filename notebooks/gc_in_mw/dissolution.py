#!/usr/bin/env python3
"""When does the cluster dissolve? Bound mass over time in the live and the static arm (numpy, CPU).

The bound mass is an iterative energy cut of the cluster's own (Plummer-softened, eps_c) self-gravity
in its centre-of-mass frame, from the densest clump: stars with E = v^2 / 2 + Phi_bound > 0 leave the set
until it converges. Tides are ignored, so it is an upper bound on what a tidal cut would keep.

Live arm: every snapshot (members + stripped stars). Also, for the strip rule: of the stars nornax has
STRIPPED (beyond 2 r_J), how many are still energetically bound to the members -- a stripped star is never
recaptured (nornax section 9), so a large bound fraction would mean the rule speeds dissolution.
Static arm: every chunk (all stars; no stripping bookkeeping).

usage: python dissolution.py --live runs/pal5_live_1gyr --static runs/pal5_static_1gyr --out figs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcmw  # noqa: E402

MSUN_PER_CODE = 1e10


def _phi(x, m, eps):
    d = x[:, None, :] - x[None, :, :]
    r2 = np.sum(d * d, -1) + eps * eps
    inv = 1.0 / np.sqrt(r2)
    np.fill_diagonal(inv, 0.0)
    return -(inv * m[None, :]).sum(1)


def bound_set(x, v, m, eps, n_iter=30):
    """Indices of the energetically bound stars (iterative unbinding; G = 1)."""
    # start: the densest clump (shrinking spheres)
    c = np.average(x, 0, m)
    r = np.percentile(np.linalg.norm(x - c, axis=1), 90)
    while True:
        sel = np.linalg.norm(x - c, axis=1) < r
        if sel.sum() < 50:
            break
        c = np.average(x[sel], 0, m[sel])
        r *= 0.8
    keep = np.linalg.norm(x - c, axis=1) < max(5 * r, 1e-4)
    if keep.sum() < 2:
        return np.zeros(len(m), bool)
    for _ in range(n_iter):
        idx = np.flatnonzero(keep)
        if len(idx) < 2:
            return np.zeros(len(m), bool)
        mb = m[idx]
        vc = np.average(v[idx], 0, mb)
        # potential of the bound set at every star (bound and not)
        phi = np.empty(len(m))
        d = x[:, None, :] - x[None, idx, :]
        r2 = np.sum(d * d, -1) + eps * eps
        inv = 1.0 / np.sqrt(r2)
        inv[idx, np.arange(len(idx))] = 0.0
        phi = -(inv * mb[None, :]).sum(1)
        e = 0.5 * np.sum((v - vc) ** 2, 1) + phi
        new = e < 0
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


def live_series(live_dir, eps):
    rows = []
    for f in sorted(glob.glob(os.path.join(live_dir, "snap_*.npz"))):
        d = np.load(f)
        cm = d["cluster_m"].reshape(-1)
        mem = cm > 0
        xm, vm = d["cluster_x"].reshape(-1, 3)[mem], d["cluster_v"].reshape(-1, 3)[mem]
        xs, vs, ms = d["stripped_x"], d["stripped_v"], d["stripped_m"]
        x = np.concatenate([xm, xs])
        v = np.concatenate([vm, vs])
        m = np.concatenate([cm[mem], ms])
        b = bound_set(x, v, m, eps) if len(m) else np.zeros(0, bool)
        # stripped stars still bound to the MEMBERS alone (the strip rule's question)
        sb = None
        if mem.sum() >= 2 and len(ms):
            Vm = np.average(vm, 0, cm[mem])
            d2 = xs[:, None, :] - xm[None, :, :]
            phi = -(cm[mem][None, :] / np.sqrt(np.sum(d2 * d2, -1) + eps * eps)).sum(1)
            sb = int(np.sum(0.5 * np.sum((vs - Vm) ** 2, 1) + phi < 0))
        rows.append(dict(time_myr=float(d["time"]) * gcmw.TIME_MYR, members=int(mem.sum()), stripped=len(ms),
                         bound=int(b.sum()), bound_msun=float(m[b].sum() * MSUN_PER_CODE),
                         stripped_still_bound=sb))
    return rows


def static_series(static_dir, eps, every=5):
    rows = []
    files = sorted(glob.glob(os.path.join(static_dir, "chunk_*.npz")))
    for f in files[::every] + ([files[-1]] if (len(files) - 1) % every else []):
        d = np.load(f)
        s = d["state"]
        x, v = s[:, 0], s[:, 1]
        with open(os.path.join(static_dir, "run.json")) as fh:
            run = json.load(fh)
        n = len(x)
        mass = None
        for row in run.get("table", []) or []:
            mass = row.get("mass")
        m = np.full(n, (mass or 1.34e-6) / n)
        b = bound_set(x, v, m, eps)
        rows.append(dict(time_myr=float(d["time"]) * gcmw.TIME_MYR, bound=int(b.sum()),
                         bound_msun=float(m[b].sum() * MSUN_PER_CODE)))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", default=None)
    p.add_argument("--static", nargs="*", default=[])
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--mass-msun", nargs="*", type=float, default=None, help="static runs' cluster masses")
    p.add_argument("--eps", type=float, default=None, help="eps_c (default: the live run's)")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    eps = a.eps
    out = {}
    if a.live:
        with open(os.path.join(a.live, "run.json")) as fh:
            eps = eps or json.load(fh)["eps_c"]
        out["live"] = live_series(a.live, eps)
    for k, sdir in enumerate(a.static):
        with open(os.path.join(sdir, "run.json")) as fh:
            srun = json.load(fh)
        e = eps or srun["eps_c"]
        rows = static_series(sdir, e, a.every)
        if a.mass_msun:  # rescale to the run's cluster mass
            n = srun["M"]
            for r in rows:
                r["bound_msun"] = r["bound"] / n * a.mass_msun[k]
        out[(a.labels[k] if a.labels else f"static{k}")] = rows
    gcmw.write_json(os.path.join(a.out, "dissolution.json"), out)
    for k, rows in out.items():
        print(k)
        for r in rows:
            print("  ", {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in r.items()})


if __name__ == "__main__":
    main()
