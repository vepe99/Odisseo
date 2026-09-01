#!/usr/bin/env python3
"""Compare a mesh rollout's final snapshot against its initial conditions.

A quarter-orbit integration of a disc + bulge is only interesting if you can say what
changed, per component. The snapshot written by ``tools/mesh_galaxy_run.py`` carries the
``component`` labels from the IC (0 = disc, 1 = bulge) in the same row order as
``state0``, which is what makes that separation possible after the fact.

Everything is reduced in float64. The state is stored float32, and quantities like the
total angular momentum are differences of sums over 21 million terms, where a float32
reduction's own round-off (~log2(N)*eps ~ 3e-06 relative) is the same size as the
signal -- the mistake this script exists not to repeat.

Example
-------
    python tools/mesh_rollout_analysis.py \\
        --ic  /export/scratch/tbuck/odisseo_ic/disk_bulge_21m_v2.npz \\
        --final /export/scratch/tbuck/odisseo_runs/quarter_orbit/qorbit_final.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic", required=True)
    p.add_argument("--final", required=True,
                   help="<prefix>_final.npz (or _ckpt.npz) from the rollout")
    p.add_argument("--length-unit-kpc", type=float, default=10.0)
    p.add_argument("--time-unit-myr", type=float, default=149.1)
    p.add_argument("--json-out", default="",
                   help="Also write the numbers as JSON here.")
    return p.parse_args()


def _load(path):
    d = np.load(path)
    st = np.asarray(d["state0"])
    return (st[:, 0, :].astype(np.float64),
            st[:, 1, :].astype(np.float64),
            np.asarray(d["mass"]).astype(np.float64),
            d)


def _cyl(pos, vel):
    R = np.hypot(pos[:, 0], pos[:, 1])
    Rs = R + 1e-300
    vphi = (-pos[:, 1] * vel[:, 0] + pos[:, 0] * vel[:, 1]) / Rs
    vR = (pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]) / Rs
    return R, vphi, vR


def _profile(R, m, edges):
    """Surface density in annuli: mass per unit area, so it is comparable across radii."""
    h, _ = np.histogram(R, bins=edges, weights=m)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return h / np.maximum(area, 1e-300)


def main() -> None:
    args = parse_args()
    L = float(args.length_unit_kpc)
    pos0, vel0, m0, d0 = _load(args.ic)
    pos1, vel1, m1, d1 = _load(args.final)
    if len(m0) != len(m1):
        # The rollout may have trimmed N to ndev*k*leaf; compare the rows it kept.
        n = min(len(m0), len(m1))
        print(f"# IC has {len(m0):,} rows and the snapshot {len(m1):,}; comparing the "
              f"first {n:,} (the rollout trims a prefix)")
        pos0, vel0, m0 = pos0[:n], vel0[:n], m0[:n]
        pos1, vel1, m1 = pos1[:n], vel1[:n], m1[:n]

    comp = np.asarray(d1["component"]) if "component" in d1.files else (
        np.asarray(d0["component"]) if "component" in d0.files else None)
    if comp is not None:
        comp = comp[: len(m1)]
    t_end = float(d1["t"]) if "t" in d1.files else float("nan")
    step = int(d1["step"]) if "step" in d1.files else -1

    out: dict = {
        "t_code": t_end, "t_myr": t_end * float(args.time_unit_myr), "step": step,
        "n": int(len(m1)),
    }
    print(f"# snapshot at step {step}, t = {t_end:.5f} code = "
          f"{t_end * float(args.time_unit_myr):.2f} Myr")
    print(f"# N = {len(m1):,}   M_total = {m1.sum():.6f}")

    # ---- conservation, in float64 ----
    for lbl, (pp, vv) in (("initial", (pos0, vel0)), ("final", (pos1, vel1))):
        Lv = (m1[:, None] * np.cross(pp, vv)).sum(0)
        Pv = (m1[:, None] * vv).sum(0)
        KE = 0.5 * float((m1 * (vv * vv).sum(1)).sum())
        com = (m1[:, None] * pp).sum(0) / m1.sum()
        out[lbl] = {"L": Lv.tolist(), "absL": float(np.linalg.norm(Lv)),
                    "P": Pv.tolist(), "KE": KE, "com": com.tolist()}
    L0v = np.array(out["initial"]["L"]); L1v = np.array(out["final"]["L"])
    lscale = float((m1 * np.linalg.norm(np.cross(pos0, vel0), axis=1)).sum())
    out["dL_over_absL"] = float(np.linalg.norm(L1v - L0v) / max(np.linalg.norm(L0v), 1e-300))
    out["dL_over_lscale"] = float(np.linalg.norm(L1v - L0v) / max(lscale, 1e-300))
    out["dKE_over_KE"] = float(
        (out["final"]["KE"] - out["initial"]["KE"]) / max(out["initial"]["KE"], 1e-300))
    print(f"\n# conservation (float64)")
    print(f"    |L| {out['initial']['absL']:.8e} -> {out['final']['absL']:.8e}")
    print(f"    dL/|L|      {out['dL_over_absL']:.4e}   (dL/lscale {out['dL_over_lscale']:.4e})")
    print(f"    KE          {out['initial']['KE']:.8e} -> {out['final']['KE']:.8e} "
          f"({100 * out['dKE_over_KE']:+.4f} %)")
    print(f"    COM drift   {np.linalg.norm(np.array(out['final']['com']) - np.array(out['initial']['com'])):.4e}")

    # ---- per-component structure ----
    groups = [("all", np.ones(len(m1), bool))]
    if comp is not None:
        groups += [("disc", comp == 0), ("bulge", comp == 1)]
    edges = np.concatenate([[0.0], np.logspace(-2.3, 0.7, 25)])
    out["components"] = {}
    for lbl, sel in groups:
        if sel.sum() < 10:
            continue
        r0 = np.linalg.norm(pos0[sel], axis=1)
        r1 = np.linalg.norm(pos1[sel], axis=1)
        R0, vphi0, vR0 = _cyl(pos0[sel], vel0[sel])
        R1, vphi1, vR1 = _cyl(pos1[sel], vel1[sel])
        rec = {
            "n": int(sel.sum()), "mass": float(m1[sel].sum()),
            "r_half": [float(np.median(r0)), float(np.median(r1))],
            "R_half_cyl": [float(np.median(R0)), float(np.median(R1))],
            "abs_z_median": [float(np.median(np.abs(pos0[sel, 2]))),
                             float(np.median(np.abs(pos1[sel, 2])))],
            "vphi_median": [float(np.median(vphi0)), float(np.median(vphi1))],
            "sigma_R": [float(vR0.std()), float(vR1.std())],
            "sigma_z": [float(vel0[sel, 2].std()), float(vel1[sel, 2].std())],
            "prograde_frac": [float((vphi0 > 0).mean()), float((vphi1 > 0).mean())],
        }
        out["components"][lbl] = rec
        print(f"\n# {lbl}  n={rec['n']:,}  M={rec['mass']:.4f}")
        print(f"    {'quantity':18s} {'initial':>12} {'final':>12} {'ratio':>8}")
        for k in ("r_half", "R_half_cyl", "abs_z_median", "vphi_median",
                  "sigma_R", "sigma_z", "prograde_frac"):
            a, b = rec[k]
            unit = f"  ({a * L:.3f} -> {b * L:.3f} kpc)" if k in (
                "r_half", "R_half_cyl", "abs_z_median") else ""
            print(f"    {k:18s} {a:12.5f} {b:12.5f} {b / (a + 1e-300):8.4f}{unit}")

        s0 = _profile(R0, m1[sel], edges)
        s1 = _profile(R1, m1[sel], edges)
        rec["sigma_profile"] = {
            "R_centres": (0.5 * (edges[1:] + edges[:-1])).tolist(),
            "sigma_initial": s0.tolist(), "sigma_final": s1.tolist(),
        }

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n# wrote {args.json_out}")


if __name__ == "__main__":
    main()
