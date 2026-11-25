#!/usr/bin/env python3
"""The K ~ 150 bundle's functional run: verdicts, membership / ledger / energy columns, device peak, a map.

Reads run_live.py's output (rows.jsonl, run.json, the last snap_*.npz); writes bundle.png and
bundle_summary.json into --out. numpy + matplotlib only.

usage: python analyse_bundle.py --run runs/bundle_functional --out figs
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
from analyse import read_rows  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.run, "run.json")) as fh:
        run = json.load(fh)
    rows = read_rows(os.path.join(a.run, "rows.jsonl"))
    snaps = sorted(glob.glob(os.path.join(a.run, "snap_*.npz")))
    first, last = np.load(snaps[0]), np.load(snaps[-1])
    m0 = first["cluster_m"].sum(1)
    m1 = last["cluster_m"].sum(1)
    table = []
    for r in rows:
        table.append(dict(
            segment=r["segment"], time_myr=r["time"] * gcmw.TIME_MYR, retries=r["retries"], replans=r["replans"],
            n_stripped=r.get("n_stripped"), strip_transfers=r.get("strip_transfers"),
            dissolved=int(np.sum(r["dissolved"])) if r.get("dissolved") is not None else None,
            reserve=r.get("reserve"), n_live=r.get("n_live"),
            energy_ledgered_drift=r.get("energy_ledgered_drift"), momentum_drift=r.get("momentum_drift"),
            strip_ledger=r.get("strip_ledger"), W_cc_inter=r.get("W_cc_inter"),
            peak_gb=((r.get("mem") or {}).get("peak") or 0) / 1e9, wall_s=r.get("wall_s"),
        ))
    lost = 1.0 - m1 / np.where(m0 > 0, m0, 1.0)
    order = np.argsort(-lost)
    names = [row["name"] for row in run["table"]]
    summary = dict(
        K=run["K"], M=run["M"], N_galaxy=run["N_galaxy"], n_sub=run["n_sub"], eps_c=run["eps_c"], eps_g=run["eps_g"],
        dt=run["dt"], steps_per_segment=run["steps_per_segment"], edges=table,
        verdicts_ok=all(int(t["retries"] or 0) == 0 for t in table), device_peak_gb=max(t["peak_gb"] for t in table),
        mass_lost_top=[dict(name=names[k], lost_fraction=float(lost[k]),
                            r_gc_kpc=float(np.linalg.norm(first["cluster_x"][k].mean(0)) * gcmw.KPC))
                       for k in order[:10]],
        clusters_losing_any=int(np.sum(lost > 0)),
    )
    gcmw.write_json(os.path.join(a.out, "bundle_summary.json"), summary)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), tight_layout=True)
    cx = last["cluster_x"]
    for j, (i0, i1, lab) in enumerate(((0, 1, "y"), (0, 2, "z"))):
        axx = ax[j]
        axx.imshow(np.log10(1 + last["galaxy_xy" if j == 0 else "galaxy_xz"].T), origin="lower",
                   extent=[-30, 30, -30, 30], cmap="Greys", aspect="equal")
        mem = last["cluster_m"] > 0
        axx.scatter(cx[mem][:, i0] * gcmw.KPC, cx[mem][:, i1] * gcmw.KPC, s=0.2, c="tab:blue", label="members")
        sx = last["stripped_x"]
        axx.scatter(sx[:, i0] * gcmw.KPC, sx[:, i1] * gcmw.KPC, s=0.5, c="tab:red", label="stripped")
        axx.set_xlim(-30, 30)
        axx.set_ylim(-30, 30)
        axx.set_xlabel("x [kpc]")
        axx.set_ylabel(f"{lab} [kpc]")
    ax[0].legend(markerscale=10, loc="upper right")
    ax[0].set_title(f"K = {run['K']} clusters x {run['M']} stars, t = {float(last['time']) * gcmw.TIME_MYR:.1f} Myr")
    t = [x["time_myr"] for x in table]
    ax[2].plot(t, [x["n_stripped"] for x in table], "o-", label="stripped stars")
    ax[2].set_xlabel("t [Myr]")
    ax[2].set_ylabel("stripped stars")
    ax2 = ax[2].twinx()
    ax2.plot(t, [abs(x["energy_ledgered_drift"] or 0) for x in table], "s--", c="tab:red", label="|E - ledger| drift")
    ax2.set_yscale("log")
    ax2.set_ylabel("|(E - ledger) drift|", color="tab:red")
    fig.savefig(os.path.join(a.out, "bundle.png"), dpi=110)
    print(json.dumps({k: v for k, v in summary.items() if k != "edges"}, indent=1))
    for row in table:
        print(row)


if __name__ == "__main__":
    main()
