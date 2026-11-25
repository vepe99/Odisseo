#!/usr/bin/env python3
"""Step 4 of the GC-in-MW demo: conservation, stripping, and the stream track live vs static.

Reads prepare.py's, run_live.py's and run_static.py's outputs; writes figures and
``summary.json`` into --out. numpy + matplotlib only (no JAX).

The comparison (the sanity check):
- the progenitors' present positions, live vs static, against the separation of the
  POINT orbits in the live model's potential vs the static fit (prepare.py): that
  separation is the orbit-level size of the live-vs-static potential difference;
- the stream tracks in each progenitor's own orbital frame (phi1 along the orbit,
  phi2 out of its plane, r / r_progenitor), which removes the orbital phase: their
  differences against the stream's own width.

usage: python analyse.py --prepare runs/prepare --live runs/pal5_live --static runs/pal5_static --out figs
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


def read_rows(path):
    rows = {}
    if os.path.exists(path):
        for ln in open(path):
            if ln.strip():
                r = json.loads(ln)
                rows[r.get("segment", r.get("chunk"))] = r  # a resumed run rewrites edges: keep the last
    return [rows[k] for k in sorted(rows)]


def shrinking_centre(x, v, m, r0=None, frac=0.8, n_min=50):
    """Centre of the densest clump (shrinking spheres), and the mean velocity of its members."""
    c = np.average(x, 0, m)
    r = r0 or np.percentile(np.linalg.norm(x - c, axis=1), 90)
    while True:
        sel = np.linalg.norm(x - c, axis=1) < r
        if sel.sum() < n_min:
            break
        c = np.average(x[sel], 0, m[sel])
        r *= frac
    sel = np.linalg.norm(x - c, axis=1) < r / frac
    return c, np.average(v[sel], 0, m[sel])


def live_final(live_dir):
    snaps = sorted(glob.glob(os.path.join(live_dir, "snap_*.npz")))
    d = np.load(snaps[-1])
    cm = d["cluster_m"].reshape(-1)
    member = cm > 0
    xm, vm = d["cluster_x"].reshape(-1, 3)[member], d["cluster_v"].reshape(-1, 3)[member]
    x = np.concatenate([xm, d["stripped_x"]])
    v = np.concatenate([vm, d["stripped_v"]])
    m = np.concatenate([cm[member], d["stripped_m"]])
    return dict(time=float(d["time"]), x=x, v=v, m=m, n_members=int(member.sum()),
                n_stripped=len(d["stripped_m"]), snaps=snaps)


def static_final(static_dir, t=None):
    """The static run's state at the chunk end closest to ``t`` (default: its last)."""
    chunks = sorted(glob.glob(os.path.join(static_dir, "chunk_*.npz")))
    times = np.array([float(np.load(c)["time"]) for c in chunks])
    k = len(chunks) - 1 if t is None else int(np.argmin(np.abs(times - t)))
    d = np.load(chunks[k])
    s = d["state"]
    with open(os.path.join(static_dir, "run.json")) as fh:
        run = json.load(fh)
    m = np.full(len(s), 1.0)
    return dict(time=float(d["time"]), x=s[:, 0], v=s[:, 1], m=m, run=run)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--static", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.prepare, "prepare.json")) as fh:
        prep = json.load(fh)
    pz = np.load(os.path.join(a.prepare, "prepare.npz"))
    with open(os.path.join(a.live, "run.json")) as fh:
        live_run = json.load(fh)
    rows = read_rows(os.path.join(a.live, "rows.jsonl"))
    srows = read_rows(os.path.join(a.static, "rows.jsonl"))
    L = live_final(a.live)
    S = static_final(a.static, L["time"])  # the comparison is at the live run's last snapshot
    t_end = prep["T_code"]
    complete = abs(L["time"] - t_end) < 1e-6
    myr = gcmw.TIME_MYR

    # ---- conservation and stripping (live) ---------------------------------------
    t = np.array([r["time"] for r in rows]) * myr
    def col(name):
        return np.array([np.nan if r.get(name) is None else float(np.ravel(r.get(name))[0]) for r in rows])
    e_led = col("energy_ledgered_drift")
    mom = col("momentum_drift")
    nstrip = col("n_stripped")
    ts = np.array([r["time"] for r in srows]) * myr
    es = np.array([r["energy"] for r in srows])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), tight_layout=True)
    ax[0].plot(t, e_led, ".-", label="live: (E - ledger) drift")
    if len(es) > 1:
        upto = ts <= t.max() + 1e-9  # the live run's span
        ax[0].plot(ts[upto], ((es - es[0]) / abs(es[0]))[upto], "s-", ms=3, label="static: cluster E drift")
    ax[0].set_xlabel("t [Myr]")
    ax[0].set_ylabel("relative energy drift")
    ax[0].legend()
    ax[1].semilogy(t, np.abs(mom), ".-")
    ax[1].set_xlabel("t [Myr]")
    ax[1].set_ylabel("|momentum drift| (runner)")
    ax[2].plot(t, nstrip, ".-")
    ax[2].set_xlabel("t [Myr]")
    ax[2].set_ylabel("stripped stars")
    ax[2].set_title(f"of {live_run['K'] * live_run['M']} stars")
    fig.savefig(os.path.join(a.out, "conservation.png"), dpi=110)

    # ---- the potential --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), tight_layout=True)
    ax[0].plot(pz["R"] * gcmw.KPC, pz["vc_live"] * gcmw.KMS, label="live model (agama SCM)")
    ax[0].plot(pz["R"] * gcmw.KPC, pz["vc_static"] * gcmw.KMS, "--", label="static fit (Odisseo)")
    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("v_c [km/s]")
    ax[0].legend()
    sep = np.linalg.norm(pz["fwd_live"][:, 0, 0] - pz["fwd_static"][:, 0, 0], axis=-1) * gcmw.KPC
    tt = np.linspace(0, prep["t_gyr"] * 1e3, len(sep))
    ax[1].plot(tt, sep)
    ax[1].axvline(L["time"] * myr, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("t [Myr]")
    ax[1].set_ylabel("point-orbit separation, live model vs static [kpc]")
    fig.savefig(os.path.join(a.out, "potential.png"), dpi=110)

    # ---- the stream -----------------------------------------------------------------
    nt = len(pz["fwd_live"])
    tq = np.linspace(0.0, t_end, nt)
    point_live = np.array([np.interp(L["time"], tq, pz["fwd_live"][:, 0, 0, i]) for i in range(3)])
    point_static = np.array([np.interp(L["time"], tq, pz["fwd_static"][:, 0, 0, i]) for i in range(3)])
    cL, vL = shrinking_centre(L["x"], L["v"], L["m"])
    cS, vS = shrinking_centre(S["x"], S["v"], S["m"])
    today = np.array(prep["today"]["x"])[0] if complete else point_live
    p1L, p2L, rL = gcmw.stream_coordinates(L["x"], cL, vL)
    p1S, p2S, rS = gcmw.stream_coordinates(S["x"], cS, vS)
    lo, hi = np.percentile(np.concatenate([p1L, p1S]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 31)
    tL2, tLr, nL = gcmw.stream_track(p1L, p2L, rL / np.linalg.norm(cL), bins)
    tS2, tSr, nS = gcmw.stream_track(p1S, p2S, rS / np.linalg.norm(cS), bins)
    both = np.isfinite(tL2) & np.isfinite(tS2)
    width = float(np.std(p2L[np.abs(p1L) > 1.0])) if np.any(np.abs(p1L) > 1.0) else float("nan")
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), tight_layout=True)
    fl, fs = pz["fwd_live"][:, 0, 0] * gcmw.KPC, pz["fwd_static"][:, 0, 0] * gcmw.KPC
    for j, (i0, i1, lab) in enumerate(((0, 1, "y"), (0, 2, "z"))):
        axx = ax[0, j]
        axx.plot(fl[:, i0], fl[:, i1], "k-", lw=0.6, alpha=0.5, label="point orbit, live model")
        axx.plot(fs[:, i0], fs[:, i1], "g-", lw=0.6, alpha=0.5, label="point orbit, static fit")
        axx.scatter(S["x"][:, i0] * gcmw.KPC, S["x"][:, i1] * gcmw.KPC, s=1, c="tab:orange", label="static N-body")
        axx.scatter(L["x"][:, i0] * gcmw.KPC, L["x"][:, i1] * gcmw.KPC, s=1, c="tab:blue", label="live MW")
        axx.plot(today[i0] * gcmw.KPC, today[i1] * gcmw.KPC, "r*", ms=9,
                 label="Pal 5 today" if complete else f"point orbit (live model) at {L['time'] * myr:.0f} Myr")
        axx.set_xlabel("x [kpc]")
        axx.set_ylabel(f"{lab} [kpc]")
        axx.set_aspect("equal", "datalim")
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].scatter(p1S, p2S, s=1, c="tab:orange", label="static")
    ax[1, 0].scatter(p1L, p2L, s=1, c="tab:blue", label="live")
    mid = 0.5 * (bins[1:] + bins[:-1])
    ax[1, 0].plot(mid, tL2, "b-", mid, tS2, "-", color="darkorange")
    ax[1, 0].set_xlabel("phi1 [deg] (along the orbit, progenitor at 0)")
    ax[1, 0].set_ylabel("phi2 [deg]")
    ax[1, 0].legend(markerscale=4)
    ax[1, 1].plot(mid, tLr, "b.-", label="live")
    ax[1, 1].plot(mid, tSr, ".-", color="darkorange", label="static")
    ax[1, 1].set_xlabel("phi1 [deg]")
    ax[1, 1].set_ylabel("r / r_progenitor (median)")
    ax[1, 1].legend()
    fig.savefig(os.path.join(a.out, "stream.png"), dpi=110)

    summary = dict(
        live=dict(time_myr=L["time"] * myr, n_members=L["n_members"], n_stripped=L["n_stripped"],
                  progenitor_kpc=cL * gcmw.KPC, energy_ledgered_drift_max=float(np.nanmax(np.abs(e_led))),
                  momentum_drift_max=float(np.nanmax(np.abs(mom))), replans=sum(len(r.get("replans") or [])
                                                                               for r in rows),
                  retries=sum(int(r.get("retries") or 0) for r in rows),
                  device_peak_gb=max((r["mem"] or {}).get("peak") or 0 for r in rows) / 1e9),
        static=dict(time_myr=S["time"] * myr, progenitor_kpc=cS * gcmw.KPC,
                    energy_drift_max=float(np.max(np.abs((es - es[0]) / es[0]))) if len(es) > 1 else None),
        today_kpc=today * gcmw.KPC,
        progenitor_offset_live_vs_static_kpc=float(np.linalg.norm(cL - cS) * gcmw.KPC),
        complete=complete, compared_at_myr=L["time"] * myr, static_chunk_time_myr=S["time"] * myr,
        point_orbit_offset_live_model_vs_static_kpc=float(np.linalg.norm(point_live - point_static) * gcmw.KPC),
        point_orbit_offset_at_present_kpc=float(prep["present_offset_static_vs_live_kpc"][0]),
        live_progenitor_vs_point_orbit_kpc=float(np.linalg.norm(cL - point_live) * gcmw.KPC),
        static_progenitor_vs_point_orbit_static_kpc=float(np.linalg.norm(cS - point_static) * gcmw.KPC),
        live_progenitor_vs_today_kpc=float(np.linalg.norm(cL - today) * gcmw.KPC),
        static_progenitor_vs_today_kpc=float(np.linalg.norm(cS - today) * gcmw.KPC),
        track=dict(bins_deg=[float(bins[0]), float(bins[-1]), len(bins) - 1], bins_compared=int(both.sum()),
                   dphi2_median_deg=float(np.median(np.abs(tL2 - tS2)[both])) if both.any() else None,
                   dphi2_max_deg=float(np.max(np.abs(tL2 - tS2)[both])) if both.any() else None,
                   dr_rel_median=float(np.median(np.abs(tLr - tSr)[both])) if both.any() else None,
                   stream_width_phi2_deg=width,
                   extent_live_deg=[float(np.percentile(p1L, 1)), float(np.percentile(p1L, 99))],
                   extent_static_deg=[float(np.percentile(p1S, 1)), float(np.percentile(p1S, 99))]),
        static_fit=prep["fit_stats"], vc_8kpc=prep["vc_kms_at_8kpc"],
        # mass loss on equal terms (the static run has no stripping bookkeeping): the fraction of
        # the stars farther than R from their own progenitor
        outside_fraction={f"{int(r_pc)}pc": dict(
            live=float(np.mean(np.linalg.norm(L["x"] - cL, axis=1) > r_pc / 1e4)),
            static=float(np.mean(np.linalg.norm(S["x"] - cS, axis=1) > r_pc / 1e4)))
            for r_pc in (50.0, 100.0, 200.0)},
    )
    gcmw.write_json(os.path.join(a.out, "summary.json"), summary)
    print(json.dumps(summary, indent=1, default=gcmw._jsonable))


if __name__ == "__main__":
    main()
