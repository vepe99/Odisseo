#!/usr/bin/env python3
"""Step 1 of the GC-in-MW demo: the static potential and the GCs' orbits back in time (CPU, needs agama).

1. Fit Odisseo's static MW (NFW + MN3 disc + Hernquist bulge, gcmw.static_config_params)
   to the live model's own potential -- the agama self-consistent potential that
   ``tools/a4_galaxy_ics.py --export-potential`` wrote with the live galaxy -- over the
   region the clusters' orbits cover. Free: the halo's mass and scale radius and the
   disc's mass (the SCM's halo is adiabatically contracted and cut off; the analytic
   NFW is neither). The residual force difference is "the live-vs-static potential
   difference" of the sanity check.
2. Today's Galactocentric position/velocity of each GC (catalogue + astropy's frame),
   integrated BACK by ``--t-gyr`` in the agama potential (the live galaxy's mean field).
   That is where the live run starts the clusters.
3. The same back point integrated FORWARD again in the agama potential (it returns to
   today) and in the static fit: their separation at the present is the orbit-level
   size of the live-vs-static difference the stream tracks are compared against.

Writes ``prepare.json`` (fit, residuals, back points, orbit tracks) and ``prepare.npz``.

usage: python prepare.py --potential mw_1000000.pot.ini [--names "Pal 5"] [--t-gyr 2]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcmw  # noqa: E402

from odisseo import globular_clusters as gc  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--potential", required=True, help="agama .ini of the live model")
    p.add_argument("--catalogue", default=os.environ.get("ODISSEO_GC_DATA"), help="fetch_baumgardt.py --out dir")
    p.add_argument("--names", nargs="+", default=["Pal 5"])
    p.add_argument("--t-gyr", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=2e-4, help="orbit step (code units)")
    p.add_argument("--out", default=None, help="default $ODISSEO_GC_RUNS/prepare")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import agama

    out = a.out or gcmw.runs_dir("prepare")
    os.makedirs(out, exist_ok=True)
    pot = agama.Potential(a.potential)

    def acc_live(x):
        return np.asarray(pot.force(np.atleast_2d(x)))

    # --- the orbits in the live model's potential ----------------------------
    cat = gc.load_baumgardt(directory=a.catalogue)
    frame = gc.galactocentric_frame()
    table = gc.cluster_table(a.names, cat, frame=frame)
    x_now = np.array([row["x"] for row in table])
    v_now = np.array([row["v"] for row in table])
    T = a.t_gyr * 1e3 / gcmw.TIME_MYR
    xb, vb, back = gc.integrate_orbit(x_now, v_now, acc_live, -T, a.dt, n_out=400)

    # --- the fit: where the orbits go ----------------------------------------
    # points along the back-integrated orbits, jittered by ~1 kpc, plus 20 % spread
    # over 3-30 kpc (so the fit stays a galaxy)
    rng = np.random.default_rng(a.seed)
    track = back[:, :, 0].reshape(-1, 3)
    on = track[rng.integers(0, len(track), 2400)] + rng.normal(scale=0.1, size=(2400, 3))
    r = np.exp(rng.uniform(np.log(0.3), np.log(3.0), 600))
    mu = rng.uniform(-1, 1, 600)
    ph = rng.uniform(0, 2 * np.pi, 600)
    pts = np.concatenate([on, np.stack([r * np.sqrt(1 - mu**2) * np.cos(ph), r * np.sqrt(1 - mu**2) * np.sin(ph),
                                        r * mu], -1)])
    a_live = acc_live(pts)
    an = np.linalg.norm(a_live, axis=1)
    # free: the halo (the SCM's halo is contracted and cut off) and the disc within physical bounds
    # (MN3 only approximates an exponential disc); the bulge is the model's own Hernquist, fixed
    names = ["halo_M", "halo_rs", "disc_M", "disc_hr", "disc_hz"]
    lo = np.log([10.0, 0.5, 4.0, 0.15, 0.01])
    hi = np.log([300.0, 6.0, 9.0, 0.45, 0.1])

    def model(theta):
        q = dict(gcmw.STATIC_NOMINAL)
        q.update(zip(names, np.exp(theta)))
        return q

    def resid(theta):
        return ((gcmw.static_accel(model(theta))(pts) - a_live) / an[:, None]).ravel()

    from scipy.optimize import least_squares

    nominal = gcmw.STATIC_NOMINAL
    theta0 = np.log([nominal[k] for k in names])
    fit = least_squares(resid, theta0, bounds=(lo, hi), diff_step=1e-4, xtol=1e-8, ftol=1e-10)
    best = model(fit.x)
    stats = {}
    on_orbit = np.arange(len(pts)) < 2400
    for label, q in (("nominal", dict(nominal)), ("fit", best)):
        rel = np.linalg.norm(gcmw.static_accel(q)(pts) - a_live, axis=1) / an
        stats[label] = {}
        for region, sel in (("orbit", on_orbit), ("3-30kpc", ~on_orbit)):
            stats[label][region] = dict(median=float(np.median(rel[sel])), p90=float(np.percentile(rel[sel], 90)),
                                        max=float(rel[sel].max()))
        print(f"[fit] {label}: {stats[label]}", flush=True)
    acc_static = gcmw.static_accel(best)

    # circular velocity curves (plane)
    R = np.linspace(0.05, 3.0, 60)
    xp = np.stack([R, 0 * R, 0 * R], -1)
    vc_live = np.sqrt(R * -acc_live(xp)[:, 0])
    vc_static = np.sqrt(R * -acc_static(xp)[:, 0])

    xf_live, vf_live, fwd_live = gc.integrate_orbit(xb, vb, acc_live, T, a.dt, n_out=400)
    xf_st, vf_st, fwd_st = gc.integrate_orbit(xb, vb, acc_static, T, a.dt, n_out=400)
    sep = np.linalg.norm(fwd_live[:, :, 0] - fwd_st[:, :, 0], axis=-1)  # [n_out + 1, K]
    rel_orbit = np.linalg.norm(acc_static(fwd_live[:, 0, 0]) - acc_live(fwd_live[:, 0, 0]), axis=1) / np.linalg.norm(
        acc_live(fwd_live[:, 0, 0]), axis=1)
    rec = dict(
        potential=os.path.abspath(a.potential), names=[row["name"] for row in table], t_gyr=a.t_gyr, T_code=T,
        dt=a.dt, frame=gc.frame_parameters(frame), fit_params=best, fit_free=names, fit_stats=stats,
        fit_region="2400 points along the back orbits (1 kpc jitter) + 600 over 3-30 kpc", nominal=nominal,
        today=dict(x=x_now, v=v_now), back=dict(x=xb, v=vb),
        return_error_live=np.linalg.norm(xf_live - x_now, axis=1),
        present_offset_static_vs_live_kpc=sep[-1] * gcmw.KPC,
        max_offset_static_vs_live_kpc=sep.max(0) * gcmw.KPC,
        force_diff_along_orbit=dict(median=float(np.median(rel_orbit)), max=float(rel_orbit.max())),
        vc_kms_at_8kpc=dict(live=float(np.interp(0.8, R, vc_live) * gcmw.KMS),
                            static=float(np.interp(0.8, R, vc_static) * gcmw.KMS)),
        table=[{k: v for k, v in row.items()} for row in table],
    )
    gcmw.write_json(os.path.join(out, "prepare.json"), rec)
    np.savez(os.path.join(out, "prepare.npz"), back_track=back, fwd_live=fwd_live, fwd_static=fwd_st,
             R=R, vc_live=vc_live, vc_static=vc_static, fit_points=pts, a_live=a_live,
             a_static=acc_static(pts))
    print(f"[orbit] {rec['names']}: back {a.t_gyr} Gyr -> x {xb.tolist()}; present offset static vs live "
          f"{rec['present_offset_static_vs_live_kpc']} kpc (max {rec['max_offset_static_vs_live_kpc']}); "
          f"vc(8 kpc) live {rec['vc_kms_at_8kpc']['live']:.1f} static {rec['vc_kms_at_8kpc']['static']:.1f} km/s",
          flush=True)


if __name__ == "__main__":
    main()
