#!/usr/bin/env python
"""T2.0 / T2.1 -- interaction budget vs error for both codes, on one axis.

Reads ``interaction_budget_*.json`` (jaccpot, per leaf x theta) and the
``compare_force*.json`` pkdgrav3 rows, and prints:

* per code, per configuration: direct sources per target, share of N, aggL2,
  wall time -- the MAC-efficiency table;
* the theta-sensitivity of each code's error, **with pkdgrav3's theta rescaled**
  (plan T2.1): pkdgrav3 opens on ``(bMax_c + bMax_k)/d < theta/1.5`` while
  jaccpot's dehnen/bh test is ``(r_t + r_s)/d < theta``, so a pkdgrav3 theta of
  ``x`` is a jaccpot theta of ``x/1.5``.  Slopes are reported per 0.1 of
  *jaccpot-theta*, and pkdgrav3's are taken from theta >= 0.6 only, because its
  0.4/0.5 points sit on the fp32 P-P floor and flatten the fit;
* the pkdgrav3 budget from its own ``P-P per active`` report (both readings of
  the sink-bucket question, see ``common/budget.py``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PKD_THETA_SCALE = 1.5  # pkdgrav3 X_Open = 1.5 * bMax / theta  (gravity/opening.cxx)


def slope_per_0p1(thetas, errs):
    """Multiplicative error growth per +0.1 theta from a log-linear fit."""
    t = np.asarray(thetas, float)
    e = np.log(np.asarray(errs, float))
    if len(t) < 2:
        return float("nan")
    b = np.polyfit(t, e, 1)[0]
    return math.exp(0.1 * b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default=str(ROOT / "artifacts" / "interaction_budget_plummer200000.json"))
    ap.add_argument("--compare", default=str(ROOT / "artifacts" / "compare_force_plummer200k.json"))
    ap.add_argument("--pkd-log", default=None,
                    help="pkdgrav3 stdout to parse P-P/P-C from when the compare JSON predates "
                         "the budget columns")
    args = ap.parse_args()

    B = json.load(open(args.budget))
    n = B["ic"]["n"]
    rows = [r for r in B["rows"] if "failed" not in r]
    print(f"jaccpot  N={n}  IC={B['ic']['name']}  order={rows[0]['order']}  GPU={B['devices']}")
    print(f"{'leaf':>5} {'theta':>5} {'ms':>8} {'aggL2':>10} {'nbr/leaf':>8} {'max':>5} "
          f"{'direct/tgt':>10} {'share':>6} {'p90 share':>9} {'flags'}")
    by_leaf: dict[int, list] = {}
    for r in rows:
        b = r["budget"]
        by_leaf.setdefault(r["leaf"], []).append(r)
        print(f"{r['leaf']:>5} {r['theta']:>5.2f} {r['force_eval_s']['min']*1e3:>8.2f} "
              f"{r['errors']['aggL2']:>10.3e} {b['mean_neighbor_leaves_per_leaf']:>8.1f} "
              f"{b['max_neighbor_leaves_per_leaf']:>5} {b['direct_sources_per_target_mean']:>10.0f} "
              f"{b['direct_share_of_N']:>6.3f} {b['direct_sources_per_target_p90']/n:>9.3f} "
              f"{','.join(r['contention']['flags']) or '-'}")
    print("\njaccpot error growth per +0.1 theta (log-linear fit over the sweep):")
    for leaf, rs in sorted(by_leaf.items()):
        rs = sorted(rs, key=lambda r: r["theta"])
        th = [r["theta"] for r in rs]
        er = [r["errors"]["aggL2"] for r in rs]
        print(f"  leaf {leaf:>4}: x{slope_per_0p1(th, er):.2f}  over theta {th[0]:.1f}-{th[-1]:.1f}")

    # ---- pkdgrav3 ----
    C = json.load(open(args.compare))
    pkd = [r for r in C.get("pkdgrav3", []) if r.get("ndev") == 1]
    if not pkd:
        print("\nno single-GPU pkdgrav3 rows in", args.compare)
        return
    row = pkd[0]
    nb = row["summary"].get("n_bucket", 16)
    reports = row.get("run", {}).get("gravity_reports") or []
    if args.pkd_log:
        import sys
        sys.path.insert(0, str(ROOT))
        from common.pkdgrav3 import parse_gravity_lines
        reports = parse_gravity_lines(Path(args.pkd_log).read_text())
    results = row["summary"]["results"]
    per_theta = row["summary"].get("warmup", 1) + results[0]["repeats"]
    print(f"\npkdgrav3  N={n}  nBucket={nb}  cores={row.get('cores')}  ({len(reports)} gravity reports)")
    print(f"{'theta':>5} {'theta/1.5':>9} {'grav ms':>8} {'aggL2':>10} {'P-P/act':>8} {'P-C/act':>8} "
          f"{'share':>6} {'share+own':>9}")
    pk_th, pk_er = [], []
    for i, res in enumerate(results):
        rep = None
        blk = reports[i * per_theta:(i + 1) * per_theta]
        blk = [b for b in blk if "pp_per_active" in b]
        if blk:
            rep = blk[-1]
        pp = rep["pp_per_active"]["avg"] if rep else float("nan")
        pc = rep["pc_per_active"]["avg"] if rep else float("nan")
        share = pp / n
        print(f"{res['theta']:>5.2f} {res['theta']/PKD_THETA_SCALE:>9.3f} "
              f"{res['gravity_s']['min']*1e3:>8.2f} {res['errors']['aggL2']:>10.3e} "
              f"{pp:>8.1f} {pc:>8.1f} {share:>6.4f} {(pp+nb-1)/n:>9.4f}")
        pk_th.append(res["theta"]); pk_er.append(res["errors"]["aggL2"])
    sel = [(t, e) for t, e in zip(pk_th, pk_er) if t >= 0.6]
    if len(sel) >= 2:
        t, e = zip(*sel)
        nominal = slope_per_0p1(t, e)
        rescaled = slope_per_0p1([x / PKD_THETA_SCALE for x in t], e)
        print(f"\npkdgrav3 error growth per +0.1 of ITS theta (theta>=0.6): x{nominal:.2f}")
        print(f"pkdgrav3 error growth per +0.1 of jaccpot-equivalent theta:  x{rescaled:.2f}"
              f"   (= x{nominal:.2f}^{PKD_THETA_SCALE})")
    all_ = slope_per_0p1(pk_th, pk_er)
    print(f"(pkdgrav3 over the whole sweep incl. the fp32-floor points: x{all_:.2f} nominal)")


if __name__ == "__main__":
    main()
