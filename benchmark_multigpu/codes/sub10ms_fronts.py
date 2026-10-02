#!/usr/bin/env python
"""Phase 0.3 / G0 of the sub-10 ms plan: both codes' accuracy-time fronts on one plot.

Reads the jz-fmm rows (``artifacts/jzfmm/jzfmm_front_<tag>.json``) and the
jaccpot step rows (``artifacts/sub10ms/jaccpot_*.json`` from ``smallleaf_baseline``,
``eval_only`` + ``scan_full``), writes ``artifacts/smallleaf/fronts_2026-09.json``
and a figure, and prints the G0 table: for each N the Pareto front of each code
(time vs aggL2), the matched-error ratio, and jz-fmm's accuracy at its defaults.

Times compared: jz-fmm ``force()`` (tree build + lists + force, one call) against
jaccpot ``strict_run_v2`` per step (tree + lists + force + KDK integrator) -- the
plan's like-for-like. jaccpot ``eval_only`` (fixed lists) is listed separately.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent


def _pareto(rows, tkey, ekey):
    rows = sorted(rows, key=lambda r: (r[ekey], r[tkey]))
    out, best = [], float("inf")
    for r in rows:
        if r[tkey] < best:
            out.append(r)
            best = r[tkey]
    return out


def _time_at_error(front, err, tkey="ms", ekey="aggL2"):
    """Time of the cheapest point on ``front`` with error <= ``err`` (None if none)."""
    ok = [r for r in front if r[ekey] <= err]
    return min((r[tkey] for r in ok), default=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jz-glob", default=str(ROOT / "artifacts" / "jzfmm" / "jzfmm_front_plummer*.json"))
    ap.add_argument("--jac-glob", default=str(ROOT / "artifacts" / "sub10ms" / "jaccpot_*.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "smallleaf" / "fronts_2026-09.json"))
    ap.add_argument("--fig", default=str(ROOT / "artifacts" / "smallleaf" / "fronts_2026-09.png"))
    ap.add_argument("--match-error", type=float, nargs="+", default=[1.2e-3, 4.6e-4])
    args = ap.parse_args()

    jz_rows, jac_rows = [], []
    for f in sorted(glob.glob(args.jz_glob)):
        d = json.load(open(f))
        if "smoke" in d.get("tag", ""):
            continue
        for r in d["rows"]:
            if "timing" not in r:
                continue
            jz_rows.append(dict(code="jzfmm", n=d["n"], leaf=r["leaf"], p=r["p"], theta=r["theta"],
                                ms=r["timing"]["min"] * 1e3, aggL2=r["error"]["aggL2"], p90=r["error"]["p90"],
                                direct_share=r.get("lists", {}).get("direct_share_of_N"),
                                launches=r.get("trace", {}).get("per_call", {}).get("launches"),
                                flags=r.get("contention", {}).get("flags")))
    for f in sorted(glob.glob(args.jac_glob)):
        d = json.load(open(f))
        if "scan_full" not in d or "eval_only" not in d:
            continue
        e = d["eval_only"]
        mode = d.get("env_overrides", {}).get("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "aabb")
        if f.endswith("_bound.json"):
            mode += "_bound"  # child-sphere internal radii (superseded by exact)
        if (d.get("tree") or {}).get("leaf_partition") == "cells":
            mode += "_cells"
        tr = d["scan_full"].get("trace", {})
        jac_rows.append(dict(code=f"jaccpot_{mode}", n=d["n"], leaf=d["leaf"], p=d["order"], theta=d["theta"],
                             ms=d["scan_full"]["ms_per_step_min"], eval_ms=e["timing"]["min"] * 1e3,
                             aggL2=e["aggL2"], direct_share=e["lists"].get("direct_share_of_N"),
                             far_pairs=e["lists"].get("far_pairs"),
                             launches=tr.get("per_step", {}).get("launches"),
                             stages={k: round(v["ms_per_step"], 2) for k, v in tr.get("stages", {}).items()},
                             flags=d["scan_full"].get("contention", {}).get("flags")))

    result = dict(jzfmm=jz_rows, jaccpot=jac_rows, fronts={}, g0={})
    ns = sorted({r["n"] for r in jz_rows} | {r["n"] for r in jac_rows})
    codes = sorted({r["code"] for r in jz_rows + jac_rows})
    for n in ns:
        result["fronts"][str(n)] = {}
        for code in codes:
            rows = [r for r in jz_rows + jac_rows if r["n"] == n and r["code"] == code]
            if rows:
                result["fronts"][str(n)][code] = _pareto(rows, "ms", "aggL2")
        g0 = {}
        for err in args.match_error:
            t = {code: _time_at_error(result["fronts"][str(n)].get(code, []), err) for code in codes}
            g0[f"time_ms_at_aggL2<={err:g}"] = t
            if t.get("jzfmm") and t.get("jaccpot_aabb"):
                g0[f"ratio_jaccpot_aabb_over_jzfmm_at_{err:g}"] = t["jaccpot_aabb"] / t["jzfmm"]
            if t.get("jzfmm") and t.get("jaccpot_com"):
                g0[f"ratio_jaccpot_com_over_jzfmm_at_{err:g}"] = t["jaccpot_com"] / t["jzfmm"]
        default = [r for r in jz_rows if r["n"] == n and r["p"] == 5 and abs(r["theta"] - 0.8) < 1e-9]
        g0["jzfmm_default_p5_theta0.8"] = [dict(leaf=r["leaf"], ms=r["ms"], aggL2=r["aggL2"], p90=r["p90"]) for r in default]
        result["g0"][str(n)] = g0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    # ---- table
    for n in ns:
        print(f"\n=== N = {n} ===")
        for code, front in result["fronts"][str(n)].items():
            print(f"  {code} Pareto front (time vs aggL2):")
            for r in front:
                extra = f" direct {r['direct_share']:.4f}N" if r.get("direct_share") is not None else ""
                print(f"    leaf{r['leaf']:>3} p{r['p']} th{r['theta']:<4g} {r['ms']:8.2f} ms  aggL2 {r['aggL2']:.3e}{extra}"
                      f"{'  launches ' + str(int(r['launches'])) if r.get('launches') else ''}")
        for k, v in result["g0"][str(n)].items():
            print(f"  G0 {k}: {v}")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(ns), figsize=(6.5 * len(ns), 5), squeeze=False)
        markers = {32: "s", 64: "o", 128: "^", 256: "D"}
        colors = {"jzfmm": "tab:blue", "jaccpot_aabb": "tab:red", "jaccpot_com": "tab:green", "jaccpot_com_bound": "tab:olive", "jaccpot_com_cells": "tab:purple"}
        for ax, n in zip(axes[0], ns):
            for code in codes:
                rows = [r for r in jz_rows + jac_rows if r["n"] == n and r["code"] == code]
                for r in rows:
                    ax.scatter(r["aggL2"], r["ms"], marker=markers.get(r["leaf"], "x"), color=colors.get(code, "k"),
                               alpha=0.45, s=28)
                front = result["fronts"][str(n)].get(code, [])
                if front:
                    ax.plot([r["aggL2"] for r in front], [r["ms"] for r in front], "-", color=colors.get(code, "k"),
                            label=f"{code} front")
            for err in args.match_error:
                ax.axvline(err, color="grey", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("aggregate relative L2 force error (4096 targets, fp64 reference)")
            ax.set_ylabel("ms per force (tree build included)")
            ax.set_title(f"Plummer N = {n:,}, one A100 (idle card)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.text(0.01, 0.01, "markers: leaf size (square 32, circle 64); jz-fmm force() vs jaccpot strict_run_v2 step",
                 fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(args.fig, dpi=130)
        print(f"\nwrote {args.out} and {args.fig}")
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
