#!/usr/bin/env python
"""jz-fmm's Fig. 9 (arXiv:2609.09307) redrawn on our box, with our two codes.

Their axes and their metric: evaluation time against the **90th percentile of the
per-particle relative force error**, both logarithmic, one curve per expansion
order, each point labelled with its opening angle. Their figure is 4x10^7
Hernquist on four A100s against gadget4 and pkdgrav3; this is the same axes and
the same metric at our operating point -- 2x10^5 on ONE A100, jaccpot against
jz-fmm -- so it is comparable in shape and slope, never point-for-point with
theirs.

What each time means (both codes, same protocol as the rest of the record):
jz-fmm's is one ``force()`` call including tree build; jaccpot's is one fused
``strict_run_v2`` step, which also carries a KDK integrator (~0.6 ms of it). The
error is identical for both, measured on the same 4096 targets (seed 12345)
against an fp64 direct sum. Softening is 1e-7 for both codes and both
references, NOT the zero their paper states: jz-fmm returns NaN forces at exactly
eps=0 (its Plummer-kernel self pair is 0/0, and removing the self interaction
masks after the divide). At 1e-7 the softening is orders of magnitude below any
pair separation here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent


def _jaccpot_rows(pattern: str) -> list[dict]:
    """Rows from ``smallleaf_baseline`` JSON: step time and the p90 error."""
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sf = d.get("scan_full") or {}
        eo = d.get("eval_only") or {}
        err = eo.get("error") or {}
        ms = sf.get("ms_per_step_min")
        if ms is None or not err.get("p90"):
            continue
        out.append(
            dict(
                code="jaccpot",
                p=int(d["order"]),
                theta=float(d["theta"]),
                ms=float(ms),
                eval_ms=float((eo.get("timing") or {}).get("min", 0.0)) * 1e3,
                p90=float(err["p90"]),
                aggL2=float(err["aggL2"]),
                load=_load(sf),
            )
        )
    return out


def _load(section: dict) -> float:
    """Host load recorded with a row (0.0 when the harness did not flag one)."""
    for flag in (section.get("contention") or {}).get("flags", []) or []:
        if flag.startswith("loadavg>=8 ("):
            return float(flag.split("(")[1].rstrip(")"))
    return 0.0


def _jzfmm_rows(pattern: str) -> list[dict]:
    """Rows from ``jzfmm_force_eval`` JSON: one force call, same error metric."""
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for r in d.get("rows", []):
            err = r.get("error") or {}
            if not err.get("p90"):
                continue
            ms = float((r.get("timing") or {}).get("min", 0.0)) * 1e3
            out.append(
                dict(
                    code="jz-fmm",
                    p=int(r["p"]),
                    theta=float(r["theta"]),
                    ms=ms,
                    eval_ms=ms,
                    p90=float(err["p90"]),
                    aggL2=float(err["aggL2"]),
                    load=float((r.get("contention") or {}).get("loadavg1_max", 0.0) or 0.0),
                )
            )
    return out


#: one colour per code, one marker/dash per expansion order -- generated rather
#: than tabulated so a run at an order this file has never seen still plots.
_COLOR = {"jaccpot": "#1f77b4", "jz-fmm": "#d62728"}
_MARKERS = ("o", "s", "^", "D", "v", "P")
_DASHES = ("-", "--", "-.", ":")


def _style(code: str, p: int, orders: list[int]) -> dict:
    """Line style for one (code, order) curve.

    Parameters
    ----------
    code : str
        Code name, which picks the colour.
    p : int
        Expansion order, which picks the marker and dash.
    orders : list[int]
        Every order in the figure, ascending; ``p``'s position in it indexes the
        marker and dash cycles.

    Returns
    -------
    dict
        Keyword arguments for ``Axes.plot``.
    """
    i = orders.index(int(p))
    return dict(
        color=_COLOR.get(code, "#555555"),
        marker=_MARKERS[i % len(_MARKERS)],
        ls=_DASHES[i % len(_DASHES)],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ic", default="hernquist")
    ap.add_argument("--jac-glob", default=None)
    ap.add_argument("--jz-glob", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()
    jac_glob = args.jac_glob or str(ROOT / "artifacts" / "sub10ms" / f"jaccpot_fig9_{args.ic}_jac_*.json")
    jz_glob = args.jz_glob or str(ROOT / "artifacts" / "jzfmm" / f"jzfmm_front_fig9_{args.ic}_jz_*.json")
    rows = _jaccpot_rows(jac_glob) + _jzfmm_rows(jz_glob)
    if not rows:
        raise SystemExit(f"no rows matched\n  {jac_glob}\n  {jz_glob}")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    orders = sorted({int(r["p"]) for r in rows})
    for code in sorted({r["code"] for r in rows}):
        for p in orders:
            pts = sorted(
                [r for r in rows if r["code"] == code and int(r["p"]) == p],
                key=lambda r: r["ms"],
            )
            if not pts:
                continue
            style = _style(code, p, orders)
            ax.plot([r["ms"] for r in pts], [r["p90"] for r in pts], label=f"{code}  p={p}",
                    markersize=6, linewidth=1.4, **style)
            for r in pts:
                ax.annotate(f"{r['theta']:g}", (r["ms"], r["p90"]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7.5, color=style["color"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time per force [ms]  (tree build included; jaccpot also one KDK step, ~0.6 ms)")
    ax.set_ylabel("relative force error, 90th percentile")
    ax.set_title(f"N = 2x10$^5$ {args.ic}, one A100  (jz-fmm Fig. 9 axes)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.015, "labels = opening angle; error vs an fp64 direct sum, softening 1e-7, "
             "4096 targets (seed 12345)", ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figpath = args.fig or str(ROOT / "artifacts" / "smallleaf" / f"fig9_{args.ic}.png")
    outpath = args.out or str(ROOT / "artifacts" / "smallleaf" / f"fig9_{args.ic}.json")
    fig.savefig(figpath, dpi=160)
    json.dump(rows, open(outpath, "w"), indent=1)
    print(f"{'code':8} {'p':>2} {'theta':>5} {'ms':>7} {'p90':>10} {'aggL2':>10}  load")
    for r in sorted(rows, key=lambda r: (r["code"], r["p"], -r["theta"])):
        print(f"{r['code']:8} {r['p']:2} {r['theta']:5} {r['ms']:7.2f} {r['p90']:10.3e} "
              f"{r['aggL2']:10.3e}  {r['load'] or '-'}")
    print(f"\nwrote {figpath} and {outpath}")


if __name__ == "__main__":
    main()
