"""Summarise the benchmark CSVs and draw the figure.

Inputs (default, those that exist in --results-dir):
  odisseo_throughput.csv  Odisseo throughput sweep   (metric_basis = interactions)
  odisseo_matched.csv     Odisseo at matched |dE/E|  (metric_basis = wallclock_to_fixed_time)
  reference.csv           reference code (PeTar)     (metric_basis = wallclock_to_fixed_time)
For each (series, N) the latest run_id with successful rows is used, unless --run-id is given.

Outputs: summary.csv (median and interquartile range over repeats for each
file/series/N) and throughput.{pdf,png}.

Error bars: min-max over repeats (IQR is written to summary.csv).
Panel (a): pairwise interactions / s vs N, log-log, only rows with
metric_basis = interactions. fp64 = filled markers, solid lines; fp32 = open
markers, dashed lines; colour + marker shape = scheme.
Panel (b), drawn if any wallclock_to_fixed_time rows exist: wall-clock to the
same physical t_end vs N, for the reference code and for Odisseo runs whose
dt was tuned to the reference |dE/E|. Odisseo points whose accuracy is NOT
within the match factor are drawn as grey crosses; each reference point is
annotated with its |dE/E|. Wall-clock rows are never converted into
interaction rates.

Numbers only: no ratios, speed-ups or interpretive text are produced.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR  # noqa: E402

# Okabe-Ito colours (colour-blind safe); identity also carried by marker shape.
STYLE = {
    "DIRECT_ACC": dict(color="#0072B2", marker="o"),
    "DIRECT_ACC_LAXMAP": dict(color="#D55E00", marker="s"),
}
REF_STYLE = dict(color="#000000", marker="D")
FILES = ("odisseo_throughput.csv", "odisseo_matched.csv", "reference.csv")
NUM = ("n_steps", "dt", "t_end", "n_force_evals", "softening", "compile_s", "wall_s",
       "interactions_per_s", "gflops", "dE_E_final", "target_dE_E", "accuracy_ratio")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_file(p: Path, run_id: str | None):
    """Rows of one CSV. Default: for every (series, N), the latest run_id that
    has at least one ``ok`` row (so a later run that adds a new N, e.g. 2^17,
    is combined with earlier runs at the other N; each point still comes from
    a single run). With ``run_id``, only that run."""
    rows = list(csv.DictReader(open(p, newline="")))
    if not rows:
        return []
    if run_id:
        rows = [r for r in rows if r["run_id"] == run_id]
    else:
        best = {}
        for r in rows:
            k = (r["series"], r["N"])
            ok = r["status"] == "ok"
            cand = (ok, r["run_id"])
            if k not in best or cand > best[k]:
                best[k] = cand
        rows = [r for r in rows if r["run_id"] == best[(r["series"], r["N"])][1]]
    for r in rows:
        for k in NUM:
            r[k] = _f(r.get(k))
        r["N"] = int(r["N"])
        r["source"] = p.name
        r["accuracy_matched"] = r.get("accuracy_matched", "")
    return rows


def _pct(arr):
    arr = np.asarray(arr, float)
    if np.all(np.isnan(arr)):
        return (np.nan,) * 3
    return tuple(np.nanpercentile(arr, [25, 50, 75]))


def summarise(rows):
    keys = ("source", "run_id", "code", "series", "precision", "scheme", "metric_basis", "N")
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, g in groups.items():
        d = dict(zip(keys, key))
        f = g[0]
        d.update(n_repeats=len(g), **{k: f[k] for k in (
            "n_steps", "dt", "t_end", "n_force_evals", "softening", "compile_s", "dE_E_final",
            "target_dE_E", "accuracy_ratio", "accuracy_matched")})
        if f["code"] != "odisseo":  # reference repeats may differ in accuracy
            d["dE_E_final"] = float(np.nanmedian([r["dE_E_final"] for r in g]))
        for name, col in (("wall_s", "wall_s"), ("ips", "interactions_per_s"), ("gflops", "gflops")):
            vals = [r[col] for r in g]
            q1, med, q3 = _pct(vals)
            lo = float(np.nanmin(vals)) if not np.all(np.isnan(vals)) else np.nan
            hi = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else np.nan
            d.update({f"{name}_min": lo, f"{name}_q25": q1, f"{name}_median": med,
                      f"{name}_q75": q3, f"{name}_max": hi})
        out.append(d)
    out.sort(key=lambda d: (d["source"], d["code"], d["precision"], d["scheme"], d["N"]))
    return out


def _by(summary, keys):
    g = defaultdict(list)
    for d in summary:
        g[tuple(d[k] for k in keys)].append(d)
    return {k: sorted(v, key=lambda d: d["N"]) for k, v in sorted(g.items())}


def _arr(g, k):
    return np.array([d[k] for d in g], float)


def _xticks(ax, Ns) -> None:
    Ns = sorted({int(n) for n in Ns})
    ax.set_xticks(Ns, [f"$2^{{{int(round(np.log2(n)))}}}$" for n in Ns])
    ax.minorticks_off()
    ax.set_xlim(Ns[0] / 1.5, Ns[-1] * 1.5)
    ax.set_xlabel("N")


def plot(summary, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inter = [d for d in summary if d["metric_basis"] == "interactions"]
    wall = [d for d in summary if d["metric_basis"] == "wallclock_to_fixed_time"]
    ncols = 2 if wall else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols + (2.6 if wall else 0), 4.2), squeeze=False)

    ax = axes[0, 0]
    for (code, scheme, prec), g in _by(inter, ("code", "scheme", "precision")).items():
        st = STYLE.get(scheme, REF_STYLE)
        fp64 = prec == "fp64"
        med = _arr(g, "ips_median")
        yerr = np.vstack([med - _arr(g, "ips_min"), _arr(g, "ips_max") - med])
        ax.errorbar(_arr(g, "N"), med, yerr=yerr, color=st["color"], marker=st["marker"], ms=6,
                    lw=1.8, ls="-" if fp64 else "--", capsize=2,
                    mfc=st["color"] if fp64 else "white", mec=st["color"],
                    label=f"{code} {scheme} {prec}")
    ax.set(xscale="log", yscale="log", xlabel="N", ylabel="pairwise interactions / s")
    if inter:
        _xticks(ax, [d["N"] for d in inter])
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"${y / 10 ** np.floor(np.log10(y)):.0f}\\times10^{{{int(np.floor(np.log10(y)))}}}$"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, color="0.88", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("(a) Odisseo throughput; median, bars = min-max of repeats", fontsize=9, loc="left")

    if wall:
        ax2 = axes[0, 1]
        for (code, scheme, prec), g in _by(wall, ("code", "scheme", "precision")).items():
            med = _arr(g, "wall_s_median")
            yerr = np.vstack([med - _arr(g, "wall_s_min"), _arr(g, "wall_s_max") - med])
            N = _arr(g, "N")
            if code == "odisseo":
                st = STYLE.get(scheme, REF_STYLE)
                fp64 = prec == "fp64"
                ok = np.array([str(d["accuracy_matched"]) == "True" for d in g])
                ax2.errorbar(N[ok], med[ok], yerr=yerr[:, ok], color=st["color"], marker=st["marker"],
                             ms=6, lw=1.8, ls="-" if fp64 else "--", capsize=2,
                             mfc=st["color"] if fp64 else "white", mec=st["color"],
                             label=f"odisseo {scheme} {prec} (dt tuned)")
                if (~ok).any():
                    ax2.plot(N[~ok], med[~ok], "x", color="0.5", ms=7,
                             label="odisseo, accuracy not matched")
            else:
                ax2.errorbar(N, med, yerr=yerr, ls="none", capsize=2, ms=7,
                             label=f"{code} ({g[0]['series']})", **REF_STYLE)
                for d in g:
                    ax2.annotate(f"|dE/E|={d['dE_E_final']:.1e}", (d["N"], d["wall_s_median"]),
                                 fontsize=7, xytext=(0, 8), textcoords="offset points", ha="center", va="bottom")
        t_end = sorted({d["t_end"] for d in wall})
        ax2.set(xscale="log", yscale="log", ylabel=f"wall-clock to t_end = {', '.join(map(str, t_end))} [s]")
        _xticks(ax2, [d["N"] for d in wall])
        _lo, _hi = ax2.get_ylim(); ax2.set_ylim(_lo, _hi * 3)
        ax2.grid(True, color="0.88", lw=0.6)
        ax2.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax2.set_title("(b) same IC, softening, t_end; Odisseo dt tuned to reference |dE/E|;\nmedian, bars = min-max of repeats",
                      fontsize=9, loc="left")

    fig.text(0.005, 0.005, "Error bars: min-max over repeats (n = 5); where not visible they are "
             "smaller than the marker. Interquartile ranges are in summary.csv.", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    print(f"[plot] {out.with_suffix('.pdf')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--csv", nargs="+", default=None, help="explicit CSV files (default: see above)")
    ap.add_argument("--run-id", default=None, help="use only this run_id (default: latest successful run per series and N)")
    a = ap.parse_args(argv)
    rdir = Path(a.results_dir)
    paths = [Path(p) for p in a.csv] if a.csv else [rdir / f for f in FILES if (rdir / f).exists()]
    if not paths:
        raise SystemExit(f"no CSV found in {rdir}")
    rows = [r for p in paths for r in read_file(p, a.run_id)]

    seen = set()
    for r in rows:
        k = (r["series"], r["N"], r["status"])
        if r["status"] not in ("ok", "ladder") and k not in seen:
            seen.add(k)
            print(f"[plot] not plotted: {r['series']} N={r['N']} status={r['status']} {r.get('error', '')}")

    summary = summarise(rows)
    cols = list(summary[0]) if summary else []
    with open(rdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    print(f"{'series':38s} {'N':>7s} {'rep':>3s} {'n_steps':>8s} {'compile_s':>9s} {'wall_med':>10s} "
          f"{'wall_q25':>10s} {'wall_q75':>10s} {'ips_med':>10s} {'GFLOP/s':>8s} {'|dE/E|':>9s}")
    for d in summary:
        print(f"{d['series']:38s} {d['N']:7d} {d['n_repeats']:3d} {d['n_steps']:8.0f} {d['compile_s']:9.3f} "
              f"{d['wall_s_median']:10.4f} {d['wall_s_q25']:10.4f} {d['wall_s_q75']:10.4f} "
              f"{d['ips_median']:10.3e} {d['gflops_median']:8.1f} {d['dE_E_final']:9.2e}")
    print(f"[plot] {rdir / 'summary.csv'}")
    plot(summary, rdir / "throughput")


if __name__ == "__main__":
    main()
