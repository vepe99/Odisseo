#!/usr/bin/env python
"""Render paper figures from the benchmark JSON artifacts (no GPU needed).

Thin, reproducible plotting layer: heavy compute lives in the harness scripts,
which write ``artifacts/*.json``; this reads them and emits publication figures
so plots regenerate on any machine without a GPU.

    /export/home/tbuck/micromamba/envs/odisseo/bin/python \
        benchmark_multigpu/notebooks/make_figures.py --which all
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")


def _load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        print(f"  (skip) missing artifact {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fig_accuracy():
    data = _load("accuracy.json")
    if not data:
        return
    rows = data["results"]
    ics = sorted({r["ic"] for r in rows})
    fig, axes = plt.subplots(1, len(ics), figsize=(5 * len(ics), 4), squeeze=False)
    for ax, ic in zip(axes[0], ics):
        sub = [r for r in rows if r["ic"] == ic and not r.get("error")]
        orders = sorted({r["order"] for r in sub})
        for p in orders:
            pts = sorted([r for r in sub if r["order"] == p], key=lambda r: r["theta"])
            ax.plot([r["theta"] for r in pts], [r["median"] for r in pts],
                    "o-", label=f"p={p}")
        ax.set_yscale("log")
        ax.set_xlabel(r"opening angle $\theta$")
        ax.set_ylabel("median rel. accel. error")
        ax.set_title(f"{ic}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="multipole order")
    fig.suptitle("Distributed FMM accuracy vs. direct sum")
    fig.tight_layout()
    out = os.path.join(ART, "fig_accuracy.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_performance():
    data = _load("performance.json")
    if not data:
        return
    rows = [r for r in data["results"] if not r.get("error")]
    ndevs = [r["ndev"] for r in rows]
    tmin = [r["min_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([str(d) for d in ndevs], tmin, color="#4C72B0", label="distributed FMM")
    b = data.get("bonsai_single_gpu")
    if b and np.isfinite(b.get("ms_per_step", np.nan)):
        ax.axhline(b["ms_per_step"], color="#C44E52", ls="--",
                   label=f"Bonsai 1-GPU ({b['ms_per_step']:.1f} ms)")
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("force-eval wall time [ms]")
    ax.set_title(f"FMM performance (N={rows[0]['n'] if rows else '?'})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig_performance.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_scaling():
    data = _load("scaling.json")
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    strong = data.get("strong") or []
    if strong:
        nd = [r["ndev"] for r in strong]
        sp = [r.get("speedup", np.nan) for r in strong]
        axes[0].plot(nd, sp, "o-", label="measured")
        axes[0].plot(nd, nd, "k--", alpha=0.5, label="ideal")
        axes[0].set_xlabel("# GPUs")
        axes[0].set_ylabel("speedup")
        axes[0].set_title(f"Strong scaling (N={strong[0]['n']})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    weak = data.get("weak") or []
    if weak:
        nd = [r["ndev"] for r in weak]
        eff = [r.get("efficiency", np.nan) for r in weak]
        axes[1].plot(nd, eff, "s-", color="#55A868", label="parallel efficiency")
        axes[1].axhline(1.0, color="k", ls="--", alpha=0.5, label="ideal")
        axes[1].set_xlabel("# GPUs")
        axes[1].set_ylabel("weak-scaling efficiency")
        axes[1].set_ylim(0, 1.15)
        axes[1].set_title("Weak scaling")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig_scaling.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_tree():
    data = _load("tree.json")
    if not data:
        return
    rows = [r for r in data["results"] if not r.get("error")]
    ops = sorted({r["op"] for r in rows})
    fig, axes = plt.subplots(1, len(ops), figsize=(5 * len(ops), 4), squeeze=False)
    for ax, op in zip(axes[0], ops):
        by_lib = defaultdict(list)
        for r in rows:
            if r["op"] == op:
                by_lib[r["lib"]].append(r)
        for lib, pts in by_lib.items():
            pts = sorted(pts, key=lambda r: r["n"])
            ax.plot([r["n"] for r in pts], [r["min_ms"] for r in pts], "o-", label=lib)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N particles")
        ax.set_ylabel("time [ms]")
        ax.set_title(op)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("yggdrax vs. jztree")
    fig.tight_layout()
    out = os.path.join(ART, "fig_tree.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+",
                    default=["accuracy", "performance", "scaling", "tree"],
                    choices=["accuracy", "performance", "scaling", "tree", "all"])
    args = ap.parse_args()
    which = ["accuracy", "performance", "scaling", "tree"] if "all" in args.which else args.which
    fns = {"accuracy": fig_accuracy, "performance": fig_performance,
           "scaling": fig_scaling, "tree": fig_tree}
    for w in which:
        fns[w]()


if __name__ == "__main__":
    main()
