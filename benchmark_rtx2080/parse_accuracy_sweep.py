#!/usr/bin/env python3
"""Build the FMM accuracy<->cost table from the accuracy_sweep reports, and the
accuracy-matched comparison vs Bonsai.

For each reports_<tag> dir (tag = p<order>_th<theta>_<basis>):
  - force error: galaxy_disk_initial_acceleration_*.json -> fmm_vs_direct_rel_err
  - per-step time: galaxy_disk_profile_*.json -> perf_measured_median_step_seconds
Bonsai (BH monopole+quadrupole ~ order 2, theta=0.5) per-step time is parsed
from bonsai_reference/bonsai_reference.log ("Loop alone took" / last iter).
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "accuracy_sweep")


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def parse_tag(tag):
    m = re.match(r"p(\d+)_th([0-9.]+)_(\w+)", tag)
    return (int(m.group(1)), float(m.group(2)), m.group(3)) if m else (None, None, tag)


def fmm_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(SWEEP, "reports_p*"))):
        tag = os.path.basename(d).replace("reports_", "")
        order, theta, basis = parse_tag(tag)
        acc_f = _latest(os.path.join(d, "galaxy_disk_initial_acceleration_*.json"))
        prof_f = _latest(os.path.join(d, "*profile*.json"))
        p50 = p90 = step_ms = None
        if acc_f:
            a = json.load(open(acc_f))
            re_ = a.get("fmm_vs_direct_rel_err", {})
            p50, p90 = re_.get("p50"), re_.get("p90")
        if prof_f:
            pr = json.load(open(prof_f))
            s = pr.get("perf_measured_median_step_seconds")
            if s is None:
                med = pr.get("perf_measured_median_seconds")
                n = pr.get("num_steps")
                s = (med / n) if (med and n) else None
            step_ms = (s * 1e3) if s else None
        rows.append(dict(order=order, theta=theta, basis=basis,
                         p50=p50, p90=p90, step_ms=step_ms))
    rows.sort(key=lambda r: (r["theta"] or 0, r["order"] or 0, r["basis"]))
    return rows


def bonsai_step_ms():
    log = os.path.join(HERE, "bonsai_reference", "bonsai_reference.log")
    if not os.path.exists(log):
        return None, None
    txt = open(log, errors="ignore").read()
    m = re.search(r"Loop alone took:\s*([0-9.]+)", txt)
    loop = float(m.group(1)) if m else None
    iters = re.findall(r"iter=(\d+)\s*:", txt)
    n = int(iters[-1]) if iters else None
    return (loop, (loop / n * 1e3) if (loop and n) else None)


def main():
    rows = fmm_rows()
    bloop, bstep = bonsai_step_ms()
    print(f"\nBonsai (BH monopole+quadrupole, theta=0.5) on RTX 2080 Ti: "
          f"{bstep:.2f} ms/step  (loop {bloop:.1f}s / 4000 steps)\n" if bstep
          else "\nBonsai timing not found\n")
    print(f"{'basis':8} {'order':>5} {'theta':>5} {'relerr_p50':>11} {'relerr_p90':>11} "
          f"{'ms/step':>9} {'vs Bonsai':>10}")
    print("-" * 70)
    for r in rows:
        ratio = (r["step_ms"] / bstep) if (r["step_ms"] and bstep) else None
        print(f"{r['basis']:8} {str(r['order']):>5} {str(r['theta']):>5} "
              f"{(f'{r['p50']*100:.3f}%' if r['p50'] is not None else '-'):>11} "
              f"{(f'{r['p90']*100:.3f}%' if r['p90'] is not None else '-'):>11} "
              f"{(f'{r['step_ms']:.1f}' if r['step_ms'] else '-'):>9} "
              f"{(f'{ratio:.1f}x' if ratio else '-'):>10}")
    out = os.path.join(HERE, "accuracy_sweep_summary.json")
    json.dump(dict(bonsai_step_ms=bstep, bonsai_loop_s=bloop, fmm=rows), open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
