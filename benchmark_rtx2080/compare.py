#!/usr/bin/env python3
"""Summarize the RTX 2080 Ti ODISSEO-FMM vs Bonsai benchmark.

Reads:
  - benchmark_rtx2080/reports_odisseo_perf/*.json   (ODISSEO perf timing)
  - benchmark_rtx2080/bonsai_reference/bonsai_reference.log  (Bonsai loop time)
Writes summary.json + summary.md and prints a comparison table.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_FULL = 4000  # the showcase / bonsai run length


def latest_json(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_odisseo():
    j = latest_json(os.path.join(HERE, "reports_odisseo_perf", "*.json"))
    if not j:
        return None
    with open(j) as f:
        d = json.load(f)
    step = d.get("perf_measured_median_step_seconds")
    med = d.get("perf_measured_median_seconds")
    nsteps = d.get("num_steps")
    if step is None and med and nsteps:
        step = med / nsteps
    return {
        "report": os.path.basename(j),
        "median_seconds": med,
        "num_steps": nsteps,
        "step_seconds": step,
        "measured_run_seconds": d.get("perf_measured_run_seconds"),
        "pallas": d.get("used_pallas", "off (sm_75)"),
        "full_run_estimate_seconds": (step * STEPS_FULL) if step else None,
    }


def load_bonsai():
    log = os.path.join(HERE, "bonsai_reference", "bonsai_reference.log")
    if not os.path.exists(log):
        return None
    loop = total = None
    niter = None
    with open(log, errors="ignore") as f:
        txt = f.read()
    m = re.search(r"Loop alone took:\s*([0-9.]+)", txt)
    if m:
        loop = float(m.group(1))
    m = re.search(r"Took in total:\s*([0-9.]+)\s*sec", txt)
    if m:
        total = float(m.group(1))
    iters = re.findall(r"iter=(\d+)\s*:", txt)
    if iters:
        niter = int(iters[-1])
    return {
        "loop_seconds": loop,
        "total_seconds": total,
        "num_steps": niter,
        "step_seconds": (loop / niter) if (loop and niter) else None,
    }


def main():
    od = load_odisseo()
    bo = load_bonsai()
    summary = {"n_particles": 200000, "steps_full": STEPS_FULL,
               "hardware": "RTX 2080 Ti (sm_75)", "odisseo": od, "bonsai": bo}
    if od and bo and od.get("step_seconds") and bo.get("step_seconds"):
        summary["odisseo_over_bonsai_per_step"] = od["step_seconds"] / bo["step_seconds"]

    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("# RTX 2080 Ti benchmark - ODISSEO jaccpot FMM vs Bonsai")
    lines.append("")
    lines.append("200k-particle disk, live self-gravity + static NFW halo, G=1, same IC.")
    lines.append("")
    lines.append("| Code | per-step (ms) | full 4000-step (s) |")
    lines.append("|---|---|---|")
    if od and od.get("step_seconds"):
        full = od.get("full_run_estimate_seconds")
        lines.append(f"| ODISSEO jaccpot FMM (Pallas off) | {od['step_seconds']*1e3:.1f} "
                     f"| {full:.0f} (est. = per-step x 4000) |")
    else:
        lines.append("| ODISSEO jaccpot FMM (Pallas off) | (no report yet) | - |")
    if bo and bo.get("step_seconds"):
        lines.append(f"| Bonsai (BH-tree) | {bo['step_seconds']*1e3:.1f} "
                     f"| {bo.get('loop_seconds', float('nan')):.1f} (measured loop) |")
    else:
        lines.append("| Bonsai (BH-tree) | (no log yet) | - |")
    lines.append("")
    if "odisseo_over_bonsai_per_step" in summary:
        r = summary["odisseo_over_bonsai_per_step"]
        lines.append(f"**ODISSEO is {r:.1f}x {'slower' if r>=1 else 'faster'} "
                     f"than Bonsai per step on the RTX 2080 Ti.**")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    sys.exit(main())
