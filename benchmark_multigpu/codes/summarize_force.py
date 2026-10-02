#!/usr/bin/env python
"""Turn a ``compare_force.py`` artifact into the comparison that is actually valid.

Costs are matched **stage for stage**, because the two jaccpot lanes draw the
line in different places:

    partition_for_devices   <->  pkdgrav3 domain_decompose   (excluded both sides)
    prepare_state (fused)   <->  pkdgrav3 build_tree
    eval_fn                 <->  pkdgrav3 gravity

The single-GPU jaccpot lane (``strict_fused_prepared_eval_fn``) is **eval only**
with the tree already built, so it is compared against pkdgrav3's ``gravity``
alone.  The distributed lane rebuilds the tree inside every call, so it is
compared against pkdgrav3's ``build_tree + gravity``.  Domain decomposition is
excluded on both sides: it is the partitioning step whose jaccpot analogue
(``partition_for_devices``) the driver runs once outside its timing loop, and in
production pkdgrav3 amortises it too (``dFracNoDomainDecomp``).

Reports, per device count:
  * every measured (accuracy, cost) point for both codes;
  * each code's **Pareto front** -- the points not beaten on both axes at once;
  * an **accuracy-matched** cost ratio: for each pkdgrav3 theta, the cheapest
    jaccpot configuration that is at least as accurate, and vice versa.

The accuracy-matched ratio is the only headline number worth quoting. Comparing
default-vs-default would compare two arbitrary points on two different curves,
and pkdgrav3 has no expansion-order knob to match jaccpot's, so the codes cannot
be aligned by configuration -- only by delivered accuracy.

No GPU or JAX needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pkd_points(payload, ndev):
    out = []
    for row in payload.get("pkdgrav3", []):
        if row["ndev"] != ndev:
            continue
        for res in row["summary"]["results"]:
            if "errors" not in res:
                continue
            # PRIMARY cost = build_tree + gravity, NOT force_eval.
            # force_eval also contains domain_decompose, which is pkdgrav3's
            # spatial partitioning across ranks/devices -- the analogue of
            # jaccpot's partition_for_devices, which compare_force.py runs ONCE
            # outside its timing loop. Counting it per call on one side only
            # would penalise pkdgrav3 for work the other side excluded, and in
            # production it is amortised anyway (dFracNoDomainDecomp skips it on
            # most steps). Both numbers are carried so the choice stays visible.
            out.append(
                dict(
                    label=f"theta={res['theta']:g}",
                    theta=res["theta"],
                    err=res["errors"]["aggL2"],
                    ms=(res["tree_s"]["min"] + res["gravity_s"]["min"]) * 1e3,
                    ms_eval_only=res["gravity_s"]["min"] * 1e3,
                    ms_tree_eval=(res["tree_s"]["min"] + res["gravity_s"]["min"]) * 1e3,
                    grav_ms=res["gravity_s"]["min"] * 1e3,
                    decomp_ms=res["domain_s"]["min"] * 1e3,
                    with_decomp_ms=res["force_eval_s"]["min"] * 1e3,
                    signflip=res["errors"]["aggL2_signflip"],
                )
            )
    return sorted(out, key=lambda r: r["err"])


def jac_points(payload, ndev):
    out = []
    for row in payload.get("jaccpot", []):
        if row["ndev"] != ndev or "errors" not in row:
            continue
        out.append(
            dict(
                label=f"p={row['order']} theta={row['theta']:g}"
                      + (" [OVERFLOW]" if row.get("overflow") else ""),
                overflow=row.get("overflow") or [],
                order=row["order"],
                theta=row["theta"],
                err=row["errors"]["aggL2"],
                ms=row["force_eval_s"]["min"] * 1e3,
                stage=("eval_only" if row.get("lane") == "single_gpu_fused"
                       else "tree_eval"),
                fused=(row.get("fastlane") or {}).get("fused_mode_active"),
                decomp_ms=0.0,
                with_decomp_ms=row["force_eval_s"]["min"] * 1e3,
                signflip=row["errors"]["aggL2_signflip"],
                lane=row.get("lane", "distributed"),
                fastlane=row.get("fastlane"),
            )
        )
    return sorted(out, key=lambda r: r["err"])


def pareto(points):
    """Points not dominated on (error, time) simultaneously."""
    front = []
    for p in sorted(points, key=lambda r: r["ms"]):
        if all(not (q["ms"] <= p["ms"] and q["err"] <= p["err"] and q is not p) for q in front):
            front.append(p)
    return sorted(front, key=lambda r: r["err"])


def cheapest_at_least_as_accurate(points, target_err):
    ok = [p for p in points if p["err"] <= target_err]
    return min(ok, key=lambda r: r["ms"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    args = ap.parse_args()
    payload = json.loads(Path(args.artifact).read_text())

    ic = payload["ic"]
    print(f"IC {ic['name']}  N={ic['n']}  eps={ic['softening']:g} "
          f"(moves reference by {ic.get('softening_reference_shift_aggL2', float('nan')):.2e})")
    print(f"devices: {payload['devices']}")

    ndevs = sorted({r["ndev"] for r in payload.get("pkdgrav3", [])}
                   | {r["ndev"] for r in payload.get("jaccpot", [])})

    for ndev in ndevs:
        pk = pkd_points(payload, ndev)
        jc = jac_points(payload, ndev)
        # align pkdgrav3's cost column with whatever stage jaccpot measured here
        stage = jc[0]["stage"] if jc else "tree_eval"
        for p_ in pk:
            p_["ms"] = p_["ms_eval_only"] if stage == "eval_only" else p_["ms_tree_eval"]
        print(f"  [stage matched: jaccpot {stage} vs pkdgrav3 "
              f"{'gravity' if stage == 'eval_only' else 'build_tree+gravity'}]")
        print(f"\n{'='*72}\n{ndev} GPU(s)\n{'='*72}")

        for name, pts in (("pkdgrav3", pk), ("jaccpot", jc)):
            if not pts:
                print(f"\n  {name}: no points")
                continue
            bad = [p for p in pts if p["signflip"] < p["err"]]
            if bad:
                print(f"  !! {name}: sign convention looks inverted -- comparison is "
                      f"wired wrong, do not report these numbers")
            front = {id(p) for p in pareto(pts)}
            print(f"\n  {name}   (* = on the Pareto front)")
            print(f"    {'config':<26} {'aggL2 err':>11} {'matched stage':>15}")
            for p in pts:
                star = "*" if id(p) in front else " "
                flag = ""
                if p.get("fused") is False:
                    flag = "  !! NOT FUSED"
                print(f"  {star} {p['label']:<26} {p['err']:>11.3e} "
                      f"{p['ms']:>12.2f} ms{flag}")

        if pk and jc:
            print("\n  accuracy-matched (cheapest config at least as accurate):")
            print("    reference points are pkdgrav3's PARETO FRONT only -- matching "
                  "against a\n    dominated point (beaten on both axes) invents a "
                  "win that is not there.")
            print(f"    {'reference point':<28} {'matched config':<24} {'ratio':>18}")
            for p in pareto(pk):
                m = cheapest_at_least_as_accurate(pareto(jc), p["err"])
                if m is None:
                    print(f"    pkdgrav3 {p['label']:<19} {'-- jaccpot cannot reach this accuracy':<24}")
                    continue
                ratio = m["ms"] / p["ms"]
                verdict = f"jaccpot {ratio:.2f}x " + ("slower" if ratio > 1 else "FASTER")
                print(f"    pkdgrav3 {p['label']:<19} {m['label']:<24} {verdict:>18}")

    # multi-GPU scaling. Only pkdgrav3 can be scaled honestly here: it runs the
    # same code on 1 and 2 GPUs, so gravity-only is comparable across both. The
    # jaccpot columns are DIFFERENT LANES (fused eval-only on one device, the
    # shard_map pipeline with an in-call tree rebuild on two), so dividing one by
    # the other would not be a speedup -- it is reported per lane instead.
    if len(ndevs) > 1:
        print(f"\n{'='*72}\nscaling {ndevs[0]} -> {ndevs[-1]} GPU(s)\n{'='*72}")
        a, b = pkd_points(payload, ndevs[0]), pkd_points(payload, ndevs[-1])
        if a and b:
            by_theta_a = {p["theta"]: p for p in a}
            print("  pkdgrav3 (gravity only, same code both sides):")
            for pb in sorted(b, key=lambda r: -r["theta"]):
                pa = by_theta_a.get(pb["theta"])
                if pa is None:
                    continue
                print(f"    theta={pb['theta']:.1f}: {pa['ms_eval_only']:7.2f} -> "
                      f"{pb['ms_eval_only']:7.2f} ms   "
                      f"speedup {pa['ms_eval_only']/pb['ms_eval_only']:.2f}x")
        print("  jaccpot: 1-GPU fused eval-only and 2-GPU distributed are different\n"
              "           lanes measuring different stages -- no scaling ratio is "
              "meaningful.")


if __name__ == "__main__":
    main()
