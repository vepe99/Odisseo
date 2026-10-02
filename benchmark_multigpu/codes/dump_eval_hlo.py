#!/usr/bin/env python
"""Dump the optimized HLO of one fused eval call and list its while loops.

Companion to ``profile_eval.py``: the profile showed 782 launches each of five
tiny fusions at N=200k / leaf 256 (= one per leaf).  This finds the loop that
emits them -- its trip count, its body's fusions and their shapes -- so the fix
targets the right operator.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common.gpu_guard import pick_idle_gpus, set_cuda_visible  # noqa: E402
from compare_force import apply_fast_lane_env  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=256)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1)
    set_cuda_visible(devices)
    apply_fast_lane_env(args.n)

    import jax
    from jaccpot import (FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig,
                         NearFieldConfig, TreeConfig)

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    P = jax.numpy.asarray(pos, jax.numpy.float32)
    M = jax.numpy.asarray(mass, jax.numpy.float32)
    s = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
        G=1.0, softening=1e-7, working_dtype=jax.numpy.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                                   farfield=FarFieldConfig(mode="auto"),
                                   nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=args.order)
    prepared, ev = s.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)
    compiled = ev.lower(prepared).compile()
    hlo = compiled.as_text()
    tag = f"plummer{args.n}_leaf{args.leaf}_th{args.theta:g}_p{args.order}"
    out = Path(args.out or (ROOT / "artifacts" / "traces" / f"eval_hlo_{tag}.txt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(hlo)
    print(f"wrote {out} ({len(hlo)/1e6:.1f} MB)")

    # while loops: find each `while(` instruction, its condition computation and
    # any constant trip count XLA annotated.
    for m in re.finditer(r"^\s*(\S+) = \S+ while\((.*?)\), condition=(\S+), body=(\S+)(.*)$",
                         hlo, re.MULTILINE):
        name, _args, cond, body, rest = m.groups()
        trip = re.search(r"trip_count[=:]\s*\"?(\d+)", rest)
        print(f"\nWHILE {name}: condition={cond} body={body} "
              f"trip_count={trip.group(1) if trip else '?'}")
        # body fusions
        bm = re.search(rf"^{re.escape(body)} .*?\{{(.*?)^\}}", hlo, re.MULTILINE | re.DOTALL)
        if bm:
            for line in bm.group(1).splitlines():
                if " fusion(" in line or " custom-call(" in line or "dynamic-update-slice" in line:
                    print("   ", line.strip()[:200])
        cm = re.search(rf"^{re.escape(cond)} .*?\{{(.*?)^\}}", hlo, re.MULTILINE | re.DOTALL)
        if cm:
            for line in cm.group(1).splitlines():
                if "compare" in line or "constant" in line:
                    print("   cond:", line.strip()[:160])
    n_fusions = len(re.findall(r" fusion\(", hlo))
    n_custom = re.findall(r'custom_call_target="([^"]+)"', hlo)
    print(f"\ntop-level fusion instructions: {n_fusions}; custom calls: "
          f"{sorted(set(n_custom))}")


if __name__ == "__main__":
    main()
