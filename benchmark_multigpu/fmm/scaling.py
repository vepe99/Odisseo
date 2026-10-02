#!/usr/bin/env python
"""Phase 3 -- distributed FMM strong & weak scaling.

  strong: fixed total N, GPUs in {1,2,3}  -> speedup = t(1)/t(p), eff = speedup/p
  weak:   fixed N per GPU, GPUs in {1,2,3} -> ideal = flat t,      eff = t(1)/t(p)

Only 3 GPUs are free on this box, so curves run 1-2-3 (no clean power-of-two, but
efficiency curves are meaningful).  Records comms/halo diagnostics per point so a
communication-volume-vs-GPUs panel can be made.

Run:
    CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o -q) JAX_ENABLE_X64=1 \
      /export/home/tbuck/micromamba/envs/odisseo/bin/python \
      benchmark_multigpu/fmm/scaling.py --mode both --strong-n 300000 --weak-per-gpu 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.env import capture_provenance
from common.timing import timed_call


def _time_point(n, ndev, args):
    """Time one (N, ndev) force evaluation; return (timing_dict, diag_summary)."""
    import jax.numpy as jnp

    from jaccpot.distributed import (
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
    )
    from jaccpot.distributed.fmm import DIAG_FIELDS
    from yggdrax.dtypes import INDEX_DTYPE
    from yggdrax.distributed import make_mesh

    from common.ic import IC_GENERATORS

    mesh = make_mesh(ndev)
    if args.ic == "clusters":
        pos, mass = IC_GENERATORS["clusters"](ndev, n // ndev, seed=0)
    elif args.ic == "disk":
        pos, mass = IC_GENERATORS["disk"](subsample=n, seed=0)
    else:
        pos, mass = IC_GENERATORS[args.ic](n, seed=0)

    cfg = DistributedFMMConfig(
        order=args.order, theta=args.theta, theta_cross=args.theta,
        leaf_size=args.leaf_size, softening=args.softening,
        max_interactions_per_node=512, max_neighbors_per_leaf=256,
        max_pair_queue=1 << 18,
        cross_max_interactions_per_node=512, cross_max_neighbors_per_leaf=256,
        cross_max_pair_queue=1 << 18,
    )
    part = partition_for_devices(pos, mass, ndev, leaf_size=cfg.leaf_size)
    cap = part["cap"]
    dev_args = (
        jnp.asarray(part["pos_flat"]), jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]), jnp.asarray(part["counts"], INDEX_DTYPE),
    )
    evaluate = make_force_evaluator(cfg, ndev, cap, mesh, jit=True)
    t = timed_call(evaluate, *dev_args, repeats=args.repeats, warmup=args.warmup)
    _, _, diag = evaluate(*dev_args)
    diag = np.asarray(diag)
    dsum = {k: float(np.max(diag[:, i])) for i, k in enumerate(DIAG_FIELDS)}
    return t, dsum, int(pos.shape[0]), cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strong", "weak", "both"], default="both")
    ap.add_argument("--strong-n", type=int, default=300000)
    ap.add_argument("--weak-per-gpu", type=int, default=100000)
    ap.add_argument("--ndevs", type=int, nargs="+", default=None)
    ap.add_argument("--ic", default="uniform")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--leaf-size", type=int, default=16)
    ap.add_argument("--softening", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax  # noqa: F401

    from yggdrax.distributed import device_count

    n_visible = device_count()
    ndevs = args.ndevs or list(range(1, n_visible + 1))
    print(f"visible devices: {n_visible}; scaling over ndevs={ndevs}")

    strong, weak = [], []
    if args.mode in ("strong", "both"):
        print(f"\n=== STRONG scaling: fixed N={args.strong_n} ===")
        for ndev in ndevs:
            if ndev > n_visible:
                continue
            t, dsum, n_actual, cap = _time_point(args.strong_n, ndev, args)
            strong.append(dict(ndev=ndev, n=n_actual, cap=cap, **t.as_dict(), diag=dsum))
            print(f"  ndev={ndev}: min={t.min_ms:.2f}ms")
        if strong:
            t1 = strong[0]["min_ms"]
            for r in strong:
                r["speedup"] = t1 / r["min_ms"]
                r["efficiency"] = r["speedup"] / r["ndev"]

    if args.mode in ("weak", "both"):
        print(f"\n=== WEAK scaling: {args.weak_per_gpu} particles / GPU ===")
        for ndev in ndevs:
            if ndev > n_visible:
                continue
            n = args.weak_per_gpu * ndev
            t, dsum, n_actual, cap = _time_point(n, ndev, args)
            weak.append(dict(ndev=ndev, n=n_actual, cap=cap, **t.as_dict(), diag=dsum))
            print(f"  ndev={ndev} N={n_actual}: min={t.min_ms:.2f}ms")
        if weak:
            t1 = weak[0]["min_ms"]
            for r in weak:
                r["efficiency"] = t1 / r["min_ms"]  # ideal = 1 (flat)

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "scaling.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "provenance": capture_provenance({"benchmark": "fmm_scaling", "args": vars(args)}),
        "strong": strong,
        "weak": weak,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
