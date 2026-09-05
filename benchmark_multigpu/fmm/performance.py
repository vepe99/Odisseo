#!/usr/bin/env python
"""Phase 2 -- distributed FMM performance (steady-state force-eval wall time).

Times a single jitted distributed-FMM force evaluation at fixed N across device
counts (1/2/3 GPUs) using warmup + block_until_ready + min-of-repeats -- the
natural per-force-evaluation metric, comparable to a treecode's per-step gravity
cost.  Optionally runs the single-GPU Bonsai baseline on the same particle set
for an absolute ms/step reference.

This is the FIRST performance characterization of the multi-GPU path; the
single-GPU FMM is known to be launch-latency-bound, so a comms/launch-bound
multi-GPU result is a legitimate finding, not a failure.

Run:
    CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o -q) JAX_ENABLE_X64=1 \
      /export/home/tbuck/micromamba/envs/odisseo/bin/python \
      benchmark_multigpu/fmm/performance.py --n 200000 --ndevs 1 2 3 --ic disk
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


def _make_ic(name, n, ndev, seed):
    from common.ic import IC_GENERATORS

    if name == "clusters":
        return IC_GENERATORS["clusters"](ndev, n // ndev, seed=seed)
    if name == "disk":
        return IC_GENERATORS["disk"](subsample=n, seed=seed)
    return IC_GENERATORS[name](n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--ndevs", type=int, nargs="+", default=None)
    ap.add_argument("--ic", default="disk")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--leaf-size", type=int, default=16)
    ap.add_argument("--softening", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax  # noqa: F401
    import jax.numpy as jnp

    from jaccpot.distributed import (
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
    )
    from jaccpot.distributed.fmm import DIAG_FIELDS
    from yggdrax.dtypes import INDEX_DTYPE
    from yggdrax.distributed import device_count, make_mesh

    n_visible = device_count()
    ndevs = args.ndevs or list(range(1, n_visible + 1))
    print(f"visible devices: {n_visible}; timing ndevs={ndevs}")

    results = []
    for ndev in ndevs:
        if ndev > n_visible:
            print(f"skip ndev={ndev} (only {n_visible} visible)")
            continue
        mesh = make_mesh(ndev)
        pos, mass = _make_ic(args.ic, args.n, ndev, seed=0)
        cfg = DistributedFMMConfig(
            order=args.order, theta=args.theta, theta_cross=args.theta,
            leaf_size=args.leaf_size, softening=args.softening,
            max_interactions_per_node=512, max_neighbors_per_leaf=256,
            max_pair_queue=1 << 18,
            cross_max_interactions_per_node=512, cross_max_neighbors_per_leaf=256,
            cross_max_pair_queue=1 << 18,
        )
        if args.calibrate:
            from common.capacity import calibrate_caps
            cfg, _ = calibrate_caps(pos, mass, mesh=mesh, ndev=ndev,
                                    base_config=cfg, verbose=True)

        part = partition_for_devices(pos, mass, ndev, leaf_size=cfg.leaf_size)
        cap = part["cap"]
        args_dev = (
            jnp.asarray(part["pos_flat"]), jnp.asarray(part["mass_flat"]),
            jnp.asarray(part["gid_flat"]), jnp.asarray(part["counts"], INDEX_DTYPE),
        )
        evaluate = make_force_evaluator(cfg, ndev, cap, mesh, jit=True)

        t = timed_call(evaluate, *args_dev, repeats=args.repeats, warmup=args.warmup)
        # one extra call to grab diagnostics (overflow / pair counts)
        _, _, diag = evaluate(*args_dev)
        diag = np.asarray(diag)
        overflow = bool(
            np.any(diag[:, [DIAG_FIELDS.index(k) for k in DIAG_FIELDS if "overflow" in k]] > 0)
        )
        row = dict(
            ic=args.ic, n=int(pos.shape[0]), ndev=ndev, cap=cap,
            order=args.order, theta=args.theta, leaf_size=cfg.leaf_size,
            overflow=overflow, **t.as_dict(),
        )
        results.append(row)
        print(
            f"  ndev={ndev} N={pos.shape[0]}: min={t.min_ms:.2f}ms mean={t.mean_ms:.2f}"
            f"±{t.std_ms:.2f} overflow={overflow}"
        )

    # Optional single-GPU Bonsai baseline (disk IC has a matching tipsy already)
    bonsai = None
    if args.ic == "disk":
        try:
            from common.bonsai import parse_bonsai_log
            # Reuse a previously produced bonsai log if present; running Bonsai is
            # left to run_all.sh (needs a free GPU picked outside this jax process).
            prev = os.path.join(os.path.dirname(__file__), "..", "artifacts", "bonsai.log")
            if os.path.exists(prev):
                bonsai = parse_bonsai_log(prev)
                print(f"  bonsai (from log): {bonsai.get('ms_per_step'):.2f} ms/step")
        except Exception as exc:  # pragma: no cover
            print(f"  bonsai baseline unavailable: {exc}")

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "performance.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "provenance": capture_provenance({"benchmark": "fmm_performance", "args": vars(args)}),
        "results": results,
        "bonsai_single_gpu": bonsai,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
