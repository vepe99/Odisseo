#!/usr/bin/env python
"""Phase 1 -- distributed FMM accuracy vs. a direct-sum ground truth.

For each initial condition we sweep the multipole order ``p`` and the opening
angle ``theta`` (used for BOTH the local self MAC and the cross-domain far MAC --
the standard single-knob FMM accuracy control) and record the per-particle
relative acceleration error against the exact chunked direct sum.  We also record
the error at each device count so a "multi-GPU adds no error beyond LET
truncation" plot can be made.

Direct sum (float64) is the ONLY valid accuracy reference: Bonsai is itself an
approximate treecode, so it is a *performance* competitor, not an accuracy
reference (see performance.py).  Bonsai-vs-direct on the same IC, if Bonsai is
made to dump forces, would be overlaid at plot time.

Run (3 GPUs):
    CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o -q) JAX_ENABLE_X64=1 \
      /export/home/tbuck/micromamba/envs/odisseo/bin/python \
      benchmark_multigpu/fmm/accuracy.py --n 20000 --ics uniform plummer
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.env import capture_provenance
from common.ic import IC_GENERATORS
from common.reference import direct_accelerations


def _make_ic(name, n, ndev, seed):
    if name == "uniform":
        return IC_GENERATORS["uniform"](n, seed=seed)
    if name == "plummer":
        return IC_GENERATORS["plummer"](n, seed=seed)
    if name == "clusters":
        return IC_GENERATORS["clusters"](ndev, n // ndev, seed=seed)
    if name == "disk":
        return IC_GENERATORS["disk"](subsample=n, seed=seed)
    raise ValueError(name)


def _rel_errors(a_fmm, a_ref):
    num = np.linalg.norm(a_fmm - a_ref, axis=1)
    den = np.linalg.norm(a_ref, axis=1) + 1e-30
    per = num / den
    aggl2 = float(np.linalg.norm(a_fmm - a_ref) / (np.linalg.norm(a_ref) + 1e-30))
    return {
        "median": float(np.median(per)),
        "p90": float(np.percentile(per, 90)),
        "max": float(np.max(per)),
        "aggL2": aggl2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ndevs", type=int, nargs="+", default=None,
                    help="device counts to test; default = [all visible]")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.7])
    ap.add_argument("--ics", nargs="+", default=["uniform", "plummer"])
    ap.add_argument("--softening", type=float, default=0.02)
    ap.add_argument("--leaf-size", type=int, default=8)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true",
                    help="auto-grow traversal caps until no overflow (slower)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # jax imports deferred until after arg parse (so --help is instant)
    import jax
    from jaccpot.distributed import DistributedFMMConfig, distributed_fmm_accelerations
    from yggdrax.distributed import device_count, make_mesh

    n_visible = device_count()
    ndevs = args.ndevs or [n_visible]
    print(f"visible devices: {n_visible}; testing ndevs={ndevs}")

    results = []
    for ic_name in args.ics:
        for ndev in ndevs:
            if ndev > n_visible:
                print(f"skip ndev={ndev} (only {n_visible} visible)")
                continue
            mesh = make_mesh(ndev)
            pos, mass = _make_ic(ic_name, args.n, ndev, args.seed)
            print(f"\n=== IC={ic_name} N={pos.shape[0]} ndev={ndev} ===")
            print("computing direct reference (float64)...")
            a_ref = direct_accelerations(
                pos, mass, G=args.G, softening=args.softening, block_size=1024
            )

            base_caps = dict(
                max_interactions_per_node=512,
                max_neighbors_per_leaf=256,
                max_pair_queue=1 << 17,
                cross_max_interactions_per_node=512,
                cross_max_neighbors_per_leaf=256,
                cross_max_pair_queue=1 << 17,
            )
            for order in args.orders:
                for theta in args.thetas:
                    cfg = DistributedFMMConfig(
                        order=order,
                        theta=theta,
                        theta_cross=theta,
                        leaf_size=args.leaf_size,
                        softening=args.softening,
                        G=args.G,
                        **base_caps,
                    )
                    if args.calibrate:
                        from common.capacity import calibrate_caps
                        cfg, _ = calibrate_caps(
                            pos, mass, mesh=mesh, ndev=ndev, base_config=cfg,
                            verbose=False,
                        )
                    res = distributed_fmm_accelerations(
                        pos, mass, config=cfg, mesh=mesh, ndev=ndev, jit=False
                    )
                    errs = _rel_errors(res.accelerations, a_ref)
                    row = dict(
                        ic=ic_name, n=int(pos.shape[0]), ndev=ndev,
                        order=order, theta=theta, overflow=res.overflow, **errs,
                    )
                    results.append(row)
                    print(
                        f"  p={order} theta={theta:.2f}: median={errs['median']:.2e} "
                        f"p90={errs['p90']:.2e} aggL2={errs['aggL2']:.2e} "
                        f"overflow={res.overflow}"
                    )

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "accuracy.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "provenance": capture_provenance({"benchmark": "fmm_accuracy", "args": vars(args)}),
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    np.savez(out.replace(".json", ".npz"), results=np.array(results, dtype=object),
             args=json.dumps(vars(args)))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
