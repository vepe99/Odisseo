#!/usr/bin/env python
"""Phase 4 -- yggdrax tree vs. jztree (build, traversal, distributed decomposition).

Matched initial conditions and leaf sizes.  Comparable quantities:
  * single-GPU tree BUILD time vs N   (jztree ``zsort_and_tree`` vs yggdrax ``Tree.from_particles``)
  * single-GPU TRAVERSAL / neighbour find (jztree ``knn`` dual-walk vs yggdrax
    ``build_interactions_and_neighbors``) + list sizes -- note these are DIFFERENT
    queries (k-NN vs FMM interaction lists), so we report both times AND the list
    cardinalities rather than claiming a like-for-like speedup.
  * distributed SFC decomposition load balance + time (jztree ``distr_zsort`` vs
    yggdrax ``sfc_decompose``).

jztree is credited in yggdrax's distributed/ as the design it followed, so this is
the head-to-head for the yggdrax paper.

Run:
    CUDA_VISIBLE_DEVICES=$(autocvd -n 1 -l -o -q) JAX_ENABLE_X64=1 \
      /export/home/tbuck/micromamba/envs/odisseo/bin/python \
      benchmark_multigpu/tree/tree_bench.py --mode build --ns 10000 100000 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.env import capture_provenance
from common.timing import block, timed_call


# ---------------------------------------------------------------- yggdrax ----
def bench_yggdrax_build(pos, mass, leaf_size, repeats, warmup):
    import jax
    from yggdrax import Tree

    def build(x, m):
        t = Tree.from_particles(
            x, m, tree_type="radix", leaf_size=leaf_size, return_reordered=True
        )
        return t.node_ranges

    jbuild = jax.jit(build)
    import jax.numpy as jnp
    x, m = jnp.asarray(pos), jnp.asarray(mass)
    t = timed_call(jbuild, x, m, repeats=repeats, warmup=warmup)
    nodes = int(np.asarray(block(jbuild(x, m))).shape[0])
    return t, {"n_nodes": nodes}


def bench_yggdrax_traverse(pos, mass, leaf_size, theta, repeats, warmup):
    import jax
    import jax.numpy as jnp
    from yggdrax import (
        Tree,
        build_interactions_and_neighbors,
        compute_tree_geometry,
    )

    x, m = jnp.asarray(pos), jnp.asarray(mass)
    tree = Tree.from_particles(x, m, tree_type="radix", leaf_size=leaf_size,
                               return_reordered=True)
    geom = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=leaf_size)

    def walk():
        inter, nbr = build_interactions_and_neighbors(
            tree, geom, theta=theta, mac_type="bh"
        )
        return inter.counts, nbr.counts

    jwalk = jax.jit(walk)
    t = timed_call(jwalk, repeats=repeats, warmup=warmup)
    ic, nc = (np.asarray(a) for a in block(jwalk()))
    return t, {"far_pairs": int(ic.sum()), "near_pairs": int(nc.sum())}


# ----------------------------------------------------------------- jztree ----
def bench_jztree_build(pos, leaf_size, repeats, warmup):
    import jax.numpy as jnp
    import jztree

    cfg = jztree.config.TreeConfig(max_leaf_size=leaf_size)
    part = jztree.data.Pos(pos=jnp.asarray(pos))

    def build(p):
        _, th = jztree.tree.zsort_and_tree.jit(p, cfg)
        return th.ispl_l2p if hasattr(th, "ispl_l2p") else th

    t = timed_call(build, part, repeats=repeats, warmup=warmup)
    return t, {}


def bench_jztree_knn(pos, leaf_size, k, repeats, warmup):
    import jax.numpy as jnp
    import jztree

    cfg = jztree.config.KNNConfig(tree=jztree.config.TreeConfig(max_leaf_size=leaf_size))
    part = jztree.data.Pos(pos=jnp.asarray(pos))

    def do_knn(p):
        rnn, inn = jztree.knn.knn(p, k=k, cfg=cfg)
        return rnn, inn

    import jax
    jknn = jax.jit(do_knn) if not hasattr(do_knn, "jit") else do_knn
    t = timed_call(jknn, part, repeats=repeats, warmup=warmup)
    return t, {"k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "traverse", "both"], default="both")
    ap.add_argument("--ns", type=int, nargs="+", default=[10000, 100000, 1000000])
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leaf-size", type=int, default=32)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--knn-k", type=int, default=32)
    ap.add_argument("--libs", nargs="+", default=["yggdrax", "jztree"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax  # noqa: F401
    from common.ic import IC_GENERATORS

    rows = []
    for n in args.ns:
        pos, mass = IC_GENERATORS[args.ic](n, seed=0)
        print(f"\n=== N={n} ({args.ic}) ===")
        for lib in args.libs:
            try:
                if args.mode in ("build", "both"):
                    if lib == "yggdrax":
                        t, extra = bench_yggdrax_build(pos, mass, args.leaf_size,
                                                       args.repeats, args.warmup)
                    else:
                        t, extra = bench_jztree_build(pos, args.leaf_size,
                                                      args.repeats, args.warmup)
                    rows.append(dict(op="build", lib=lib, n=n, leaf_size=args.leaf_size,
                                     **t.as_dict(), **extra))
                    print(f"  [{lib}] build: min={t.min_ms:.3f}ms {extra}")
                if args.mode in ("traverse", "both"):
                    if lib == "yggdrax":
                        t, extra = bench_yggdrax_traverse(pos, mass, args.leaf_size,
                                                          args.theta, args.repeats, args.warmup)
                        op = "interactions"
                    else:
                        t, extra = bench_jztree_knn(pos, args.leaf_size, args.knn_k,
                                                    args.repeats, args.warmup)
                        op = "knn"
                    rows.append(dict(op=op, lib=lib, n=n, leaf_size=args.leaf_size,
                                     **t.as_dict(), **extra))
                    print(f"  [{lib}] {op}: min={t.min_ms:.3f}ms {extra}")
            except Exception as exc:
                print(f"  [{lib}] FAILED at N={n}: {type(exc).__name__}: {exc}")
                rows.append(dict(op=args.mode, lib=lib, n=n, error=str(exc)))

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "tree.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "provenance": capture_provenance({"benchmark": "tree_vs_jztree", "args": vars(args)}),
        "results": rows,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
