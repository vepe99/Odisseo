#!/usr/bin/env python
"""Does leaf COMPACTNESS (Morton-cell leaves) remove the near-field volume? (sub-10 ms plan, Phase 1.2)

``probe_near_volume.py`` showed that with the COM-consistent MAC at theta 0.8,
341 radix leaves of radius > 4x the median (11 % of the particles) sit in 56 % of
the direct volume: 64-particle Morton BUCKETS in the low-density shell (r ~ 3-5)
straddle several cells and become neighbours of most of the tree. jz-tree's
leaves are Morton CELLS (bounded extent, sparse in the tail), so its near field
at the same theta is 11x smaller.

This probe runs the SAME mutual walk (``_build_flat_walk_artifacts_strict_streamed``,
COM centres, exact particle radii about them) on several trees of the same
particles and compares list volumes:

* static radix, leaf 64 (today's fused lane) and leaf 32;
* yggdrax adaptive octree (Morton cells with <= leaf_size particles), leaf 64 / 32;

and for the radix tree also the conservative internal-node bound of
``com_mac_geometry`` against exact radii, to price the bound in far pairs.
Reports far pairs, near edges, direct share of N, leaf count / occupancy / radius
statistics and the walk's eager wall time. No fused-lane change is made before
these numbers are known.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import pick_idle_gpus, set_cuda_visible  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402


def _exact_rmax(node_ranges, pos, centers):
    out = np.zeros(node_ranges.shape[0])
    for i, (a, b) in enumerate(node_ranges):
        if b >= a:
            seg = pos[a : b + 1]
            out[i] = np.sqrt(np.max(np.sum((seg - centers[i]) ** 2, axis=1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.8, 0.6])
    ap.add_argument("--trees", nargs="+", default=["cells64", "cells32", "cells16"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("YGGDRAX_INDEX_PRECISION", "int32")
    os.environ.setdefault("JACCPOT_INDEX_PRECISION", "int32")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        set_cuda_visible(pick_idle_gpus(1))

    import jax
    import jax.numpy as jnp
    from yggdrax.geometry import TreeGeometry
    from yggdrax.tree import Tree
    from yggdrax.tree_moments import compute_tree_mass_moments

    from jaccpot.runtime._interaction_cache import _build_flat_walk_artifacts_strict_streamed
    from jaccpot.runtime._mac_geometry import com_mac_geometry

    pos0, mass0 = IC_GENERATORS[args.ic](args.n, seed=0)
    P = jnp.asarray(pos0, jnp.float32)
    M = jnp.asarray(mass0, jnp.float32)
    n = int(P.shape[0])
    result = dict(n=n, ic=args.ic, rows=[])
    out_path = Path(args.out or (ROOT / "artifacts" / "sub10ms" / f"tree_volume_{args.ic}{n}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from yggdrax import _tree_impl
    from yggdrax.bounds import infer_bounds
    from yggdrax.morton import morton_encode

    def cell_partition(codes_sorted: np.ndarray, leaf_size: int, max_depth: int = 21):
        """Adaptive Morton-CELL leaves (jz-tree style): each leaf is the coarsest cell
        holding <= leaf_size particles; cells at max depth are leaves whatever they hold."""
        n = codes_sorted.shape[0]
        depth = np.full(n, max_depth, np.int64)
        assigned = np.zeros(n, bool)
        for d in range(0, max_depth + 1):
            cell = codes_sorted >> np.uint64(63 - 3 * d) if d > 0 else np.zeros(n, np.uint64)
            # run lengths of equal cells (cells are contiguous in Morton order)
            change = np.concatenate([[True], cell[1:] != cell[:-1]])
            starts = np.flatnonzero(change)
            ends = np.concatenate([starts[1:], [n]])
            run_len = np.repeat(ends - starts, ends - starts)
            fit = (run_len <= leaf_size) & ~assigned
            depth[fit] = d
            assigned |= fit
            if assigned.all():
                break
        # leaf key: the particle's cell at its leaf depth (vectorised shift)
        shifts = (63 - 3 * depth).astype(np.uint64)
        key_cell = codes_sorted >> shifts
        change = np.concatenate([[True], (key_cell[1:] != key_cell[:-1]) | (depth[1:] != depth[:-1])])
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], [n]])
        return starts, ends, depth[starts]

    def build(name):
        kind, leaf = name.rstrip("0123456789"), int("".join(ch for ch in name if ch.isdigit()))
        if kind == "radix":
            return Tree.from_particles(P, M, tree_type="radix", build_mode="static_radix", leaf_size=leaf), leaf
        if kind == "octree":
            return Tree.from_particles(P, M, tree_type="octree", build_mode="adaptive", leaf_size=leaf), leaf
        if kind == "cells":
            bounds = infer_bounds(P)
            codes = morton_encode(P, bounds)
            sorted_indices = jnp.argsort(codes, stable=True)
            sorted_codes = codes[sorted_indices]
            starts, ends, depths = cell_partition(np.asarray(sorted_codes).astype(np.uint64), leaf)
            out = _tree_impl._build_tree_from_leaf_partitions(
                P, M, sorted_indices, sorted_codes,
                jnp.asarray(starts, jnp.int32), jnp.asarray(ends, jnp.int32), bounds,
                leaf_size=None, return_reordered=True, workspace=None, return_workspace=False,
            )
            topo_, ps_, ms_ = out[0], out[1], out[2]

            class _Shim:  # what the flat builder and the geometry read off a Tree
                topology = topo_
                positions_sorted = ps_
                masses_sorted = ms_

                def __getattr__(self, name):
                    return getattr(self.topology, name)

            print(f"[tree {name}] cell partition: {starts.size} leaves, depth p50 {np.median(depths):.0f} max {depths.max()}", flush=True)
            return _Shim(), leaf
        if kind == "bcells":
            # cell leaves inside the fused lane's FIXED balanced bucket structure,
            # padded with empty leaves to a power-of-two capacity
            bounds = infer_bounds(P)
            codes = morton_encode(P, bounds)
            sorted_indices = jnp.argsort(codes, stable=True)
            sorted_codes = codes[sorted_indices]
            starts, ends, depths = cell_partition(np.asarray(sorted_codes).astype(np.uint64), leaf)
            k = int(starts.size)
            cap = 1 << int(np.ceil(np.log2(1.25 * k)))
            starts_p = np.full(cap, n, np.int64); ends_p = np.full(cap, n, np.int64)
            starts_p[:k] = starts; ends_p[:k] = ends
            (parent_np, left_np, right_np, lil_np, ril_np, node_ranges_np, node_level_np,
             level_offsets_np, nodes_by_level_np, num_levels) = _tree_impl._build_balanced_bucket_structure(starts_p, ends_p)
            ps_, ms_, inv_ = _tree_impl.reorder_particles_by_indices(P, M, sorted_indices)
            I = jnp.int32
            topo_ = _tree_impl.RadixTree(
                parent=jnp.asarray(parent_np, I), left_child=jnp.asarray(left_np, I), right_child=jnp.asarray(right_np, I),
                left_is_leaf=jnp.asarray(lil_np), right_is_leaf=jnp.asarray(ril_np),
                particle_indices=jnp.asarray(sorted_indices, I), morton_codes=sorted_codes,
                node_ranges=jnp.asarray(node_ranges_np, I), num_particles=n, num_internal_nodes=cap - 1,
                node_level=jnp.asarray(node_level_np, I), level_offsets=jnp.asarray(level_offsets_np, I),
                nodes_by_level=jnp.asarray(nodes_by_level_np, I), num_levels=jnp.asarray(num_levels, I),
                bounds_min=jnp.asarray(bounds[0], P.dtype), bounds_max=jnp.asarray(bounds[1], P.dtype),
                leaf_codes=sorted_codes[jnp.asarray(np.minimum(starts_p, n - 1), I)],
                leaf_depths=jnp.asarray(np.concatenate([depths, np.full(cap - k, -1)]), I),
                use_morton_geometry=jnp.asarray(False), leaf_size=int(leaf),
            )

            class _Shim2:
                topology = topo_
                positions_sorted = ps_
                masses_sorted = ms_

                def __getattr__(self, name):
                    return getattr(self.topology, name)

            print(f"[tree {name}] cell partition: {k} live leaves in cap {cap} (balanced structure), depth p50 {np.median(depths):.0f} max {depths.max()}", flush=True)
            return _Shim2(), leaf
        raise ValueError(name)

    for name in args.trees:
        try:
            t0 = time.perf_counter()
            tree, leaf = build(name)
            jax.block_until_ready(tree.node_ranges)
            build_s = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            print(f"[tree {name}] BUILD FAILED: {str(exc).splitlines()[0][:300]}", flush=True)
            result["rows"].append(dict(tree=name, failed=str(exc)[:500]))
            continue
        topo = tree.topology
        ps = jnp.asarray(tree.positions_sorted)
        ms = jnp.asarray(tree.masses_sorted)
        node_ranges = np.asarray(topo.node_ranges).astype(np.int64)
        num_internal = int(np.asarray(topo.left_child).shape[0])
        total_nodes = int(node_ranges.shape[0])
        L = total_nodes - num_internal
        # leaves must be the last L nodes for the flat builder
        lil = np.asarray(topo.left_is_leaf)
        lc = np.asarray(topo.left_child)
        ok_order = bool(np.all((lc >= num_internal) == lil))
        com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
        com_np = np.asarray(com, np.float64)
        pos_np = np.asarray(ps, np.float64)
        rmax = _exact_rmax(node_ranges, pos_np, com_np)
        leaf_ranges = node_ranges[num_internal:]
        occ = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
        occ = np.where(leaf_ranges[:, 1] >= leaf_ranges[:, 0], occ, 0)
        live_leaf = occ > 0
        r_leaf = rmax[num_internal:][live_leaf]
        info = dict(
            tree=name, leaf_size=leaf, build_s=build_s, num_leaves=L, num_internal=num_internal,
            leaves_last=ok_order,
            occupancy=dict(mean=float(occ[live_leaf].mean()), p50=float(np.median(occ[live_leaf])), min=int(occ[live_leaf].min()),
                           max=int(occ.max()), empty=int(np.sum(occ == 0))),
            leaf_radius=dict(p50=float(np.median(r_leaf)), p90=float(np.percentile(r_leaf, 90)),
                             p99=float(np.percentile(r_leaf, 99)), max=float(r_leaf.max())),
            walks=[],
        )
        print(f"[tree {name}] leaves {L} (empty {int(np.sum(occ == 0))}) internal {num_internal} occ mean {occ[live_leaf].mean():.1f} min {occ[live_leaf].min()} "
              f"radius p50 {np.median(r_leaf):.3f} p99 {np.percentile(r_leaf, 99):.3f} max {r_leaf.max():.3f} "
              f"leaves_last {ok_order} build {build_s:.2f}s", flush=True)
        if not ok_order:
            result["rows"].append(info)
            continue
        geoms = {
            "com_exact": TreeGeometry(jnp.asarray(com_np, jnp.float32),
                                      jnp.broadcast_to(jnp.asarray(rmax, jnp.float32)[:, None], (total_nodes, 3)),
                                      jnp.asarray(rmax, jnp.float32), jnp.asarray(rmax, jnp.float32)),
        }
        if name.startswith("radix"):
            geoms["com_bound"] = com_mac_geometry(topo, ps, com, leaf_cap=leaf)
        for theta in args.thetas:
            for gname, geom in geoms.items():
                try:
                    t0 = time.perf_counter()
                    art = _build_flat_walk_artifacts_strict_streamed(
                        tree=tree, geometry=geom, theta=theta, mac_type="dehnen", dehnen_radius_scale=1.0,
                        compact_far_pair_capacity=1 << 24, near_edge_capacity=1 << 24, max_pair_queue=1 << 20,
                        far_named=False, near_edge_named=False,
                    )
                    jax.block_until_ready(art.compact_far_pairs.far_pair_count)
                    walk_s = time.perf_counter() - t0
                    far_directed = int(art.compact_far_pairs.far_pair_count)
                    nl = art.neighbor_list
                    counts = np.asarray(nl.counts).astype(np.int64)
                    offsets = np.asarray(nl.offsets).astype(np.int64)
                    nbr = np.asarray(nl.neighbors).astype(np.int64)
                    row = np.repeat(np.arange(L), counts)
                    edge_idx = np.concatenate([np.arange(offsets[l], offsets[l] + counts[l]) for l in range(L)]) if counts.sum() else np.zeros(0, np.int64)
                    src = nbr[edge_idx] - num_internal
                    per_target = np.bincount(row, weights=occ[src], minlength=L) + np.maximum(occ - 1, 0)
                    expanded = np.repeat(per_target, occ)
                    p2p = float(np.sum(per_target * occ))
                    rec = dict(theta=theta, geometry=gname, walk_s=walk_s, far_pairs_directed=far_directed,
                               near_edges_directed=int(counts.sum()), rows_max=int(counts.max()),
                               direct_share_of_N=float(expanded.mean() / n),
                               direct_p50=float(np.median(expanded) / n), p2p_pair_evaluations=p2p,
                               indegree_max=int(np.bincount(src, minlength=L).max()) if src.size else 0)
                    print(f"[tree {name}] th{theta:g} {gname:9s}: far {far_directed:>9} dir, near edges {int(counts.sum()):>9}, "
                          f"direct {rec['direct_share_of_N']:.4f}N (p50 {rec['direct_p50']:.4f}), p2p {p2p:.3e}, "
                          f"rows max {int(counts.max())}, indeg max {rec['indegree_max']}, walk {walk_s:.2f}s", flush=True)
                except Exception as exc:  # noqa: BLE001
                    rec = dict(theta=theta, geometry=gname, failed=str(exc).splitlines()[0][:400])
                    print(f"[tree {name}] th{theta:g} {gname}: WALK FAILED {rec['failed'][:200]}", flush=True)
                info["walks"].append(rec)
        result["rows"].append(info)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
