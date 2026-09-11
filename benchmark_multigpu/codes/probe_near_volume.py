#!/usr/bin/env python
"""Where does the near-field volume come from? (sub-10 ms plan, Phase 1.2)

With the COM-consistent MAC (``JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com``) the
fused lane at N=2x10^5 / leaf 64 / theta 0.8 still sums 0.13 N sources per target
directly, jz-fmm 0.0116 N at the same theta -- 11x. Hypothesis: radix leaves hold
a FIXED 64 particles, so the leaves in the Plummer tail are enormous (radius
comparable to the core's distance) and the mutual MAC makes each of them near to
EVERY leaf; each core target then sums 64 x (number of such leaves) sources.
jz-tree's leaves are Morton cells: bounded in SIZE, sparse in the tail, so a
"global" tail cell costs a handful of particles per target.

This probe takes the prepared state and reports, for the near CSR:
* per-leaf in-degree (how many rows a leaf appears in) and its radius / distance
  from the centre; the share of the direct volume carried by leaves in the top
  in-degree percentiles;
* the share of the direct volume from pairs where at least one leaf's COM radius
  exceeds k x the median leaf radius (k = 2, 4, 8);
* per-target volume percentiles (median vs mean says structural vs outlier).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import pick_idle_gpus, set_cuda_visible  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402
from compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        set_cuda_visible(pick_idle_gpus(1))
    extra = dict(kv.split("=", 1) for kv in args.env)
    overrides = fast_lane_overrides_for_leaf(args.leaf, args.n, extra)
    overrides.update(extra)
    preset_trav = dict((FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {}))
    apply_fast_lane_env(args.n, overrides=overrides)

    import jax.numpy as jnp
    from jaccpot import (
        FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig,
        RuntimePolicyConfig, TraversalOverrides, TreeConfig,
    )

    pos0, mass0 = IC_GENERATORS[args.ic](args.n, seed=0)
    P = jnp.asarray(pos0, jnp.float32)
    M = jnp.asarray(mass0, jnp.float32)
    runtime_cfg = RuntimePolicyConfig()
    if preset_trav:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(**{k: int(v) for k, v in preset_trav.items()}))
    solver = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
        G=1.0, softening=args.softening, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"),
            runtime=runtime_cfg, mac_type="dehnen"),
        fixed_order=args.order)
    prepared, _ = solver.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)

    tree = prepared.tree
    pos = np.asarray(prepared.positions_sorted, np.float64)
    mass = np.asarray(prepared.masses_sorted, np.float64)
    node_ranges = np.asarray(tree.node_ranges).astype(np.int64)
    num_internal = int(np.asarray(tree.left_child).shape[0])
    L = node_ranges.shape[0] - num_internal
    leaf_ranges = node_ranges[num_internal:]
    occ = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    # leaf COM and radius about it
    pm = np.concatenate([[0.0], np.cumsum(mass)])
    pmx = np.concatenate([np.zeros((1, 3)), np.cumsum(mass[:, None] * pos, axis=0)], axis=0)
    a, b = leaf_ranges[:, 0], leaf_ranges[:, 1] + 1
    com = (pmx[b] - pmx[a]) / (pm[b] - pm[a])[:, None]
    rad = np.array([np.sqrt(np.max(np.sum((pos[a[i]:b[i]] - com[i]) ** 2, axis=1))) for i in range(L)])
    dist = np.linalg.norm(com, axis=1)

    nl = prepared.neighbor_list
    counts = np.asarray(nl.counts).astype(np.int64)
    offsets = np.asarray(nl.offsets).astype(np.int64)
    nbr_nodes = np.asarray(nl.neighbors).astype(np.int64)
    row = np.repeat(np.arange(L), counts)
    edge_idx = np.concatenate([np.arange(offsets[l], offsets[l] + counts[l]) for l in range(L)])
    src = nbr_nodes[edge_idx] - num_internal
    assert np.all((src >= 0) & (src < L))
    # directed volume: target particles of row x source particles
    vol = occ[row] * occ[src]
    total = float(vol.sum()) + float(np.sum(occ * (occ - 1)))
    n = int(pos.shape[0])
    per_target_direct = (np.bincount(row, weights=occ[src], minlength=L) + occ - 1)
    expanded = np.repeat(per_target_direct, occ)

    indeg = np.bincount(src, minlength=L)
    order = np.argsort(-indeg)
    cum_vol_by_src = np.bincount(src, weights=vol, minlength=L)
    res = dict(
        n=n, leaf=args.leaf, theta=args.theta, mac_geometry=os.environ.get("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "aabb"),
        num_leaves=L, near_edges_directed=int(counts.sum()), direct_share_of_N=float(expanded.mean() / n),
        per_target=dict(p50=float(np.median(expanded) / n), p90=float(np.percentile(expanded, 90) / n),
                        max=float(expanded.max() / n)),
        leaf_radius=dict(p50=float(np.median(rad)), p90=float(np.percentile(rad, 90)), p99=float(np.percentile(rad, 99)),
                         max=float(rad.max())),
        indegree=dict(p50=float(np.median(indeg)), p90=float(np.percentile(indeg, 90)), p99=float(np.percentile(indeg, 99)),
                      max=int(indeg.max()), leaves_seen_by_half_of_all_rows=int(np.sum(indeg > L / 2))),
        volume_share_from_top_indegree_leaves={},
        volume_share_from_pairs_with_big_leaf={},
        big_leaves={},
    )
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        k = max(1, int(frac * L))
        top = order[:k]
        res["volume_share_from_top_indegree_leaves"][f"top{frac:g}"] = dict(
            leaves=k, volume_share=float(cum_vol_by_src[top].sum() / total),
            particles=int(occ[top].sum()), radius_p50=float(np.median(rad[top])),
            dist_p50=float(np.median(dist[top])), indegree_min=int(indeg[top].min()))
    med = float(np.median(rad))
    for kk in (2, 4, 8, 16):
        big = rad > kk * med
        pairs_big = big[row] | big[src]
        res["volume_share_from_pairs_with_big_leaf"][f"r>{kk}xmedian"] = dict(
            big_leaves=int(big.sum()), big_leaf_particles=int(occ[big].sum()),
            volume_share=float(vol[pairs_big].sum() / total),
            volume_share_big_as_source_only=float(vol[big[src] & ~big[row]].sum() / total))
    res["big_leaves"] = dict(
        radius_gt_2median=int(np.sum(rad > 2 * med)), radius_gt_4median=int(np.sum(rad > 4 * med)),
        radius_gt_8median=int(np.sum(rad > 8 * med)),
        dist_of_leaves_radius_gt_4median_p50=float(np.median(dist[rad > 4 * med])) if np.any(rad > 4 * med) else None,
    )
    out_path = Path(args.out or (ROOT / "artifacts" / "sub10ms" /
                                 f"near_volume_{args.ic}{args.n}_leaf{args.leaf}_th{args.theta:g}_{res['mac_geometry']}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
