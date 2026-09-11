#!/usr/bin/env python
"""Phase 1.2 of the sub-10 ms plan: is the MAC consistent with the expansion centres?

Observation (queue0, 2026-09-11, N=200k Plummer, leaf 64): the force error at
theta 0.8 barely moves with order (5.3e-3 / 4.1e-3 / 3.5e-3 for p = 4/5/6) and
at theta 1.0 it RISES (1.7e-2 / 1.8e-2 / 3.7e-2), while jz-fmm at theta 0.8 p5
reaches 4.6e-4 with 16x fewer direct pairs. An error that grows with the order is
a divergent expansion: some accepted far pair has (r_A + r_B) / d >= 1 about the
centres the expansion actually uses.

Our walk tests the MAC with the bounding-box geometry (``TreeGeometry.center``
and the sphere radius ``_build_mac_extents`` derives from it), but the native
real-basis upward sweep expands about the CENTRE OF MASS (``center_mode='com'``
is the only mode it accepts), and M2L/L2L/L2P use those COM centres. The two
sets of centres differ by up to a leaf radius in a skewed leaf, so a pair that
passes the MAC about the box centres can sit outside the convergence radius
about the COMs.

This probe takes the fused lane's prepared state (the real far-pair list) and
measures, per accepted far pair, the convergence ratio about three geometries:

* ``mac``  -- what the walk tested: MAC radii and box centres (should be <= theta);
* ``aabb`` -- exact particle radii about the box centres;
* ``com``  -- exact particle radii about the COMs and COM-to-COM distance: the
  ratio that governs the expansion actually evaluated.

Reports the max, the fraction >= 1 (divergent) and >= theta, and how many far
pairs a COM-consistent MAC at the same theta would reject. No kernel is written
before this number is known (plan 1.2).
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


def _node_com_and_rmax(node_ranges, pos, mass, centers_alt=None):
    """Per-node centre of mass and exact max particle distance about it (and about ``centers_alt``)."""
    n_nodes = node_ranges.shape[0]
    pm = np.concatenate([[0.0], np.cumsum(mass)])
    pmx = np.concatenate([np.zeros((1, 3)), np.cumsum(mass[:, None] * pos, axis=0)], axis=0)
    a = node_ranges[:, 0]
    b = node_ranges[:, 1] + 1
    msum = pm[b] - pm[a]
    com = (pmx[b] - pmx[a]) / np.maximum(msum, 1e-300)[:, None]
    rmax_com = np.zeros(n_nodes)
    rmax_alt = np.zeros(n_nodes) if centers_alt is not None else None
    for i in range(n_nodes):
        seg = pos[a[i]:b[i]]
        if seg.shape[0] == 0:
            continue
        rmax_com[i] = np.sqrt(np.max(np.sum((seg - com[i]) ** 2, axis=1)))
        if centers_alt is not None:
            rmax_alt[i] = np.sqrt(np.max(np.sum((seg - centers_alt[i]) ** 2, axis=1)))
    return com, rmax_com, rmax_alt, msum


def _stats(ratio, theta):
    q = np.percentile(ratio, [50, 90, 99, 99.9])
    return dict(
        n=int(ratio.size), max=float(ratio.max()), p50=float(q[0]), p90=float(q[1]),
        p99=float(q[2]), p999=float(q[3]),
        frac_ge_1=float(np.mean(ratio >= 1.0)), frac_ge_theta=float(np.mean(ratio > theta)),
        count_ge_1=int(np.sum(ratio >= 1.0)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.6, 0.8, 1.0])
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

    import jax
    import jax.numpy as jnp
    from jaccpot import (
        FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig,
        RuntimePolicyConfig, TraversalOverrides, TreeConfig,
    )
    from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled
    from yggdrax._interactions_impl import _build_mac_extents

    pos0, mass0 = IC_GENERATORS[args.ic](args.n, seed=0)
    P = jnp.asarray(pos0, jnp.float32)
    M = jnp.asarray(mass0, jnp.float32)
    runtime_cfg = RuntimePolicyConfig()
    if preset_trav:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(**{k: int(v) for k, v in preset_trav.items()}))
    result = dict(n=args.n, ic=args.ic, leaf=args.leaf, order=args.order, env=overrides, thetas={})
    out_path = Path(args.out or (ROOT / "artifacts" / "sub10ms" / f"mac_consistency_{args.ic}{args.n}_leaf{args.leaf}_{os.environ.get('JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY', 'aabb')}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for theta in args.thetas:
        solver = FastMultipoleMethod(
            preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta,
            G=1.0, softening=args.softening, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"),
                runtime=runtime_cfg, mac_type="dehnen"),
            fixed_order=args.order)
        prepared, _ = solver.strict_fused_prepared_eval_fn(
            positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=theta)
        tree = prepared.tree
        pos = np.asarray(prepared.positions_sorted, np.float64)
        mass = np.asarray(prepared.masses_sorted, np.float64)
        node_ranges = np.asarray(tree.node_ranges).astype(np.int64)
        parent = np.asarray(tree.parent)
        num_internal = int(np.asarray(tree.left_child).shape[0])
        geom = compute_tree_geometry_compiled(tree, jnp.asarray(prepared.positions_sorted))
        c_aabb = np.asarray(geom.center, np.float64)
        r_geom = np.asarray(geom.radius, np.float64)
        mac_ext, _ = _build_mac_extents(
            jnp.asarray(tree.parent), geom, num_internal, "dehnen", 1.0)
        mac_ext = np.asarray(mac_ext, np.float64)

        cfp = prepared.compact_far_pairs
        cnt = int(np.asarray(cfp.far_pair_count))
        src = np.asarray(cfp.sources)[:cnt].astype(np.int64)
        tgt = np.asarray(cfp.targets)[:cnt].astype(np.int64)
        keep = (src >= 0) & (tgt >= 0)
        src, tgt = src[keep], tgt[keep]
        # interleaved [b->a, a->b]: keep one direction of each canonical pair
        canon = src < tgt
        src_c, tgt_c = src[canon], tgt[canon]

        com, rmax_com, rmax_aabb, msum = _node_com_and_rmax(node_ranges, pos, mass, centers_alt=c_aabb)

        d_aabb = np.linalg.norm(c_aabb[src_c] - c_aabb[tgt_c], axis=1)
        d_com = np.linalg.norm(com[src_c] - com[tgt_c], axis=1)
        ratio_mac = (mac_ext[src_c] + mac_ext[tgt_c]) / d_aabb
        ratio_aabb = (rmax_aabb[src_c] + rmax_aabb[tgt_c]) / d_aabb
        ratio_com = (rmax_com[src_c] + rmax_com[tgt_c]) / d_com
        # the mixed case the code actually evaluates: expansions about the COMs
        # (radii about COM, distance COM-to-COM) -- that is ratio_com.
        off = np.linalg.norm(com - c_aabb, axis=1)
        # how loose is the COM geometry's internal-node bound on THIS tree?
        from jaccpot.runtime._mac_geometry import com_mac_geometry
        com_geom = com_mac_geometry(tree, jnp.asarray(prepared.positions_sorted),
                                    jnp.asarray(com, jnp.float32), leaf_cap=args.leaf)
        r_bound = np.asarray(com_geom.radius, np.float64)
        nz = rmax_com > 0
        loose = r_bound[nz] / rmax_com[nz]
        loose_int = loose[: int(np.sum(nz[:num_internal]))]
        bound_stats = dict(
            all_ge_exact=bool(np.all(r_bound + 1e-6 * np.maximum(rmax_com, 1) >= rmax_com)),
            internal_p50=float(np.median(loose_int)), internal_p90=float(np.percentile(loose_int, 90)),
            internal_p99=float(np.percentile(loose_int, 99)), internal_max=float(loose_int.max()),
        )
        rec = dict(
            far_pairs_canonical=int(src_c.size), nodes=int(node_ranges.shape[0]), num_internal=num_internal,
            geometry_radius_vs_exact_aabb=dict(
                ratio_max=float(np.max(r_geom / np.maximum(rmax_aabb, 1e-300))),
                ratio_min=float(np.min(np.where(rmax_aabb > 0, r_geom / np.maximum(rmax_aabb, 1e-300), np.inf))),
            ),
            com_offset_over_rmax_aabb=dict(
                p50=float(np.median(off / np.maximum(rmax_aabb, 1e-300))),
                p99=float(np.percentile(off / np.maximum(rmax_aabb, 1e-300), 99)),
                max=float(np.max(off / np.maximum(rmax_aabb, 1e-300))),
            ),
            rmax_com_over_rmax_aabb=dict(
                p50=float(np.median(rmax_com / np.maximum(rmax_aabb, 1e-300))),
                p99=float(np.percentile(rmax_com / np.maximum(rmax_aabb, 1e-300), 99)),
                max=float(np.max(rmax_com / np.maximum(rmax_aabb, 1e-300))),
            ),
            com_bound_over_exact=bound_stats,
            ratio_mac=_stats(ratio_mac, theta),
            ratio_aabb_exact=_stats(ratio_aabb, theta),
            ratio_com_exact=_stats(ratio_com, theta),
            far_pairs_rejected_by_com_mac_at_same_theta=int(np.sum(ratio_com > theta)),
        )
        # which pairs diverge: leaf-leaf, leaf-internal, internal-internal
        is_leaf = lambda i: i >= num_internal  # noqa: E731
        div = ratio_com >= 1.0
        rec["divergent_pairs_by_class"] = dict(
            leaf_leaf=int(np.sum(div & is_leaf(src_c) & is_leaf(tgt_c))),
            leaf_internal=int(np.sum(div & (is_leaf(src_c) ^ is_leaf(tgt_c)))),
            internal_internal=int(np.sum(div & ~is_leaf(src_c) & ~is_leaf(tgt_c))),
        )
        result["thetas"][str(theta)] = rec
        print(f"[mac theta {theta}] far pairs {src_c.size}: ratio_mac max {rec['ratio_mac']['max']:.3f} | "
              f"aabb-exact max {rec['ratio_aabb_exact']['max']:.3f} (>=1: {rec['ratio_aabb_exact']['frac_ge_1']:.2e}) | "
              f"COM-exact max {rec['ratio_com_exact']['max']:.3f} p99 {rec['ratio_com_exact']['p99']:.3f} "
              f">=1: {rec['ratio_com_exact']['count_ge_1']} ({rec['ratio_com_exact']['frac_ge_1']:.2e}) "
              f">theta: {rec['far_pairs_rejected_by_com_mac_at_same_theta']} ({rec['ratio_com_exact']['frac_ge_theta']:.3f}); "
              f"COM offset/r_aabb p50 {rec['com_offset_over_rmax_aabb']['p50']:.2f} max {rec['com_offset_over_rmax_aabb']['max']:.2f}; "
              f"divergent by class {rec['divergent_pairs_by_class']}; "
              f"COM bound/exact internal p50 {bound_stats['internal_p50']:.2f} p90 {bound_stats['internal_p90']:.2f} "
              f"max {bound_stats['internal_max']:.2f} ge_exact {bound_stats['all_ge_exact']}", flush=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        del solver, prepared
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
