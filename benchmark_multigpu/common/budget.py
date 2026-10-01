"""Interaction budget: how much of each force is computed by direct summation.

An implementation-independent measure of MAC efficiency, to sit beside wall time
on every row of the jaccpot-vs-pkdgrav3 comparison (plan T2.0).

jaccpot side -- from the fused lane's prepared state.  Semantics pinned from
source on 2026-09-05:

* ``yggdrax/_interactions_impl.py::_default_pair_actions_only`` emits a NEAR
  pair only for ``target_leaf & source_leaf & different_nodes`` -- a leaf is
  **never** in its own neighbour row.
* ``_dual_tree_walk_impl::_near_update`` writes every accepted pair into
  **both** endpoints' CSR rows, so ``NodeNeighborList.counts[l]`` is the number
  of *other* leaves whose particles leaf ``l`` sums directly, and
  ``recent_dual_neighbor_count = neighbors.shape[0] = sum(counts)`` (before the
  radix fast lane zero-pads ``neighbors`` to the profile cap; ``offsets`` /
  ``counts`` stay exact).
* The intra-leaf term is a separate path (``pallas/nearfield_fused_leaf.py``
  docstring: "source slots never contain the target's own leaf"), so each
  target also sums its ``occupancy - 1`` leaf-mates directly.

Hence, per target particle in leaf ``l``::

    direct_sources(l) = sum_{m in nbrs(l)} occupancy(m) + occupancy(l) - 1

pkdgrav3 side -- ``master.cxx`` prints ``P-P per active`` = mean interaction-
list-particle count per active particle (``walk2.cxx:568``:
``*pdPartSum += nActive * ilp.count()``) and ``P-C per active`` likewise for
cells.  ``pkdGravInteract`` (``grav2.cxx``) sources only from those two lists,
with the sink bucket's particles as sinks; whether the sink bucket itself is
also an *opened* checklist entry (and so its own particles are on the P-P list)
was not settled from the fragments read on 2026-09-05.  The two readings differ
by ``nBucket - 1`` (~3 % at nBucket 16, P-P 470), so both are reported.
"""

from __future__ import annotations

import numpy as np


def jaccpot_direct_budget(prepared, n_particles: int) -> dict:
    """Exact per-target direct-summation budget from a ``LargeNPreparedState``."""
    nl = prepared.neighbor_list
    counts = np.asarray(nl.counts).astype(np.int64)
    offsets = np.asarray(nl.offsets).astype(np.int64)
    neighbors = np.asarray(nl.neighbors).astype(np.int64)
    leaf_nodes = np.asarray(nl.leaf_indices).astype(np.int64)
    num_leaves = int(counts.shape[0])
    total_edges = int(counts.sum())

    mask = np.asarray(prepared.nearfield_leaf_particle_mask)
    occ = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.int64)
    assert occ.shape[0] == num_leaves, (occ.shape, num_leaves)
    assert int(occ.sum()) == int(n_particles), (int(occ.sum()), n_particles)

    # node id -> leaf position; neighbour entries are node ids
    total_nodes = int(leaf_nodes.max()) + 1 if num_leaves else 0
    node_to_leaf = np.full(total_nodes, -1, np.int64)
    node_to_leaf[leaf_nodes] = np.arange(num_leaves)

    # gather exact rows (neighbors may be zero-padded past sum(counts))
    row_of_edge = np.repeat(np.arange(num_leaves), counts)
    edge_idx = np.concatenate(
        [np.arange(offsets[l], offsets[l] + counts[l]) for l in range(num_leaves)]
    ) if total_edges else np.zeros(0, np.int64)
    nbr_leaf = node_to_leaf[neighbors[edge_idx]]
    assert np.all(nbr_leaf >= 0), "neighbour entry that is not a leaf"
    self_hits = int(np.sum(nbr_leaf == row_of_edge))

    # directed vs undirected check: is every (a,b) matched by (b,a)?
    pairs = set(zip(row_of_edge.tolist(), nbr_leaf.tolist()))
    unmatched = sum(1 for (a, b) in pairs if (b, a) not in pairs)

    cross_sources = np.bincount(row_of_edge, weights=occ[nbr_leaf], minlength=num_leaves)
    direct_per_target = cross_sources + (occ - 1)  # per particle in leaf l
    w = occ.astype(np.float64)  # weight leaves by their particle count
    mean_direct = float(np.sum(direct_per_target * w) / max(1.0, w.sum()))
    # per-particle percentiles: expand by occupancy
    expanded = np.repeat(direct_per_target, occ)
    return dict(
        num_leaves=num_leaves,
        neighbor_entries_directed=total_edges,
        neighbor_pairs_undirected=total_edges // 2,
        unmatched_directions=int(unmatched),
        self_entries=self_hits,
        mean_neighbor_leaves_per_leaf=float(counts.mean()) if num_leaves else 0.0,
        max_neighbor_leaves_per_leaf=int(counts.max()) if num_leaves else 0,
        occupancy_mean=float(occ.mean()) if num_leaves else 0.0,
        occupancy_min=int(occ.min()) if num_leaves else 0,
        occupancy_max=int(occ.max()) if num_leaves else 0,
        direct_sources_per_target_mean=mean_direct,
        direct_sources_per_target_median=float(np.median(expanded)),
        direct_sources_per_target_p90=float(np.percentile(expanded, 90)),
        direct_sources_per_target_max=int(expanded.max()),
        direct_share_of_N=mean_direct / float(n_particles),
        # P2P particle-pair evaluations actually performed (cross + intra, directed)
        p2p_pair_evaluations=float(np.sum(cross_sources * occ) + np.sum(occ * (occ - 1))),
    )


def pkdgrav3_direct_budget(pp_per_active: float, pc_per_active: float,
                           n_bucket: int, n_particles: int) -> dict:
    """pkdgrav3's budget from its own ``P-P per active`` / ``P-C per active`` report."""
    raw = float(pp_per_active)
    plus_own = raw + (int(n_bucket) - 1)
    return dict(
        pp_per_active=raw,
        pc_per_active=float(pc_per_active),
        n_bucket=int(n_bucket),
        direct_sources_per_target_mean=raw,
        direct_sources_per_target_mean_if_own_bucket_excluded_from_pp=plus_own,
        direct_share_of_N=raw / float(n_particles),
        direct_share_of_N_upper=plus_own / float(n_particles),
    )
