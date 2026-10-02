"""Overflow-retry capacity calibration for the distributed FMM driver.

The driver's traversal buffers (``max_interactions_per_node``, ``max_neighbors_per_leaf``,
``max_pair_queue`` and their cross-walk twins) are *static* -- they set fixed
array shapes and, if too small, silently truncate the interaction/neighbour
lists.  The driver surfaces this as ``*_overflow`` flags in its diagnostics.

:func:`calibrate_caps` grows the caps (by ``growth`` each try) until every
overflow flag clears, then returns the working config plus the observed pair
counts, so a per-N cap schedule can be recorded once and reused (each cap change
forces an XLA recompile, so we don't want to grow at benchmark time).
"""

from __future__ import annotations

import numpy as np

from jaccpot.distributed import DistributedFMMConfig, distributed_fmm_accelerations


def calibrate_caps(
    positions,
    masses,
    *,
    mesh,
    ndev: int,
    base_config: DistributedFMMConfig | None = None,
    growth: float = 1.6,
    max_tries: int = 6,
    verbose: bool = True,
):
    """Grow traversal caps until the driver reports no overflow.

    Returns ``(config, info)`` where ``config`` is the smallest tried config with
    no overflow and ``info`` records the tries, final diagnostics, and the
    observed max pair counts (for setting caps with margin).
    """
    config = base_config or DistributedFMMConfig()
    tries = []
    for attempt in range(max_tries):
        result = distributed_fmm_accelerations(
            positions, masses, config=config, mesh=mesh, ndev=ndev, jit=False
        )
        diag = result.diagnostics
        overflow = result.overflow
        rec = {
            "attempt": attempt,
            "caps": {
                "max_interactions_per_node": config.max_interactions_per_node,
                "max_neighbors_per_leaf": config.max_neighbors_per_leaf,
                "max_pair_queue": config.max_pair_queue,
                "cross_max_interactions_per_node": config.cross_max_interactions_per_node,
                "cross_max_neighbors_per_leaf": config.cross_max_neighbors_per_leaf,
                "cross_max_pair_queue": config.cross_max_pair_queue,
            },
            "overflow": overflow,
            "cross_far_pairs": float(np.max(diag["cross_far_pairs"])),
            "cross_near_pairs": float(np.max(diag["cross_near_pairs"])),
            "self_far_pairs": float(np.max(diag["self_far_pairs"])),
            "self_near_pairs": float(np.max(diag["self_near_pairs"])),
        }
        tries.append(rec)
        if verbose:
            print(
                f"[calibrate] try {attempt}: overflow={overflow} "
                f"caps(int/nbr/queue)={config.max_interactions_per_node}/"
                f"{config.max_neighbors_per_leaf}/{config.max_pair_queue} "
                f"cross={config.cross_max_interactions_per_node}/"
                f"{config.cross_max_neighbors_per_leaf}/{config.cross_max_pair_queue}"
            )
        if not overflow:
            return config, {"tries": tries, "final_diag": {k: np.asarray(v).tolist() for k, v in diag.items()}}
        config = config.with_scaled_caps(growth)

    raise RuntimeError(
        f"caps still overflowing after {max_tries} tries; last caps={tries[-1]['caps']}"
    )
