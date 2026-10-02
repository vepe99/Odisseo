"""Chunked O(N^2) direct-sum reference (the accuracy ground truth).

``jaccpot.autodiff.differentiable_gravitational_acceleration`` materialises the
full ``(N, N, 3)`` difference tensor, which OOMs above ~2e4 particles.  This
version loops over target blocks so the peak memory is ``block_size * N * 3``,
letting the direct reference reach ~1e5 on a single H100 while computing exactly
the same Plummer-softened sum (self-interaction excluded, ``-G`` applied).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnames=("softening", "G"))
def _block_accel(tgt_pos, src_pos, src_mass, tgt_gid, src_gid, softening, G):
    diffs = tgt_pos[:, None, :] - src_pos[None, :, :]  # (B, N, 3)
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv3 = dist2 ** (-1.5)
    # exclude exact self (matched by global id, robust to duplicate positions)
    self_mask = tgt_gid[:, None] == src_gid[None, :]
    w = jnp.where(self_mask, 0.0, src_mass[None, :] * inv3)
    return -G * jnp.einsum("ij,ijk->ik", w, diffs)


#: Target size of one block's ``(block, N)`` fp64 temporary. A few of these are
#: live at once inside the fused kernel, so this stays well under a card.
_BLOCK_BUDGET_BYTES = 2 << 30


def direct_accelerations(
    positions,
    masses,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    block_size: int = 0,
    dtype=jnp.float64,
    target_indices=None,
) -> np.ndarray:
    """Exact direct-sum accelerations, computed in blocks of ``block_size`` targets.

    Computed in ``dtype`` (float64 by default -> clean ground truth; requires
    ``JAX_ENABLE_X64=1``).  Returns a NumPy array of shape ``(N, 3)``, or
    ``(len(target_indices), 3)`` when ``target_indices`` restricts the targets
    (every source still enters each sum; only the rows evaluated are fewer).

    ``block_size`` 0 (the default) sizes the block so one block's ``(block, N)``
    fp64 temporary stays near ``_BLOCK_BUDGET_BYTES``. A FIXED block does not
    scale: 1024 targets x 4x10^6 sources in fp64 is a 30.5 GiB allocation, and at
    N = 4x10^6 that is what failed -- the ACCURACY CHECK, not the solver, and the
    traceback names this file, so read it before concluding anything about the
    FMM's own ceiling.
    """
    pos = jnp.asarray(positions, dtype)
    mass = jnp.asarray(masses, dtype)
    n = pos.shape[0]
    gid = jnp.arange(n, dtype=jnp.int64)
    if target_indices is None:
        tgt_pos, tgt_gid = pos, gid
    else:
        idx = jnp.asarray(np.asarray(target_indices, np.int64))
        tgt_pos, tgt_gid = pos[idx], gid[idx]
    m = int(tgt_pos.shape[0])
    if int(block_size) <= 0:
        per_target = max(int(n), 1) * jnp.dtype(dtype).itemsize
        block_size = int(max(1, min(m, _BLOCK_BUDGET_BYTES // max(per_target, 1))))
    out = np.empty((m, 3), np.asarray(pos).dtype)
    for s in range(0, m, block_size):
        e = min(s + block_size, m)
        acc = _block_accel(tgt_pos[s:e], pos, mass, tgt_gid[s:e], gid, softening, G)
        out[s:e] = np.asarray(acc)
    return out
