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


def direct_accelerations(
    positions,
    masses,
    *,
    G: float = 1.0,
    softening: float = 0.0,
    block_size: int = 1024,
    dtype=jnp.float64,
) -> np.ndarray:
    """Exact direct-sum accelerations, computed in blocks of ``block_size`` targets.

    Computed in ``dtype`` (float64 by default -> clean ground truth; requires
    ``JAX_ENABLE_X64=1``).  Returns a NumPy array of shape ``(N, 3)``.
    """
    pos = jnp.asarray(positions, dtype)
    mass = jnp.asarray(masses, dtype)
    n = pos.shape[0]
    gid = jnp.arange(n, dtype=jnp.int64)
    out = np.empty((n, 3), np.asarray(pos).dtype)
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        acc = _block_accel(pos[s:e], pos, mass, gid[s:e], gid, softening, G)
        out[s:e] = np.asarray(acc)
    return out
