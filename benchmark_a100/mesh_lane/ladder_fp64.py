"""Run one ceiling-ladder point with float64 POSITIONS instead of float32.

The harness's `_disc` returns float32, so every lane below it computes in fp32
while the oracle is fp64 on those same float32 coordinates. If the ~5e-3 rel_l2
at N >= 10^6 were multipole-truncation error it would fall when the expansion
order rises; it does not (order 3 -> 6 moved it 5.007e-3 -> 4.811e-3). This
isolates the other candidate: float32 accumulation over ~10^5-10^6 near sources
per target, whose net force is a small residual of a much larger sum of terms.
Nothing but the input dtype changes.
"""

import sys

import numpy as np

import bench.distributed_ceiling_ladder as L

_orig = L._disc


def _disc64(n, radius=10.0, thickness=0.2, seed=9):
    pos, mass = _orig(n, radius=radius, thickness=thickness, seed=seed)
    return pos.astype(np.float64), mass.astype(np.float64)


L._disc = _disc64

if __name__ == "__main__":
    sys.exit(L.main(sys.argv[1:]))
