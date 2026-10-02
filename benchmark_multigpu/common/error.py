"""Per-particle force-error metrics shared by every code in the comparison.

Lived in three places (``codes/compare_force.py``, ``codes/jzfmm_force_eval.py``
and a bare aggregate-L2 in ``codes/smallleaf_baseline.py``) until 2026-09-12.
Two definitions of "the error" drifting apart is exactly the failure the record
cannot afford, so the metric lives here and nothing else defines it. This module
imports numpy only -- ``jzfmm_force_eval`` runs in a venv with no jaccpot.

``p90`` is jz-fmm's headline (their Fig. 9): the 90th percentile of the
per-particle relative acceleration error against a direct sum. ``aggL2`` is
ours. Both are reported everywhere so either figure can be drawn.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rel_errors"]


def rel_errors(a: np.ndarray, a_ref: np.ndarray, idx=None) -> dict:
    """Per-particle relative acceleration error plus an aggregate L2.

    ``idx`` restricts the comparison to a subsample of targets (identical for
    both codes); the memory ``rel-l2-probe-not-comparable`` applies *across*
    subsample sizes, never within one, so ``n_ref`` is recorded on the result.

    ``aggL2_signflip`` exists purely as a convention tripwire: if a code returned
    the opposite sign convention we would otherwise report a ~2.0 "error" and
    call it an accuracy result.  If the flipped number is the small one, the
    comparison is wired wrong -- fix it, do not report it.
    """
    a = np.asarray(a, np.float64)
    ref = np.asarray(a_ref, np.float64)
    if idx is not None:
        a = a[idx]
    num = np.linalg.norm(a - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-300
    per = num / den
    denom = np.linalg.norm(ref) + 1e-300
    return dict(
        median=float(np.median(per)),
        p90=float(np.percentile(per, 90)),
        max=float(np.max(per)),
        aggL2=float(np.linalg.norm(a - ref) / denom),
        aggL2_signflip=float(np.linalg.norm(-a - ref) / denom),
        n_ref=int(ref.shape[0]),
    )
