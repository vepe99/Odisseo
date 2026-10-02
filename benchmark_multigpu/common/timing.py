"""Steady-state GPU timing helpers (warmup + block_until_ready + min-of-repeats).

Mirrors the ``timed_call`` idiom used in yggdrax's
``examples/tree_gpu_performance_scaling.ipynb`` so tree and FMM benchmarks report
comparable numbers.  We report the *minimum* over repeats (least contaminated by
scheduler/other-process noise) alongside mean and std.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import jax


def block(x):
    """Block until every leaf of a pytree of jax Arrays is materialised."""
    leaves = jax.tree_util.tree_leaves(x)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


@dataclass
class Timing:
    min_ms: float
    mean_ms: float
    std_ms: float
    repeats: int
    warmup: int

    def as_dict(self) -> dict:
        return asdict(self)


def timed_call(fn, *args, repeats: int = 10, warmup: int = 2, **kwargs) -> Timing:
    """Time ``fn(*args, **kwargs)`` on device.

    Runs ``warmup`` untimed calls (triggers compilation + steady state), then
    ``repeats`` timed calls, blocking on the result each time.  Returns min /
    mean / std wall-clock in milliseconds.
    """
    for _ in range(warmup):
        block(fn(*args, **kwargs))

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        block(fn(*args, **kwargs))
        samples.append((time.perf_counter() - t0) * 1e3)

    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / n
    return Timing(
        min_ms=min(samples),
        mean_ms=mean,
        std_ms=var**0.5,
        repeats=repeats,
        warmup=warmup,
    )
