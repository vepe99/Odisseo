"""What is jaccpot's ~90 ms floor at N=200k made of, and is Pallas even on?

Three questions the pkdgrav3 comparison could not answer because it never asked:
  1. Does loosening theta past 0.6 buy anything? (the sweep stopped at 0.6, while
     pkdgrav3's cheap points are at 0.7-0.8)
  2. Is the Pallas near-field kernel actually active? use_pallas was left at None
     (auto-detect Ampere) and never verified.
  3. Is leaf 256 the right leaf size for THIS lane on an A100? It was picked from a
     2026-07 RTX-2080 U-curve.

Prints the traversal counts alongside the time, because a time that does not move
while the pair counts do is a padding/overhead floor, not compute.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/codes")
from compare_force import apply_fast_lane_env  # noqa: E402

N = int(os.environ.get("PROBE_N", "200000"))
apply_fast_lane_env(N)

from common.ic import IC_GENERATORS  # noqa: E402

import jax  # noqa: E402
from jaccpot import (  # noqa: E402
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)

pos, mass = IC_GENERATORS["plummer"](N, seed=0)
P = jax.numpy.asarray(pos, jax.numpy.float32)
M = jax.numpy.asarray(mass, jax.numpy.float32)

DIAG_KEYS = (
    "static_radix_tree_leaf_count",
    "static_radix_far_pair_count",
    "recent_dual_neighbor_count",
    "recent_dual_far_pair_count",
    "large_n_eval_leaf_particle_slots",
    "large_n_neighbor_edges_profile_cap",
)


def run(order, theta, leaf, use_pallas, repeats=5, warmup=2):
    s = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real",
        theta=theta, G=1.0, softening=1e-7, working_dtype=jax.numpy.float32,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=leaf),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=order,
    )
    try:
        prepared, ev = s.strict_fused_prepared_eval_fn(
            positions=P, masses=M, leaf_size=leaf, max_order=order, theta=theta
        )
    except Exception as exc:
        return None, str(exc).splitlines()[0][:110], {}
    for _ in range(warmup):
        jax.block_until_ready(ev(prepared))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        a = jax.block_until_ready(ev(prepared))
        ts.append(time.perf_counter() - t0)
    d = dict(s.get_runtime_diagnostics() or {})
    return min(ts) * 1e3, np.asarray(a), {k: d.get(k) for k in DIAG_KEYS} | {
        "nearfield_mode": s.nearfield_mode, "farfield_mode": s.farfield_mode,
        "fused": d.get("strict_fused_mode_active"),
    }


def err(a, ref):
    return float(np.linalg.norm(np.asarray(a, np.float64) - ref) / np.linalg.norm(ref))


from common.reference import direct_accelerations  # noqa: E402

print(f"N={N}  computing float64 direct reference ...", flush=True)
REF = direct_accelerations(pos, mass, G=1.0, softening=1e-7, block_size=2048)

print("\n=== Q1: does theta past 0.6 buy anything?  (order 4, leaf 256, pallas auto) ===")
print(f"{'theta':>6} {'ms':>9} {'aggL2':>11} {'leaves':>8} {'near nbrs':>11} {'far pairs':>11}")
for th in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    ms, a, d = run(4, th, 256, None)
    if ms is None:
        print(f"{th:>6.1f}  FAILED: {a}")
        continue
    print(f"{th:>6.1f} {ms:>9.2f} {err(a, REF):>11.3e} "
          f"{str(d['static_radix_tree_leaf_count']):>8} "
          f"{str(d['recent_dual_neighbor_count']):>11} "
          f"{str(d['static_radix_far_pair_count']):>11}")

print("\n=== Q2: is Pallas actually on, and worth anything? (order 4, leaf 256) ===")
print(f"{'theta':>6} {'pallas':>8} {'ms':>9} {'aggL2':>11} {'nearfield_mode':>16}")
for th in (0.4, 0.6, 0.8):
    for up in (True, False):
        ms, a, d = run(4, th, 256, up)
        if ms is None:
            print(f"{th:>6.1f} {str(up):>8}  FAILED: {a}")
            continue
        print(f"{th:>6.1f} {str(up):>8} {ms:>9.2f} {err(a, REF):>11.3e} "
              f"{str(d['nearfield_mode']):>16}")

print("\n=== Q3: is leaf 256 right for this lane on an A100? (order 4, theta 0.6) ===")
print(f"{'leaf':>6} {'ms':>9} {'aggL2':>11} {'leaves':>8} {'near nbrs':>11} {'far pairs':>11}")
for lf in (64, 128, 256, 512, 1024):
    ms, a, d = run(4, 0.6, lf, None)
    if ms is None:
        print(f"{lf:>6}  FAILED: {a}")
        continue
    print(f"{lf:>6} {ms:>9.2f} {err(a, REF):>11.3e} "
          f"{str(d['static_radix_tree_leaf_count']):>8} "
          f"{str(d['recent_dual_neighbor_count']):>11} "
          f"{str(d['static_radix_far_pair_count']):>11}")
