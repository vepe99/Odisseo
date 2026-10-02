"""Decompose the 32.5 ms 'fixed' cost: is it O(1) launch overhead, or the far field?
 A) theta=1.0, N=200k, order 2/4/6 -> if time scales with order, it is far field.
 B) theta=1.0, order 4, N = 50k/100k/200k/400k -> intercept at N->0 is true O(1) overhead."""
import os, sys, time, numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/codes")
from compare_force import apply_fast_lane_env
from common.ic import IC_GENERATORS
def run(N, order, theta, leaf=256):
    apply_fast_lane_env(N)
    import jax
    from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, TreeConfig, FarFieldConfig, NearFieldConfig
    pos, mass = IC_GENERATORS["plummer"](N, seed=0)
    P = jax.numpy.asarray(pos, jax.numpy.float32); M = jax.numpy.asarray(mass, jax.numpy.float32)
    s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta, G=1.0,
        softening=1e-7, working_dtype=jax.numpy.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=leaf),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=order)
    try:
        prep, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=leaf, max_order=order, theta=theta)
    except Exception as e:
        return None, str(e).splitlines()[0][:100], {}
    for _ in range(2): jax.block_until_ready(ev(prep))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); jax.block_until_ready(ev(prep)); ts.append(time.perf_counter() - t0)
    d = s.get_runtime_diagnostics() or {}
    return min(ts) * 1e3, None, {"nbrs": d.get("recent_dual_neighbor_count"),
        "slots": d.get("large_n_eval_leaf_particle_slots"), "leaves": d.get("large_n_eval_active_leaf_count")}
print("=== A: theta=1.0, N=200k, order sweep ===", flush=True)
for p in (2, 4, 6):
    ms, err, d = run(200_000, p, 1.0)
    print(f"  order {p}: {ms if ms is None else f'{ms:8.2f} ms'}  {err or ''} {d}", flush=True)
print("=== B: theta=1.0, order 4, N sweep ===", flush=True)
for N in (50_000, 100_000, 200_000, 400_000, 800_000):
    ms, err, d = run(N, 4, 1.0)
    print(f"  N {N:>7}: {ms if ms is None else f'{ms:8.2f} ms'}  {err or ''} {d}", flush=True)
print("=== C: theta=0.6, order 4, N sweep (the working point) ===", flush=True)
for N in (50_000, 100_000, 200_000, 400_000, 800_000):
    ms, err, d = run(N, 4, 0.6)
    print(f"  N {N:>7}: {ms if ms is None else f'{ms:8.2f} ms'}  {err or ''} {d}", flush=True)
