#!/usr/bin/env python
"""Definitive check: in-scan force at step k vs an EAGER prepare+evaluate at the
scan's own position x_k (not at x0).  Also quantifies how much the force field
changes between x0 and x1 (close pairs dominate aggL2)."""
import os, sys
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible
from compare_force import apply_fast_lane_env, rel_errors
from common.ic import IC_GENERATORS
N = 200_000; LEAF = 256; ORDER = 4; THETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = [int(os.environ["BENCH_FORCE_GPU"])] if os.environ.get("BENCH_FORCE_GPU") else pick_idle_gpus(1)
set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
import jaccpot; print("jaccpot", jaccpot.__file__)
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
def solver():
    return FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
def eager_force(x):
    s = solver()
    p, ev = s.strict_fused_prepared_eval_fn(positions=jnp.asarray(x, jnp.float32), masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
    return np.asarray(jax.block_until_ready(ev(p)), np.float64)
dt = 1e-2
a_x0 = eager_force(pos)
print(f"|a| stats at x0: median {np.median(np.linalg.norm(a_x0,axis=1)):.3f}  max {np.linalg.norm(a_x0,axis=1).max():.3e}  "
      f"top-10 share of ||a||^2: {np.sort(np.sum(a_x0**2,1))[-10:].sum()/np.sum(a_x0**2):.3f}")
s = solver()
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
_, prep_out, hist = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=3, refresh_every=1, leaf_size=LEAF,
                                    max_order=ORDER, theta=THETA, prepared_state=None, return_prepared_state=True, return_history=True)
h = np.asarray(hist, np.float64)
xs = [pos.astype(np.float64)] + [h[k, :, 0, :] for k in range(3)]
print(f"max |x1-x0| = {np.abs(xs[1]-xs[0]).max():.3e}, |x2-x1| = {np.abs(xs[2]-xs[1]).max():.3e}, |x3-x2| = {np.abs(xs[3]-xs[2]).max():.3e}")
a_scan = [2 * (xs[1] - xs[0]) / dt**2] + [(xs[k + 1] - 2 * xs[k] + xs[k - 1]) / dt**2 for k in (1, 2)]
for k in range(3):
    a_e = eager_force(xs[k])
    print(f"step {k+1}: in-scan force at x{k} vs eager@x{k}: {rel_errors(a_scan[k], a_e)['aggL2']:.3e}   "
          f"(vs eager@x0: {rel_errors(a_scan[k], a_x0)['aggL2']:.3e};  eager@x{k} vs eager@x0: {rel_errors(a_e, a_x0)['aggL2']:.3e})")
rows = np.asarray(prep_out.neighbor_list.counts); print("refreshed rows max", rows.max(), "mean", round(float(rows.mean()), 1))
