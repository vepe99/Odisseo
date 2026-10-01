#!/usr/bin/env python
"""Final confirmation: within ONE strict_run_v2 call of 2 steps (history on), is the
step-2 force right?  Plus the traversal cap / overflow diagnostics."""
import os, sys
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible
from compare_force import apply_fast_lane_env, rel_errors
from common.ic import IC_GENERATORS
N = 200_000; LEAF = 256; ORDER = 4; THETA = 1.0
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = pick_idle_gpus(1); set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
ref = direct_accelerations(pos, mass, G=1.0, softening=1e-7, block_size=2048)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
    softening=1e-7, working_dtype=jnp.float32,
    advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
        farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
    fixed_order=ORDER)
dt = 1e-2
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
final, prep, hist = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=3, refresh_every=1, leaf_size=LEAF,
                                    max_order=ORDER, theta=THETA, prepared_state=None,
                                    return_prepared_state=True, return_history=True)
hist = np.asarray(hist, np.float64)  # [steps, N, 2, 3] presumably
print("history shape", hist.shape)
xs = [pos.astype(np.float64)] + [hist[k, :, 0, :] for k in range(hist.shape[0])]
a0 = 2 * (xs[1] - xs[0]) / dt**2
print(f"one call, 3 steps, theta {THETA}: step-1 force vs ref aggL2 {rel_errors(a0, ref)['aggL2']:.3e}")
for k in range(1, len(xs) - 1):
    ak = (xs[k + 1] - 2 * xs[k] + xs[k - 1]) / dt**2
    print(f"   step-{k+1} force (second difference) vs ref aggL2 {rel_errors(ak, ref)['aggL2']:.3e}")
d = s.get_runtime_diagnostics() or {}
print("caps/overflow diagnostics:", {k: v for k, v in d.items() if ("overflow" in k or "profiled" in k or "near_cap" in k.lower() or "neighbor" in k) and not isinstance(v, (list, tuple, dict))})
print("refreshed rows: max", int(np.asarray(prep.neighbor_list.counts).max()), "mean", float(np.asarray(prep.neighbor_list.counts).mean()))
