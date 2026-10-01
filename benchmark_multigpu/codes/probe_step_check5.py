#!/usr/bin/env python
"""After ONE real step (dt=1e-2), is the refreshed state right at the new positions?"""
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
import jaccpot; print("jaccpot", jaccpot.__file__)
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
    softening=1e-7, working_dtype=jnp.float32,
    advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
        farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
    fixed_order=ORDER)
prep0, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
dt = 1e-2
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
st1, prep1, _ = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=1, refresh_every=1, leaf_size=LEAF,
                                max_order=ORDER, theta=THETA, prepared_state=None, return_prepared_state=True)
x1 = np.asarray(st1[:, 0, :], np.float64)
ref1 = direct_accelerations(x1, mass, G=1.0, softening=1e-7, block_size=2048)
a_state = np.asarray(jax.block_until_ready(ev(prep1)), np.float64)
print(f"eval_fn(refreshed state after 1 real step) vs direct@x1: aggL2 {rel_errors(a_state, ref1)['aggL2']:.3e}")
# what positions does the refreshed state hold, in which order?
t0, t1 = prep0.tree, prep1.tree
for name in ("positions_sorted", "inverse_permutation", "permutation", "sort_indices"):
    a = getattr(t0, name, None); b = getattr(t1, name, None)
    if a is not None and b is not None:
        a = np.asarray(a); b = np.asarray(b)
        print(f"  tree.{name}: shape {a.shape}; changed entries {int(np.sum(np.any(a != b, axis=-1)) if a.ndim>1 else np.sum(a != b))}")
ps1 = np.asarray(getattr(t1, "positions_sorted", None), np.float64)
inv1 = getattr(t1, "inverse_permutation", None)
if inv1 is not None:
    inv1 = np.asarray(inv1)
    back = ps1[inv1]
    print(f"  positions_sorted[inverse_permutation] vs x1: max |diff| {np.abs(back - x1).max():.3e}")
    inv0 = np.asarray(t0.inverse_permutation)
    back0 = ps1[inv0]
    print(f"  positions_sorted[OLD inverse_permutation] vs x1: max |diff| {np.abs(back0 - x1).max():.3e}")
# eager rebuild at x1 for comparison
prep_e, ev_e = s.strict_fused_prepared_eval_fn(positions=jnp.asarray(x1, jnp.float32), masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
a_e = np.asarray(jax.block_until_ready(ev_e(prep_e)), np.float64)
print(f"eval_fn(eager prepare at x1) vs direct@x1: aggL2 {rel_errors(a_e, ref1)['aggL2']:.3e}")
te = prep_e.tree
print("  eager@x1 vs refreshed: leaf count", np.asarray(te.leaf_codes).shape, np.asarray(t1.leaf_codes).shape,
      "| inverse_permutation equal:", np.array_equal(np.asarray(te.inverse_permutation), np.asarray(t1.inverse_permutation)))
