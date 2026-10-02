#!/usr/bin/env python
"""Bisect the in-scan wrong force: (1) jit(refresh+evaluate) outside the scan;
(2) scan with the near field zeroed (far only) and with pairs only."""
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
from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
def solver():
    return FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
dt = 1e-2
s = solver(); e = s._impl
prep0, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
a0 = np.asarray(jax.block_until_ready(ev(prep0)), np.float64)
x1 = pos.astype(np.float64) + 0.5 * a0 * dt**2          # exact first Verlet position
X1 = jnp.asarray(x1, jnp.float32)
prep_e, ev_e = s.strict_fused_prepared_eval_fn(positions=X1, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
a_e = np.asarray(jax.block_until_ready(ev_e(prep_e)), np.float64)   # eager at x1 (reference for this probe)

# (1) traced refresh + evaluate in ONE jit, exactly as the scan does
def refresh_eval(p, x):
    pn = e._refresh_large_n_same_topology(p, x, M, bounds=None, leaf_size=LEAF, max_order=ORDER, theta=THETA,
                                          runtime_overrides_override=None, fused_device_mode=True)
    return jnp.asarray(evaluate_large_n_state(e, pn, target_indices=None, return_potential=False, max_acc_derivative_order=0))
a_jit = np.asarray(jax.block_until_ready(jax.jit(refresh_eval)(prep0, X1)), np.float64)
print(f"(1) jit(refresh+evaluate)(prep0, x1) vs eager@x1: aggL2 {rel_errors(a_jit, a_e)['aggL2']:.3e}")
# eager refresh + eager evaluate
pn = e._refresh_large_n_same_topology(prep0, X1, M, bounds=None, leaf_size=LEAF, max_order=ORDER, theta=THETA,
                                      runtime_overrides_override=None, fused_device_mode=True)
a_eag = np.asarray(jax.block_until_ready(evaluate_large_n_state(e, pn, target_indices=None, return_potential=False, max_acc_derivative_order=0)), np.float64)
print(f"    eager refresh + eager evaluate vs eager@x1: aggL2 {rel_errors(a_eag, a_e)['aggL2']:.3e}")
# traced refresh, then eager evaluate of its (returned) state
pn_jit = jax.jit(lambda p, x: e._refresh_large_n_same_topology(p, x, M, bounds=None, leaf_size=LEAF, max_order=ORDER, theta=THETA, runtime_overrides_override=None, fused_device_mode=True))(prep0, X1)
a_mix = np.asarray(jax.block_until_ready(ev(pn_jit)), np.float64)
print(f"    jit(refresh) then eager evaluate vs eager@x1: aggL2 {rel_errors(a_mix, a_e)['aggL2']:.3e}")

# (2) in-scan bisection by near-field diag mode
for mode in ("zero", "pairs_only"):
    s2 = solver(); s2._impl._large_n_nearfield_diag_mode = mode
    p2, ev2 = s2.strict_fused_prepared_eval_fn(positions=X1, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
    a_ref_mode = np.asarray(jax.block_until_ready(ev2(p2)), np.float64)      # eager component at x1
    state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
    _, _, hist = s2.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=2, refresh_every=1, leaf_size=LEAF,
                                  max_order=ORDER, theta=THETA, prepared_state=None, return_prepared_state=True, return_history=True)
    h = np.asarray(hist, np.float64)
    xs = [pos.astype(np.float64), h[0, :, 0, :], h[1, :, 0, :]]
    a1_scan = (xs[2] - 2 * xs[1] + xs[0]) / dt**2
    print(f"(2) nearfield diag '{mode}': in-scan step-2 force vs eager same-mode@x1: aggL2 {rel_errors(a1_scan, a_ref_mode)['aggL2']:.3e}  (|eager comp|/|full| = {np.linalg.norm(a_ref_mode)/np.linalg.norm(a_e):.3f})")
