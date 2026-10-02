#!/usr/bin/env python
"""Does strict_run_v2's per-step force equal eval_fn's?  (T3.0 sanity check)

strict_run_v2 came out at 46 ms/step at theta 0.6 where eval_fn alone is 90 ms.
Either the scan path uses a different (better-balanced) near-field layout, or it
drops neighbours.  One velocity-Verlet step from rest gives
``a = 2 (x1 - x0) / dt^2``; compare it with eval_fn's acceleration and with the
float64 direct sum.
"""
import os, sys, time
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible
from compare_force import apply_fast_lane_env, rel_errors
from common.ic import IC_GENERATORS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
LEAF = 256; ORDER = 4
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
set_cuda_visible(pick_idle_gpus(1)); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
ref = direct_accelerations(pos, mass, G=1.0, softening=1e-7, block_size=2048)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
for theta in (0.6, 1.0):
    s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
    prepared, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=theta)
    a_eval = np.asarray(jax.block_until_ready(ev(prepared)), np.float64)
    dt = 1e-2
    state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
    # positions in float32 limit the recovered acceleration to ~1e-3 relative; use the
    # velocity instead: after one kick-drift-kick from rest v1 = 0.5*dt*(a0 + a1) ~ dt*a0
    out = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=1, refresh_every=1,
                          leaf_size=LEAF, max_order=ORDER, theta=theta, prepared_state=None,
                          return_prepared_state=True)
    st = np.asarray(jax.block_until_ready(out[0]), np.float64)
    a_from_v = st[:, 1, :] / dt          # 0.5*(a0+a1), a1 ~ a0 for a tiny dt
    a_from_x = 2.0 * (st[:, 0, :] - pos.astype(np.float64)) / dt**2
    print(f"theta {theta}: eval_fn vs ref  aggL2 {rel_errors(a_eval, ref)['aggL2']:.3e}")
    print(f"           step(v) vs ref  aggL2 {rel_errors(a_from_v, ref)['aggL2']:.3e}   step(v) vs eval_fn {rel_errors(a_from_v, a_eval)['aggL2']:.3e}")
    print(f"           step(x) vs ref  aggL2 {rel_errors(a_from_x, ref)['aggL2']:.3e}")
    d = s.get_runtime_diagnostics() or {}
    print("           diag:", {k: d.get(k) for k in ("strict_fused_fastlane_hits", "strict_fused_fallback_count",
          "strict_fused_last_fallback_reason", "recent_dual_neighbor_count", "large_n_eval_active_leaf_count")})
