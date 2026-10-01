#!/usr/bin/env python
"""Is the force on the SECOND strict_run_v2 step (refreshed state) still right?

Two Verlet steps from rest: a1 = (x2 - 2 x1 + x0) / dt^2 is the force the
refreshed state produced.  Also: is the compact far-pair list at its cap?
"""
import os, sys
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible
from compare_force import apply_fast_lane_env, rel_errors
from common.ic import IC_GENERATORS
N = 200_000; LEAF = 256; ORDER = 4
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = pick_idle_gpus(1); set_cuda_visible(devices); apply_fast_lane_env(N)
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1); os.environ[k] = v
print("env overrides:", sys.argv[1:], "| MAX_MB", os.environ.get("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB"),
      "| MAX_PER_LEAF", os.environ.get("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"), flush=True)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
ref = direct_accelerations(pos, mass, G=1.0, softening=1e-7, block_size=2048)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
def solver(theta):
    return FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
for theta in (0.6, 1.0):
    s = solver(theta)
    dt = 1e-2
    state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
    st1, prep1, _ = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=1, refresh_every=1, leaf_size=LEAF,
                                    max_order=ORDER, theta=theta, prepared_state=None, return_prepared_state=True)
    st2, prep2, _ = s.strict_run_v2(state=st1, masses=M, dt=dt, num_steps=1, refresh_every=1, leaf_size=LEAF,
                                    max_order=ORDER, theta=theta, prepared_state=prep1, return_prepared_state=True)
    x0 = pos.astype(np.float64); x1 = np.asarray(st1[:, 0, :], np.float64); x2 = np.asarray(st2[:, 0, :], np.float64)
    a0 = 2 * (x1 - x0) / dt**2
    a1 = (x2 - 2 * x1 + x0) / dt**2
    d = s.get_runtime_diagnostics() or {}
    fp = int(np.asarray(prep1.compact_far_pairs.far_pair_count))
    print(f"theta {theta}: step-1 force (fresh prepare) vs ref aggL2 {rel_errors(a0, ref)['aggL2']:.3e}; "
          f"step-2 force (refreshed state) vs ref aggL2 {rel_errors(a1, ref)['aggL2']:.3e}")
    print(f"   refreshed state: neighbor rows max {int(np.asarray(prep2.neighbor_list.counts).max())} "
          f"mean {float(np.asarray(prep2.neighbor_list.counts).mean()):.1f}; far_pair_count {fp} "
          f"(buffer {prep1.compact_far_pairs.sources.shape[0]}); fallback_count {d.get('strict_fused_fallback_count')} "
          f"reason {d.get('strict_fused_last_fallback_reason')!r}")
# far-pair cap check on the eval seam (only in the default run)
os.environ["JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP"] = "1048576"
for theta in (() if sys.argv[1:] else (0.6, 1.0)):
    s = solver(theta)
    prep, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=theta)
    a = np.asarray(jax.block_until_ready(ev(prep)))
    fp = int(np.asarray(prep.compact_far_pairs.far_pair_count))
    print(f"theta {theta} with far cap 2^20: far_pair_count {fp} (buffer {prep.compact_far_pairs.sources.shape[0]}) "
          f"aggL2 {rel_errors(a, ref)['aggL2']:.3e}")
