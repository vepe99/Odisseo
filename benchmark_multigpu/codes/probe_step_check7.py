#!/usr/bin/env python
"""Does an eager eval-seam call before strict_run_v2 change the in-scan force?"""
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
devices = [int(os.environ["BENCH_FORCE_GPU"])] if os.environ.get("BENCH_FORCE_GPU") else pick_idle_gpus(1); set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
import jaccpot; print("jaccpot", jaccpot.__file__)
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
def solver():
    return FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
        softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=ORDER)
s0 = solver()
prep, ev = s0.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
a_ref = np.asarray(jax.block_until_ready(ev(prep)), np.float64)
dt = 1e-2
def run3(s, label):
    state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
    _, prep_out, hist = s.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=3, refresh_every=1, leaf_size=LEAF,
                                        max_order=ORDER, theta=THETA, prepared_state=None,
                                        return_prepared_state=True, return_history=True)
    h = np.asarray(hist, np.float64)
    xs = [pos.astype(np.float64)] + [h[k, :, 0, :] for k in range(h.shape[0])]
    errs = [rel_errors(2 * (xs[1] - xs[0]) / dt**2, a_ref)["aggL2"]]
    for k in range(1, len(xs) - 1):
        errs.append(rel_errors((xs[k + 1] - 2 * xs[k] + xs[k - 1]) / dt**2, a_ref)["aggL2"])
    rows = np.asarray(prep_out.neighbor_list.counts)
    print(f"{label}: step forces vs eager@x0 {['%.3e' % e for e in errs]}  refreshed rows max {rows.max()} mean {rows.mean():.1f}", flush=True)
    return prep_out
# A: fresh solver, strict_run_v2 directly
run3(solver(), "A fresh -> strict_run_v2")
# B: fresh solver, eval seam prepare (no evaluate call) -> strict_run_v2
sB = solver(); pB, evB = sB.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
run3(sB, "B eval-seam prepare only -> strict_run_v2")
# C: fresh solver, eval seam prepare + one evaluate -> strict_run_v2
sC = solver(); pC, evC = sC.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
jax.block_until_ready(evC(pC))
run3(sC, "C eval-seam prepare + evaluate -> strict_run_v2")
# D: fresh solver, strict_run_v2 with prepared_state from the eval seam
sD = solver(); pD, evD = sD.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
_, pout, hist = sD.strict_run_v2(state=state0, masses=M, dt=dt, num_steps=3, refresh_every=1, leaf_size=LEAF,
                                 max_order=ORDER, theta=THETA, prepared_state=pD, return_prepared_state=True, return_history=True)
h = np.asarray(hist, np.float64); xs = [pos.astype(np.float64)] + [h[k, :, 0, :] for k in range(3)]
errs = [rel_errors(2 * (xs[1] - xs[0]) / dt**2, a_ref)["aggL2"]] + [rel_errors((xs[k + 1] - 2 * xs[k] + xs[k - 1]) / dt**2, a_ref)["aggL2"] for k in (1, 2)]
print(f"D eval-seam state passed in -> strict_run_v2: {['%.3e' % e for e in errs]}")
