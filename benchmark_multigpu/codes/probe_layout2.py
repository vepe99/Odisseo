#!/usr/bin/env python
"""Is eval_fn(scan state) CORRECT?  And where do the two states' interaction lists differ?"""
import os, sys, dataclasses
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls
from compare_force import apply_fast_lane_env, rel_errors
from common.budget import jaccpot_direct_budget
from common.ic import IC_GENERATORS
N = 200_000; THETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6; LEAF = 256; ORDER = 4
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
prep_eval, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
a_eval = np.asarray(jax.block_until_ready(ev(prep_eval)))
d1 = dict(s.get_runtime_diagnostics() or {})
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
out = s.strict_run_v2(state=state0, masses=M, dt=1e-6, num_steps=1, refresh_every=1, leaf_size=LEAF,
                      max_order=ORDER, theta=THETA, prepared_state=None, return_prepared_state=True)
prep_scan = out[1]
d2 = dict(s.get_runtime_diagnostics() or {})
a_scan = np.asarray(jax.block_until_ready(ev(prep_scan)))
print(f"theta {THETA}: eval_fn(eval state) aggL2 {rel_errors(a_eval, ref)['aggL2']:.3e}")
print(f"           eval_fn(scan state) aggL2 {rel_errors(a_scan, ref)['aggL2']:.3e}   vs eval-state {rel_errors(a_scan, a_eval)['aggL2']:.3e}")
for name, prep in (("eval", prep_eval), ("scan", prep_scan)):
    b = jaccpot_direct_budget(prep, N)
    pl = prep.radix_fast_payload
    rows = np.asarray(pl.source_leaf_valid_mask).reshape(782, -1).sum(1)
    tb = np.asarray(prep.nearfield_target_block_valid_mask_padded).reshape(782, -1).sum(1)
    print(f"  {name}: walk CSR entries {b['neighbor_entries_directed']} (mean row {b['mean_neighbor_leaves_per_leaf']:.1f}, max {b['max_neighbor_leaves_per_leaf']}); "
          f"payload rows mean {rows.mean():.1f} max {rows.max()}; target_block_padded rows mean {tb.mean():.1f} max {tb.max()}")
    far = [f.name for f in dataclasses.fields(prep)] if dataclasses.is_dataclass(prep) else list(prep._fields)
    for k in far:
        if "far" in k.lower():
            v = getattr(prep, k)
            if hasattr(v, "shape"): print(f"      {k}: {tuple(v.shape)}")
            elif dataclasses.is_dataclass(v) or hasattr(v, "_fields"):
                sub = [f.name for f in dataclasses.fields(v)] if dataclasses.is_dataclass(v) else v._fields
                print(f"      {k}: " + ", ".join(f"{n}={tuple(getattr(v,n).shape) if hasattr(getattr(v,n),'shape') else getattr(v,n)}" for n in sub))
print("diag after eval prepare:", {k: d1.get(k) for k in ('recent_dual_neighbor_count','recent_dual_far_pair_count','static_radix_far_pair_count')})
print("diag after scan prepare:", {k: d2.get(k) for k in ('recent_dual_neighbor_count','recent_dual_far_pair_count','static_radix_far_pair_count')})
for name, prep in (("eval", prep_eval), ("scan", prep_scan)):
    _, t, cont = timed_calls(lambda: ev(prep), repeats=5, warmup=2, devices=devices, block=jax.block_until_ready)
    print(f"  eval_fn({name} state): {t['min']*1e3:.2f} ms  flags={cont.flags or '-'}")
