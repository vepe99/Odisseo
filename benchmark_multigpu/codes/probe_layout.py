#!/usr/bin/env python
"""Why is the SAME near-field kernel 5.6x faster inside strict_run_v2 than in eval_fn?

Compares the radix fast payload of the eval-seam prepared state with the one the
production scan carries (row order, row-length order, shapes), then times the
eval closure on BOTH states.  If eval_fn(scan_state) is fast, the layout is the
whole difference -- and it is the T1.2 lever, already in the codebase.
"""
import os, sys, time
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
import dataclasses
from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls
from compare_force import apply_fast_lane_env
from common.ic import IC_GENERATORS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
THETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
LEAF = 256; ORDER = 4
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = pick_idle_gpus(1); set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=THETA, G=1.0,
    softening=1e-7, working_dtype=jnp.float32,
    advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
        farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
    fixed_order=ORDER)
prep_eval, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=THETA)
state0 = jnp.stack([P, jnp.zeros((N, 3), jnp.float32)], axis=1)
out = s.strict_run_v2(state=state0, masses=M, dt=1e-6, num_steps=1, refresh_every=1, leaf_size=LEAF,
                      max_order=ORDER, theta=THETA, prepared_state=None, return_prepared_state=True)
prep_scan = out[1]

def describe(name, prep):
    pl = prep.radix_fast_payload
    ids = np.asarray(pl.source_leaf_ids); valid = np.asarray(pl.source_leaf_valid_mask)
    tl = np.asarray(pl.target_leaf_ids)
    rows = valid.reshape(valid.shape[0], -1).sum(1)
    diffs = np.diff(rows)
    order = "descending" if np.all(diffs <= 0) else ("ascending" if np.all(diffs >= 0) else "unsorted")
    print(f"{name}: source_leaf_ids {ids.shape} valid {valid.shape} target_leaf_ids {tl.shape} "
          f"(identity: {np.array_equal(tl, np.arange(tl.shape[0]))}) rows mean {rows.mean():.1f} max {rows.max()} "
          f"order {order}; first rows {rows[:6].tolist()} last {rows[-6:].tolist()}")
    names = [f.name for f in dataclasses.fields(pl)] if dataclasses.is_dataclass(pl) else list(getattr(pl, '_fields', []))
    extra = {k: (tuple(getattr(pl, k).shape) if hasattr(getattr(pl, k), 'shape') else getattr(pl, k))
             for k in names if k not in ('source_leaf_ids', 'source_leaf_valid_mask', 'target_leaf_ids',
                                         'target_particle_ids', 'target_particle_mask')}
    print("   other payload fields:", extra)
    for k in ("nearfield_target_block_leaf_ids", "nearfield_target_block_source_leaf_ids",
              "nearfield_target_block_valid_mask", "nearfield_target_block_offsets",
              "nearfield_target_block_source_leaf_ids_padded", "nearfield_target_block_valid_mask_padded"):
        v = getattr(prep, k, None)
        if v is not None and hasattr(v, "shape"):
            print(f"   {k}: {tuple(v.shape)}")
    return rows

r1 = describe("eval-seam state", prep_eval)
r2 = describe("scan state     ", prep_scan)
for name, prep in (("eval-seam state", prep_eval), ("scan state", prep_scan)):
    try:
        _, t, cont = timed_calls(lambda: ev(prep), repeats=5, warmup=2, devices=devices, block=jax.block_until_ready)
        print(f"eval_fn({name}): {t['min']*1e3:.2f} ms  flags={cont.flags or '-'}")
    except Exception as e:
        print(f"eval_fn({name}) FAILED: {str(e).splitlines()[0][:160]}")
