#!/usr/bin/env python
"""T1.1 follow-up -- is the kernel's neighbour-independent time one serial warp?

Hypothesis from the profiles + the exact budget: the leafpair Pallas kernel runs
one single-warp program per (target leaf, 32-target subtile) that loops over
that leaf's neighbour row.  In every configuration the LONGEST row has
``num_leaves - 1`` entries -- a halo leaf so extended that the mutual MAC makes
it a near neighbour of every other leaf -- so one program serially sums all N
sources: ~N x 95 ns = 19 ms at 200k, 38 ms at 400k, theta- and p-independent.

Test: take the prepared state, truncate every neighbour row to at most ``cap``
valid slots (a *timing* hack -- the force is wrong and is not reported), and
re-time the same eval.  If the kernel loses ~19 ms while dropping <2 % of the
neighbour entries, the intercept is the tail; if the time falls in proportion
to the entries dropped, it is not.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls  # noqa: E402
from compare_force import apply_fast_lane_env  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=256)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--caps", type=int, nargs="+", default=[0, 600, 400, 300, 200])
    args = ap.parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1)
    set_cuda_visible(devices)
    apply_fast_lane_env(args.n)

    import jax
    import jax.numpy as jnp
    from jaccpot import (FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig,
                         NearFieldConfig, TreeConfig)

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)
    s = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
        G=1.0, softening=1e-7, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                                   farfield=FarFieldConfig(mode="auto"),
                                   nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
        fixed_order=args.order)
    prepared, ev = s.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)

    # find the prepacked payload's source-leaf validity array anywhere in the state
    import dataclasses

    def walk(obj, path, out, depth=0):
        if depth > 4:
            return
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            if obj.dtype == jnp.bool_ and obj.ndim >= 2:
                out.append((path, tuple(obj.shape)))
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}", out, depth + 1)
        elif hasattr(obj, "_fields"):
            for f in obj._fields:
                walk(getattr(obj, f), f"{path}.{f}", out, depth + 1)
        elif dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                walk(getattr(obj, f.name), f"{path}.{f.name}", out, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", out, depth + 1)

    cands = []
    walk(prepared, "prepared", cands)
    print("bool arrays (ndim>=2) in prepared state:", flush=True)
    for c in cands:
        print("   ", c, flush=True)

    def get(path):
        obj = prepared
        for part in path.split(".")[1:]:
            if "[" in part:
                name, idx = part[:-1].split("[")
                obj = getattr(obj, name)[int(idx)] if name else obj[int(idx)]
            else:
                obj = getattr(obj, part)
        return obj

    def replace(path, new):
        parts = path.split(".")[1:]

        def rep(obj, parts, new):
            if not parts:
                return new
            head, rest = parts[0], parts[1:]
            child = getattr(obj, head)
            new_child = rep(child, rest, new)
            if hasattr(obj, "_replace"):
                return obj._replace(**{head: new_child})
            if dataclasses.is_dataclass(obj):
                return dataclasses.replace(obj, **{head: new_child})
            raise TypeError(f"cannot replace field {head} on {type(obj)}")

        return rep(prepared, parts, new)

    num_leaves = int(prepared.nearfield_leaf_particle_mask.shape[0])
    valid_path = None
    for name, shape in cands:
        if "valid" in name.lower() and shape[0] == num_leaves and len(shape) == 3:
            valid_path = name
    if valid_path is None:
        raise SystemExit("could not find the source-leaf validity array; see list above")
    valid_arr = np.asarray(get(valid_path))
    valid0 = valid_arr.reshape(num_leaves, -1)
    counts = valid0.sum(axis=1)
    print(f"using {valid_path} shape {valid_arr.shape}; rows: mean {counts.mean():.1f} "
          f"max {counts.max()} entries {counts.sum()}", flush=True)

    base_ms = None
    for cap in args.caps:
        if cap <= 0:
            v = valid0
        else:
            keep = np.cumsum(valid0, axis=1) <= cap
            v = valid0 & keep
        dropped = int(valid0.sum() - v.sum())
        st = replace(valid_path, jnp.asarray(v.reshape(valid_arr.shape)))
        _, t, cont = timed_calls(lambda: ev(st), repeats=5, warmup=2,  # noqa: B023
                                 devices=devices, block=jax.block_until_ready)
        ms = t["min"] * 1e3
        if base_ms is None:
            base_ms = ms
        print(f"cap {cap:>5}: {ms:8.2f} ms  (-{base_ms-ms:6.2f})  entries dropped {dropped:>7} "
              f"({100*dropped/max(1,valid0.sum()):.1f} %)  longest row {int(v.sum(1).max())}  "
              f"flags={cont.flags or '-'}", flush=True)


if __name__ == "__main__":
    main()
