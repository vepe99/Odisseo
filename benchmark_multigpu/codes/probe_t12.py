#!/usr/bin/env python
"""T1.2 gate: the two near-field levers must leave aggL2 unchanged (4 s.f.) at
theta 0.4 and 0.8 and cut the theta=1.0 time.  A/B per env setting, same GPU,
same process (fresh solver per setting; kernels differ by name so no cache clash)."""
import os, sys, time, json
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls
from compare_force import apply_fast_lane_env, rel_errors
from common.ic import IC_GENERATORS
N = int(os.environ.get("PROBE_N", "200000")); LEAF = int(os.environ.get("PROBE_LEAF", "256")); ORDER = 4
THETAS = [float(t) for t in os.environ.get("PROBE_THETAS", "0.4 0.6 0.8 1.0").split()]
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = [int(os.environ["BENCH_FORCE_GPU"])] if os.environ.get("BENCH_FORCE_GPU") else pick_idle_gpus(1)
set_cuda_visible(devices); apply_fast_lane_env(N)
import jax, jax.numpy as jnp
import jaccpot; print("jaccpot", jaccpot.__file__, "| GPU", devices, flush=True)
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
from common.reference import direct_accelerations
pos, mass = IC_GENERATORS["plummer"](N, seed=0)
ref = None if os.environ.get("PROBE_NO_REF") else direct_accelerations(pos, mass, G=1.0, softening=1e-7, block_size=2048)
M = jnp.asarray(mass, jnp.float32); P = jnp.asarray(pos, jnp.float32)
SETTINGS = [("baseline", {"JACCPOT_NEARFIELD_LEAFPAIR_SOURCE_CHUNK": "0", "JACCPOT_NEARFIELD_SELF_BATCH": "1"}),
            ("chunk64", {"JACCPOT_NEARFIELD_LEAFPAIR_SOURCE_CHUNK": "64", "JACCPOT_NEARFIELD_SELF_BATCH": "1"}),
            ("batch32", {"JACCPOT_NEARFIELD_LEAFPAIR_SOURCE_CHUNK": "0", "JACCPOT_NEARFIELD_SELF_BATCH": "32"}),
            ("both", {"JACCPOT_NEARFIELD_LEAFPAIR_SOURCE_CHUNK": "64", "JACCPOT_NEARFIELD_SELF_BATCH": "32"})]
if os.environ.get("PROBE_SETTINGS"):
    keep = os.environ["PROBE_SETTINGS"].split(); SETTINGS = [s_ for s_ in SETTINGS if s_[0] in keep]
rows = []
print(f"{'setting':>9} {'theta':>5} {'ms':>8} {'iqr':>6} {'aggL2':>11} {'flags'}")
for theta in THETAS:
    base_acc = None
    for name, env in SETTINGS:
        for k, v in env.items(): os.environ[k] = v
        s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=theta, G=1.0,
            softening=1e-7, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
                farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
            fixed_order=ORDER)
        prep, ev = s.strict_fused_prepared_eval_fn(positions=P, masses=M, leaf_size=LEAF, max_order=ORDER, theta=theta)
        out, t, cont = timed_calls(lambda: ev(prep), repeats=5, warmup=2, devices=devices, block=jax.block_until_ready)
        a = np.asarray(out, np.float64)
        e = rel_errors(a, ref)["aggL2"] if ref is not None else float("nan")
        if base_acc is None: base_acc = a
        dev = rel_errors(a, base_acc)["aggL2"]
        rows.append(dict(setting=name, theta=theta, ms=t["min"]*1e3, iqr=t["iqr"]*1e3, aggL2=e, vs_baseline=dev, flags=cont.flags))
        print(f"{name:>9} {theta:>5.2f} {t['min']*1e3:>8.2f} {t['iqr']*1e3:>6.2f} {e:>11.4e}  vs-baseline {dev:.2e}  {','.join(cont.flags) or '-'}", flush=True)
        del s, prep, ev
Path(ROOT / "artifacts" / f"probe_t12_plummer{N}_leaf{LEAF}.json").write_text(json.dumps(dict(devices=devices, rows=rows), indent=2))
