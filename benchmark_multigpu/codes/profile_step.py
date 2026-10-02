#!/usr/bin/env python
"""Profile jaccpot's PRODUCTION step (strict_run_v2) kernel by kernel.

strict_run_v2 computes the same force as eval_fn (probe_step_check.py) in 46 ms
where eval_fn takes 90 ms at theta 0.6 -- so the two paths run different
near-field layouts.  This traces one warm multi-step call and ranks device
kernels per step, reusing profile_eval.analyse.
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls
from compare_force import apply_fast_lane_env
from common.ic import IC_GENERATORS
from profile_eval import analyse, load_perfetto

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200_000); ap.add_argument("--leaf", type=int, default=256)
ap.add_argument("--theta", type=float, default=0.6); ap.add_argument("--order", type=int, default=4)
ap.add_argument("--steps", type=int, default=5); ap.add_argument("--vel-sigma", type=float, default=0.4)
ap.add_argument("--dt", type=float, default=0.01)
args = ap.parse_args()
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
devices = pick_idle_gpus(1); set_cuda_visible(devices); apply_fast_lane_env(args.n)
import jax, jax.numpy as jnp
from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig, TreeConfig
pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
M = jnp.asarray(mass, jnp.float32)
vel = np.random.default_rng(7).normal(0.0, args.vel_sigma, (args.n, 3)).astype(np.float32)
state = jnp.stack([jnp.asarray(pos, jnp.float32), jnp.asarray(vel)], axis=1)
s = FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta, G=1.0,
    softening=1e-7, working_dtype=jnp.float32,
    advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
        farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), mac_type="dehnen"),
    fixed_order=args.order)
def run(state, prepared, steps):
    out = s.strict_run_v2(state=state, masses=M, dt=args.dt, num_steps=steps, refresh_every=1,
                          leaf_size=args.leaf, max_order=args.order, theta=args.theta,
                          prepared_state=prepared, return_prepared_state=True)
    jax.block_until_ready(out[0]); return out
state, prepared, _ = run(state, None, 1)
state, prepared, _ = run(state, prepared, args.steps)   # compile the K-step scan
t0 = time.perf_counter(); state, prepared, _ = run(state, prepared, args.steps); t_warm = time.perf_counter() - t0
print(f"warm {args.steps}-step call: {t_warm*1e3/args.steps:.1f} ms/step", flush=True)
tag = f"step_plummer{args.n}_leaf{args.leaf}_th{args.theta:g}_p{args.order}"
trace_dir = ROOT / "artifacts" / "traces" / tag; trace_dir.mkdir(parents=True, exist_ok=True)
with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
    t0 = time.perf_counter(); state, prepared, _ = run(state, prepared, args.steps); t_tr = time.perf_counter() - t0
print(f"traced {args.steps}-step call: {t_tr*1e3/args.steps:.1f} ms/step", flush=True)
res = analyse(load_perfetto(trace_dir), args.steps)
pc = res["per_call"]
print(f"per step: window {pc['window_ms']:.2f} ms, device busy {pc['busy_ms']:.2f} ms "
      f"({100*pc['busy_ms']/max(1e-9,pc['window_ms']):.0f} %), kernel-sum {pc['kernel_sum_ms']:.2f} ms, launches {pc['launches']:.0f}")
print(f"{'ms/step':>8} {'share':>6} {'n/step':>7} {'mean us':>8}  kernel")
for k in res["top"][:25]:
    print(f"{k['total_ms_per_call']:>8.3f} {100*k['share']:>5.1f}% {k['count_per_call']:>7.1f} {k['mean_us']:>8.1f}  {k['name'][:100]}")
res.update(config=vars(args), warm_ms_per_step=t_warm*1e3/args.steps)
(ROOT / "artifacts" / f"profile_{tag}.json").write_text(json.dumps(res, indent=2))
