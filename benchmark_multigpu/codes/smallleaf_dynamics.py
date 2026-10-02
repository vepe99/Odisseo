#!/usr/bin/env python
"""Phase 3.3: does the fold + CSR M2L lane bias a long strict_run_v2 rollout?

Runs ``--steps`` velocity-Verlet steps of strict_run_v2 (refresh every step) at
one leaf size with the CSR M2L lane ON and OFF from the same initial state, and
reports for each: kinetic-energy drift, |v| max, angular-momentum drift, plus the
force the scan applied at the FINAL positions recovered from the trajectory
against an eager prepare+evaluate at those same positions (the only legitimate
force check -- see tests/integration/test_strict_run_v2_refresh_capacity.py),
and the position/velocity divergence between the two lanes. Two lanes computing
the same operator to fp32 summation order should diverge chaotically but keep
identical drift statistics; a bias shows up as a systematic drift difference.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__))); ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import numpy as np
from common.gpu_guard import pick_idle_gpus, set_cuda_visible, GpuMonitor
from common.ic import IC_GENERATORS
from compare_force import FAST_LANE_ENV_BY_LEAF, apply_fast_lane_env, fast_lane_overrides_for_leaf


def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--vel-sigma", type=float, default=0.4)
    ap.add_argument("--flag", default="JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
                    help="env flag toggled 0/1 between the two lanes")
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL", help="extra env for both lanes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1); set_cuda_visible(devices)
    extra = dict(kv.split("=", 1) for kv in args.env)
    apply_fast_lane_env(args.n, overrides={**fast_lane_overrides_for_leaf(args.leaf, args.n, extra), **extra})
    trav = dict((FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {}))

    import jax, jax.numpy as jnp
    from jaccpot import (FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig, NearFieldConfig,
                         RuntimePolicyConfig, TraversalOverrides, TreeConfig)

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    vel = np.random.default_rng(7).normal(0.0, args.vel_sigma, (args.n, 3)).astype(np.float32)
    M = jnp.asarray(mass, jnp.float32)
    state0 = jnp.stack([jnp.asarray(pos, jnp.float32), jnp.asarray(vel)], axis=1)
    rt = RuntimePolicyConfig(traversal_config=TraversalOverrides(**{k: int(v) for k, v in trav.items()})) if trav else RuntimePolicyConfig()

    def make():
        return FastMultipoleMethod(preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
            G=1.0, softening=1e-7, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"), runtime=rt, mac_type="dehnen"),
            fixed_order=args.order)

    def stats(st):
        x = np.asarray(st[:, 0, :], np.float64); v = np.asarray(st[:, 1, :], np.float64); m = np.asarray(mass, np.float64)
        ke = 0.5 * np.sum(m[:, None] * v * v); L = np.sum(m[:, None] * np.cross(x, v), axis=0)
        return dict(KE=float(ke), Lz=float(L[2]), vmax=float(np.abs(v).max()), com=np.sum(m[:, None] * x, axis=0).tolist())

    out = dict(n=args.n, leaf=args.leaf, theta=args.theta, order=args.order, steps=args.steps, dt=args.dt,
               flag=args.flag, env=extra, lanes={})
    finals = {}
    for csr in ("0", "1"):
        os.environ[args.flag] = csr
        s = make()
        with GpuMonitor(devices) as mon:
            t0 = time.perf_counter()
            final, prepared, hist = s.strict_run_v2(state=state0, masses=M, dt=args.dt, num_steps=args.steps, refresh_every=1,
                leaf_size=args.leaf, max_order=args.order, theta=args.theta, prepared_state=None,
                return_prepared_state=True, return_history=True)
            jax.block_until_ready(final); wall = time.perf_counter() - t0
        hist = np.asarray(hist, np.float64)  # (steps, n, 2, 3)
        xs = [np.asarray(pos, np.float64)] + [hist[k, :, 0, :] for k in range(hist.shape[0])]
        a_last = (xs[-1] - 2.0 * xs[-2] + xs[-3]) / args.dt ** 2  # force applied at x_{S-1}
        p_k, ev_k = s.strict_fused_prepared_eval_fn(positions=jnp.asarray(xs[-2], jnp.float32), masses=M,
                                                     leaf_size=args.leaf, max_order=args.order, theta=args.theta)
        a_eager = np.asarray(jax.block_until_ready(ev_k(p_k)), np.float64)
        s0, s1 = stats(np.asarray(state0)), stats(np.asarray(final))
        d = dict(s.get_runtime_diagnostics() or {})
        lane = dict(wall_s=wall, ms_per_step=1e3 * wall / args.steps, contention=mon.summary().as_dict(),
                    KE0=s0["KE"], KE1=s1["KE"], dKE_rel=(s1["KE"] - s0["KE"]) / s0["KE"], Lz0=s0["Lz"], Lz1=s1["Lz"],
                    dLz=s1["Lz"] - s0["Lz"], vmax0=s0["vmax"], vmax1=s1["vmax"],
                    force_recovered_vs_eager_rel_l2=_rel(a_last, a_eager), fallbacks=d.get("strict_fused_fallback_count"))
        out["lanes"][f"csr{csr}"] = lane
        finals[csr] = (np.asarray(final, np.float64), hist)
        print(f"{args.flag.split('_')[-1]}={csr}: {lane['ms_per_step']:.1f} ms/step  dKE/KE {lane['dKE_rel']:+.3e}  dLz {lane['dLz']:+.3e}  "
              f"vmax {s0['vmax']:.3f}->{s1['vmax']:.3f}  force(recovered vs eager @x_final) {lane['force_recovered_vs_eager_rel_l2']:.3e}  "
              f"fallbacks {lane['fallbacks']} flags={mon.summary().flags or '-'}", flush=True)
        del s, prepared, p_k, ev_k
    f0, h0 = finals["0"]; f1, h1 = finals["1"]
    div = [_rel(h1[k, :, 0, :], h0[k, :, 0, :]) for k in (0, 9, 49, 99, args.steps - 1) if k < args.steps]
    out["divergence_pos_rel_l2_at_steps"] = dict(zip([1, 10, 50, 100, args.steps], div))
    out["final_vel_rel_l2"] = _rel(f1[:, 1, :], f0[:, 1, :])
    print("position divergence CSR on vs off:", {k: f"{v:.2e}" for k, v in out["divergence_pos_rel_l2_at_steps"].items()},
          "final vel rel", f"{out['final_vel_rel_l2']:.2e}", flush=True)
    p = Path(args.out or (ROOT / "artifacts" / "smallleaf" / f"dynamics_{args.flag.split('_')[-1].lower()}_leaf{args.leaf}_th{args.theta:g}_p{args.order}_{args.steps}steps.json"))
    with open(p, "w") as fh: json.dump(out, fh, indent=2, default=str)
    print("wrote", p)


if __name__ == "__main__":
    main()
