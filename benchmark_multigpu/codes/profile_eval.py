#!/usr/bin/env python
"""T1.1 -- what is one fused ``eval_fn`` call made of, kernel by kernel?

Traces ``--calls`` warm evaluations with ``jax.profiler`` (perfetto JSON) on a
verified-idle GPU and ranks device kernels by total time, counts launches, and
measures device idle inside the traced window.  Written to attribute the
theta- and order-independent ~32 ms at N=200k (plan Track 1): O(1) launch/host
overhead amortises with N, O(N) kernel work does not, and the two need
different fixes.

Run:
    JAX_ENABLE_X64=1 /export/home/tbuck/jaccpot/.venv/bin/python \
      benchmark_multigpu/codes/profile_eval.py --n 200000 --theta 1.0 --leaf 256
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common.gpu_guard import pick_idle_gpus, set_cuda_visible, timed_calls  # noqa: E402
from compare_force import apply_fast_lane_env  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402


def load_perfetto(log_dir: Path) -> list[dict]:
    files = sorted(glob.glob(str(log_dir / "**" / "*.trace.json.gz"), recursive=True))
    if not files:
        files = sorted(glob.glob(str(log_dir / "**" / "*.json.gz"), recursive=True))
    if not files:
        raise SystemExit(f"no perfetto trace under {log_dir}")
    with gzip.open(files[-1], "rt") as fh:
        data = json.load(fh)
    return data["traceEvents"] if isinstance(data, dict) else data


def analyse(events: list[dict], calls: int) -> dict:
    """Split events into host/device by process name; rank device kernels."""
    pnames, tnames = {}, {}
    for e in events:
        if e.get("ph") == "M":
            if e.get("name") == "process_name":
                pnames[e["pid"]] = e["args"]["name"]
            elif e.get("name") == "thread_name":
                tnames[(e["pid"], e["tid"])] = e["args"]["name"]
    dev_pids = {p for p, n in pnames.items() if "GPU" in n.upper() and "HOST" not in n.upper()}
    # kernel-level events on device: thread names like "XLA Ops"; module/step
    # rows ("XLA Modules", "Steps") are containers and are excluded.
    by_name = collections.defaultdict(lambda: [0.0, 0])
    per_stream = collections.defaultdict(list)
    t_min, t_max = float("inf"), 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in dev_pids:
            continue
        tn = tnames.get((e["pid"], e["tid"]), "")
        if "module" in tn.lower() or "step" in tn.lower():
            continue
        dur = float(e.get("dur", 0.0))
        ts = float(e["ts"])
        by_name[e["name"]][0] += dur
        by_name[e["name"]][1] += 1
        per_stream[(e["pid"], e["tid"], tn)].append((ts, ts + dur))
        t_min, t_max = min(t_min, ts), max(t_max, ts + dur)
    # device busy = union of all kernel intervals across streams
    ivs = sorted(iv for lst in per_stream.values() for iv in lst)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    total_kernel = sum(v[0] for v in by_name.values())
    launches = sum(v[1] for v in by_name.values())
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])
    return dict(
        process_names=pnames,
        device_pids=sorted(dev_pids),
        streams=sorted(str(k) for k in per_stream),
        window_us=(t_max - t_min) if ivs else 0.0,
        busy_us=busy,
        kernel_time_sum_us=total_kernel,
        launches=launches,
        per_call=dict(
            window_ms=(t_max - t_min) / 1e3 / calls if ivs else 0.0,
            busy_ms=busy / 1e3 / calls,
            kernel_sum_ms=total_kernel / 1e3 / calls,
            launches=launches / calls,
        ),
        top=[
            dict(name=n, total_ms_per_call=v[0] / 1e3 / calls, count_per_call=v[1] / calls,
                 mean_us=v[0] / max(1, v[1]), share=v[0] / max(1e-9, total_kernel))
            for n, v in ranked[:60]
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leaf", type=int, default=256)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    devices = pick_idle_gpus(1)
    cvd = set_cuda_visible(devices)
    print(f"GPU: physical {devices} (CUDA_VISIBLE_DEVICES={cvd})", flush=True)
    apply_fast_lane_env(args.n)

    import jax
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    P = jax.numpy.asarray(pos, jax.numpy.float32)
    M = jax.numpy.asarray(mass, jax.numpy.float32)
    s = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real",
        theta=args.theta, G=1.0, softening=args.softening, working_dtype=jax.numpy.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen"),
        fixed_order=args.order)
    prepared, ev = s.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)
    _, timing, cont = timed_calls(lambda: ev(prepared), repeats=5, warmup=3,
                                  devices=devices, block=jax.block_until_ready)
    print(f"untraced: min {timing['min']*1e3:.2f} ms  median {timing['median']*1e3:.2f} ms  "
          f"flags={cont.flags or '-'}", flush=True)

    tag = f"{args.ic}{args.n}_leaf{args.leaf}_th{args.theta:g}_p{args.order}"
    trace_dir = Path(args.trace_dir or (ROOT / "artifacts" / "traces" / tag))
    trace_dir.mkdir(parents=True, exist_ok=True)
    import time

    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        traced = []
        for _ in range(args.calls):
            t0 = time.perf_counter()
            jax.block_until_ready(ev(prepared))
            traced.append(time.perf_counter() - t0)
    print(f"traced wall per call: {[round(t*1e3,2) for t in traced]} ms", flush=True)

    events = load_perfetto(trace_dir)
    res = analyse(events, args.calls)
    diag = dict(s.get_runtime_diagnostics() or {})
    res.update(config=dict(n=args.n, ic=args.ic, leaf=args.leaf, theta=args.theta,
                           order=args.order, calls=args.calls),
               untraced_timing=timing, contention=cont.as_dict(),
               traced_wall_ms=[t * 1e3 for t in traced],
               nbrs=diag.get("recent_dual_neighbor_count"),
               leaves=diag.get("large_n_eval_active_leaf_count"),
               far_pairs=diag.get("static_radix_far_pair_count"))
    pc = res["per_call"]
    print(f"\ndevice pids {res['device_pids']} streams {len(res['streams'])}")
    print(f"per call: window {pc['window_ms']:.2f} ms, device busy {pc['busy_ms']:.2f} ms "
          f"({100*pc['busy_ms']/max(1e-9,pc['window_ms']):.0f} %), kernel-sum {pc['kernel_sum_ms']:.2f} ms, "
          f"launches {pc['launches']:.0f}")
    print(f"{'ms/call':>8} {'share':>6} {'n/call':>7} {'mean us':>8}  kernel")
    for k in res["top"][:40]:
        print(f"{k['total_ms_per_call']:>8.3f} {100*k['share']:>5.1f}% {k['count_per_call']:>7.1f} "
              f"{k['mean_us']:>8.1f}  {k['name'][:110]}")
    out = Path(args.out or (ROOT / "artifacts" / f"profile_eval_{tag}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
