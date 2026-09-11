#!/usr/bin/env python
"""Phase 4.1 of the sub-10 ms plan: the CSR-driven leaf-pair kernel vs the shipped rectangle kernel.

Builds the REAL fused-lane prepared state (N, leaf, theta as the step benchmark
does), pulls its leaf tables and neighbour CSR, and times, on an idle A100:

* A -- the shipped kernel exactly as the lane calls it: the padded rectangle
  ``(num_leaves, S)`` of source slots, ``source_chunk=64``, self folded in;
* B -- ``nearfield_leafpair_csr_pallas`` over the CSR, a sweep of chunk sizes,
  target subtiles and accumulator widths, timed with AND without the per-step
  chunk-table build.

Parity: B against A to float32 summation order (max relative difference over
all particles), on the same lists. ns per directed pair evaluation from the
exact P2P budget (``common.budget``).

Run on an idle card::

    PYTHONPATH=.../sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-sub10ms-wt \
      YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-sub10ms-wt \
      /export/home/tbuck/jaccpot/.venv/bin/python codes/nearfield_csr_microbench.py --leaf 64 --theta 0.6
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.budget import jaccpot_direct_budget  # noqa: E402
from common.gpu_guard import GpuMonitor, pick_idle_gpus, set_cuda_visible  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402
from compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)


def _time(fn, block, *, warmup=3, repeats=10):
    for _ in range(warmup):
        block(fn())
    ts = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = block(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return out, dict(min_ms=ts[0] * 1e3, median_ms=ts[len(ts) // 2] * 1e3, max_ms=ts[-1] * 1e3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer")
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--subtiles", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--accums", nargs="+", default=["input", "wide"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        devices = pick_idle_gpus(1)
        set_cuda_visible(devices)
    else:
        devices = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]

    extra = dict(kv.split("=", 1) for kv in args.env)
    overrides = fast_lane_overrides_for_leaf(args.leaf, args.n, extra)
    overrides.update(extra)
    preset_trav = dict((FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {}))
    apply_fast_lane_env(args.n, overrides=overrides)

    import jax
    import jax.numpy as jnp
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TraversalOverrides,
        TreeConfig,
    )
    from jaccpot.pallas.nearfield_fused_leaf import nearfield_leafpair_pallas
    from jaccpot.pallas.nearfield_leafpair_csr import (
        build_leafpair_chunk_table,
        leafpair_chunk_capacity,
        nearfield_leafpair_csr_pallas,
    )

    tag = f"{args.ic}{args.n}_leaf{args.leaf}_th{args.theta:g}"
    print(f"[csr {tag}] devices={devices} worktree={os.environ.get('JACCPOT_WORKTREE')}", flush=True)
    pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    n = int(pos.shape[0])
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)
    runtime_cfg = RuntimePolicyConfig()
    if preset_trav:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(**{k: int(v) for k, v in preset_trav.items()}))
    solver = FastMultipoleMethod(
        preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
        G=1.0, softening=args.softening, working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
            farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"),
            runtime=runtime_cfg, mac_type="dehnen"),
        fixed_order=args.order)
    prepared, _eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)

    # ---- tables from the prepared state (what the lane itself gathers)
    ids = jnp.asarray(prepared.nearfield_leaf_particle_indices)
    lmask = jnp.asarray(prepared.nearfield_leaf_particle_mask, dtype=bool)
    safe_ids = jnp.where(lmask, ids, 0)
    leaf_pos = jnp.asarray(prepared.positions_sorted)[safe_ids]
    leaf_mass = jnp.asarray(prepared.masses_sorted)[safe_ids]
    L, W = int(ids.shape[0]), int(ids.shape[1])
    nl = prepared.neighbor_list
    offsets = jnp.asarray(nl.offsets)
    counts = jnp.asarray(nl.counts)
    nbr_nodes = jnp.asarray(nl.neighbors)
    leaf_nodes = np.asarray(nl.leaf_indices)
    lookup = np.full(int(leaf_nodes.max()) + 1, -1, np.int64)
    lookup[leaf_nodes] = np.arange(L)
    lookup_j = jnp.asarray(lookup, nbr_nodes.dtype)
    nbr_leaf = jnp.where(nbr_nodes >= 0, lookup_j[jnp.clip(nbr_nodes, 0, lookup_j.shape[0] - 1)], 0)
    edge_cap = int(nbr_nodes.shape[0])
    payload = prepared.radix_fast_payload
    rect_ids = jnp.asarray(payload.source_leaf_ids)
    rect_valid = jnp.asarray(payload.source_leaf_valid_mask, dtype=bool)
    S = int(rect_ids.shape[1]) * int(rect_ids.shape[2])
    rect_ids = rect_ids.reshape(L, S)
    rect_valid = rect_valid.reshape(L, S)
    budget = jaccpot_direct_budget(prepared, n)
    p2p = float(budget["p2p_pair_evaluations"])
    soft = jnp.asarray(args.softening**2, jnp.float32)
    G = jnp.asarray(1.0, jnp.float32)
    print(f"[csr {tag}] L={L} W={W} S(rect)={S} edges={int(counts.sum())} edge_cap={edge_cap} "
          f"rows max {int(counts.max())} p2p={p2p:.3e} rect fill {int(counts.sum())/(L*S):.3f}", flush=True)

    result = dict(tag=tag, n=n, leaf=args.leaf, theta=args.theta, order=args.order, devices=devices,
                  L=L, W=W, S_rect=S, edges=int(counts.sum()), edge_cap=edge_cap,
                  rows_max=int(counts.max()), p2p_pair_evaluations=p2p, rows=[])
    out_path = Path(args.out or (ROOT / "artifacts" / "sub10ms" / f"nearfield_csr_microbench_{tag}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)

    def rel_max(a, b):
        a = np.asarray(a, np.float64)[..., :3]
        b = np.asarray(b, np.float64)[..., :3]
        num = np.linalg.norm(a - b, axis=-1)
        den = np.linalg.norm(b, axis=-1) + 1e-30
        m = np.asarray(lmask)
        return float(np.max(num[m] / den[m])), float(np.linalg.norm((a - b)[m]) / np.linalg.norm(b[m]))

    # ---- A: shipped rectangle kernel, as the lane calls it
    shipped_chunk = int(os.environ.get("JACCPOT_NEARFIELD_LEAFPAIR_SOURCE_CHUNK", "64"))
    fn_a = jax.jit(lambda: nearfield_leafpair_pallas(
        leaf_pos, leaf_mass, lmask, rect_ids, rect_valid, softening_sq=soft, G=G,
        source_chunk=shipped_chunk, include_self=True))
    with GpuMonitor(devices) as mon:
        out_a, t_a = _time(fn_a, jax.block_until_ready, repeats=args.repeats)
    row = dict(kernel="rectangle_shipped", source_chunk=shipped_chunk, **t_a,
               ns_per_pair=t_a["min_ms"] * 1e6 / p2p, contention=mon.summary().as_dict())
    result["rows"].append(row)
    print(f"[csr {tag}] A rectangle c{shipped_chunk}: {t_a['min_ms']:.2f} ms  {row['ns_per_pair']:.3f} ns/pair "
          f"flags={mon.summary().flags or '-'}", flush=True)
    write()

    # ---- B: CSR kernel sweep
    for chunk, bt, accum in itertools.product(args.chunks, args.subtiles, args.accums):
        cap = leafpair_chunk_capacity(edge_cap, L, chunk)
        try:
            tab = jax.block_until_ready(build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap))
            fn_b = jax.jit(lambda tab=tab, chunk=chunk, bt=bt, accum=accum: nearfield_leafpair_csr_pallas(
                leaf_pos, leaf_mass, lmask, nbr_leaf, tab, softening_sq=soft, G=G, chunk=chunk,
                target_subtile=bt, accum=accum, include_self=True))
            fn_bt = jax.jit(lambda chunk=chunk, bt=bt, accum=accum, cap=cap: nearfield_leafpair_csr_pallas(
                leaf_pos, leaf_mass, lmask, nbr_leaf,
                build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap),
                softening_sq=soft, G=G, chunk=chunk, target_subtile=bt, accum=accum, include_self=True))
            with GpuMonitor(devices) as mon:
                out_b, t_b = _time(fn_b, jax.block_until_ready, repeats=args.repeats)
                _, t_bt = _time(fn_bt, jax.block_until_ready, repeats=args.repeats)
            rmax, rl2 = rel_max(out_b, out_a)
            live_chunks = int(np.sum(np.asarray(tab.leaf) >= 0))
            row = dict(kernel="csr", chunk=chunk, subtile=bt, accum=accum, capacity=cap, live_chunks=live_chunks,
                       **t_b, with_table_min_ms=t_bt["min_ms"], ns_per_pair=t_b["min_ms"] * 1e6 / p2p,
                       rel_max_vs_A=rmax, rel_l2_vs_A=rl2, contention=mon.summary().as_dict())
            print(f"[csr {tag}] B csr c{chunk} bt{bt} {accum:5s}: {t_b['min_ms']:.2f} ms (+table {t_bt['min_ms']:.2f}) "
                  f"{row['ns_per_pair']:.3f} ns/pair  vs A relmax {rmax:.2e} relL2 {rl2:.2e}  "
                  f"chunks {live_chunks}/{cap} speedup {t_a['min_ms']/t_b['min_ms']:.2f}x flags={mon.summary().flags or '-'}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            row = dict(kernel="csr", chunk=chunk, subtile=bt, accum=accum, failed=str(exc).splitlines()[0][:400])
            print(f"[csr {tag}] B csr c{chunk} bt{bt} {accum}: FAILED {row['failed'][:200]}", flush=True)
        result["rows"].append(row)
        write()
    print(f"[csr {tag}] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
