#!/usr/bin/env python
"""Phase 0.2 of the sub-10 ms plan: jz-fmm's single-GPU accuracy/time front on OUR Plummer IC.

jz-fmm (Jens Stuecker, MIT) is the target curve for the plan: ~9 ms per force at
2x10^5 on one A100, tree build included.  This script measures its front on the
SAME initial conditions, softening, reference and error metric as the jaccpot
rows (``smallleaf_baseline.py`` / ``compare_force.py``), so the two codes land
on one plot:

* IC: ``common.ic.plummer_sphere(n, seed=0)`` (float32, masses 1/N, G = 1).
* Softening: Plummer in both codes (``PlummerKernel(softening=...)``), so a
  matched-error comparison is legitimate (unlike pkdgrav3's spline).
* Reference: fp64 direct sum on ``--ref-targets`` targets drawn with the same
  seed (12345) as ``smallleaf_baseline.py`` -- never compare across subsample
  sizes (memory ``rel-l2-probe-not-comparable``).
* Error: ``aggL2`` (our headline) AND the per-particle p90 (jz-fmm's headline
  in ``docs/_static/hernquist_performance_comparison.png``).
* Time: ``fast_multipole_method.jit(...)`` -> ``loc.force()``, tree build
  included (jz-fmm rebuilds per call, as our fused step does), min of
  ``--repeats`` after ``--warmup`` under the GPU monitor; a contended row is
  flagged, not trusted.
* Direct-sum share: from the leaf-level interaction list
  (``_evaluate_node_node_fmm`` returns it) and the leaf->particle splits, the
  same ``direct_sources_per_target / N`` accounting as ``common.budget``.
* Launch count: kernels per force from a ``jax.profiler`` trace (optional).

Memory (``--memory``, jaccpot fused-memory round 2, Phase 0c): per row the XLA
allocator's ``memory_stats`` (``peak_bytes_in_use`` is per PROCESS -- run one
configuration per process for a clean peak), ``memory_analysis`` of the jitted
FMM, and the process's own ``used_gpu_memory`` sampled from nvidia-smi (jz-fmm's
FFI kernels could allocate outside XLA; that number includes the CUDA context).
``--prealloc FRAC`` preallocates that fraction of the card, as jaccpot's
``bench/fused_memory_budget.py --prealloc`` does, for ceiling ladders.

Runs INSIDE the dedicated jz-fmm venv (``/export/scratch/tbuck/jzfmm-venv``;
home is quota-bound), never in envs/odisseo or the jaccpot venv::

    /export/scratch/tbuck/jzfmm-venv/bin/python codes/jzfmm_force_eval.py \
        --n 200000 --p 3 4 5 6 --theta 0.5 0.6 0.8 1.0 --leaf 32 64
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.gpu_guard import idle_gpus, pick_idle_gpus, set_cuda_visible, timed_calls  # noqa: E402
from common.error import rel_errors  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402


def _device(allow_busy: bool) -> list[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        pool = [int(x) for x in cvd.split(",") if x.strip()]
        idle, reasons = idle_gpus(settle_s=2.0, samples=4)
        busy = [d for d in pool if d not in idle]
        if busy and not allow_busy:
            raise SystemExit("CUDA_VISIBLE_DEVICES names busy card(s): "
                             + "; ".join(f"GPU {d}: {reasons.get(d)}" for d in busy))
        if busy:
            print(f"!! running on busy GPU(s) {busy}; timings are NOT for the record", flush=True)
        return pool[:1]
    chosen = pick_idle_gpus(1)
    set_cuda_visible(chosen)
    return chosen


class _ProcessMemorySampler:
    """This process's ``used_gpu_memory`` from nvidia-smi, max over a background poll."""

    def __init__(self, device: int, interval_s: float = 0.25):
        import threading

        self.device = device
        self.interval_s = interval_s
        self.max_mib = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self) -> None:
        import subprocess

        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                for line in out.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                        self.max_mib = max(self.max_mib, int(parts[1]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        import threading

        if self._thread is None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()


def _memory_row(jax, n: int, sampler: _ProcessMemorySampler) -> dict:
    stats = jax.devices()[0].memory_stats() or {}
    peak = int(stats.get("peak_bytes_in_use", 0))
    return dict(
        peak_bytes=peak,
        peak_gib=peak / 2**30,
        peak_bytes_per_particle=peak / max(n, 1),
        bytes_in_use=int(stats.get("bytes_in_use", 0)),
        largest_alloc_size=int(stats.get("largest_alloc_size", 0)),
        bytes_limit=int(stats.get("bytes_limit", 0)),
        process_max_mib=sampler.max_mib,
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer", choices=sorted(IC_GENERATORS))
    ap.add_argument("--p", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--theta", type=float, nargs="+", default=[0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--leaf", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--softening", type=float, default=1e-7,
                    help="Plummer softening; smallleaf_baseline.py's default is 1e-7 (compare like with like)")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ref-targets", type=int, default=4096)
    ap.add_argument("--alloc-fac-ilist", type=float, default=64.0,
                    help="jz-fmm's interaction-list allocation factor; doubled on overflow up to 4x")
    ap.add_argument("--no-trace", action="store_true", help="skip the launch-count trace")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--memory", action="store_true",
                    help="record memory_stats / memory_analysis / nvidia-smi per row")
    ap.add_argument("--prealloc", type=float, default=0.0,
                    help="preallocate this fraction of the card (0 = grow on demand)")
    ap.add_argument("--no-lists", action="store_true", help="skip the direct-share list pass")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.prealloc > 0:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{args.prealloc:.3f}"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")

    devices = _device(args.allow_busy)

    import jax
    import jax.numpy as jnp
    import jzfmm
    import jztree
    from jzfmm.fmm import _as_posmass, _evaluate_node_node_fmm, fast_multipole_method
    from jztree.config import TreeConfig

    from common.reference import direct_accelerations

    tag = args.tag or f"{args.ic}{args.n}"
    print(f"[jzfmm {tag}] devices={devices} jax {jax.__version__} jzfmm {getattr(jzfmm, '__version__', '?')} "
          f"jztree {getattr(jztree, '__version__', '?')} {jax.devices()}", flush=True)

    pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    n = int(pos.shape[0])
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)
    part = jzfmm.data.Particles(pos=P, mass=M, vel=jnp.zeros_like(P))

    ref_idx = None
    if args.ref_targets and args.ref_targets < n:
        ref_idx = np.sort(np.random.default_rng(12345).choice(n, args.ref_targets, replace=False))
    t0 = time.perf_counter()
    ref_cache = ROOT / "artifacts" / "reference" / (
        f"direct_fp64_{args.ic}{n}_soft{args.softening:g}_ref{0 if ref_idx is None else len(ref_idx)}_seed12345.npy")
    if ref_cache.exists():
        a_ref = np.load(ref_cache)
        src = "cached"
    else:
        # block_size 0 = sized to a fixed budget (a fixed 1024 is 64 GB at 8e6); with
        # --memory on the CPU, so the GPU peak below is jz-fmm's alone
        import contextlib

        ctx = (jax.default_device(jax.devices("cpu")[0]) if args.memory
               else contextlib.nullcontext())
        with ctx:
            a_ref = direct_accelerations(pos, mass, G=1.0, softening=args.softening,
                                         block_size=0, target_indices=ref_idx)
        ref_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref_cache, a_ref)
        src = "computed"
    print(f"[jzfmm {tag}] fp64 direct reference on {a_ref.shape[0]} targets ({src}): {time.perf_counter()-t0:.1f} s", flush=True)

    result: dict = dict(
        code="jzfmm", tag=tag, n=n, ic=args.ic, softening=args.softening, G=1.0,
        devices=devices, host=platform.node(),
        versions=dict(jax=jax.__version__, jzfmm=getattr(jzfmm, "__version__", None),
                      jztree=getattr(jztree, "__version__", None)),
        ref_targets=None if ref_idx is None else int(len(ref_idx)),
        repeats=args.repeats, warmup=args.warmup,
        defaults=dict(FMMConfig=str(jzfmm.FMMConfig())),
        rows=[],
    )
    out_path = Path(args.out or (ROOT / "artifacts" / "jzfmm" / f"jzfmm_front_{tag}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)

    sampler = _ProcessMemorySampler(devices[0]) if args.memory else None
    if args.memory and len(args.leaf) * len(args.p) * len(args.theta) > 1:
        print(f"[jzfmm {tag}] !! --memory with several configurations: peak_bytes_in_use "
              "is cumulative over the process; run one per process", flush=True)

    try:
        from profile_eval import analyse, load_perfetto  # noqa: E402
    except Exception:  # noqa: BLE001
        analyse = load_perfetto = None

    for leaf in args.leaf:
        for p in args.p:
            for theta in args.theta:
                row: dict = dict(leaf=leaf, p=p, theta=theta)
                label = f"leaf{leaf} p{p} th{theta:g}"
                alloc = args.alloc_fac_ilist
                loc = None
                for attempt in range(3):
                    cfg = jzfmm.FMMConfig(
                        tree=TreeConfig(max_leaf_size=leaf, mass_centered=False, alloc_fac_nodes=1.2,
                                        regularization=None, coarse_fac=4.0),
                        kernel=jzfmm.PlummerKernel(softening=args.softening),
                        p=p, opening=jzfmm.OpeningByAngle(theta=theta),
                        alloc_fac_ilist=alloc,
                    )
                    try:
                        t0 = time.perf_counter()
                        if sampler is not None:
                            sampler.start()
                        loc = fast_multipole_method.jit(part, cfg_fmm=cfg, G=1.0)
                        loc.values.block_until_ready()
                        row["compile_s_incl_first_call"] = time.perf_counter() - t0
                        break
                    except Exception as exc:  # noqa: BLE001
                        msg = str(exc).splitlines()[0][:300]
                        if "Interaction list allocation" in str(exc) or "alloc_fac" in str(exc):
                            print(f"[jzfmm {tag}] {label}: ilist overflow at alloc_fac_ilist={alloc}: {msg}", flush=True)
                            alloc *= 2.0
                            loc = None
                            continue
                        row["failed"] = "".join(traceback.format_exception(exc))[-3000:]
                        print(f"[jzfmm {tag}] {label} FAILED: {msg}", flush=True)
                        loc = None
                        break
                if loc is None:
                    row.setdefault("failed", f"ilist overflow up to alloc_fac_ilist={alloc/2}")
                    result["rows"].append(row)
                    write()
                    continue
                row["alloc_fac_ilist"] = alloc
                row["cfg"] = str(cfg)

                def force_call(cfg=cfg):
                    return fast_multipole_method.jit(part, cfg_fmm=cfg, G=1.0).force()

                acc, timing, cont = timed_calls(force_call, repeats=args.repeats, warmup=args.warmup,
                                                devices=devices, block=jax.block_until_ready)
                a = np.asarray(acc, np.float64)
                err = rel_errors(a if ref_idx is None else a[ref_idx], a_ref)
                row["timing"] = timing
                row["contention"] = cont.as_dict()
                row["error"] = err
                if sampler is not None:
                    row["memory"] = _memory_row(jax, n, sampler)
                    try:
                        comp = fast_multipole_method.jit.lower(part, cfg_fmm=cfg, G=1.0).compile()
                        ma = comp.memory_analysis()
                        row["memory"]["memory_analysis"] = {
                            k: int(getattr(ma, k)) for k in (
                                "argument_size_in_bytes", "output_size_in_bytes",
                                "temp_size_in_bytes", "alias_size_in_bytes",
                                "generated_code_size_in_bytes") if hasattr(ma, k)}
                    except Exception as exc:  # noqa: BLE001
                        row["memory"]["memory_analysis_failed"] = str(exc).splitlines()[0][:300]
                    m = row["memory"]
                    print(f"[jzfmm {tag}] {label}: peak {m['peak_gib']:.3f} GiB "
                          f"({m['peak_bytes_per_particle']:.0f} B/p), nvidia-smi max "
                          f"{m['process_max_mib']} MiB, temp "
                          f"{m.get('memory_analysis', {}).get('temp_size_in_bytes', 0) / 2**30:.3f} GiB",
                          flush=True)
                if err["aggL2_signflip"] < err["aggL2"]:
                    row["SIGN_CONVENTION_MISMATCH"] = True

                # ---- direct-sum share from the leaf-level interaction list
                try:
                    if args.no_lists:
                        raise RuntimeError("skipped (--no-lists)")
                    partz, th = fast_multipole_method.jit(part, cfg_fmm=cfg, G=1.0, result="partz_tree")
                    _, ilist = _evaluate_node_node_fmm.jit(_as_posmass(partz), th, cfg_fmm=cfg)
                    nleaf = int(th.num(0))
                    spl = np.asarray(th.splits_leaf_to_part())[: nleaf + 1].astype(np.int64)
                    occ = spl[1:] - spl[:-1]
                    ispl = np.asarray(ilist.ispl)[: nleaf + 1].astype(np.int64)
                    isrc = np.asarray(ilist.isrc)[: ispl[-1]].astype(np.int64)
                    row_of = np.repeat(np.arange(nleaf), ispl[1:] - ispl[:-1])
                    self_hits = int(np.sum(isrc == row_of))
                    src_particles = np.bincount(row_of, weights=occ[isrc], minlength=nleaf)
                    # jz-fmm's list contains the leaf itself when self_hits == nleaf; then the
                    # per-target count is sum(occ) - 1 (remove_self_interaction); otherwise + occ - 1
                    if self_hits == nleaf:
                        direct = src_particles - 1
                    else:
                        direct = src_particles + occ - 1
                    w = occ.astype(np.float64)
                    mean_direct = float(np.sum(direct * w) / w.sum())
                    row["lists"] = dict(
                        num_leaves=nleaf, planes=int(th.num_planes()), size_leaves_alloc=int(th.size_leaves),
                        near_entries_directed=int(ispl[-1]), ilist_capacity=int(ilist.size()),
                        self_entries=self_hits, occupancy_mean=float(occ.mean()), occupancy_max=int(occ.max()),
                        mean_neighbor_leaves_per_leaf=float((ispl[1:] - ispl[:-1]).mean()),
                        max_neighbor_leaves_per_leaf=int((ispl[1:] - ispl[:-1]).max()),
                        direct_sources_per_target_mean=mean_direct,
                        direct_share_of_N=mean_direct / n,
                        p2p_pair_evaluations=float(np.sum(direct * occ)),
                    )
                except Exception as exc:  # noqa: BLE001
                    row["lists_failed"] = str(exc).splitlines()[0][:300]

                # ---- launch count per force
                if not args.no_trace and analyse is not None:
                    try:
                        tdir = ROOT / "artifacts" / "traces" / f"jzfmm_{tag}_leaf{leaf}_p{p}_th{theta:g}"
                        tdir.mkdir(parents=True, exist_ok=True)
                        with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
                            for _ in range(3):
                                jax.block_until_ready(force_call())
                        res = analyse(load_perfetto(tdir), 3)
                        row["trace"] = dict(per_call=res["per_call"], top=res["top"][:25])
                    except Exception as exc:  # noqa: BLE001
                        row["trace_failed"] = str(exc).splitlines()[0][:300]

                result["rows"].append(row)
                write()
                lists = row.get("lists", {})
                tr = row.get("trace", {}).get("per_call", {})
                print(f"[jzfmm {tag}] {label}: min {timing['min']*1e3:.2f} ms (IQR {timing['iqr']*1e3:.2f}) "
                      f"aggL2 {err['aggL2']:.3e} p90 {err['p90']:.3e} med {err['median']:.3e} "
                      f"direct {lists.get('direct_share_of_N', float('nan')):.4f}N leaves {lists.get('num_leaves')} "
                      f"planes {lists.get('planes')} launches {tr.get('launches', '-')} "
                      f"flags={cont.flags or '-'}", flush=True)
    print(f"[jzfmm {tag}] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
