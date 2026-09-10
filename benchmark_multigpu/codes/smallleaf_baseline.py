#!/usr/bin/env python
"""Phase 0.3 of the small-leaves plan: an ATTRIBUTED per-step baseline, one config per process.

Question this answers (gate G0): is the small-leaf penalty launch overhead
(self-leaf ``lax.scan``, anonymous XLA fusions, launch-bound M2L) or does the
Pallas leaf-pair kernel itself have a small-W efficiency cliff?  The host
``refresh_*_seconds`` timers are structurally dead on the fused lane (a
``perf_counter`` around a traced scan body), so three mechanisms that do work
are crossed here:

1. **Perfetto kernel table** of one warm ``strict_run_v2`` call and of one warm
   eval-only call, classified into stages by kernel NAME (``nearfield_leafpair_*``,
   ``m2l_real_*``, ``treecode_walk_*``, sorts) and by COUNT: any kernel launched
   once per leaf per step is the self-leaf scan family (``_self_contributions``
   is a ``lax.scan`` over leaves, ~5 fusions of ~2 us each per leaf).
2. **Cumulative refresh diag modes** (``JACCPOT_STRICT_REFRESH_DIAG_MODE``):
   ``integrator_only <= tree_only <= upward_only <= downward_only <= full``;
   stage = difference of neighbours (``bench/profile_fused_stage_ablation.py``).
3. **Within-full ablations**: ``JACCPOT_LARGE_N_EVAL_DIAG_MODE`` in {zero,
   permutation_only, far_only, near_only}, ``JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE``
   in {self_only, pairs_only}, ``JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE`` in
   {m2l_only, l2l_only}.  Each mode is read at solver construction and compiles
   its own runner (~30-60 s).

Every diag mode is timed with the same protocol (``--reps`` repeats of a
``--steps``-step warm scan, min and IQR reported) under a ``GpuMonitor`` so a
contended row is flagged rather than trusted.  The within-mode spread is the
noise floor for any attributed difference (35 % has been measured on this
workload; see the ablation script's docstring).

Also recorded: aggL2 of the eager force against a float64 direct sum on
``--ref-targets`` targets (the accuracy of small leaves at fixed theta is NOT
the same as at leaf 256), list sizes and the direct-sum share.

Caps come from ``compare_force.FAST_LANE_ENV_BY_LEAF`` (fitted by
``codes/fit_smallleaf_caps.py``); ``--env`` adds or overrides.

Run on an idle card (the guard picks one; ``CUDA_VISIBLE_DEVICES`` is honoured
and checked)::

    PYTHONPATH=.../sitecustom_wt JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-smallleaf-wt \
      /export/home/tbuck/jaccpot/.venv/bin/python codes/smallleaf_baseline.py --leaf 64 --theta 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.budget import jaccpot_direct_budget  # noqa: E402
from common.gpu_guard import GpuMonitor, idle_gpus, pick_idle_gpus, set_cuda_visible, timed_calls  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402
from compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    _nearfield_kernel_layout,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)
from profile_eval import analyse, load_perfetto  # noqa: E402

#: (env var, value, label) -- every ablation mode timed on top of ``full``.
REFRESH_MODES = ["integrator_only", "tree_only", "upward_only", "downward_only"]
EVAL_MODES = ["zero", "permutation_only", "far_only", "near_only"]
NEARFIELD_MODES = ["self_only", "pairs_only"]
DETAIL_MODES = ["m2l_only", "l2l_only"]

_NAME_STAGES = (
    ("nearfield_leafpair", "nearfield_pallas_leafpair"),
    ("nearfield_fused_leaf", "nearfield_pallas_streaming"),
    ("nearfield", "nearfield_other"),
    ("m2l_real", "m2l_pallas"),
    ("m2l", "m2l_other"),
    ("treecode_walk", "treecode_walk"),
    ("dual_tree", "dual_tree_walk"),
    ("sort", "sort"),
    ("radix", "sort"),
    ("scan", "cumsum_scan"),
    ("scatter", "scatter"),
    ("gather", "gather"),
    ("memcpy", "memcpy"),
    ("memset", "memset"),
    ("reduce", "reduce"),
)


def classify_kernels(top: list[dict], *, leaves: int, steps: int) -> dict:
    """Stage table from ``profile_eval.analyse``'s ranked kernel list.

    ``count_per_call`` is per traced call (``steps`` scan steps); a kernel whose
    per-step count is within 2 % of the leaf count is the per-leaf launch family
    (the self-leaf scan) regardless of its anonymous fusion name.
    """
    stages: dict[str, dict] = {}
    per_leaf_family = []
    for k in top:
        per_step_count = float(k["count_per_call"]) / max(1, steps)
        name = k["name"]
        low = name.lower()
        stage = None
        if leaves > 0 and abs(per_step_count - leaves) <= max(2.0, 0.02 * leaves) and not low.startswith(
            ("nearfield", "m2l", "treecode")
        ):
            stage = "per_leaf_launch_family"
            per_leaf_family.append(name)
        else:
            for needle, label in _NAME_STAGES:
                if needle in low:
                    stage = label
                    break
        if stage is None:
            stage = "xla_fusion_other" if "fusion" in low else "other"
        s = stages.setdefault(stage, dict(ms_per_step=0.0, launches_per_step=0.0, kernels=0))
        s["ms_per_step"] += float(k["total_ms_per_call"]) / max(1, steps)
        s["launches_per_step"] += per_step_count
        s["kernels"] += 1
    return dict(stages=stages, per_leaf_family=per_leaf_family)


def _rel_l2(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer", choices=sorted(IC_GENERATORS))
    ap.add_argument("--leaf", type=int, required=True)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=5, help="scan steps per timed call")
    ap.add_argument("--reps", type=int, default=3, help="timed calls per mode")
    ap.add_argument("--eval-repeats", type=int, default=7)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--vel-sigma", type=float, default=0.4)
    ap.add_argument("--softening", type=float, default=1e-7)
    ap.add_argument("--ref-targets", type=int, default=4096)
    ap.add_argument("--env", nargs="+", default=[], metavar="KEY=VAL")
    ap.add_argument("--no-leaf-preset", action="store_true")
    ap.add_argument("--modes", default="refresh,eval,nearfield,detail",
                    help="comma list of ablation families to run; '' = none")
    ap.add_argument("--m2l-chunk", type=int, default=0,
                    help="FarFieldConfig(m2l_chunk_size=...); 0 = jaccpot default (4096 pairs per chunk)")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")

    devices = _device(args.allow_busy)
    overrides: dict[str, str] = {}
    preset_trav: dict = {}
    extra = dict(kv.split("=", 1) for kv in args.env)
    if not args.no_leaf_preset:
        overrides.update(fast_lane_overrides_for_leaf(args.leaf, args.n, extra))
        preset_trav = dict((FAST_LANE_ENV_BY_LEAF.get(args.leaf) or {}).get("_traversal_overrides", {}))
    overrides.update(extra)
    env = apply_fast_lane_env(args.n, overrides=overrides)

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

    from common.reference import direct_accelerations

    tag = args.tag or f"{args.ic}{args.n}_leaf{args.leaf}_th{args.theta:g}_p{args.order}"
    print(f"[{tag}] devices={devices} overrides={json.dumps(overrides)} traversal={preset_trav}", flush=True)

    pos, mass = IC_GENERATORS[args.ic](args.n, seed=0)
    n = int(pos.shape[0])
    P = jnp.asarray(pos, jnp.float32)
    M = jnp.asarray(mass, jnp.float32)
    vel = np.random.default_rng(7).normal(0.0, args.vel_sigma, (n, 3)).astype(np.float32)
    state0 = jnp.stack([P, jnp.asarray(vel)], axis=1)

    runtime_cfg = RuntimePolicyConfig()
    if preset_trav:
        runtime_cfg = RuntimePolicyConfig(
            traversal_config=TraversalOverrides(**{k: int(v) for k, v in preset_trav.items()})
        )

    def build_solver():
        return FastMultipoleMethod(
            preset="large_n_gpu", runtime_path="large_n", basis="real", theta=args.theta,
            G=1.0, softening=args.softening, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=args.leaf),
                farfield=FarFieldConfig(mode="auto", **({"m2l_chunk_size": int(args.m2l_chunk)} if args.m2l_chunk else {})),
                nearfield=NearFieldConfig(mode="auto"),
                runtime=runtime_cfg, mac_type="dehnen"),
            fixed_order=args.order)

    result: dict = dict(
        m2l_chunk=int(args.m2l_chunk) or None,
        tag=tag, n=n, ic=args.ic, leaf=args.leaf, order=args.order, theta=args.theta,
        steps=args.steps, reps=args.reps, dt=args.dt, vel_sigma=args.vel_sigma,
        devices=devices, env_overrides=overrides, traversal_overrides=preset_trav,
        fast_lane_env=env, worktree=os.environ.get("JACCPOT_WORKTREE"),
        num_leaves=int(-(-n // args.leaf)),
    )
    out_path = Path(args.out or (ROOT / "artifacts" / "smallleaf" / f"baseline_{tag}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def write():
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)

    # ------------------------------------------------------------- eval-only
    solver = build_solver()
    solver._refresh_timing_active = True
    t0 = time.perf_counter()
    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=args.leaf, max_order=args.order, theta=args.theta)
    a_eager = np.asarray(jax.block_until_ready(eval_fn(prepared)), np.float64)
    prepare_s = time.perf_counter() - t0
    out, timing, cont = timed_calls(lambda: eval_fn(prepared), repeats=args.eval_repeats,
                                    warmup=2, devices=devices, block=jax.block_until_ready)
    diag = dict(solver.get_runtime_diagnostics() or {})
    rows = np.asarray(prepared.neighbor_list.counts)
    validated = dict(getattr(solver._impl, "_strict_fused_validated_caps", None) or {})
    budget = jaccpot_direct_budget(prepared, n)
    ref_idx = None
    if args.ref_targets and args.ref_targets < n:
        ref_idx = np.sort(np.random.default_rng(12345).choice(n, args.ref_targets, replace=False))
    a_ref = direct_accelerations(pos, mass, G=1.0, softening=args.softening,
                                 block_size=1024, target_indices=ref_idx)
    agg = _rel_l2(a_eager if ref_idx is None else a_eager[ref_idx], a_ref)
    result["eval_only"] = dict(
        timing=timing, contention=cont.as_dict(), prepare_s_incl_compile=prepare_s,
        aggL2=agg, ref_targets=None if ref_idx is None else int(len(ref_idx)),
        nearfield_kernel_layout=_nearfield_kernel_layout(prepared),
        fused_mode_active=bool(diag.get("strict_fused_mode_active")),
        lists=dict(neighbor_rows_max=int(rows.max()), neighbor_edges_total=int(rows.sum()),
                   far_pairs=validated.get("far_pair_count"),
                   direct_share_of_N=budget.get("direct_share_of_N"),
                   active_leaves=diag.get("large_n_eval_active_leaf_count")),
        validated_caps=validated,
    )
    print(f"[{tag}] eval-only min {timing['min']*1e3:.2f} ms (IQR {timing['iqr']*1e3:.2f}) aggL2 {agg:.4e} "
          f"direct {budget.get('direct_share_of_N'):.3f}N rows max {rows.max()} far {validated.get('far_pair_count')} "
          f"layout {result['eval_only']['nearfield_kernel_layout']} flags={cont.flags or '-'}", flush=True)
    write()

    leaves = int(diag.get("large_n_eval_active_leaf_count") or result["num_leaves"])

    if not args.no_trace:
        tdir = ROOT / "artifacts" / "traces" / f"smallleaf_eval_{tag}"
        tdir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
            for _ in range(3):
                jax.block_until_ready(eval_fn(prepared))
        res = analyse(load_perfetto(tdir), 3)
        cls = classify_kernels(res["top"], leaves=leaves, steps=1)
        result["eval_only"]["trace"] = dict(per_call=res["per_call"], stages=cls["stages"],
                                           per_leaf_family=cls["per_leaf_family"], top=res["top"][:30])
        print(f"[{tag}] eval trace: busy {res['per_call']['busy_ms']:.2f}/{res['per_call']['window_ms']:.2f} ms, "
              f"launches {res['per_call']['launches']:.0f}; stages "
              + ", ".join(f"{k}={v['ms_per_step']:.2f}" for k, v in
                          sorted(cls["stages"].items(), key=lambda kv: -kv[1]["ms_per_step"])), flush=True)
        write()
    del prepared, eval_fn

    # ------------------------------------------------------- strict_run_v2
    def scan_timing(s, label: str) -> dict:
        """Compile, then time ``reps`` warm ``steps``-step calls of strict_run_v2."""
        def run(state, prepared, k):
            out = s.strict_run_v2(state=state, masses=M, dt=args.dt, num_steps=k, refresh_every=1,
                                  leaf_size=args.leaf, max_order=args.order, theta=args.theta,
                                  prepared_state=prepared, return_prepared_state=True)
            jax.block_until_ready(out[0])
            return out
        t0 = time.perf_counter()
        state, prepared, _ = run(state0, None, 1)
        state, prepared, _ = run(state, prepared, args.steps)
        compile_s = time.perf_counter() - t0
        samples = []
        with GpuMonitor(devices) as mon:
            for _ in range(args.reps):
                t0 = time.perf_counter()
                state, prepared, _ = run(state, prepared, args.steps)
                samples.append((time.perf_counter() - t0) / args.steps)
        cont = mon.summary()
        d = dict(s.get_runtime_diagnostics() or {})
        srt = sorted(samples)
        rec = dict(
            label=label, ms_per_step_min=srt[0] * 1e3, ms_per_step_median=srt[len(srt) // 2] * 1e3,
            ms_per_step_max=srt[-1] * 1e3, samples_ms=[x * 1e3 for x in samples],
            spread=(srt[-1] - srt[0]) / max(1e-12, srt[0]), compile_s=compile_s,
            contention=cont.as_dict(), fallback_count=d.get("strict_fused_fallback_count"),
            last_fallback_reason=d.get("strict_fused_last_fallback_reason"),
            compile_count=d.get("strict_runner_compile_count"),
        )
        print(f"[{tag}] {label:>22}: {rec['ms_per_step_min']:8.2f} ms/step (spread {100*rec['spread']:.0f} %, "
              f"compile {compile_s:.0f} s) fallbacks {rec['fallback_count']} flags={cont.flags or '-'}", flush=True)
        return rec, state, prepared, run

    try:
        full, state, prepared, run = scan_timing(solver, "full")
    except Exception as exc:  # noqa: BLE001
        result["scan_failed"] = "".join(traceback.format_exception(exc))[-6000:]
        print(f"[{tag}] strict_run_v2 FAILED: {str(exc).splitlines()[0][:300]}", flush=True)
        write()
        return 2
    result["scan_full"] = full
    write()

    if not args.no_trace:
        tdir = ROOT / "artifacts" / "traces" / f"smallleaf_step_{tag}"
        tdir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
            state, prepared, _ = run(state, prepared, args.steps)
        res = analyse(load_perfetto(tdir), 1)
        cls = classify_kernels(res["top"], leaves=leaves, steps=args.steps)
        pc = res["per_call"]
        result["scan_full"]["trace"] = dict(
            per_step=dict(window_ms=pc["window_ms"] / args.steps, busy_ms=pc["busy_ms"] / args.steps,
                          kernel_sum_ms=pc["kernel_sum_ms"] / args.steps, launches=pc["launches"] / args.steps),
            stages=cls["stages"], per_leaf_family=cls["per_leaf_family"], top=res["top"][:40])
        print(f"[{tag}] step trace: busy {pc['busy_ms']/args.steps:.2f}/{pc['window_ms']/args.steps:.2f} ms/step, "
              f"launches {pc['launches']/args.steps:.0f}/step; stages "
              + ", ".join(f"{k}={v['ms_per_step']:.2f}" for k, v in
                          sorted(cls["stages"].items(), key=lambda kv: -kv[1]["ms_per_step"])), flush=True)
        write()
    del solver, state, prepared

    # ------------------------------------------------------- ablation modes
    families = [f for f in args.modes.split(",") if f]
    plan = []
    if "refresh" in families:
        plan += [("JACCPOT_STRICT_REFRESH_DIAG_MODE", m) for m in REFRESH_MODES]
    if "eval" in families:
        plan += [("JACCPOT_LARGE_N_EVAL_DIAG_MODE", m) for m in EVAL_MODES]
    if "nearfield" in families:
        plan += [("JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE", m) for m in NEARFIELD_MODES]
    if "detail" in families:
        plan += [("JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE", m) for m in DETAIL_MODES]
    result["ablations"] = {}
    for var, mode in plan:
        os.environ[var] = mode
        try:
            s = build_solver()
            rec, st, pr, _ = scan_timing(s, f"{var.split('_DIAG_MODE')[0].replace('JACCPOT_', '').lower()}={mode}")
            del s, st, pr
        except Exception as exc:  # noqa: BLE001
            rec = dict(label=mode, failed=str(exc).splitlines()[0][:300])
            print(f"[{tag}] {var}={mode} FAILED: {rec['failed']}", flush=True)
        finally:
            os.environ[var] = "full"
        result["ablations"][f"{var}={mode}"] = rec
        write()

    # ------------------------------------------------------- attribution
    def ms(key):
        r = result["ablations"].get(key) or {}
        return r.get("ms_per_step_min")

    full_ms = full["ms_per_step_min"]
    attr = dict(full_ms=full_ms)
    cum = {m: ms(f"JACCPOT_STRICT_REFRESH_DIAG_MODE={m}") for m in REFRESH_MODES}
    if all(v is not None for v in cum.values()):
        attr["refresh_cumulative"] = dict(
            integrator=cum["integrator_only"],
            tree=cum["tree_only"] - cum["integrator_only"],
            upward=cum["upward_only"] - cum["tree_only"],
            downward=cum["downward_only"] - cum["upward_only"],
            eval_plus_nearfield=full_ms - cum["downward_only"],
        )
    ev = {m: ms(f"JACCPOT_LARGE_N_EVAL_DIAG_MODE={m}") for m in EVAL_MODES}
    if all(v is not None for v in ev.values()):
        attr["eval_split"] = dict(
            everything_but_eval=ev["zero"], permutation_floor=ev["permutation_only"] - ev["zero"],
            far_L2P=ev["far_only"] - ev["permutation_only"], near=ev["near_only"] - ev["permutation_only"],
            full_minus_near_only=full_ms - ev["near_only"],
        )
    nf = {m: ms(f"JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE={m}") for m in NEARFIELD_MODES}
    if all(v is not None for v in nf.values()):
        attr["nearfield_split"] = dict(
            self_scan_saved=full_ms - nf["pairs_only"], pairs_saved=full_ms - nf["self_only"])
    dt_ = {m: ms(f"JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE={m}") for m in DETAIL_MODES}
    if all(v is not None for v in dt_.values()):
        attr["downward_split"] = dict(m2l_only=dt_["m2l_only"], l2l_only=dt_["l2l_only"])
    result["attribution"] = attr
    print(f"[{tag}] attribution: {json.dumps(attr, indent=None)}", flush=True)
    write()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
