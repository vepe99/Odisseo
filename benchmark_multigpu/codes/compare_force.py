#!/usr/bin/env python
"""jaccpot vs pkdgrav3: one force evaluation, same particles, same reference.

Both codes are FMMs, but they expose *different* accuracy knobs -- pkdgrav3's
expansion order is fixed at compile time (4th-order multipoles / 5th-order local
expansions, ``gravity/moments.h``) and its only runtime control is ``dTheta``,
while jaccpot has ``(order, theta)``.  A single timing number per code is
therefore meaningless.  This script produces the raw material for the only fair
comparison: an **accuracy-vs-cost Pareto front** per code, both measured against
the *same* float64 direct sum on the *same* particles.

jaccpot uses Plummer softening (``r^2 + eps^2``) and pkdgrav3 a compact-support
spline (``gravity/pp.h::EvalPP``), so at any *dynamically meaningful* ``eps`` the
two codes compute different physics and the comparison would measure the
softening kernel rather than the algorithm.  Exactly zero is not usable either:
jaccpot's distributed lane pads each device to a capacity, and those coincident
zero-mass rows produce NaN at ``eps=0``.  So the softening is a regularization
epsilon, small enough that both kernels are Newtonian far below the measurement
floor -- and the script *verifies* that by measuring how far the epsilon moves
the direct-sum reference, refusing to run if it is not negligible.

Ordering of the two phases matters: every pkdgrav3 subprocess runs *before* JAX
is imported, so the two codes never contend for the same GPU and neither one's
timings are polluted by the other's memory pool.

What is being compared -- stated once (plan T3.0).  A single force evaluation on
fixed particles, no integrator, no I/O.  Stage correspondence:

    partition_for_devices   <->  pkdgrav3 domain_decompose
    prepare_state (fused)   <->  pkdgrav3 build_tree        (+ jaccpot's MAC walk!)
    eval_fn                 <->  pkdgrav3 gravity           (which INCLUDES pkdgrav3's walk)

jaccpot's interaction lists are built in ``prepare_state`` and pkdgrav3's opening
decisions happen inside ``gravity()``, so ``eval_fn`` alone flatters jaccpot.  Every
jaccpot row therefore carries ``walk_s`` (the ``refresh_dual_*`` prepare-stage
timings of the MAC walk) so the artifact can show eval-only *and* eval+walk.

Every row also carries its **interaction budget** -- how many source particles
each target sums directly (``common/budget.py``) -- and a **contention record**
from ``common/gpu_guard.py``: the device(s) are chosen among cards that are 0 %
utilised with no compute process, and a sampler flags any row during which a
foreign process touched the device or the load average exceeded 8.

Run (single idle GPU picked by the guard; jaccpot side needs the jax-0.10.2 venv):
    JAX_ENABLE_X64=1 /export/home/tbuck/jaccpot/.venv/bin/python \
      benchmark_multigpu/codes/compare_force.py --n 200000 --ndevs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.budget import jaccpot_direct_budget, pkdgrav3_direct_budget
from common.gpu_guard import GpuMonitor, idle_gpus, pick_idle_gpus, set_cuda_visible, timed_calls
from common.ic import IC_GENERATORS
from common.pkdgrav3 import (
    PKDGRAV3_BIN,
    parse_gravity_lines,
    read_tipsy_dark,
    run_pkdgrav3,
    write_tipsy_dark,
)

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
PKD_SCRIPT = HERE / "pkdgrav3_force_eval.py"

#: pkdgrav3 opens on ``(bMax_c + bMax_k)/d < theta/1.5`` (``gravity/opening.cxx``,
#: ``X_Open = 1.5*bMax/theta``); jaccpot's bh/dehnen test is ``(r_t + r_s)/d < theta``.
#: So the same number means a 1.5x looser criterion in pkdgrav3 -- report both.
PKD_THETA_SCALE = 1.5

#: prepare-stage timings that make up jaccpot's MAC walk + list build (excluded
#: from eval_fn, included in pkdgrav3's gravity()).
WALK_KEYS = (
    "refresh_dual_setup_seconds",
    "refresh_dual_artifact_build_seconds",
    "refresh_dual_select_interactions_seconds",
    "refresh_dual_far_pair_plan_seconds",
    "refresh_dual_split_combined_seconds",
    "refresh_dual_split_far_pairs_seconds",
    "refresh_dual_split_leaf_neighbors_seconds",
    "refresh_dual_raw_combined_seconds",
    "refresh_tree_build_seconds",
    "refresh_tree_upward_seconds",
    "refresh_nearfield_seconds",
    "refresh_total_seconds",
)


def make_ic(name: str, n: int, ndev: int, seed: int):
    if name == "clusters":
        return IC_GENERATORS["clusters"](ndev, n // ndev, seed=seed)
    if name == "disk":
        return IC_GENERATORS["disk"](subsample=n, seed=seed)
    return IC_GENERATORS[name](n, seed=seed)


def rel_errors(a: np.ndarray, a_ref: np.ndarray, idx=None) -> dict:
    """Per-particle relative acceleration error plus an aggregate L2.

    ``idx`` restricts the comparison to a subsample of targets (identical for
    both codes); the memory ``rel-l2-probe-not-comparable`` applies *across*
    subsample sizes, never within one, so ``n_ref`` is recorded on the result.

    ``aggL2_signflip`` exists purely as a convention tripwire: if a code returned
    the opposite sign convention we would otherwise report a ~2.0 "error" and
    call it an accuracy result.  If the flipped number is the small one, the
    comparison is wired wrong -- fix it, do not report it.
    """
    a = np.asarray(a, np.float64)
    ref = np.asarray(a_ref, np.float64)
    if idx is not None:
        a = a[idx]
    num = np.linalg.norm(a - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-300
    per = num / den
    denom = np.linalg.norm(ref) + 1e-300
    return dict(
        median=float(np.median(per)),
        p90=float(np.percentile(per, 90)),
        max=float(np.max(per)),
        aggL2=float(np.linalg.norm(a - ref) / denom),
        aggL2_signflip=float(np.linalg.norm(-a - ref) / denom),
        n_ref=int(ref.shape[0]),
    )


def device_pool(args) -> list[int]:
    """Physical GPU indices to use, all verified idle (plan T3.1).

    An explicit ``CUDA_VISIBLE_DEVICES`` is honoured but still checked; the guard
    refuses a busy card unless ``--allow-busy`` says the numbers are not for the
    record.
    """
    need = max(args.ndevs)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        pool = [int(x) for x in cvd.split(",") if x.strip()]
        idle, reasons = idle_gpus(settle_s=2.0, samples=4)
        busy = [d for d in pool if d not in idle]
        if busy and not args.allow_busy:
            raise SystemExit(
                "CUDA_VISIBLE_DEVICES names busy card(s) "
                + "; ".join(f"GPU {d}: {reasons.get(d)}" for d in busy)
                + " -- pick idle ones (unset CUDA_VISIBLE_DEVICES to let the guard choose) "
                "or pass --allow-busy for a non-record run"
            )
        if busy:
            print(f"!! running on busy GPU(s) {busy}; timings are NOT for the record")
        return pool[:need] if len(pool) >= need else pool
    return pick_idle_gpus(need)


# --------------------------------------------------------------------------
# phase 1: pkdgrav3 (subprocesses, before JAX exists in this process)
# --------------------------------------------------------------------------


def _attach_gravity_reports(summary: dict, stdout: str, thetas: list[float], n: int) -> None:
    """Map pkdgrav3's per-call ``Gravity Calculated`` blocks onto the theta rows.

    The script runs ``warmup + repeats`` gravity calls per theta in order, so
    block ``i`` belongs to theta ``i // per_theta``; the interaction-list
    statistics (``P-P per active`` etc.) are identical across the repeats of one
    theta, so the last block of each group is attached together with the budget
    it implies.
    """
    reports = parse_gravity_lines(stdout)
    results = summary["results"]
    per_theta = int(summary.get("warmup", results[0]["warmup"])) + int(results[0]["repeats"])
    ok = len(reports) == len(results) * per_theta
    for i, res in enumerate(results):
        blk = reports[i * per_theta:(i + 1) * per_theta] if ok else []
        blk = [b for b in blk if "pp_per_active" in b]
        res["gravity_report"] = blk[-1] if blk else None
        res["gravity_reports_matched"] = ok
        if blk:
            res["budget"] = pkdgrav3_direct_budget(
                blk[-1]["pp_per_active"]["avg"], blk[-1]["pc_per_active"]["avg"],
                summary.get("n_bucket", 16), n,
            )
        res["theta_jaccpot_equivalent"] = float(res["theta"]) / PKD_THETA_SCALE
    if not ok:
        print(f"  !! pkdgrav3 emitted {len(reports)} gravity reports, expected "
              f"{len(results) * per_theta}; P-P/P-C not attached", flush=True)


def run_pkdgrav3_phase(args, ic_path: Path, workdir: Path, pool: list[int], n: int) -> list[dict]:
    rows = []
    variants = [("gpu", True, args.pkd_cores)]
    if args.pkd_cpu:
        variants.append(("cpu", False, args.pkd_cpu_cores))
    for ndev in args.ndevs:
        devs = ",".join(str(d) for d in pool[:ndev])
        for nbucket in args.pkd_nbucket:
            for ngroup in args.pkd_ngroup:
                for tag, gpu, cores in variants:
                    if not gpu and ndev != args.ndevs[0]:
                        continue  # the CPU path does not use the GPU count
                    out_prefix = workdir / f"pkd_{tag}_ndev{ndev}_nb{nbucket}_ng{ngroup}"
                    env_extra = {
                        "CUDA_VISIBLE_DEVICES": devs,
                        "PKD_IC": str(ic_path),
                        "PKD_OUT": str(out_prefix),
                        "PKD_THETAS": ",".join(f"{t:g}" for t in args.pkd_thetas),
                        "PKD_REPEATS": str(args.repeats),
                        "PKD_WARMUP": str(args.warmup),
                        "PKD_SOFT": str(args.softening),
                        "PKD_NBUCKET": str(nbucket),
                        "PKD_NGROUP": str(ngroup),
                        "PKD_GPU": "1" if gpu else "0",
                    }
                    print(f"\n=== pkdgrav3 [{tag}] ndev={ndev} nBucket={nbucket} nGroup={ngroup or 'default'} "
                          f"cores={cores} (CUDA_VISIBLE_DEVICES={devs}) ===", flush=True)
                    with GpuMonitor(pool[:ndev]) as mon:
                        run = run_pkdgrav3(
                            PKD_SCRIPT, env_extra=env_extra, cores=cores, timeout_s=args.timeout
                        )
                    cont = mon.summary()
                    (workdir / f"{out_prefix.name}_stdout.log").write_text(run.stdout)
                    (workdir / f"{out_prefix.name}_stderr.log").write_text(run.stderr)
                    print(run.stdout[-2500:], flush=True)
                    summary_path = f"{out_prefix}_summary.json"
                    # pkdgrav3 exits 0 even when the embedded interpreter raises, so the
                    # summary file -- not the return code -- is what says the run succeeded.
                    if run.returncode != 0 or not os.path.exists(summary_path):
                        print(run.stderr[-4000:], file=sys.stderr)
                        raise SystemExit(
                            f"pkdgrav3 produced no summary for {out_prefix.name} (rc={run.returncode}); "
                            "see the captured stdout above for the embedded-interpreter traceback"
                        )
                    with open(summary_path) as fh:
                        summary = json.load(fh)
                    summary["warmup"] = args.warmup
                    _attach_gravity_reports(summary, run.stdout, args.pkd_thetas, n)
                    if cont.flags:
                        print(f"  !! contention during pkdgrav3 run: {cont.flags}", flush=True)
                    rows.append(
                        dict(
                            code="pkdgrav3",
                            variant=tag,
                            ndev=ndev,
                            devices=devs,
                            cores=cores,
                            n_bucket=nbucket,
                            n_group=ngroup,
                            summary=summary,
                            run=run.as_dict(),
                            contention=cont.as_dict(),
                            stdout_log=str(workdir / f"{out_prefix.name}_stdout.log"),
                        )
                    )
    return rows


# --------------------------------------------------------------------------
# phase 2: jaccpot (in-process; imports JAX only once pkdgrav3 is done)
# --------------------------------------------------------------------------


def _reduce_overflow_flags(diag) -> list[str]:
    """Which traversal buffers overflowed, if any.

    An overflowed buffer silently drops interactions -- in this project's history
    a queue-cap overflow once cost 90% force error while every other signal
    looked healthy -- so a row that trips this is not a measurement.
    """
    from jaccpot.distributed.fmm import DIAG_FIELDS, _OVERFLOW_FIELDS

    tripped = []
    for name in _OVERFLOW_FIELDS:
        col = DIAG_FIELDS.index(name)
        if np.any(np.asarray(diag)[:, col] > 0):
            tripped.append(name)
    return tripped


def _peak_bytes(ndev):
    import jax

    peak = {}
    for i, dev in enumerate(jax.local_devices()[:ndev]):
        try:
            peak[i] = int(dev.memory_stats().get("peak_bytes_in_use", 0))
        except Exception:
            pass
    return peak


#: The device-only fused fast lane, exactly as ``benchmark_a100/env_fused.sh`` and
#: ``odisseo/jaccpot_coupling.py::_large_n_environment_overrides`` set it.  Without
#: these jaccpot runs a ~10x slower path (``fastlane_attempts=0``) and a comparison
#: would understate it by more than an order of magnitude -- measured here at
#: N=20k: 2.5 s/call on the plain ``compute_accelerations`` entry point.
#: Several of them are read in ``FastMultipoleMethod.__init__`` and captured as
#: instance state, so they MUST be in the environment before the solver is built.
#:
#: ``RADIX_FAST_PAYLOAD_MAX_MB=0`` is this harness's addition (2026-09-06).  With
#: jaccpot's default of 1024 the lane materialises a per-particle source payload
#: whenever ``leaves * slots * W * 9 B <= 1 GB`` and then runs
#: ``nearfield_fused_leaf_t32_k...``, which streams ALL padded sources per target
#: -- a masked all-pairs sum, flat in theta: leaf 1024 at 200k cost 175 ms at
#: every theta, leaf 512 176 ms, leaf 256 at N<=100k likewise.  With the payload
#: cap at 0 the gather kernel (``nearfield_leafpair_...``) runs the actual
#: neighbour lists: leaf 512 -> 59 ms, leaf 1024 -> 87 ms at theta 1.0, same
#: forces.  Every jaccpot row records which kernel layout it got.
FAST_LANE_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "131072",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    # int32 indices for the fused single-GPU lane (plan "tree walk" 0.2): halves every
    # queue byte of the traced walk. Both are read at IMPORT; yggdrax falls back to the
    # jaccpot variable but not the other way round, so both are set. Ranges at leaf 32,
    # N=200k: leaves x neighbour cap 1e8, nodes x far cap 2e8 -- far below 2^31.
    "YGGDRAX_INDEX_PRECISION": "int32",
    "JACCPOT_INDEX_PRECISION": "int32",
}


def apply_fast_lane_env(n: int, *, edge_cap: int | None = None,
                        overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Install the fused fast-lane environment; must run before jaccpot is imported.

    ``edge_cap`` sets ``JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP``: the
    200k default (2^21) does not fit N=800k at theta 0.6 / leaf 256 (~2.2M edges),
    and the failure is a hard error, not a silent fallback -- but a sweep that
    hits it looks like "jaccpot cannot scale".  Default here: ``2^21 * ceil(N / 200k)``.
    """
    env = dict(FAST_LANE_ENV)
    env["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = str(int(n))
    if edge_cap is None:
        edge_cap = (1 << 21) * max(1, -(-int(n) // 200_000))
    env["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = str(int(edge_cap))
    if overrides:
        env.update({k: str(v) for k, v in overrides.items()})
    for k, v in env.items():
        os.environ[k] = v
    return env


#: Per-leaf capacity fits (plan "small leaves", Phase 0.2, ``codes/fit_smallleaf_caps.py``).
#: ``FAST_LANE_ENV`` is the leaf-256 fit.  Smaller leaves have 3-10x more far
#: pairs, ~2x the neighbour edges and a longest neighbour row of ``num_leaves-1``
#: (a Plummer tail leaf sees every other leaf), and each of those saturates a
#: fixed cap that the fused lane cannot widen inside the compiled scan -- a
#: saturated cap TRUNCATES the lists silently (PR #333).  Every measurement at a
#: given leaf must use the same caps, so they live here.  Values are env-var
#: strings; the pseudo-key ``_traversal_overrides`` holds ``TraversalOverrides``
#: fields (an explicit override bypasses the 2048 neighbours-per-leaf clamp of
#: ``fmm_overrides.py``).  Caps are for N=200k; ``fast_lane_overrides_for_leaf``
#: scales the two list caps with N.
FAST_LANE_ENV_BY_LEAF: dict[int, dict] = {
    # Fitted 2026-09-07 (artifacts/smallleaf/fit_leaf*_th0.6*.json), N=200k Plummer,
    # p=4, theta 0.6.  Rules that came out of the fitting:
    #  * compact far-pair cap: pow2 >= ~1.5x the far-pair count
    #    (101k / 363k / 992k / 2.34M at leaf 256 / 128 / 64 / 32);
    #  * neighbour-edge cap: inside the compiled scan the edge count is the PADDED
    #    shape num_leaves x traced_neighbour_cap, and #333 sets the traced cap to
    #    pow2(1.5 x longest eager row + 1) with the longest row = num_leaves - 1,
    #    so the cap must be >= num_leaves x pow2(1.5 num_leaves): 782x2048 = 1.6M
    #    (2^21 ok), 1563x4096 = 6.4M (2^23), 3125x8192 = 25.6M (2^25),
    #    6250x16384 = 102M (2^27) -- the leaf-256 default 2^21 fails at leaf 128;
    #  * max_neighbors_per_leaf must be explicit above 2048 (the clamp) -- the
    #    longest row IS num_leaves - 1 (a Plummer tail leaf sees every leaf);
    #  * leaf 32 also overflows the far-field per-node interaction cap (8192 clamp)
    #    and needs an explicit max_interactions_per_node.
    256: {},
    128: {
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": str(1 << 20),
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": str(1 << 23),
    },
    64: {
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": str(1 << 21),
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": str(1 << 25),
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "auto",
        # theta 0.4 overflows the 8192 per-node far cap at leaf 64 too (U-curve 2026-09-09)
        "_traversal_overrides": {"max_neighbors_per_leaf": 4096,
                                 "max_interactions_per_node": 16384},
    },
    32: {
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": str(1 << 22),
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": str(1 << 27),
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "auto",
        "_traversal_overrides": {"max_neighbors_per_leaf": 8192,
                                 "max_interactions_per_node": 16384},
    },
}


def fast_lane_overrides_for_leaf(leaf: int, n: int) -> dict[str, str]:
    """Env overrides (strings only) for ``leaf`` at particle count ``n``.

    The two list caps are the 200k fit scaled by ``ceil(n / 200k)`` and rounded
    up to a power of two; the pseudo-key ``_traversal_overrides`` is dropped
    (callers read it from ``FAST_LANE_ENV_BY_LEAF`` directly).
    """
    entry = FAST_LANE_ENV_BY_LEAF.get(int(leaf))
    if entry is None:
        return {}
    scale = max(1, -(-int(n) // 200_000))
    out: dict[str, str] = {}
    for k, v in entry.items():
        if k.startswith("_"):
            continue
        if k in ("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
                 "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP") and scale > 1:
            val = int(v) * scale
            v = str(1 << (val - 1).bit_length())
        out[k] = str(v)
    return out


def _nearfield_kernel_layout(prepared) -> str:
    """Which near-field kernel the fused lane will run for this state."""
    payload = getattr(prepared, "radix_fast_payload", None)
    if payload is None:
        return "unknown"
    spid = getattr(payload, "source_particle_ids", None)
    if spid is not None and int(np.prod(spid.shape)) > 0:
        return f"streaming_all_sources(k={int(spid.shape[1]) * int(spid.shape[2])})"
    return "leafpair_gather"


def run_jaccpot_single_gpu(args, pos, mass, a_ref, ref_idx, softening, devices):
    """ndev=1 uses jaccpot's strict FUSED eval seam -- the only one that is fast.

    Two separate discoveries forced this entry point, both verified by reading
    ``jaccpot/runtime/fmm_strict_run.py`` and by measurement at N=200k:

    * The distributed driver cannot run on one device at all: its LET stage
      builds a coarse tree from the *remote* particle set, empty with no remote
      devices ("Need at least one particle" out of yggdrax).
    * ``_strict_fused_mode_active`` is assigned in exactly two places -- inside
      ``strict_run_v2`` and inside ``strict_fused_prepared_eval_fn``.  It is NOT
      set by ``compute_accelerations``, by ``prepare_state`` /
      ``evaluate_prepared_state``, or by ``strict_prepare_refresh_and_evaluate``.
      Measured at N=200k, p=3, theta=0.5 on an A100: those paths cost ~2.9-3.0 s
      per call with ``strict_fused_mode_active=False``, while the fused eval is
      **110 ms** -- a factor of 27 -- for a bit-comparable answer (aggL2 8.940e-4
      either way, so the fused lane is accuracy-neutral).

    ``strict_fused_prepared_eval_fn`` exists precisely for this benchmark; its own
    docstring calls it a seam "for apples-to-apples benchmarking against
    functional FMM eval APIs".  It returns ``(prepared_state, eval_fn)`` where
    ``eval_fn`` runs the self-force evaluation with **no refresh and no Verlet
    update** -- pkdgrav3's ``gravity()`` with the tree built AND the walk done.
    """
    import time

    import jax

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TraversalOverrides,
        TreeConfig,
    )

    # At N=1M, theta<=0.6, leaf 256 the preset's far-field cap of 1024 accepted
    # pairs per node overflows ("Interaction list capacity exceeded") and the
    # strict streamed retry does not grow that one, so the run dies with a
    # "could not fit" report while the card has 29 GiB free. A TraversalOverrides
    # raises only the named capacity and leaves the preset's N-sizing alone.
    rows = []
    n = int(pos.shape[0])
    pos_j = jax.numpy.asarray(pos, jax.numpy.float32)
    mass_j = jax.numpy.asarray(mass, jax.numpy.float32)
    for leaf in args.jac_leaf_single:
        # The leaf preset's traversal overrides (FAST_LANE_ENV_BY_LEAF, the only
        # non-env part of a capacity fit); --jac-max-interactions wins when given.
        trav = dict((FAST_LANE_ENV_BY_LEAF.get(int(leaf)) or {}).get("_traversal_overrides", {}))
        if args.jac_max_interactions:
            trav["max_interactions_per_node"] = int(args.jac_max_interactions)
        runtime_cfg = RuntimePolicyConfig()
        if trav:
            runtime_cfg = RuntimePolicyConfig(
                traversal_config=TraversalOverrides(**{k: int(v) for k, v in trav.items()})
            )
        for order in args.orders:
            for theta in args.jac_thetas:
                solver = FastMultipoleMethod(
                    preset="large_n_gpu",
                    runtime_path="large_n",
                    basis="real",
                    theta=theta,
                    G=args.G,
                    softening=softening,
                    working_dtype=jax.numpy.float32,
                    advanced=FMMAdvancedConfig(
                        tree=TreeConfig(mode="static_radix", leaf_target=leaf),
                        farfield=FarFieldConfig(mode="auto"),
                        nearfield=NearFieldConfig(mode="auto"),
                        runtime=runtime_cfg,
                        mac_type="dehnen",
                    ),
                    fixed_order=order,
                )
                # prepare-stage timings (the MAC walk) are recorded only under this
                # flag; strict_run_v2 sets it, the eval-only seam does not.
                solver._refresh_timing_active = True
                try:
                    t0 = time.perf_counter()
                    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
                        positions=pos_j,
                        masses=mass_j,
                        leaf_size=leaf,
                        max_order=order,
                        theta=theta,
                    )
                    prepare_s = time.perf_counter() - t0  # includes compile; see note
                    budget = jaccpot_direct_budget(prepared, n)
                    layout = _nearfield_kernel_layout(prepared)
                    out, timing, cont = timed_calls(
                        lambda: eval_fn(prepared), repeats=args.repeats,  # noqa: B023
                        warmup=args.warmup, devices=devices, block=jax.block_until_ready,
                    )
                except Exception as exc:
                    full = str(exc)
                    msg = full.splitlines()[0][:200]
                    # jaccpot's capacity report (buffer sizes + a config that fits) is
                    # in the FOLLOWING lines; keep them, a first line alone is useless.
                    print(f"  jaccpot[1gpu] leaf={leaf} p={order} theta={theta:.2f}: FAILED -- {msg}",
                          flush=True)
                    for line in full.splitlines()[1:14]:
                        print(f"      {line[:200]}", flush=True)
                    causes = []
                    cause = exc.__cause__ or exc.__context__
                    while cause is not None and len(causes) < 4:
                        causes.append(f"{type(cause).__name__}: {str(cause)[:1500]}")
                        cause = cause.__cause__ or cause.__context__
                    for c in causes:
                        print(f"      caused by {c[:300]}", flush=True)
                    rows.append(dict(code="jaccpot", lane="single_gpu_fused", ndev=1,
                                     order=order, theta=theta, leaf_size=leaf, failed=msg,
                                     failed_full=full[:6000], failed_causes=causes))
                    continue

                diag = dict(solver.get_runtime_diagnostics() or {})
                fused_active = bool(diag.get("strict_fused_mode_active"))
                a = np.asarray(out)
                walk = {k: diag.get(k) for k in WALK_KEYS}
                walk_s = sum(float(walk.get(k) or 0.0) for k in (
                    "refresh_dual_setup_seconds", "refresh_dual_artifact_build_seconds",
                    "refresh_dual_select_interactions_seconds", "refresh_dual_far_pair_plan_seconds"))
                row = dict(
                    code="jaccpot",
                    lane="single_gpu_fused",
                    ndev=1,
                    order=order,
                    theta=theta,
                    theta_pkd_equivalent=theta * PKD_THETA_SCALE,
                    leaf_size=leaf,
                    traversal_overrides=trav,
                    force_eval_s=timing,
                    walk_s=walk_s,
                    walk_stage_s=walk,
                    prepare_s=prepare_s,
                    peak_bytes_in_use=_peak_bytes(1),
                    errors=rel_errors(a, a_ref, ref_idx),
                    budget=budget,
                    nearfield_kernel_layout=layout,
                    contention=cont.as_dict(),
                    diagnostics=dict(
                        recent_dual_neighbor_count=diag.get("recent_dual_neighbor_count"),
                        static_radix_far_pair_count=diag.get("static_radix_far_pair_count"),
                        large_n_eval_active_leaf_count=diag.get("large_n_eval_active_leaf_count"),
                    ),
                    fastlane=dict(
                        fused_mode_active=fused_active,
                        fastlane_hits=diag.get("strict_fused_fastlane_hits"),
                        fastlane_attempts=diag.get("strict_fused_fastlane_attempts"),
                        fallback_count=diag.get("strict_fused_fallback_count"),
                        last_fallback_reason=diag.get("strict_fused_last_fallback_reason"),
                    ),
                )
                rows.append(row)
                # A silently-unfused row is ~27x slow and would misrepresent jaccpot,
                # so it is flagged in the output, not just recorded.
                warn = "" if fused_active else "  !! NOT FUSED -- timing is not representative"
                if cont.flags:
                    warn += f"  !! {','.join(cont.flags)}"
                print(
                    f"  jaccpot[1gpu] leaf={leaf} p={order} theta={theta:.2f}: "
                    f"{timing['min']*1e3:8.2f} ms (+walk {walk_s*1e3:6.1f})  "
                    f"aggL2={row['errors']['aggL2']:.3e}  direct {budget['direct_share_of_N']:.3f}N  "
                    f"{layout}{warn}",
                    flush=True,
                )
                del prepared, eval_fn, solver
    return rows


def run_jaccpot_phase(args, pos, mass, a_ref, ref_idx, pool, softening):
    import jax
    from jaccpot.distributed import (
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
        scatter_to_input_order,
    )
    from yggdrax.distributed import make_mesh

    rows = []
    for ndev in args.ndevs:
        if ndev > jax.device_count():
            print(f"skip jaccpot ndev={ndev} (only {jax.device_count()} visible)")
            continue
        if ndev == 1:
            rows += run_jaccpot_single_gpu(args, pos, mass, a_ref, ref_idx, softening, pool[:1])
            continue
        mesh = make_mesh(ndev)
        part = partition_for_devices(
            pos, mass, ndev, leaf_size=args.jac_leaf_dist, partitioner=args.partitioner
        )
        pos_f = jax.numpy.asarray(part["pos_flat"])
        mass_f = jax.numpy.asarray(part["mass_flat"])
        gid_f = jax.numpy.asarray(part["gid_flat"])
        counts = jax.numpy.asarray(part["counts"])
        cap = int(part["cap"])

        for order in args.orders:
            for theta in args.jac_thetas:
                cfg = DistributedFMMConfig(
                    order=order,
                    theta=theta,
                    leaf_size=args.jac_leaf_dist,
                    softening=softening,
                    G=args.G,
                    partitioner=args.partitioner,
                )
                fn = make_force_evaluator(cfg, ndev, cap, mesh, jit=True)
                out, timing, cont = timed_calls(
                    lambda: fn(pos_f, mass_f, gid_f, counts), repeats=args.repeats,  # noqa: B023
                    warmup=args.warmup, devices=pool[:ndev], block=jax.block_until_ready,
                )
                accel, gid, diag = out
                a = scatter_to_input_order(np.asarray(accel), np.asarray(gid), pos.shape[0])
                overflow = _reduce_overflow_flags(np.asarray(diag))
                row = dict(
                    code="jaccpot",
                    lane="distributed",
                    ndev=ndev,
                    overflow=overflow,
                    order=order,
                    theta=theta,
                    theta_pkd_equivalent=theta * PKD_THETA_SCALE,
                    leaf_size=args.jac_leaf_dist,
                    partitioner=args.partitioner,
                    force_eval_s=timing,
                    peak_bytes_in_use=_peak_bytes(ndev),
                    errors=rel_errors(a, a_ref, ref_idx),
                    contention=cont.as_dict(),
                    diagnostics=np.asarray(diag).tolist(),
                )
                rows.append(row)
                print(
                    f"  jaccpot ndev={ndev} p={order} theta={theta:.2f}: "
                    f"{timing['min']*1e3:8.2f} ms  aggL2={row['errors']['aggL2']:.3e}"
                    + ("  !! TRAVERSAL OVERFLOW: " + ",".join(overflow) if overflow else "")
                    + (f"  !! {','.join(cont.flags)}" if cont.flags else ""),
                    flush=True,
                )
    return rows


def _fan_out_per_leaf(args) -> None:
    """Run ``main`` once per single-GPU leaf size in a child process and merge."""
    import subprocess
    import tempfile
    import time

    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = Path(args.out or (root / "artifacts" / "compare_force.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = sys.argv[1:]

    def _strip(argv, flag, nargs_plus=True):
        res, skip = [], False
        for tok in argv:
            if skip:
                if tok.startswith("--"):
                    skip = False
                else:
                    continue
            if tok == flag:
                skip = nargs_plus
                continue
            res.append(tok)
        return res

    base = _strip(_strip(argv, "--jac-leaf-single"), "--out")
    merged = None
    tmpdir = Path(tempfile.mkdtemp(prefix="compare_force_leaf_"))
    for i, leaf in enumerate(args.jac_leaf_single):
        child_out = tmpdir / f"leaf{leaf}.json"
        cmd = [sys.executable, os.path.abspath(__file__), *base,
               "--jac-leaf-single", str(leaf), "--out", str(child_out)]
        if i > 0 and "--skip-pkdgrav3" not in cmd:
            cmd.append("--skip-pkdgrav3")
        print(f"\n=== compare_force child leaf={leaf}: {' '.join(cmd[2:])}", flush=True)
        # The previous child's utilisation lingers in nvidia-smi for a few seconds and
        # the idle guard then rejects the very card that child just released (it cost
        # the 1M leaf-256 rows on 2026-09-10): settle, and retry the guard a few times.
        if i > 0:
            time.sleep(15)
        rc = 1
        for attempt in range(4):
            rc = subprocess.run(cmd).returncode
            if rc == 0 or child_out.exists():
                break
            print(f"   child leaf={leaf} rc={rc}; retrying in 30 s ({attempt + 1}/4)", flush=True)
            time.sleep(30)
        if rc != 0 or not child_out.exists():
            print(f"!! child for leaf {leaf} failed (rc={rc}); its rows are missing", flush=True)
            continue
        with open(child_out) as fh:
            payload = json.load(fh)
        if merged is None:
            merged = payload
            merged["fast_lane_env_by_leaf"] = {str(leaf): payload.get("fast_lane_env")}
        else:
            merged["jaccpot"] += payload.get("jaccpot", [])
            merged["pkdgrav3"] += payload.get("pkdgrav3", [])
            merged["fast_lane_env_by_leaf"][str(leaf)] = payload.get("fast_lane_env")
    if merged is None:
        raise SystemExit("every per-leaf child failed")
    merged["provenance"]["args"]["jac_leaf_single"] = list(args.jac_leaf_single)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"\nwrote {out} (merged {len(args.jac_leaf_single)} per-leaf children)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer", choices=sorted(IC_GENERATORS))
    ap.add_argument("--ndevs", type=int, nargs="+", default=[1])
    ap.add_argument("--pkd-thetas", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--jac-thetas", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--jac-leaf-single", type=int, nargs="+", default=[256],
                    help="leaf size(s) for jaccpot's single-GPU fused lane. 256 is the "
                         "validated 200k production value (see FAST_LANE_ENV for the "
                         "streaming-kernel trap at leaf >= 512)")
    ap.add_argument("--jac-leaf-dist", type=int, default=64,
                    help="leaf size for the distributed lane (DistributedFMMConfig default)")
    ap.add_argument("--jac-edge-cap", type=int, default=None,
                    help="JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP; default 2^21 per 200k of N")
    ap.add_argument("--jac-max-interactions", type=int, default=0,
                    help="TraversalOverrides(max_interactions_per_node=...) for the single-GPU "
                         "lane; 0 = preset (1024 at large N, which overflows at N=1M theta<=0.6)")
    ap.add_argument("--jac-env", nargs="+", default=[], metavar="KEY=VAL",
                    help="extra fast-lane environment overrides")
    ap.add_argument("--partitioner", default="rcb")
    ap.add_argument("--pkd-nbucket", type=int, nargs="+", default=[16],
                    help="pkdgrav3 nBucket value(s); several = its own leaf sweep (plan T3.0)")
    ap.add_argument("--pkd-ngroup", type=int, nargs="+", default=[0],
                    help="pkdgrav3 nGroup value(s); 0 = pkdgrav3 default")
    ap.add_argument("--pkd-cores", type=int, default=16,
                    help="mdl -sz; pkdgrav3 uses the host CPU heavily, jaccpot does not, "
                         "so this number must be reported with any timing")
    ap.add_argument("--pkd-cpu", action="store_true",
                    help="also run pkdgrav3's CPU/double path (bGPU=False) as a second curve "
                         "without the fp32 P-P floor")
    ap.add_argument("--pkd-cpu-cores", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ref-targets", type=int, default=0,
                    help="0 = full O(N^2) float64 reference; else this many target rows "
                         "(seeded, identical for both codes) -- required above ~400k")
    ap.add_argument("--softening", type=float, default=1e-7,
                    help="Regularization epsilon, NOT a physical softening. jaccpot's "
                         "distributed lane pads each device up to a capacity and those "
                         "coincident zero-mass rows produce NaN at exactly 0 (measured: "
                         "294 NaN rows, 6 of them real particles, at N=20k/2 GPUs). The "
                         "default is small enough that both codes' kernels are Newtonian "
                         "far below the measurement floor; --verify-softening checks it.")
    ap.add_argument("--softening-tolerance", type=float, default=1e-6,
                    help="Fail if the epsilon moves the direct-sum reference by more "
                         "than this (aggL2), i.e. if it is not negligible for this IC")
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if CUDA_VISIBLE_DEVICES names a busy card (never for the record)")
    ap.add_argument("--skip-pkdgrav3", action="store_true")
    ap.add_argument("--skip-jaccpot", action="store_true")
    args = ap.parse_args()

    # Several single-GPU leaf sizes: one PROCESS per leaf. The fused lane's caps
    # (FAST_LANE_ENV_BY_LEAF) are read at import / solver construction and differ
    # per leaf, so a leaf sweep inside one process would run every leaf under the
    # first leaf's caps -- and a saturated cap truncates lists silently (#333).
    # pkdgrav3 runs in the first child only; the children's JSONs are merged here.
    if len(args.jac_leaf_single) > 1 and not args.skip_jaccpot:
        return _fan_out_per_leaf(args)

    # honest GPU memory numbers require this off; JAX otherwise grabs ~75% of VRAM
    # up front and every "peak" reading is the preallocation, not the algorithm.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")
    overrides = {}
    if len(args.jac_leaf_single) == 1:
        overrides.update(fast_lane_overrides_for_leaf(args.jac_leaf_single[0], args.n))
    overrides.update(dict(kv.split("=", 1) for kv in args.jac_env))  # --jac-env wins
    fast_lane_env = apply_fast_lane_env(args.n, edge_cap=args.jac_edge_cap, overrides=overrides)
    if len(args.jac_leaf_single) == 1:
        # the leaf preset's edge cap must not be undone by the N-scaled default
        preset_edge = fast_lane_overrides_for_leaf(args.jac_leaf_single[0], args.n).get(
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP")
        if preset_edge is not None and args.jac_edge_cap is None:
            os.environ["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = preset_edge
            fast_lane_env["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = preset_edge

    pool = device_pool(args)
    set_cuda_visible(pool)
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workdir = Path(args.workdir or (root / "artifacts" / "compare_force"))
    workdir.mkdir(parents=True, exist_ok=True)

    pos, mass = make_ic(args.ic, args.n, max(args.ndevs), args.seed)
    n = pos.shape[0]
    print(f"IC={args.ic} N={n} devices={pool} loadavg={os.getloadavg()} cores={os.cpu_count()}")

    ic_path = workdir / f"{args.ic}_n{n}_seed{args.seed}.tipsy"
    write_tipsy_dark(ic_path, pos, mass, softening=args.softening,
                     allow_softening=True)
    rb_pos, rb_mass, _, rb_eps, hdr = read_tipsy_dark(ic_path)
    assert hdr["nbodies"] == n, (hdr, n)
    assert np.array_equal(rb_pos, pos.astype(np.float32)), "tipsy position round-trip failed"
    assert np.array_equal(rb_mass, mass.astype(np.float32)), "tipsy mass round-trip failed"
    assert np.allclose(rb_eps, np.float32(args.softening))
    print(f"wrote + verified {ic_path} ({ic_path.stat().st_size/1e6:.1f} MB)")

    pkd_rows = []
    if not args.skip_pkdgrav3:
        if not Path(PKDGRAV3_BIN).exists():
            raise SystemExit(f"no pkdgrav3 binary at {PKDGRAV3_BIN}")
        pkd_rows = run_pkdgrav3_phase(args, ic_path, workdir, pool, n)

    # -------- reference + jaccpot (JAX enters the process only now) --------
    from common.reference import direct_accelerations

    ref_idx = None
    if args.ref_targets and args.ref_targets < n:
        ref_idx = np.sort(np.random.default_rng(12345).choice(n, args.ref_targets, replace=False))
    print(f"\ncomputing float64 direct-sum reference (eps={args.softening:g}, "
          f"targets={'all' if ref_idx is None else len(ref_idx)})...", flush=True)
    a_ref = direct_accelerations(pos, mass, G=args.G, softening=args.softening,
                                 block_size=1024, target_indices=ref_idx)
    # The two codes use different softening kernels (Plummer vs compact-support
    # spline), so the epsilon is only legitimate while it is dynamically
    # irrelevant. Measure that rather than assuming it: how far does it move the
    # reference away from the unsoftened Newtonian answer?
    a_ref0 = direct_accelerations(pos, mass, G=args.G, softening=0.0, block_size=1024,
                                  target_indices=ref_idx)
    eps_shift = float(np.linalg.norm(a_ref - a_ref0) / np.linalg.norm(a_ref0))
    print(f"  epsilon moves the reference by aggL2 {eps_shift:.3e} "
          f"(tolerance {args.softening_tolerance:g})")
    if eps_shift > args.softening_tolerance:
        raise SystemExit(
            f"softening {args.softening:g} is NOT negligible for this IC "
            f"(shifts the reference by {eps_shift:.3e}); the Plummer-vs-spline "
            "kernel difference would contaminate the comparison. Lower --softening."
        )

    for row in pkd_rows:
        for res in row["summary"]["results"]:
            path = res.get("arrays")
            if not path or not os.path.exists(path):
                continue
            z = np.load(path)
            res["errors"] = rel_errors(z["acc"], a_ref, ref_idx)
            b = res.get("budget") or {}
            print(
                f"  pkdgrav3[{row['variant']}] ndev={row['ndev']} nB={row['n_bucket']} "
                f"theta={res['theta']:.2f} (jac-eq {res['theta']/PKD_THETA_SCALE:.3f}): "
                f"grav {res['gravity_s']['min']*1e3:8.2f} ms  "
                f"aggL2={res['errors']['aggL2']:.3e}"
                f"  P-P {b.get('pp_per_active', float('nan')):.1f} P-C {b.get('pc_per_active', float('nan')):.1f}"
                f"  (signflip {res['errors']['aggL2_signflip']:.3e})",
                flush=True,
            )

    jac_rows = []
    if not args.skip_jaccpot:
        jac_rows = run_jaccpot_phase(args, pos, mass, a_ref, ref_idx, pool, args.softening)

    out = Path(args.out or (root / "artifacts" / "compare_force.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    from common.env import capture_provenance

    payload = dict(
        provenance=capture_provenance({"benchmark": "compare_force", "args": vars(args),
                                       "loadavg": os.getloadavg(), "cores": os.cpu_count()}),
        ic=dict(name=args.ic, n=n, seed=args.seed, softening=args.softening,
                softening_reference_shift_aggL2=eps_shift, G=args.G,
                reference_targets=None if ref_idx is None else int(len(ref_idx))),
        fast_lane_env=fast_lane_env,
        devices=pool,
        pkd_theta_scale=PKD_THETA_SCALE,
        pkdgrav3=pkd_rows,
        jaccpot=jac_rows,
    )
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
