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

Run (2 GPUs picked by autocvd, jaccpot side needs the jax-0.10.2 venv):
    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o -q) JAX_ENABLE_X64=1 \
      /export/home/tbuck/jaccpot/.venv/bin/python \
      benchmark_multigpu/codes/compare_force.py --n 200000 --ndevs 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.ic import IC_GENERATORS
from common.pkdgrav3 import (
    PKDGRAV3_BIN,
    read_tipsy_dark,
    run_pkdgrav3,
    write_tipsy_dark,
)

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
PKD_SCRIPT = HERE / "pkdgrav3_force_eval.py"


def make_ic(name: str, n: int, ndev: int, seed: int):
    if name == "clusters":
        return IC_GENERATORS["clusters"](ndev, n // ndev, seed=seed)
    if name == "disk":
        return IC_GENERATORS["disk"](subsample=n, seed=seed)
    return IC_GENERATORS[name](n, seed=seed)


def rel_errors(a: np.ndarray, a_ref: np.ndarray) -> dict:
    """Per-particle relative acceleration error plus an aggregate L2.

    ``aggL2_signflip`` exists purely as a convention tripwire: if a code returned
    the opposite sign convention we would otherwise report a ~2.0 "error" and
    call it an accuracy result.  If the flipped number is the small one, the
    comparison is wired wrong -- fix it, do not report it.
    """
    a = np.asarray(a, np.float64)
    ref = np.asarray(a_ref, np.float64)
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
    )


def device_pool() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    pool = [x.strip() for x in cvd.split(",") if x.strip()]
    if not pool:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is unset. Pick free GPUs explicitly, e.g.\n"
            "  CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o -q) ... compare_force.py"
        )
    return pool


# --------------------------------------------------------------------------
# phase 1: pkdgrav3 (subprocesses, before JAX exists in this process)
# --------------------------------------------------------------------------


def run_pkdgrav3_phase(args, ic_path: Path, workdir: Path, pool: list[str]) -> list[dict]:
    rows = []
    for ndev in args.ndevs:
        devs = ",".join(pool[:ndev])
        out_prefix = workdir / f"pkd_ndev{ndev}"
        env_extra = {
            "CUDA_VISIBLE_DEVICES": devs,
            "PKD_IC": str(ic_path),
            "PKD_OUT": str(out_prefix),
            "PKD_THETAS": ",".join(f"{t:g}" for t in args.pkd_thetas),
            "PKD_REPEATS": str(args.repeats),
            "PKD_WARMUP": str(args.warmup),
            "PKD_SOFT": str(args.softening),
            "PKD_NBUCKET": str(args.pkd_nbucket),
            "PKD_NGROUP": str(args.pkd_ngroup),
            "PKD_GPU": "1",
        }
        print(f"\n=== pkdgrav3, {ndev} GPU(s) (CUDA_VISIBLE_DEVICES={devs}) ===", flush=True)
        run = run_pkdgrav3(
            PKD_SCRIPT, env_extra=env_extra, cores=args.pkd_cores, timeout_s=args.timeout
        )
        print(run.stdout[-4000:], flush=True)
        summary_path = f"{out_prefix}_summary.json"
        # pkdgrav3 exits 0 even when the embedded interpreter raises, so the
        # summary file -- not the return code -- is what says the run succeeded.
        if run.returncode != 0 or not os.path.exists(summary_path):
            print(run.stderr[-4000:], file=sys.stderr)
            raise SystemExit(
                f"pkdgrav3 produced no summary for ndev={ndev} (rc={run.returncode}); "
                "see the captured stdout above for the embedded-interpreter traceback"
            )
        with open(summary_path) as fh:
            summary = json.load(fh)
        rows.append(
            dict(
                code="pkdgrav3",
                ndev=ndev,
                devices=devs,
                cores=args.pkd_cores,
                summary=summary,
                run=run.as_dict(),
            )
        )
    return rows


# --------------------------------------------------------------------------
# phase 2: jaccpot (in-process; imports JAX only once pkdgrav3 is done)
# --------------------------------------------------------------------------


def _time_calls(fn, repeats, warmup):
    import time

    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn())
        samples.append(time.perf_counter() - t0)
    s = np.array(samples)
    return out, dict(min=float(s.min()), mean=float(s.mean()), std=float(s.std()))


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
}


def apply_fast_lane_env(n: int) -> dict[str, str]:
    """Install the fused fast-lane environment; must run before jaccpot is imported."""
    env = dict(FAST_LANE_ENV)
    env["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = str(int(n))
    for k, v in env.items():
        os.environ[k] = v
    return env


def run_jaccpot_single_gpu(args, pos, mass, a_ref, softening):
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
    update** -- which is exactly pkdgrav3's ``gravity()`` with the tree already
    built.  The stage correspondence across the whole comparison is therefore:

        partition_for_devices   <->  pkdgrav3 domain_decompose
        prepare_state (fused)   <->  pkdgrav3 build_tree
        eval_fn                 <->  pkdgrav3 gravity
    """
    import time

    import jax

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    rows = []
    pos_j = jax.numpy.asarray(pos, jax.numpy.float32)
    mass_j = jax.numpy.asarray(mass, jax.numpy.float32)
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
                    tree=TreeConfig(
                        mode="static_radix", leaf_target=args.jac_leaf_single
                    ),
                    farfield=FarFieldConfig(mode="auto"),
                    nearfield=NearFieldConfig(mode="auto"),
                    mac_type="dehnen",
                ),
                fixed_order=order,
            )
            try:
                t0 = time.perf_counter()
                prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
                    positions=pos_j,
                    masses=mass_j,
                    leaf_size=args.jac_leaf_single,
                    max_order=order,
                    theta=theta,
                )
                prepare_s = time.perf_counter() - t0  # includes compile; see note
                out, timing = _time_calls(
                    lambda: eval_fn(prepared), args.repeats, args.warmup  # noqa: B023
                )
            except Exception as exc:
                msg = str(exc).splitlines()[0][:200]
                print(
                    f"  jaccpot[1gpu] p={order} theta={theta:.2f}: FAILED -- {msg}",
                    flush=True,
                )
                rows.append(
                    dict(
                        code="jaccpot",
                        lane="single_gpu_fused",
                        ndev=1,
                        order=order,
                        theta=theta,
                        leaf_size=args.jac_leaf_single,
                        failed=msg,
                    )
                )
                continue

            diag = dict(solver.get_runtime_diagnostics() or {})
            fused_active = bool(diag.get("strict_fused_mode_active"))
            a = np.asarray(out)
            row = dict(
                code="jaccpot",
                lane="single_gpu_fused",
                ndev=1,
                order=order,
                theta=theta,
                leaf_size=args.jac_leaf_single,
                force_eval_s=timing,
                prepare_s=prepare_s,
                peak_bytes_in_use=_peak_bytes(1),
                errors=rel_errors(a, a_ref),
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
            print(
                f"  jaccpot[1gpu] p={order} theta={theta:.2f}: "
                f"{timing['min']*1e3:8.2f} ms  aggL2={row['errors']['aggL2']:.3e}"
                f"  fused={fused_active}{warn}",
                flush=True,
            )
    return rows


def run_jaccpot_phase(args, pos, mass, a_ref, pool, softening):
    import time

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
            rows += run_jaccpot_single_gpu(args, pos, mass, a_ref, softening)
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

        for dev in jax.local_devices()[:ndev]:
            try:
                dev.memory_stats()  # touch; peak is read after the sweep
            except Exception:
                pass

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

                for _ in range(args.warmup):
                    out = fn(pos_f, mass_f, gid_f, counts)
                    jax.block_until_ready(out)
                samples = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    out = fn(pos_f, mass_f, gid_f, counts)
                    jax.block_until_ready(out)
                    samples.append(time.perf_counter() - t0)
                accel, gid, diag = out
                a = scatter_to_input_order(np.asarray(accel), np.asarray(gid), pos.shape[0])

                s = np.array(samples)
                peak = {}
                for i, dev in enumerate(jax.local_devices()[:ndev]):
                    try:
                        peak[i] = int(dev.memory_stats().get("peak_bytes_in_use", 0))
                    except Exception:
                        pass
                overflow = _reduce_overflow_flags(np.asarray(diag))
                row = dict(
                    code="jaccpot",
                    ndev=ndev,
                    overflow=overflow,
                    order=order,
                    theta=theta,
                    leaf_size=args.jac_leaf_dist,
                    partitioner=args.partitioner,
                    force_eval_s=dict(
                        min=float(s.min()), mean=float(s.mean()), std=float(s.std())
                    ),
                    peak_bytes_in_use=peak,
                    errors=rel_errors(a, a_ref),
                    diagnostics=np.asarray(diag).tolist(),
                )
                rows.append(row)
                print(
                    f"  jaccpot ndev={ndev} p={order} theta={theta:.2f}: "
                    f"{row['force_eval_s']['min']*1e3:8.2f} ms  "
                    f"aggL2={row['errors']['aggL2']:.3e}"
                    + ("  !! TRAVERSAL OVERFLOW: " + ",".join(overflow) if overflow else ""),
                    flush=True,
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--ic", default="plummer", choices=sorted(IC_GENERATORS))
    ap.add_argument("--ndevs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--pkd-thetas", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--jac-thetas", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--jac-leaf-single", type=int, default=256,
                    help="leaf size for jaccpot's single-GPU fused lane. 256 is the "
                         "validated 200k production value (benchmark_a100 SUMMARY.md, "
                         "static_radix + env_fused.sh); 64 overflows the interaction "
                         "list at this N")
    ap.add_argument("--jac-leaf-dist", type=int, default=64,
                    help="leaf size for the distributed lane (DistributedFMMConfig default)")
    ap.add_argument("--partitioner", default="rcb")
    ap.add_argument("--preset", default="large_n_gpu",
                    help="jaccpot single-GPU preset (ndev=1 lane only)")
    ap.add_argument("--pkd-nbucket", type=int, default=16)
    ap.add_argument("--pkd-ngroup", type=int, default=0, help="0 = pkdgrav3 default")
    ap.add_argument("--pkd-cores", type=int, default=16,
                    help="mdl -sz; pkdgrav3 uses the host CPU heavily, jaccpot does not, "
                         "so this number must be reported with any timing")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
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
    ap.add_argument("--skip-pkdgrav3", action="store_true")
    ap.add_argument("--skip-jaccpot", action="store_true")
    args = ap.parse_args()

    # honest GPU memory numbers require this off; JAX otherwise grabs ~75% of VRAM
    # up front and every "peak" reading is the preallocation, not the algorithm.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if "jax" in sys.modules:
        raise SystemExit("JAX was imported before this ran; restart the process")
    fast_lane_env = apply_fast_lane_env(args.n)

    pool = device_pool()
    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workdir = Path(args.workdir or (root / "artifacts" / "compare_force"))
    workdir.mkdir(parents=True, exist_ok=True)

    pos, mass = make_ic(args.ic, args.n, max(args.ndevs), args.seed)
    n = pos.shape[0]
    print(f"IC={args.ic} N={n} devices={pool}")

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
        pkd_rows = run_pkdgrav3_phase(args, ic_path, workdir, pool)

    # -------- reference + jaccpot (JAX enters the process only now) --------
    from common.reference import direct_accelerations

    print(f"\ncomputing float64 direct-sum reference (eps={args.softening:g})...", flush=True)
    a_ref = direct_accelerations(pos, mass, G=args.G, softening=args.softening,
                                 block_size=1024)
    # The two codes use different softening kernels (Plummer vs compact-support
    # spline), so the epsilon is only legitimate while it is dynamically
    # irrelevant. Measure that rather than assuming it: how far does it move the
    # reference away from the unsoftened Newtonian answer?
    a_ref0 = direct_accelerations(pos, mass, G=args.G, softening=0.0, block_size=1024)
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
            res["errors"] = rel_errors(z["acc"], a_ref)
            print(
                f"  pkdgrav3 ndev={row['ndev']} theta={res['theta']:.2f}: "
                f"{res['force_eval_s']['min']*1e3:8.2f} ms  "
                f"aggL2={res['errors']['aggL2']:.3e}"
                f"  (signflip {res['errors']['aggL2_signflip']:.3e})",
                flush=True,
            )

    jac_rows = []
    if not args.skip_jaccpot:
        jac_rows = run_jaccpot_phase(args, pos, mass, a_ref, pool, args.softening)

    out = Path(args.out or (root / "artifacts" / "compare_force.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    from common.env import capture_provenance

    payload = dict(
        provenance=capture_provenance({"benchmark": "compare_force", "args": vars(args)}),
        ic=dict(name=args.ic, n=n, seed=args.seed, softening=args.softening,
                softening_reference_shift_aggL2=eps_shift, G=args.G),
        fast_lane_env=fast_lane_env,
        devices=pool,
        pkdgrav3=pkd_rows,
        jaccpot=jac_rows,
    )
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
