"""Odisseo (JAX/CUDA) throughput benchmark on a single NVIDIA A100.

Structure
---------
The script is an orchestrator that never imports JAX itself. It
  1. selects one free GPU with ``autocvd`` (repo convention, shared machine),
  2. asserts it is an A100 (40 GB or 80 GB) with no other process on it,
  3. generates any missing initial conditions (``ic.py``, CPU-only subprocess),
  4. launches one worker subprocess per precision: fp64 runs with
     ``jax_enable_x64=True``; fp32 runs with x64 disabled (the odisseo force
     kernel is not dtype-polymorphic inside an x64 process),
  5. writes a provenance JSON.

Each worker, for every (N, scheme):
  * uses one ``num_timesteps`` per N, calibrated on the first series (fp64
    DIRECT_ACC by default) so one integration takes ~``--target-seconds`` and
    shared by all series (same dt and t_end at a given N),
  * AOT-compiles the integration (``jit(...).lower(...).compile()``) and
    records the compile time separately,
  * checks the lowered HLO for the precision (fp32 must contain no f64 ops),
  * runs one warm-up integration, evaluates |dE/E| of its final state in
    float64 with numpy (same softening), then
  * runs ``--repeats`` timed integrations, each ended by
    ``.block_until_ready()``, writing one CSV row per repeat immediately.

Metric: interactions/s = N^2 * n_force_evals / wall_s, with
n_force_evals = 2 * n_steps (``odisseo.integrators.leapfrog`` evaluates the
acceleration twice per step). N^2 includes the i == j pair, which the
direct kernels evaluate (and zero) like every other pair.

The script prints and stores numbers only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ic as ic_mod  # noqa: E402

DEFAULT_N = [2 ** k for k in range(10, 18)]
SCHEMES = {"DIRECT_ACC": 0, "DIRECT_ACC_LAXMAP": 1}  # checked against option_classes in the worker
# Initial guess of throughput used only to size the first calibration run.
GUESS_IPS = {"fp64": 1.5e10, "fp32": 5e10}


# =========================================================================== #
# Worker (runs inside a process that owns the GPU)                            #
# =========================================================================== #
def worker(args) -> None:
    precision = args.worker
    import jax
    jax.config.update("jax_enable_x64", precision == "fp64")
    import jax.numpy as jnp
    import jaxlib
    from odisseo import option_classes as oc
    from odisseo.option_classes import SimulationConfig, SimulationParams
    from odisseo.time_integration import time_integration

    for name, sid in SCHEMES.items():
        assert getattr(oc, name) == sid, f"{name} id changed in odisseo.option_classes"

    dtype = np.float64 if precision == "fp64" else np.float32
    dev = jax.devices()[0]
    if dev.platform != "gpu" or "A100" not in dev.device_kind:
        raise SystemExit(f"FATAL: worker sees {dev} ({dev.device_kind}), not an A100 GPU")
    gpu = json.loads(os.environ["BENCH_GPU_INFO"])

    common.write_json(Path(args.out_dir) / f"provenance_{args.run_id}_{precision}.json", {
        "precision": precision,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_backend": jax.default_backend(),
        "device": str(dev), "device_kind": dev.device_kind,
        "backend_platform_version": getattr(dev.client, "platform_version", ""),
        "env": common.env_snapshot(),
        "jax_config_relevant": {k: str(getattr(jax.config, k)) for k in (
            "jax_enable_x64", "jax_default_matmul_precision", "jax_numpy_dtype_promotion")},
    })

    csv_path = Path(args.csv)
    budget_s = args.target_seconds
    steps_file = Path(args.out_dir) / f"steps_{args.run_id}.json"

    for N in args.N:
        ic = ic_mod.load_ic(N, args.seed)
        E0 = ic_mod.softened_E0(N, args.softening, args.seed)
        state_np = np.stack([ic["pos"], ic["vel"]], axis=1).astype(dtype)  # (N, 2, 3)
        mass_np = ic["mass"].astype(dtype)
        state = jax.device_put(jnp.asarray(state_np), dev)
        mass = jax.device_put(jnp.asarray(mass_np), dev)

        for scheme in args.schemes:
            base = dict(run_id=args.run_id, code="odisseo", series=f"odisseo_{scheme}_{precision}",
                        N=N, precision=precision, scheme=scheme, scheme_id=SCHEMES[scheme],
                        batch_size=args.batch_size if scheme == "DIRECT_ACC_LAXMAP" else "",
                        integrator="LEAPFROG (kick-drift-kick, 2 force evals/step)",
                        softening=args.softening, metric_basis="interactions",
                        flops_per_interaction=common.FLOPS_PER_INTERACTION,
                        E0=E0, ic_file=str(ic["path"].relative_to(common.BENCH_DIR)),
                        ic_sha256=ic["sha256"], gpu_name=gpu["name"],
                        gpu_mem_total_MiB=gpu["memory.total"])

            def make(n_steps):
                cfg = SimulationConfig(
                    N_particles=N, dimensions=3, return_snapshots=False, fixed_timestep=True,
                    num_timesteps=int(n_steps), softening=float(args.softening),
                    integrator=oc.LEAPFROG, acceleration_scheme=SCHEMES[scheme],
                    batch_size=args.batch_size, double_map=False, external_accelerations=(),
                    gradient_horizon=0, progress_bar=False)
                prm = SimulationParams(G=dtype(1.0), t_end=dtype(args.t_end))
                t0 = time.perf_counter()
                lowered = time_integration.lower(state, mass, cfg, prm)
                compiled = lowered.compile()
                t_compile = time.perf_counter() - t0
                return cfg, prm, lowered, compiled, t_compile

            def timed(compiled, prm):
                t0 = time.perf_counter()
                out = compiled(state, mass, prm)
                out.block_until_ready()
                return out, time.perf_counter() - t0

            try:
                # ---- n_steps: calibrated ONCE per N (first series run, fp64 by
                # default) for ~budget_s per integration, then reused by every
                # series, so all series at a given N integrate the same problem
                # (same dt, same t_end). N^2 * n_steps is then ~constant.
                steps_table = json.loads(steps_file.read_text()) if steps_file.exists() else {}
                if str(N) in steps_table:
                    n_steps = int(steps_table[str(N)]["n_steps"])
                else:
                    n_cal = int(np.clip(GUESS_IPS[precision] * budget_s / 10 / (2 * N * N), 1, args.max_steps))
                    while True:
                        _, prm, _, comp, _ = make(n_cal)
                        timed(comp, prm)                      # warm
                        _, t_cal = timed(comp, prm)
                        if t_cal >= 0.05 * budget_s or n_cal >= args.max_steps:
                            break
                        n_cal = int(min(args.max_steps, n_cal * max(2.0, 0.1 * budget_s / max(t_cal, 1e-6))))
                    n_steps = int(np.clip(round(n_cal * budget_s / t_cal), 1, args.max_steps))
                    steps_table[str(N)] = dict(n_steps=n_steps, calibrated_on=f"{scheme}_{precision}",
                                               calibration_steps=n_cal, calibration_wall_s=t_cal,
                                               target_seconds=budget_s)
                    common.write_json(steps_file, steps_table)

                # ---- final configuration: compile (timed separately) ----
                cfg, prm, lowered, compiled, t_compile = make(n_steps)
                hlo = lowered.as_text()
                if precision == "fp32" and "f64" in hlo:
                    raise RuntimeError("fp32 series lowered HLO contains f64 ops")
                if precision == "fp64" and "f64" not in hlo:
                    raise RuntimeError("fp64 series lowered HLO contains no f64 ops")

                # ---- warm-up + accuracy (float64 energy of the final state) ----
                out, _ = timed(compiled, prm)
                if out.dtype != dtype:
                    raise RuntimeError(f"output dtype {out.dtype} != {dtype}")
                fin = np.asarray(out, dtype=np.float64)
                if not np.all(np.isfinite(fin)):
                    raise RuntimeError("non-finite final state")
                E1 = common.total_energy(fin[:, 0], fin[:, 1], ic["mass"], args.softening)
                dE = abs((E1 - E0) / E0)

                n_force = 2 * n_steps
                inter = float(N) * float(N) * n_force
                flags = dict(SimulationConfig={k: getattr(cfg, k) for k in (
                    "N_particles", "num_timesteps", "softening", "integrator", "acceleration_scheme",
                    "batch_size", "double_map", "fixed_timestep", "return_snapshots")},
                    XLA_FLAGS=os.environ.get("XLA_FLAGS", ""), jax_enable_x64=precision == "fp64")
                others_before = [a for a in common.compute_apps_on(gpu["uuid"]) if a["pid"] != str(os.getpid())]
                walls = []
                for r in range(args.repeats):
                    _, w = timed(compiled, prm)
                    walls.append(w)
                    ips = inter / w
                    others = others_before + [a for a in common.compute_apps_on(gpu["uuid"])
                                              if a["pid"] != str(os.getpid())]
                    common.append_row(csv_path, dict(
                        base, timestamp_utc=common.utcnow(), repeat=r,
                        status="shared_gpu" if others else "ok",
                        error=json.dumps(others) if others else "",
                        n_steps=n_steps, dt=args.t_end / n_steps, t_end=args.t_end, n_force_evals=n_force,
                        wall_s=w, compile_s=t_compile, interactions=inter, interactions_per_s=ips,
                        gflops=ips * common.FLOPS_PER_INTERACTION / 1e9,
                        E_final=E1, dE_E_final=dE, flags=json.dumps(flags)))
                q1, med, q3 = np.percentile(walls, [25, 50, 75])
                print(f"[{precision}] {scheme:18s} N={N:7d} steps={n_steps:7d} compile={t_compile:6.2f}s "
                      f"wall median={med:8.3f}s IQR=[{q1:.3f},{q3:.3f}] "
                      f"int/s={inter / med:.3e} |dE/E|={dE:.2e}", flush=True)
            except Exception as e:  # OOM or failure at large N: record and continue the sweep
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
                status = "oom" if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower() else "error"
                common.append_row(csv_path, dict(base, timestamp_utc=common.utcnow(), repeat="",
                                                 status=status, error=msg, t_end=args.t_end))
                print(f"[{precision}] {scheme} N={N}: {status}: {msg}", flush=True)
                traceback.print_exc()
            finally:
                jax.clear_caches()
        del state, mass


# =========================================================================== #
# Orchestrator (no JAX import)                                                #
# =========================================================================== #
def orchestrate(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    csv_path = Path(args.csv) if args.csv else out_dir / "odisseo_throughput.csv"

    t0 = time.perf_counter()
    ic_mod.ensure_ic(args.N, args.seed)
    t_ic = time.perf_counter() - t0
    for N in args.N:  # pre-compute (and cache) the softened E0 outside the timed workers
        ic_mod.softened_E0(N, args.softening, args.seed)

    # GPU selection AFTER the CPU-only preparation, right before the workers start.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from autocvd import autocvd
    sel = autocvd(num_gpus=1, timeout=args.gpu_timeout)
    idx = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    gpu = common.nvidia_smi_query(idx)
    variant = common.assert_a100(gpu)
    others = common.compute_apps_on(gpu["uuid"])
    if others and not args.allow_shared_gpu:
        raise SystemExit(f"FATAL: other processes on GPU {idx}: {others}")
    print(f"[bench] GPU {idx}: {gpu['name']} ({variant}), uuid {gpu['uuid']}", flush=True)


    prov = {
        "run_id": run_id, "created_utc": common.utcnow(), "script": "run_odisseo.py",
        "argv": sys.argv, "args": vars(args),
        "gpu": dict(gpu, variant=variant, autocvd_selection=sel, other_processes_at_start=others),
        "cuda": common.cuda_versions(), "nvidia_smi_q": common.nvidia_smi_full(),
        "host": common.host_info(), "odisseo_git": common.git_info(),
        "env": common.env_snapshot(), "ic_generation_s": t_ic,
        "ic_files": {N: ic_mod.load_ic(N, args.seed)["meta"] for N in args.N},
        "flops_per_interaction": common.FLOPS_PER_INTERACTION,
        "flops_convention": "Nyland, Harris & Prins (2007), GPU Gems 3 ch. 31",
        "csv": str(csv_path), "workers": {},
    }
    prov_path = out_dir / f"provenance_{run_id}.json"
    common.write_json(prov_path, prov)

    env = dict(os.environ, BENCH_GPU_INFO=json.dumps(gpu))
    rc = 0
    for prec in args.precisions:
        cmd = [sys.executable, __file__, "--worker", prec, "--run-id", run_id, "--csv", str(csv_path),
               "--out-dir", str(out_dir), "--N", *map(str, args.N), "--schemes", *args.schemes,
               "--target-seconds", str(args.target_seconds), "--repeats", str(args.repeats),
               "--t-end", repr(args.t_end), "--softening", repr(args.softening), "--seed", str(args.seed),
               "--batch-size", str(args.batch_size), "--max-steps", str(args.max_steps)]
        print(f"[bench] worker {prec}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, env=env)
        rc |= r.returncode
        wp = out_dir / f"provenance_{run_id}_{prec}.json"
        prov["workers"][prec] = dict(returncode=r.returncode,
                                     **(json.loads(wp.read_text()) if wp.exists() else {}))
        prov["gpu"]["other_processes_after_" + prec] = common.compute_apps_on(gpu["uuid"])
        common.write_json(prov_path, prov)
    sf = out_dir / f"steps_{run_id}.json"
    prov["steps_table"] = json.loads(sf.read_text()) if sf.exists() else {}
    prov["finished_utc"] = common.utcnow()
    common.write_json(prov_path, prov)
    print(f"[bench] rows -> {csv_path}\n[bench] provenance -> {prov_path}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=DEFAULT_N)
    ap.add_argument("--precisions", nargs="+", default=["fp64", "fp32"], choices=["fp64", "fp32"])
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    ap.add_argument("--target-seconds", type=float, default=30.0,
                    help="target wall-clock of ONE timed integration of the FIRST series at each N "
                         "(fp64 DIRECT_ACC by default); n_steps is calibrated to it once per N and "
                         "reused by all other series")
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats after one warm-up (>= 5)")
    ap.add_argument("--t-end", type=float, default=1.0,
                    help="physical end time in Henon units, identical for every N and series; "
                         "dt = t_end / n_steps")
    ap.add_argument("--softening", type=float, default=2.0 ** -7)
    ap.add_argument("--seed", type=int, default=ic_mod.DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=10_000,
                    help="SimulationConfig.batch_size (used by DIRECT_ACC_LAXMAP)")
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--out-dir", default=str(common.RESULTS_DIR))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gpu-timeout", type=int, default=None, help="autocvd wait timeout [s]")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="do not abort if other processes are on the selected GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="smallest N only, short runs, results under results/dryrun/")
    ap.add_argument("--worker", choices=["fp64", "fp32"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        worker(args)
        return 0
    if args.repeats < 5 and not args.dry_run:
        ap.error("--repeats must be >= 5")
    if args.dry_run:
        args.N = [min(args.N)]
        args.target_seconds = min(args.target_seconds, 2.0)
        args.out_dir = str(Path(args.out_dir) / "dryrun")
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
