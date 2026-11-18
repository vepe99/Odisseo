"""Odisseo at matched accuracy against the reference code.

For each N that has successful reference rows (results/reference.csv), the
target accuracy is the median |dE/E| of the reference repeats, both measured
with the same float64 estimator (common.total_energy, same softening) over the
same IC file and the same physical end time t_end.

For each (precision, scheme) Odisseo walks a ladder of step counts
n_steps = round(n0 * sqrt(2)^k) (dt shrinks by sqrt(2) per rung, i.e. the
leapfrog energy error by ~2x) and selects the smallest n_steps for which this
rung AND the next one both give |dE/E| <= target (guards against a lucky
non-monotonic point at coarse dt). Ladder points are written as rows with
``status = ladder``. The selected configuration is then timed with
``--repeats`` runs (compile excluded, warm-up, block_until_ready), one row per
repeat with ``metric_basis = wallclock_to_fixed_time``, the measured
|dE/E|, the target, their ratio and ``accuracy_matched`` (ratio within
[1/f, f], f = ``--match-factor``). The interaction count of the run is stored
too (exact for shared fixed steps).

GPU selection, A100 assertion, and fp64/fp32 worker processes follow
run_odisseo.py. Numbers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ic as ic_mod  # noqa: E402

SCHEMES = {"DIRECT_ACC": 0, "DIRECT_ACC_LAXMAP": 1}
EXTRA = ["target_dE_E", "accuracy_ratio", "accuracy_matched", "reference_run_id", "ladder"]
COLUMNS = common.CSV_COLUMNS + EXTRA


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def reference_targets(path: Path, run_id: str | None) -> dict:
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if r["code"] != "odisseo"]
    if not rows:
        return {}
    rid = run_id or max(r["run_id"] for r in rows)
    out = {}
    for N in sorted({int(r["N"]) for r in rows if r["run_id"] == rid}):
        ok = [float(r["dE_E_final"]) for r in rows
              if r["run_id"] == rid and int(r["N"]) == N and r["status"] == "ok" and r["dE_E_final"]]
        if ok:
            t_end = {float(r["t_end"]) for r in rows if r["run_id"] == rid and int(r["N"]) == N}
            soft = {float(r["softening"]) for r in rows if r["run_id"] == rid and int(r["N"]) == N}
            assert len(t_end) == 1 and len(soft) == 1
            out[N] = dict(target=float(np.median(ok)), t_end=t_end.pop(), softening=soft.pop(),
                          run_id=rid, n=len(ok), code=rows[0]["code"])
    return out


def worker(a) -> None:
    precision = a.worker
    import jax
    jax.config.update("jax_enable_x64", precision == "fp64")
    import jax.numpy as jnp
    from odisseo import option_classes as oc
    from odisseo.option_classes import SimulationConfig, SimulationParams
    from odisseo.time_integration import time_integration

    dtype = np.float64 if precision == "fp64" else np.float32
    dev = jax.devices()[0]
    if dev.platform != "gpu" or "A100" not in dev.device_kind:
        raise SystemExit(f"FATAL: worker sees {dev.device_kind}, not an A100 GPU")
    gpu = json.loads(os.environ["BENCH_GPU_INFO"])
    targets = {int(k): v for k, v in json.loads(os.environ["BENCH_TARGETS"]).items()}
    csv_path = Path(a.csv)

    for N, tg in sorted(targets.items()):
        ic = ic_mod.load_ic(N, a.seed)
        eps, t_end = tg["softening"], tg["t_end"]
        E0 = ic_mod.softened_E0(N, eps, a.seed)
        state = jax.device_put(jnp.asarray(np.stack([ic["pos"], ic["vel"]], 1).astype(dtype)), dev)
        mass = jax.device_put(jnp.asarray(ic["mass"].astype(dtype)), dev)
        for scheme in a.schemes:
            base = dict(run_id=a.run_id, code="odisseo", series=f"odisseo_matched_{scheme}_{precision}",
                        N=N, precision=precision, scheme=scheme, scheme_id=SCHEMES[scheme],
                        batch_size=a.batch_size if scheme == "DIRECT_ACC_LAXMAP" else "",
                        integrator="LEAPFROG (kick-drift-kick, 2 force evals/step)", softening=eps,
                        t_end=t_end, E0=E0, ic_file=str(ic["path"].relative_to(common.BENCH_DIR)),
                        ic_sha256=ic["sha256"], gpu_name=gpu["name"], gpu_mem_total_MiB=gpu["memory.total"],
                        flops_per_interaction=common.FLOPS_PER_INTERACTION,
                        target_dE_E=tg["target"], reference_run_id=tg["run_id"])

            def make(n):
                cfg = SimulationConfig(N_particles=N, num_timesteps=int(n), softening=float(eps),
                                       integrator=oc.LEAPFROG, acceleration_scheme=SCHEMES[scheme],
                                       batch_size=a.batch_size, fixed_timestep=True, return_snapshots=False)
                prm = SimulationParams(G=dtype(1.0), t_end=dtype(t_end))
                t0 = time.perf_counter()
                low = time_integration.lower(state, mass, cfg, prm)
                comp = low.compile()
                return cfg, prm, comp, time.perf_counter() - t0, low

            def run(comp, prm):
                t0 = time.perf_counter()
                out = comp(state, mass, prm)
                out.block_until_ready()
                return out, time.perf_counter() - t0

            def dE_of(out):
                fin = np.asarray(out, np.float64)
                if not np.all(np.isfinite(fin)):
                    return np.inf, np.nan
                E1 = common.total_energy(fin[:, 0], fin[:, 1], ic["mass"], eps)
                return abs((E1 - E0) / E0), E1

            try:
                ladder, chosen = [], None
                k = 0
                while True:
                    n = max(1, int(round(a.n0 * np.sqrt(2.0) ** k)))
                    if ladder and n == ladder[-1]["n_steps"]:
                        k += 1
                        continue
                    _, prm, comp, tc, _ = make(n)
                    out, w = run(comp, prm)
                    dE, E1 = dE_of(out)
                    ladder.append(dict(n_steps=n, dE_E=dE, wall_s_single=w, compile_s=tc))
                    append(csv_path, dict(base, timestamp_utc=common.utcnow(), status="ladder", n_steps=n,
                                          dt=t_end / n, n_force_evals=2 * n, dE_E_final=dE, E_final=E1,
                                          compile_s=tc, metric_basis="wallclock_to_fixed_time"))
                    print(f"[{precision}] {scheme} N={N} ladder n={n} |dE/E|={dE:.3e} target={tg['target']:.3e}",
                          flush=True)
                    if len(ladder) >= 2 and ladder[-2]["dE_E"] <= tg["target"] and dE <= tg["target"]:
                        chosen = ladder[-2]["n_steps"]
                        break
                    if n >= a.max_steps:
                        break
                    k += 1
                if chosen is None:
                    raise RuntimeError(f"target |dE/E| {tg['target']:.3e} not reached up to {a.max_steps} steps")

                cfg, prm, comp, _, low = make(chosen)  # cached: report the cold compile from the ladder
                tc = next(x["compile_s"] for x in ladder if x["n_steps"] == chosen)
                hlo = low.as_text()
                if (precision == "fp32") == ("f64" in hlo):
                    raise RuntimeError("HLO precision check failed")
                out, _ = run(comp, prm)  # warm-up
                dE, E1 = dE_of(out)
                ratio = dE / tg["target"]
                matched = (1.0 / a.match_factor) <= ratio <= a.match_factor
                inter = float(N) * N * 2 * chosen
                for r in range(a.repeats):
                    _, w = run(comp, prm)
                    others = [x for x in common.compute_apps_on(gpu["uuid"]) if x["pid"] != str(os.getpid())]
                    append(csv_path, dict(
                        base, timestamp_utc=common.utcnow(), repeat=r, status="shared_gpu" if others else "ok",
                        error=json.dumps(others) if others else "", n_steps=chosen, dt=t_end / chosen,
                        n_force_evals=2 * chosen, wall_s=w, compile_s=tc,
                        metric_basis="wallclock_to_fixed_time", interactions=inter,
                        interactions_per_s=inter / w, gflops=inter / w * common.FLOPS_PER_INTERACTION / 1e9,
                        E_final=E1, dE_E_final=dE, accuracy_ratio=ratio, accuracy_matched=matched,
                        ladder=json.dumps(ladder),
                        flags=json.dumps(dict(num_timesteps=chosen, softening=eps, t_end=t_end,
                                              acceleration_scheme=scheme, jax_enable_x64=precision == "fp64",
                                              XLA_FLAGS=os.environ.get("XLA_FLAGS", "")))))
                    print(f"[{precision}] {scheme} N={N} n={chosen} wall={w:.4f}s |dE/E|={dE:.3e} "
                          f"ratio={ratio:.2f} matched={matched}", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
                append(csv_path, dict(base, timestamp_utc=common.utcnow(), status="error", error=msg))
                print(f"[{precision}] {scheme} N={N}: error {msg}", flush=True)
            finally:
                jax.clear_caches()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-csv", default=str(common.RESULTS_DIR / "reference.csv"))
    ap.add_argument("--reference-run-id", default=None)
    ap.add_argument("--precisions", nargs="+", default=["fp64", "fp32"], choices=["fp64", "fp32"])
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n0", type=int, default=8, help="first rung of the n_steps ladder")
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--match-factor", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=ic_mod.DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=10_000)
    ap.add_argument("--out-dir", default=str(common.RESULTS_DIR))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gpu-timeout", type=int, default=None)
    ap.add_argument("--exclude-gpu", type=int, nargs="*", default=None)
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--worker", choices=["fp64", "fp32"], help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        worker(a)
        return 0
    if a.dry_run:
        a.out_dir = str(Path(a.out_dir) / "dryrun")
        if a.reference_csv == str(common.RESULTS_DIR / "reference.csv"):
            a.reference_csv = str(Path(a.out_dir) / "reference.csv")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(a.csv) if a.csv else out_dir / "odisseo_matched.csv"
    run_id = a.run_id or time.strftime("%Y%m%dT%H%M%S")

    targets = reference_targets(Path(a.reference_csv), a.reference_run_id) \
        if Path(a.reference_csv).exists() else {}
    if a.dry_run and targets:
        targets = {min(targets): targets[min(targets)]}
    if not targets:
        print(f"[matched] no successful reference rows in {a.reference_csv}; nothing to match")
        return 0
    ic_mod.ensure_ic(list(targets), a.seed)
    for N, tg in targets.items():
        ic_mod.softened_E0(N, tg["softening"], a.seed)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from autocvd import autocvd
    sel = autocvd(num_gpus=1, exclude=a.exclude_gpu, timeout=a.gpu_timeout)
    idx = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    gpu = common.nvidia_smi_query(idx)
    variant = common.assert_a100(gpu)
    others = common.compute_apps_on(gpu["uuid"])
    if others and not a.allow_shared_gpu:
        raise SystemExit(f"FATAL: other processes on GPU {idx}: {others}")
    print(f"[matched] GPU {idx}: {gpu['name']} ({variant}); targets: "
          + ", ".join(f"N={N}: {t['target']:.3e}" for N, t in targets.items()), flush=True)

    prov = dict(run_id=run_id, created_utc=common.utcnow(), script="run_matched.py", argv=sys.argv,
                args=vars(a), targets=targets, gpu=dict(gpu, variant=variant, autocvd_selection=sel),
                cuda=common.cuda_versions(), host=common.host_info(), odisseo_git=common.git_info(),
                env=common.env_snapshot(), workers={})
    prov_path = out_dir / f"provenance_matched_{run_id}.json"
    common.write_json(prov_path, prov)
    env = dict(os.environ, BENCH_GPU_INFO=json.dumps(gpu), BENCH_TARGETS=json.dumps(targets))
    rc = 0
    for prec in a.precisions:
        cmd = [sys.executable, __file__, "--worker", prec, "--run-id", run_id, "--csv", str(csv_path),
               "--schemes", *a.schemes, "--repeats", str(a.repeats), "--n0", str(a.n0),
               "--max-steps", str(a.max_steps), "--match-factor", str(a.match_factor),
               "--seed", str(a.seed), "--batch-size", str(a.batch_size)]
        p = subprocess.run(cmd, env=env)
        rc |= p.returncode
        prov["workers"][prec] = dict(returncode=p.returncode)
        common.write_json(prov_path, prov)
    import importlib.metadata as md
    prov["jax_version"] = md.version("jax")
    prov["jaxlib_version"] = md.version("jaxlib")
    prov["finished_utc"] = common.utcnow()
    common.write_json(prov_path, prov)
    print(f"[matched] rows -> {csv_path}\n[matched] provenance -> {prov_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
