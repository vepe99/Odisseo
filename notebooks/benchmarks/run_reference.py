"""Reference-code runner: PeTar (GPU build) on the same initial conditions.

For each N it
  1. reads the shared IC file (``ic.py``) and converts it, without regenerating
     it, to PeTar's ASCII input format (``-i 1``; one header line
     ``0 N 0`` and 20 columns per particle as listed by ``petar -h``),
     writing every float with ``repr`` (round-trip exact),
  2. checks the CPU load and waits until ``loadavg + OMP threads`` fits in
     ``--max-cpu-fraction`` of the cores (the machine is shared),
  3. selects one free GPU with ``autocvd`` and asserts it is an A100,
  4. runs PeTar to ``--t-end`` under a hard wall-clock cap
     (``--wallclock-cap``), timing the whole process externally,
  5. asserts that PeTar's own time-0 snapshot (``data.0``) equals the IC file
     to round-off (what PeTar ingested is what Odisseo ingested),
  6. evaluates |dE/E| with the SAME float64 direct-summation estimator as
     run_odisseo.py (common.total_energy, same Plummer softening) on PeTar's
     time-0 and final snapshots; PeTar's self-reported energy error is
     stored alongside in separate columns,
  7. parses PeTar's step statistics (tree steps, Hermite steps, AR steps,
     particle-particle and particle-superparticle tree interactions).

PeTar is not an all-pairs code (tree long-range force + Hermite/SDAR short
range), so no N^2 interaction rate is computed for it:
``metric_basis = wallclock_to_fixed_time``. The interaction counters are
recorded as numbers only.

Every flag passed to PeTar is checked against the installed binary's ``-h``
output before running; the help text is saved with the results.

The script reports numbers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import ic as ic_mod  # noqa: E402

DEFAULT_PETAR = Path.home() / "petar_build" / "install" / "bin" / "petar"
DEFAULT_PETAR_SRC = Path.home() / "petar_build"
DEFAULT_N = [2 ** 12, 2 ** 14, 2 ** 16]

REF_EXTRA = [
    "omp_threads", "mpi_ranks", "gpu_index", "loadavg_before", "loadavg_after", "cpu_cores_used",
    "petar_internal_s", "petar_E0_reported", "petar_dE_E_reported",
    "tree_steps", "h4_steps_total", "ar_steps_total", "ep_ep_total", "ep_sp_total",
    "ic_ingest_max_rel_diff", "E0_snapshot",
]
COLUMNS = common.CSV_COLUMNS + REF_EXTRA


def append_ref_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(sorted(unknown))
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# PeTar I/O                                                                    #
# --------------------------------------------------------------------------- #
def write_petar_input(path: Path, pos, vel, mass) -> None:
    """ASCII input for ``-i 1``: header 'File_ID N Time' then 20 columns:
    mass, x, y, z, vx, vy, vz, bin_stat, r_search, id, mass_bk, status,
    r_in, r_out, acc_soft[3], pot_tot, pot_soft, n_b (see ``petar -h``)."""
    n = len(mass)
    with open(path, "w") as f:
        f.write(f"0 {n} 0\n")
        for i in range(n):
            vals = " ".join(repr(float(x)) for x in (mass[i], *pos[i], *vel[i]))
            f.write(f"{vals} 0 0 {i + 1} 0 0 0 0 0 0 0 0 0 0\n")


def read_petar_snapshot(path: Path):
    """Read an ASCII PeTar snapshot: header 'File_ID N Time', then rows whose
    first 7 columns are mass, pos[3], vel[3] and whose 10th column is the id."""
    with open(path) as f:
        head = f.readline().split()
        t = float(head[2])
        data = np.loadtxt(f, usecols=(0, 1, 2, 3, 4, 5, 6, 9), ndmin=2)
    order = np.argsort(data[:, 7])
    data = data[order]
    return t, data[:, 0], data[:, 1:4], data[:, 4:7], data[:, 7].astype(np.int64)


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_petar_log(text: str) -> dict:
    """Parse energy and per-step statistics blocks of PeTar's stdout."""
    out = dict(E0=np.nan, dE_cum_last=np.nan, E_last=np.nan, tree_steps=0, h4=0.0, ar=0.0,
               epep=0.0, epsp=0.0, internal_s=0.0, blocks=0)
    blocks = re.split(r"\n(?=Time: )", text)
    first = True
    for b in blocks:
        if not b.startswith("Time: "):
            continue
        m = re.search(r"\nPhysic:\s+(.*)", b)
        if m:
            vals = [float(v) for v in re.findall(_NUM, m.group(1))]
            # columns: Error/Total Error Error_cum Total Kinetic Potential ...
            if first:
                out["E0"] = vals[3]
                first = False
            out["dE_cum_last"], out["E_last"] = vals[2], vals[3]
        ms = re.search(r"Tree step number:\s*(\d+)", b)
        if not ms:
            continue
        k = int(ms.group(1))
        out["tree_steps"] += k
        out["blocks"] += 1
        mn = re.search(r"\*\*\*\* Number per step \(global\):\n\s*(.*)\n\s*(.*)", b)
        if mn:
            names, vals = mn.group(1).split(), [float(v) for v in mn.group(2).split()]
            d = dict(zip(names, vals))
            out["h4"] += k * d.get("H4_step_sum", np.nan)
            out["ar"] += k * d.get("AR_step_sum", np.nan)
            out["epep"] += k * d.get("Ep-Ep_sum", np.nan)
            out["epsp"] += k * d.get("Ep-Sp_sum", np.nan)
        mw = re.search(r"\*\*\*\* Wallclock time per step \(local\): \[Min/Max\]\n\s*(.*)\n\s*(.*)\n\s*(.*)", b)
        if mw:
            names = mw.group(1).split()
            vmax = [float(v) for v in mw.group(3).split()]
            out["internal_s"] += k * dict(zip(names, vmax)).get("Total", np.nan)
    return out


def check_flags_in_help(flags: list[str], help_text: str) -> list[str]:
    missing = []
    for f in flags:
        if not f.startswith("-"):
            continue
        tok = f.split("=")[0]
        pat = rf"(^|\s){re.escape(tok)}(\s|$)"
        if not re.search(pat, help_text, flags=re.M):
            missing.append(tok)
    return missing


def wait_for_cpu(threads: int, frac: float, timeout: float) -> float:
    ncpu = os.cpu_count() or 1
    t0 = time.time()
    while True:
        load = os.getloadavg()[0]
        if load + threads <= frac * ncpu:
            return load
        if time.time() - t0 > timeout:
            raise SystemExit(f"FATAL: CPU busy (load {load:.1f} + {threads} threads > "
                             f"{frac:.2f} x {ncpu} cores) for {timeout:.0f} s")
        print(f"[ref] waiting for CPU: load {load:.1f}", flush=True)
        time.sleep(30)


def build_info(src: Path) -> dict:
    info = {}
    for name in ("PeTar", "FDPS", "SDAR"):
        d = src / name
        if d.exists():
            info[name] = common.git_info(d)
            info[name]["describe"] = common._run(["git", "-C", str(d), "describe", "--tags", "--always"])
    mk = src / "PeTar" / "Makefile"
    if mk.exists():
        info["makefile_compile_lines"] = [l for l in mk.read_text().splitlines()
                                          if re.match(r"^(NVCC|CXX|CXXFLAGS|OPTFLAGS|CUDAFLAGS|MT_FLAGS|"
                                                      r"FDPSFLAGS|SIMD_TYPE|CUDA_\w+)\s*[+:]?=", l)]
        conf = src / "PeTar" / "Makefile.configured"
        if conf.exists():
            info["makefile_diff_vs_configure"] = common._run(
                ["diff", str(conf), str(mk)])
    cl = src / "PeTar" / "config.log"
    if cl.exists():
        info["configure_command"] = next((l.strip() for l in cl.read_text().splitlines()
                                          if l.strip().startswith("$ ./configure")), "")
    cu = src / "PeTar" / "src" / "force_gpu_cuda.cu"
    if cu.exists():
        txt = cu.read_text()
        info["gpu_kernel_precision_evidence"] = {
            "file": str(cu), "n_float": len(re.findall(r"\bfloat[34]?\b", txt)),
            "n_double": len(re.findall(r"\bdouble[34]?\b", txt)),
            "uses_rsqrtf": "rsqrtf" in txt,
        }
    return info


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", default=DEFAULT_N)
    ap.add_argument("--t-end", type=float, default=1.0, help="physical end time (Henon units)")
    ap.add_argument("--softening", type=float, default=2.0 ** -7)
    ap.add_argument("--seed", type=int, default=ic_mod.DEFAULT_SEED)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--wallclock-cap", type=float, default=3600.0, help="hard cap per run [s]")
    ap.add_argument("--omp-threads", type=int, default=16)
    ap.add_argument("--max-cpu-fraction", type=float, default=0.8,
                    help="run only if 1-min loadavg + omp threads <= this fraction of cores")
    ap.add_argument("--cpu-wait-timeout", type=float, default=3600.0)
    ap.add_argument("--petar-bin", default=str(DEFAULT_PETAR))
    ap.add_argument("--petar-src", default=str(DEFAULT_PETAR_SRC))
    ap.add_argument("--gpu-timeout", type=int, default=None)
    ap.add_argument("--exclude-gpu", type=int, nargs="*", default=None,
                    help="GPU indices autocvd must not select")
    ap.add_argument("--out-dir", default=str(common.RESULTS_DIR))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--keep-work", action="store_true", help="keep PeTar work directories")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        a.N = [min(a.N)]
        a.repeats = 1
        a.out_dir = str(Path(a.out_dir) / "dryrun")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = a.run_id or time.strftime("%Y%m%dT%H%M%S")
    csv_path = out_dir / "reference.csv"

    petar = shutil.which(a.petar_bin) or (a.petar_bin if Path(a.petar_bin).exists() else None)
    base_common = dict(run_id=run_id, code="petar", series="petar_gpu", precision="mixed",
                       scheme="P3T tree(GPU)+Hermite/SDAR", integrator="PeTar kdk tree + 4th-order Hermite + SDAR",
                       softening=a.softening, t_end=a.t_end, metric_basis="wallclock_to_fixed_time",
                       omp_threads=a.omp_threads, mpi_ranks=1)
    if petar is None:
        for N in a.N:
            append_ref_row(csv_path, dict(base_common, N=N, status="skipped",
                                          error=f"PeTar binary not found: {a.petar_bin}",
                                          timestamp_utc=common.utcnow()))
        print(f"[ref] PeTar not found at {a.petar_bin}; rows marked skipped")
        return 0

    help_text = subprocess.run([petar, "-h"], capture_output=True, text=True).stdout
    (out_dir / "petar_help.txt").write_text(help_text)

    ic_mod.ensure_ic(a.N, a.seed)
    prov = dict(run_id=run_id, created_utc=common.utcnow(), script="run_reference.py", argv=sys.argv,
                args=vars(a), petar_binary=str(Path(petar).resolve()), build=build_info(Path(a.petar_src)),
                host=common.host_info(), env=common.env_snapshot(), cuda=common.cuda_versions(),
                nvidia_smi_q=common.nvidia_smi_full(), runs=[])
    prov_path = out_dir / f"provenance_reference_{run_id}.json"
    common.write_json(prov_path, prov)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from autocvd import autocvd

    for N in a.N:
        ic = ic_mod.load_ic(N, a.seed)
        E0_ref = ic_mod.softened_E0(N, a.softening, a.seed)
        flags = ["-i", "1", "-u", "0", "-G", "1", "-t", repr(a.t_end), "-o", repr(a.t_end),
                 "--soft-eps", repr(a.softening), "-b", "0", "-w", "1", "-a", "0", "-f", "data"]
        missing = check_flags_in_help(flags, help_text)
        base = dict(base_common, N=N, E0=E0_ref, ic_file=str(ic["path"].relative_to(common.BENCH_DIR)),
                    ic_sha256=ic["sha256"], flags=" ".join(flags))
        if missing:
            append_ref_row(csv_path, dict(base, status="skipped", timestamp_utc=common.utcnow(),
                                          error=f"flags not in petar -h: {missing}"))
            continue

        for r in range(a.repeats):
            load0 = wait_for_cpu(a.omp_threads, a.max_cpu_fraction, a.cpu_wait_timeout)
            gpu_idx = autocvd(num_gpus=1, exclude=a.exclude_gpu, timeout=a.gpu_timeout,
                              set_env=False, progress=False)[0]
            gpu = common.nvidia_smi_query(gpu_idx)
            common.assert_a100(gpu)
            others = common.compute_apps_on(gpu["uuid"])
            work = out_dir / "petar_work" / f"{run_id}_N{N}_r{r}"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            write_petar_input(work / "input.dat", ic["pos"], ic["vel"], ic["mass"])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_idx), OMP_NUM_THREADS=str(a.omp_threads),
                       OMP_STACKSIZE="128M")
            row = dict(base, repeat=r, gpu_name=gpu["name"], gpu_mem_total_MiB=gpu["memory.total"],
                       gpu_index=gpu_idx, loadavg_before=load0)
            ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            t0 = time.perf_counter()
            status, err = "ok", ""
            try:
                with open(work / "petar.log", "w") as logf:
                    p = subprocess.run([petar, *flags, "input.dat"], cwd=work, env=env, stdout=logf,
                                       stderr=subprocess.STDOUT, timeout=a.wallclock_cap)
                if p.returncode != 0:
                    status, err = "error", f"petar exit code {p.returncode}"
            except subprocess.TimeoutExpired:
                status, err = "timeout", f"exceeded wall-clock cap {a.wallclock_cap} s"
            wall = time.perf_counter() - t0
            ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
            row.update(timestamp_utc=common.utcnow(), wall_s=wall, loadavg_after=os.getloadavg()[0],
                       cpu_cores_used=cpu_s / wall if wall > 0 else np.nan)
            if others:
                status, err = "shared_gpu", json.dumps(others)

            log = (work / "petar.log").read_text(errors="replace")
            st = parse_petar_log(log)
            row.update(petar_E0_reported=st["E0"], tree_steps=st["tree_steps"], h4_steps_total=st["h4"],
                       ar_steps_total=st["ar"], ep_ep_total=st["epep"], ep_sp_total=st["epsp"],
                       petar_internal_s=st["internal_s"], n_steps=st["tree_steps"],
                       petar_dE_E_reported=abs(st["dE_cum_last"] / st["E0"]) if st["E0"] == st["E0"] else np.nan)
            try:
                # what PeTar ingested == the shared IC file (to round-off)
                t_0, m0, x0, v0, ids0 = read_petar_snapshot(work / "data.0")
                assert t_0 == 0.0 and len(ids0) == N and np.array_equal(ids0, np.arange(1, N + 1))
                ref = np.concatenate([ic["mass"][:, None], ic["pos"], ic["vel"]], 1)
                got = np.concatenate([m0[:, None], x0, v0], 1)
                rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))
                row["ic_ingest_max_rel_diff"] = rel
                if rel > 1e-14:
                    raise AssertionError(f"PeTar time-0 snapshot differs from IC file: max rel diff {rel:.3e}")
                E0_snap = common.total_energy(x0, v0, m0, a.softening)
                row["E0_snapshot"] = E0_snap
                if status == "ok":
                    snaps = sorted(work.glob("data.[0-9]*"), key=lambda p: int(p.suffix[1:]))
                    t_1, m1, x1, v1, ids1 = read_petar_snapshot(snaps[-1])
                    if abs(t_1 - a.t_end) > 1e-12 * max(1.0, a.t_end) or len(ids1) != N:
                        raise AssertionError(f"final snapshot t={t_1}, N={len(ids1)}")
                    E1 = common.total_energy(x1, v1, m1, a.softening)
                    row.update(E_final=E1, dE_E_final=abs((E1 - E0_snap) / E0_snap))
            except Exception as e:  # noqa: BLE001
                status, err = ("error", f"{type(e).__name__}: {e}") if status == "ok" else (status, err)
            row.update(status=status, error=err)
            append_ref_row(csv_path, row)
            print(f"[ref] N={N:6d} r={r} status={status} wall={wall:9.2f}s GPU={gpu_idx} "
                  f"cores_used={row['cpu_cores_used']:.1f} load {load0:.1f}->{row['loadavg_after']:.1f} "
                  f"tree_steps={st['tree_steps']} |dE/E|={row.get('dE_E_final', float('nan'))} "
                  f"(petar reported {row['petar_dE_E_reported']:.2e}) {err}", flush=True)

            prov["runs"].append(dict(N=N, repeat=r, gpu=gpu, work_dir=str(work), status=status))
            shutil.copy(work / "petar.log", out_dir / "petar_work" / f"{run_id}_N{N}_r{r}.log")
            if (work / "input.par").exists():
                shutil.copy(work / "input.par", out_dir / "petar_work" / f"{run_id}_N{N}_r{r}.input.par")
            if not a.keep_work:
                shutil.rmtree(work)
            common.write_json(prov_path, prov)
            if status == "timeout":
                break  # further repeats at this N would also time out
    prov["finished_utc"] = common.utcnow()
    common.write_json(prov_path, prov)
    print(f"[ref] rows -> {csv_path}\n[ref] provenance -> {prov_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
