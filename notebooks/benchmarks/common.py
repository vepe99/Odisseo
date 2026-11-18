"""Shared helpers for the Odisseo GPU benchmark.

Nothing in this module imports JAX: it is used both by the orchestrator (which must
not initialise CUDA) and by the per-precision workers.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
IC_DIR = BENCH_DIR / "data" / "ic"
RESULTS_DIR = BENCH_DIR / "results"

# Flop-count convention used for every code: 20 flops per pairwise force
# interaction, the convention of Nyland, Harris & Prins (GPU Gems 3, ch. 31).
# It counts the softened acceleration only (no potential, no jerk), so it is a
# convention, not a measured hardware flop count.
FLOPS_PER_INTERACTION = 20

CSV_COLUMNS = [
    "run_id", "timestamp_utc", "code", "series", "N", "precision", "scheme",
    "scheme_id", "batch_size", "integrator", "repeat", "status", "error",
    "n_steps", "dt", "t_end", "n_force_evals", "softening",
    "wall_s", "compile_s", "metric_basis", "interactions", "interactions_per_s",
    "flops_per_interaction", "gflops",
    "E0", "E_final", "dE_E_final",
    "ic_file", "ic_sha256", "gpu_name", "gpu_mem_total_MiB", "flags",
]


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def append_row(csv_path: Path, row: dict) -> None:
    """Append one row and flush it to disk immediately (crash-safe sweep)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row) - set(CSV_COLUMNS)
    if unknown:
        raise KeyError(f"unknown CSV columns: {sorted(unknown)}")
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Energies, always evaluated in float64 on the CPU with numpy, independently of #
# the precision of the integration.                                            #
# --------------------------------------------------------------------------- #
def kinetic_energy(vel: np.ndarray, mass: np.ndarray) -> float:
    vel = np.asarray(vel, np.float64)
    mass = np.asarray(mass, np.float64)
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def potential_energy(pos: np.ndarray, mass: np.ndarray, eps: float,
                     G: float = 1.0, threads: int | None = None) -> float:
    """W = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2), in float64.

    Pairs with r_ij == 0 exactly (the self pair) are skipped, which is the same
    convention as odisseo.dynamics.single_body_acc. Row blocks are processed in
    a thread pool (numpy releases the GIL in the elementwise kernels).
    """
    pos = np.ascontiguousarray(pos, np.float64)
    mass = np.ascontiguousarray(mass, np.float64)
    n = pos.shape[0]
    x, y, z = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    eps2 = float(eps) ** 2
    block = max(1, (1 << 21) // n)
    threads = threads or min(32, os.cpu_count() or 1)

    def work(i0: int) -> float:
        i1 = min(n, i0 + block)
        dx = x[i0:i1, None] - x[None, :]
        r2 = dx * dx
        dx = y[i0:i1, None] - y[None, :]
        r2 += dx * dx
        dx = z[i0:i1, None] - z[None, :]
        r2 += dx * dx
        zero = r2 == 0.0
        r2 += eps2
        np.sqrt(r2, out=r2)
        with np.errstate(divide="ignore"):
            np.divide(1.0, r2, out=r2)
        r2[zero] = 0.0
        return float(mass[i0:i1] @ (r2 @ mass))

    with ThreadPoolExecutor(threads) as ex:
        total = sum(ex.map(work, range(0, n, block)))
    return -0.5 * G * total


def total_energy(pos, vel, mass, eps, G=1.0) -> float:
    return kinetic_energy(vel, mass) + potential_energy(pos, mass, eps, G)


# --------------------------------------------------------------------------- #
# GPU / environment provenance                                                 #
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=60).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:  # pragma: no cover
        return f"<unavailable: {e}>"


def nvidia_smi_query(index: str | int) -> dict:
    """Query one GPU by its nvidia-smi index (PCI bus order)."""
    fields = ["index", "name", "uuid", "memory.total", "driver_version",
              "pci.bus_id", "clocks.max.sm", "power.limit"]
    out = _run(["nvidia-smi", f"--id={index}",
                f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    vals = [v.strip() for v in out.split(",")]
    if len(vals) != len(fields):
        raise RuntimeError(f"nvidia-smi query failed for GPU {index}: {out!r}")
    return dict(zip(fields, vals))


def compute_apps_on(uuid: str) -> list[dict]:
    out = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0] == uuid:
            apps.append(dict(pid=parts[1], process_name=parts[2], used_memory_MiB=parts[3]))
    return apps


def assert_a100(info: dict) -> str:
    """Fail loudly unless the GPU is an A100; return '40GB' or '80GB'."""
    name = info["name"]
    if "A100" not in name:
        raise SystemExit(f"FATAL: benchmark requires an NVIDIA A100, got {name!r}")
    mem = int(float(info["memory.total"]))
    if 39000 <= mem <= 41000:
        return "40GB"
    if 79000 <= mem <= 82000:
        return "80GB"
    raise SystemExit(f"FATAL: unrecognised A100 memory size {mem} MiB")


def cuda_versions() -> dict:
    smi = _run(["nvidia-smi"])
    cuda = ""
    for tok in smi.split("|"):
        if "CUDA Version" in tok:
            cuda = tok.split("CUDA Version:")[-1].strip()
    return {"driver_cuda_version": cuda}


def git_info(path: Path = REPO_ROOT) -> dict:
    return {
        "commit": _run(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(path), "status", "--porcelain", "--", "odisseo"])),
        "describe": _run(["git", "-C", str(path), "describe", "--always", "--dirty"]),
    }


def host_info() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_model": next((l.split(":", 1)[1].strip() for l in
                           Path("/proc/cpuinfo").read_text().splitlines()
                           if l.startswith("model name")), "") if Path("/proc/cpuinfo").exists() else "",
    }


def env_snapshot() -> dict:
    keep = ("XLA_", "JAX_", "CUDA_", "TF_", "NVIDIA_", "OMP_", "LD_LIBRARY_PATH")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(keep)}


def nvidia_smi_full() -> str:
    return _run(["nvidia-smi", "-q"])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
