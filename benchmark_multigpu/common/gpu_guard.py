"""Pick idle GPUs and prove they *stayed* idle while a number was being taken.

Every timing in the jaccpot-vs-pkdgrav3 comparison is a single force evaluation
of 30-300 ms on a shared 8xA100 box.  The first round of that comparison found a
1.62x run-to-run spread at identical configuration; the follow-up (memory
``contended-timings-poison-ratios``) traced ~60 % of a "missing overhead" to
another user's job on the same card.  A timing taken on a card that someone
else is using is not a measurement, and no amount of min-of-N repairs it.

Rules this module enforces (plan T3.1, and the standing instruction that timings
come only from unoccupied GPUs):

* A GPU is *idle* only if it shows **0 % utilisation in every settle sample AND
  hosts no compute process owned by another user**.  Memory held by a foreign
  idle process still disqualifies the card -- that job can wake up mid-run.
* Selection goes through ``autocvd`` (the site tool) with the non-idle cards
  passed as ``--exclude``, and its answer is re-verified against the idle set.
* While a timing runs, a background sampler records utilisation and the
  process list of every GPU plus the 1-minute load average.  The summary says
  whether a foreign process ever touched the device(s) in use and how busy the
  rest of the machine was, so a row can be *flagged* rather than trusted.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    ).stdout


def _uuid_to_index() -> dict[str, int]:
    out = {}
    for line in _run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    ).strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            out[parts[1]] = int(parts[0])
    return out


def _pid_owner(pid: int) -> str:
    try:
        return _run(["ps", "-o", "user=", "-p", str(pid)]).strip() or "?"
    except Exception:
        return "?"


@dataclass
class GpuSnapshot:
    t: float
    loadavg1: float
    util: dict[int, int]  # physical index -> utilization.gpu %
    mem_used_mib: dict[int, int]
    procs: dict[int, list[tuple[int, str, int]]]  # index -> [(pid, user, mib)]


def snapshot(owner_cache: dict[int, str] | None = None) -> GpuSnapshot:
    """One nvidia-smi sample of every GPU on the box (physical indices)."""
    owner_cache = {} if owner_cache is None else owner_cache
    util: dict[int, int] = {}
    mem: dict[int, int] = {}
    for line in _run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ).strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx = int(parts[0])
        util[idx] = int(parts[1]) if parts[1].isdigit() else -1
        mem[idx] = int(parts[2]) if parts[2].isdigit() else -1
    u2i = _uuid_to_index()
    procs: dict[int, list[tuple[int, str, int]]] = {i: [] for i in util}
    for line in _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    ).strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        pid = int(parts[0])
        mib = int(parts[1]) if parts[1].isdigit() else -1
        idx = u2i.get(parts[2])
        if idx is None:
            continue
        if pid not in owner_cache:
            owner_cache[pid] = _pid_owner(pid)
        procs.setdefault(idx, []).append((pid, owner_cache[pid], mib))
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    return GpuSnapshot(time.time(), load1, util, mem, procs)


def idle_gpus(
    *, settle_s: float = 2.0, samples: int = 4, exclude: tuple[int, ...] = ()
) -> tuple[list[int], dict[int, str]]:
    """GPUs with 0 % utilisation in every sample and no foreign compute process.

    Returns ``(idle, reasons)`` where ``reasons`` explains every excluded card.
    """
    me = getpass.getuser()
    owner_cache: dict[int, str] = {}
    snaps = []
    for k in range(samples):
        snaps.append(snapshot(owner_cache))
        if k + 1 < samples:
            time.sleep(settle_s / max(1, samples - 1))
    all_idx = sorted(snaps[0].util)
    idle, reasons = [], {}
    for i in all_idx:
        if i in exclude:
            reasons[i] = "excluded by caller"
            continue
        max_util = max(s.util.get(i, -1) for s in snaps)
        # ANY compute process disqualifies the card -- a foreign job because it
        # can wake up mid-run, one of our own because two of our benchmark
        # processes must never share a device (the autocvd double-booking trap).
        procs = sorted(
            {(pid, user) for s in snaps for (pid, user, _m) in s.procs.get(i, [])
             if pid != os.getpid()}
        )
        if procs:
            kind = "foreign" if any(u != me for _p, u in procs) else "own"
            reasons[i] = f"{kind} process " + ", ".join(f"{p}({u})" for p, u in procs)
        elif max_util != 0:
            reasons[i] = f"utilisation {max_util} %"
        else:
            idle.append(i)
    return idle, reasons


def _autocvd_bin() -> str | None:
    for cand in (
        shutil.which("autocvd"),
        os.path.join(os.path.dirname(os.sys.executable), "autocvd"),
        "/export/home/tbuck/jaccpot/.venv/bin/autocvd",
        "/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd",
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def pick_idle_gpus(
    n: int,
    *,
    exclude: tuple[int, ...] = (),
    settle_s: float = 2.0,
    samples: int = 4,
    timeout_s: float = 30.0,
) -> list[int]:
    """Choose ``n`` idle GPUs via ``autocvd`` restricted to the verified-idle set.

    ``BENCH_GPU_EXCLUDE=3,4`` in the environment adds cards to ``exclude``.

    Raises ``RuntimeError`` (never silently degrades to a busy card) when fewer
    than ``n`` cards are idle or autocvd returns something outside the idle set.
    """
    env_excl = tuple(
        int(x) for x in os.environ.get("BENCH_GPU_EXCLUDE", "").split(",") if x.strip()
    )
    idle, reasons = idle_gpus(
        settle_s=settle_s, samples=samples, exclude=tuple(exclude) + env_excl
    )
    if len(idle) < n:
        detail = "; ".join(f"GPU {i}: {r}" for i, r in sorted(reasons.items()))
        raise RuntimeError(
            f"only {len(idle)} idle GPU(s) {idle}, need {n}. Non-idle: {detail}"
        )
    not_idle = [i for i in reasons]  # everything that is not idle, incl. excluded
    autocvd = _autocvd_bin()
    chosen: list[int]
    if autocvd is not None:
        cmd = [autocvd, "-n", str(n), "-o", "-q", "-t", str(int(timeout_s))]
        if not_idle:
            cmd += ["-x", *map(str, not_idle)]
        out = _run(cmd, timeout=timeout_s + 10).strip()
        try:
            chosen = [int(x) for x in out.split(",") if x.strip() != ""]
        except ValueError:
            chosen = []
        if len(chosen) != n or any(c not in idle for c in chosen):
            raise RuntimeError(
                f"autocvd returned {out!r}, not {n} device(s) from the idle set {idle}"
            )
    else:
        chosen = idle[:n]
    return chosen


def set_cuda_visible(devices: list[int]) -> str:
    """Set ``CUDA_VISIBLE_DEVICES``; must run before JAX / the pkdgrav3 child starts."""
    val = ",".join(str(d) for d in devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = val
    return val


@dataclass
class ContentionSummary:
    devices: list[int]
    n_samples: int
    duration_s: float
    foreign_pids_on_devices: list[tuple[int, int, str]]  # (gpu, pid, user)
    own_util_max: dict[int, int]
    other_gpu_util_max: int
    loadavg1_max: float
    loadavg1_min: float
    cores: int = field(default_factory=lambda: os.cpu_count() or 0)

    @property
    def contaminated(self) -> bool:
        return bool(self.foreign_pids_on_devices)

    @property
    def flags(self) -> list[str]:
        f = []
        if self.foreign_pids_on_devices:
            f.append("foreign-process-on-device")
        if self.loadavg1_max >= 8.0:
            f.append(f"loadavg>=8 ({self.loadavg1_max:.1f})")
        if self.other_gpu_util_max > 0:
            f.append(f"other-gpus-busy ({self.other_gpu_util_max}%)")
        return f

    def as_dict(self) -> dict:
        return dict(
            devices=self.devices,
            n_samples=self.n_samples,
            duration_s=self.duration_s,
            foreign_pids_on_devices=[list(x) for x in self.foreign_pids_on_devices],
            own_util_max=self.own_util_max,
            other_gpu_util_max=self.other_gpu_util_max,
            loadavg1_max=self.loadavg1_max,
            loadavg1_min=self.loadavg1_min,
            cores=self.cores,
            contaminated=self.contaminated,
            flags=self.flags,
        )


class GpuMonitor:
    """Background sampler; use as a context manager around the timed region.

    ``devices`` are *physical* indices (what ``CUDA_VISIBLE_DEVICES`` was set to).
    The summary distinguishes foreign processes on those devices (which void the
    timing) from load elsewhere on the box (which is recorded and flagged).
    """

    def __init__(self, devices: list[int], interval_s: float = 0.5):
        self.devices = list(devices)
        self.interval_s = interval_s
        self._snaps: list[GpuSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._owner_cache: dict[int, str] = {}
        self._t0 = 0.0
        self._t1 = 0.0

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._snaps.append(snapshot(self._owner_cache))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._t0 = time.time()
        self._snaps = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4 + 25)
        # one final sample so even a sub-interval region has at least one
        try:
            self._snaps.append(snapshot(self._owner_cache))
        except Exception:
            pass
        self._t1 = time.time()
        return False

    def summary(self) -> ContentionSummary:
        me = getpass.getuser()
        foreign = set()
        own_util = {d: 0 for d in self.devices}
        other_util = 0
        loads = [s.loadavg1 for s in self._snaps if s.loadavg1 == s.loadavg1]
        for s in self._snaps:
            for d in self.devices:
                own_util[d] = max(own_util[d], s.util.get(d, 0))
                for pid, user, _m in s.procs.get(d, []):
                    if user != me:
                        foreign.add((d, pid, user))
            for d, u in s.util.items():
                if d not in self.devices:
                    other_util = max(other_util, u)
        return ContentionSummary(
            devices=self.devices,
            n_samples=len(self._snaps),
            duration_s=self._t1 - self._t0,
            foreign_pids_on_devices=sorted(foreign),
            own_util_max=own_util,
            other_gpu_util_max=other_util,
            loadavg1_max=max(loads) if loads else float("nan"),
            loadavg1_min=min(loads) if loads else float("nan"),
        )


def timed_calls(fn, *, repeats: int, warmup: int, devices: list[int], block=None):
    """min / median / IQR of ``repeats`` calls after ``warmup``, under a GpuMonitor.

    ``block`` is applied to the result before the clock stops (``jax.block_until_ready``
    for JAX callers).  Returns ``(last_output, timing_dict, ContentionSummary)``.
    """
    import statistics

    block = block or (lambda x: x)
    with GpuMonitor(devices) as mon:
        for _ in range(warmup):
            block(fn())
        samples = []
        out = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = block(fn())
            samples.append(time.perf_counter() - t0)
    s = sorted(samples)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0], s[len(s) // 2], s[-1]]
    timing = dict(
        min=float(s[0]),
        median=float(statistics.median(s)),
        iqr=float(q[2] - q[0]),
        max=float(s[-1]),
        mean=float(statistics.fmean(s)),
        std=float(statistics.pstdev(s)),
        samples=[float(x) for x in samples],
    )
    return out, timing, mon.summary()
