"""pkdgrav3 side of the code-vs-code comparison: IC bridge + subprocess runner.

pkdgrav3 (Potter, Stadel & Teyssier 2017) is the natural comparator for jaccpot:
it is itself an **FMM** -- ``gravity/moments.h`` carries 4th-order reduced
multipoles (``MOMR``) and 5th-order local expansions (``LOCR``) -- rather than a
Barnes-Hut treecode like Bonsai, and unlike Bonsai it drives **every visible GPU
from a single process** (``mdl2/cuda/mdlcuda.cu``: ``CUDA::initialize`` builds one
``Device`` per ``cudaGetDeviceCount`` entry and ``CUDA::launch`` dispatches each
work packet to the least-busy one).  So the device count is selected purely by
``CUDA_VISIBLE_DEVICES`` -- the same knob the jaccpot harness uses -- and a
single-node multi-GPU run needs exactly one MPI rank.

Two traps this module exists to keep us out of:

1. **The softening kernels differ.**  jaccpot is Plummer (``r^2 + eps^2``);
   pkdgrav3 is a compact-support spline (``gravity/pp.h::EvalPP`` applies a
   polynomial correction inside ``2h`` and is exactly Newtonian outside).  At
   matched ``eps`` the two codes therefore compute *different physics*, and a
   force-error comparison would be measuring the softening, not the algorithm.
   The accuracy plane must run at ``eps = 0`` on a smooth IC; ``write_tipsy_dark``
   defaults to that and refuses to silently write a nonzero global softening.

2. **The Tipsy record layout is not Bonsai's.**  The existing ODISSEO->Bonsai
   writer emits *star* records ending in a 64-bit id; pkdgrav3's ``tipsyDark``
   (``io/fio.c``) is ``mass, pos[3], vel[3], eps, phi`` = 36 bytes, and the last
   header word is ``nPad`` (which must be 0), not Bonsai's ``version``.
   pkdgrav3 auto-detects native (little-endian) vs standard (XDR big-endian)
   from ``nDim in [1,3]`` (``io/fio.c::tipsyDetectHeader``), so we write native.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PKDGRAV3_ROOT = Path(os.environ.get("PKDGRAV3_ROOT", "/export/home/tbuck/pkdgrav3"))
PKDGRAV3_BIN = PKDGRAV3_ROOT / "build" / "pkdgrav3"
PKDGRAV3_ENV_SH = PKDGRAV3_ROOT / "env_pkdgrav3.sh"

# tipsyHdr: double dTime; unsigned nBodies, nDim, nSph, nDark, nStar, nPad -> 32 B
_TIPSY_HDR = "<dIIIIII"
_TIPSY_HDR_SIZE = 32
# tipsyDark: float mass, pos[3], vel[3], eps, phi -> 36 B
_TIPSY_DARK_FLOATS = 9
_TIPSY_DARK_SIZE = 36


def _dark_dtype() -> np.dtype:
    dt = np.dtype(
        [
            ("mass", "<f4"),
            ("pos", "<f4", (3,)),
            ("vel", "<f4", (3,)),
            ("eps", "<f4"),
            ("phi", "<f4"),
        ]
    )
    assert dt.itemsize == _TIPSY_DARK_SIZE, dt.itemsize
    return dt


def write_tipsy_dark(
    path: str | Path,
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    velocities: np.ndarray | None = None,
    softening: float = 0.0,
    time_: float = 0.0,
    allow_softening: bool = False,
) -> Path:
    """Write ``(positions, masses)`` as a native-endian Tipsy file of dark particles.

    ``softening`` is written into each particle's ``eps`` field.  It defaults to
    zero and a nonzero value raises unless ``allow_softening=True``: pkdgrav3's
    spline softening and jaccpot's Plummer softening are different kernels, so a
    shared nonzero ``eps`` produces different forces in the two codes and
    silently invalidates any accuracy comparison.  pkdgrav3 is safe at
    ``eps = 0`` (``EvalPP`` clamps with ``minSoftening = 1e-18f``).
    """
    if softening != 0.0 and not allow_softening:
        raise ValueError(
            "refusing to write a nonzero softening: pkdgrav3 uses a compact-support "
            "spline kernel and jaccpot uses Plummer, so matched eps != matched forces. "
            "Run the accuracy plane at softening=0.0, or pass allow_softening=True if "
            "you are deliberately doing a timing-only run."
        )
    import struct

    pos = np.ascontiguousarray(positions, dtype=np.float32)
    mass = np.ascontiguousarray(masses, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {pos.shape}")
    n = pos.shape[0]
    if mass.shape != (n,):
        raise ValueError(f"masses must be (N,), got {mass.shape} for N={n}")
    vel = (
        np.zeros((n, 3), np.float32)
        if velocities is None
        else np.ascontiguousarray(velocities, dtype=np.float32)
    )

    arr = np.zeros(n, dtype=_dark_dtype())
    arr["mass"] = mass
    arr["pos"] = pos
    arr["vel"] = vel
    arr["eps"] = np.float32(softening)
    arr["phi"] = 0.0

    path = Path(path)
    with open(path, "wb") as fh:
        # nDim=3 in [1,3] is what makes pkdgrav3 read this as native, not XDR.
        fh.write(struct.pack(_TIPSY_HDR, float(time_), n, 3, 0, n, 0, 0))
        fh.write(arr.tobytes())
    return path


def read_tipsy_dark(path: str | Path):
    """Read back a native Tipsy dark-particle file (round-trip validation)."""
    import struct

    data = Path(path).read_bytes()
    time_, nbodies, ndim, nsph, ndark, nstar, npad = struct.unpack_from(
        _TIPSY_HDR, data, 0
    )
    if not (1 <= ndim <= 3):
        raise ValueError(f"not a native tipsy file (nDim={ndim}); XDR not supported here")
    if nsph or nstar or ndark != nbodies:
        raise ValueError(f"expected dark-only, got nSph={nsph} nDark={ndark} nStar={nstar}")
    expect = _TIPSY_HDR_SIZE + nbodies * _TIPSY_DARK_SIZE
    if len(data) != expect:
        raise ValueError(f"file is {len(data)} B, expected {expect} B for N={nbodies}")
    arr = np.frombuffer(data, dtype=_dark_dtype(), count=nbodies, offset=_TIPSY_HDR_SIZE)
    hdr = dict(time=time_, nbodies=nbodies, ndim=ndim, ndark=ndark, npad=npad)
    return arr["pos"].copy(), arr["mass"].copy(), arr["vel"].copy(), arr["eps"].copy(), hdr


# --------------------------------------------------------------------------
# running the binary
# --------------------------------------------------------------------------

# "Gravity Calculated, Wallclock: 0.12345 secs, Gflops:123.4, Total Gflop:1.23e+03"
_GRAVITY_LINE = re.compile(
    r"Gravity Calculated, Wallclock:\s*([0-9.eE+-]+)\s*secs"
    r"(?:.*?Gflops:\s*([0-9.eE+-]+|unknown))?"
    r"(?:.*?Total Gflop:\s*([0-9.eE+-]+))?"
)


# "  P-P per active: max=  482.86 @   13 avg=  470.72 of    15 std-dev=    8.08"
_STAT_LINE = re.compile(
    r"^\s*(P-P per active|P-C per active|actives   load|particle  load):"
    r"\s*max=\s*([0-9.eE+-]+)\s*@\s*\d+\s*avg=\s*([0-9.eE+-]+)\s*of\s*(\d+)"
    r"\s*std-dev=\s*([0-9.eE+-]+)",
    re.MULTILINE,
)
_STAT_KEYS = {
    "P-P per active": "pp_per_active",
    "P-C per active": "pc_per_active",
    "actives   load": "actives_per_thread",
    "particle  load": "particles_per_thread",
}


def parse_gravity_lines(stdout: str) -> list[dict]:
    """Extract pkdgrav3's own per-call gravity report (master.cxx ``Gravity Calculated``).

    Each report also carries the interaction-list statistics printed right after
    it: ``P-P per active`` (particles on the P-P list per active sink particle,
    ``walk2.cxx:568`` ``pdPartSum += nActive * ilp.count()``) and ``P-C per
    active`` (cells on the P-C list).  Those two are pkdgrav3's *interaction
    budget* -- the implementation-independent MAC-efficiency measure the
    comparison puts beside wall time (plan T2.0).  Stored as ``{avg, max,
    std, threads}`` per statistic, attached to the preceding gravity report.
    """
    out = []
    starts = [m.start() for m in _GRAVITY_LINE.finditer(stdout)] + [len(stdout)]
    for i, m in enumerate(_GRAVITY_LINE.finditer(stdout)):
        secs, gflops, total = m.groups()
        block = stdout[m.end() : starts[i + 1]]
        stats = {}
        for sm in _STAT_LINE.finditer(block):
            label, mx, avg, nthreads, sd = sm.groups()
            stats[_STAT_KEYS[label]] = dict(
                avg=float(avg), max=float(mx), std=float(sd), threads=int(nthreads)
            )
        out.append(
            dict(
                wallclock_s=float(secs),
                gflops=None if gflops in (None, "unknown") else float(gflops),
                total_gflop=None if total is None else float(total),
                **stats,
            )
        )
    return out


@dataclass
class GpuSample:
    t: float
    used_mib: dict[int, int]


@dataclass
class PkdRun:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    gravity_reports: list[dict] = field(default_factory=list)
    peak_gpu_mib: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(
            returncode=self.returncode,
            wall_s=self.wall_s,
            gravity_reports=self.gravity_reports,
            peak_gpu_mib=self.peak_gpu_mib,
        )


def _visible_devices() -> list[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [int(x) for x in cvd.split(",") if x.strip() != ""]


def _sample_gpu_memory(pid: int, devices: list[int]) -> dict[int, int]:
    """MiB in use on ``devices`` by ``pid`` and its children, via nvidia-smi.

    Returns physical device indices (nvidia-smi is not affected by
    CUDA_VISIBLE_DEVICES remapping), so callers get absolute device ids.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except Exception:
        return {}
    # map uuid -> index once per call (cheap enough at ~1 Hz)
    try:
        idx_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except Exception:
        return {}
    uuid_to_index = {}
    for line in idx_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid_to_index[parts[1]] = int(parts[0])

    used: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        apid, mib, uuid = parts
        if int(apid) != pid:
            continue
        dev = uuid_to_index.get(uuid)
        if dev is None:
            continue
        used[dev] = max(used.get(dev, 0), int(mib))
    return used


def run_pkdgrav3(
    script: str | Path,
    *,
    env_extra: dict[str, str] | None = None,
    cores: int | None = None,
    binary: Path = PKDGRAV3_BIN,
    deps_prefix: str = "/export/home/tbuck/micromamba/envs/pkdgrav3-deps",
    timeout_s: float = 3600.0,
    sample_gpu_memory: bool = True,
    sample_interval_s: float = 0.25,
) -> PkdRun:
    """Run ``binary script`` (pkdgrav3 analysis mode) and capture output + GPU peak.

    ``CUDA_VISIBLE_DEVICES`` is inherited from the caller's environment -- that is
    the *only* device-count control pkdgrav3 needs, since one process drives every
    visible GPU.  ``cores`` maps to mdl's ``-sz`` (``mdl2/mpi/mdl.cxx``); leave it
    ``None`` to let mdl pick, but record whatever you used, because pkdgrav3 uses
    the host CPU heavily and jaccpot does not.
    """
    binary = Path(binary)
    if not binary.exists():
        raise FileNotFoundError(
            f"pkdgrav3 binary not found at {binary}; build it first (see {PKDGRAV3_ENV_SH})"
        )
    env = dict(os.environ)
    # boost/fftw/mpich (and the CUDA runtime it was built against) live in the env
    env["LD_LIBRARY_PATH"] = f"{deps_prefix}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["MPICH_CC"] = "gcc"
    env["MPICH_CXX"] = "g++"
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    cmd = [str(binary)]
    if cores is not None:
        cmd += ["-sz", str(cores)]
    cmd += [str(script)]

    devices = _visible_devices()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )

    peak: dict[int, int] = {}
    if sample_gpu_memory and devices:
        # poll while the child runs; cheap and does not perturb the run
        while proc.poll() is None:
            for dev, mib in _sample_gpu_memory(proc.pid, devices).items():
                peak[dev] = max(peak.get(dev, 0), mib)
            time.sleep(sample_interval_s)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise
    wall = time.perf_counter() - t0

    return PkdRun(
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_s=wall,
        gravity_reports=parse_gravity_lines(stdout),
        peak_gpu_mib=peak,
    )
