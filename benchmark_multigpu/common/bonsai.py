"""Single-GPU Bonsai baseline: tipsy writer, runner, and log parser.

Bonsai is the *performance* competitor (a single-GPU Barnes-Hut treecode), NOT an
accuracy reference.  We run pure self-gravity (no external halo) for a fixed
number of steps and read the per-step wall time from Bonsai's ``--log`` output.

NOTE (untested pending the GPU-node reset): the tipsy writer and the exact CLI
were derived from ``benchmark_a100/run_a100_rerun.sh`` and a captured Bonsai log,
but have not been re-exercised since the node was wedged.  Validate
``write_tipsy`` round-trips through :func:`common.ic._read_tipsy_standard` and
that Bonsai reads it before trusting timings.

The MPI/multi-GPU Bonsai path is intentionally NOT used here: its multi-GPU
runtime hit CUDA error 700 and wedged the whole node (see the plan's incident
note).  Single-GPU only.
"""

from __future__ import annotations

import re
import struct
import subprocess
from pathlib import Path

import numpy as np

BONSAI_BIN = "/export/home/tbuck/Bonsai/runtime/build/bonsai2_slowdust"


def write_tipsy(path: str, pos: np.ndarray, mass: np.ndarray) -> None:
    """Write positions/masses as a standard-format tipsy file (all dark matter).

    Header ``double time; int nbodies,ndim,nsph,ndark,nstar`` (+4 pad = 32 bytes);
    each dark particle = ``mass, pos[3], vel[3], eps, phi`` (9 float32).
    """
    pos = np.asarray(pos, np.float32)
    mass = np.asarray(mass, np.float32)
    n = pos.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<diiiii", 0.0, n, 3, 0, n, 0))
        f.write(struct.pack("<i", 0))  # pad to 32-byte header
        buf = np.zeros((n, 9), np.float32)
        buf[:, 0] = mass
        buf[:, 1:4] = pos
        # vel[3]=0, eps (col 7)=softening placeholder 0, phi (col 8)=0
        f.write(buf.tobytes())


def run_bonsai_single_gpu(
    tipsy_path: str,
    *,
    binary: str = BONSAI_BIN,
    dt: float = 0.0005,
    t_end: float = 0.05,
    theta: float = 0.5,
    softening: float = 0.02,
    gpu: str = "0",
    log_path: str = "bonsai.log",
    timeout_s: int = 1800,
) -> dict:
    """Run Bonsai self-gravity on ``tipsy_path`` for ``t_end/dt`` steps, one GPU.

    ``gpu`` is the CUDA_VISIBLE_DEVICES value (pick a free device with autocvd).
    Returns the parsed timing dict (see :func:`parse_bonsai_log`).
    """
    import os

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        binary, "-i", str(tipsy_path), "--dev", "0",
        "-t", str(dt), "-T", str(t_end), "-e", str(softening),
        "-o", str(theta), "-r", "1", "--log",
    ]
    with open(log_path, "w") as lf:
        subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                       timeout=timeout_s, check=False,
                       cwd=str(Path(tipsy_path).parent))
    return parse_bonsai_log(log_path)


def parse_bonsai_log(log_path: str) -> dict:
    """Extract per-step timing from a Bonsai ``--log`` file.

    Reads ``Loop alone took: <s>``, the last ``iter=<N>``, and the
    ``TIME [00] TOTAL: .. Grav: .. Build: ..`` summary line.
    """
    text = Path(log_path).read_text(errors="ignore")
    loop = re.search(r"Loop alone took:\s*([\d.]+)", text)
    iters = re.findall(r"iter=(\d+)", text)
    summ = re.search(
        r"TIME \[00\] TOTAL:\s*([\d.]+).*?Grav:\s*([\d.]+).*?Build:\s*([\d.]+)", text
    )
    loop_s = float(loop.group(1)) if loop else float("nan")
    n_steps = int(iters[-1]) if iters else 0
    out = {
        "loop_s": loop_s,
        "n_steps": n_steps,
        "ms_per_step": (loop_s * 1e3 / n_steps) if n_steps else float("nan"),
    }
    if summ:
        total, grav, build = map(float, summ.groups())
        out.update(
            total_s=total, grav_s=grav, build_s=build,
            grav_ms_per_step=(grav * 1e3 / n_steps) if n_steps else float("nan"),
            build_ms_per_step=(build * 1e3 / n_steps) if n_steps else float("nan"),
        )
    return out
