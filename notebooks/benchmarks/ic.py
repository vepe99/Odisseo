"""Initial conditions shared by every code in the benchmark.

A virialised Plummer sphere in Henon units (G = M = 1, E_tot = -1/4, computed
without softening) is generated ONCE per (N, seed) with odisseo's
``Plummer_sphere`` and written to ``data/ic/plummer_N{N}_seed{seed}.npz``.
Every runner reads that file; nothing regenerates it.

Generation steps:
 1. ``odisseo.initial_condition.Plummer_sphere`` in float64 on the CPU, with
    Plummer scale radius a = 3*pi/16 (the Henon-unit value for M = G = 1).
 2. Shift to the centre-of-mass frame (position and velocity).
 3. Rescale positions and velocities so that, without softening, W = -1/2 and
    T = 1/4 exactly (virial ratio Q = 1/2, E = -1/4) for this finite sample.

Usage:  python ic.py --N 1024 4096 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IC_DIR, kinetic_energy, potential_energy, sha256_arrays, utcnow  # noqa: E402

DEFAULT_SEED = 42
PLUMMER_A_HENON = 3.0 * np.pi / 16.0


def ic_path(N: int, seed: int = DEFAULT_SEED) -> Path:
    return IC_DIR / f"plummer_N{N}_seed{seed}.npz"


def generate(N: int, seed: int = DEFAULT_SEED) -> Path:
    """Generate and write the IC file. Must run in a process where JAX has not
    been initialised on the GPU (it forces the CPU backend in float64)."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    from odisseo.initial_condition import Plummer_sphere
    from odisseo.option_classes import PlummerParams, SimulationConfig, SimulationParams

    config = SimulationConfig(N_particles=N)
    params = SimulationParams(G=1.0, Plummer_params=PlummerParams(a=PLUMMER_A_HENON, Mtot=1.0))
    pos, vel, mass = Plummer_sphere(jax.random.PRNGKey(seed), config, params)
    pos, vel, mass = (np.asarray(a, dtype=np.float64) for a in (pos, vel, mass))
    assert pos.dtype == np.float64 and mass.shape == (N,)

    # Centre-of-mass frame.
    M = mass.sum()
    pos = pos - (mass[:, None] * pos).sum(0) / M
    vel = vel - (mass[:, None] * vel).sum(0) / M

    # Exact Henon normalisation of this sample (unsoftened energies).
    T = kinetic_energy(vel, mass)
    W = potential_energy(pos, mass, eps=0.0)
    pos = pos * (W / -0.5)
    vel = vel * np.sqrt(0.25 / T)
    T1 = kinetic_energy(vel, mass)
    W1 = potential_energy(pos, mass, eps=0.0)

    meta = dict(
        N=N, seed=seed, generator="odisseo.initial_condition.Plummer_sphere",
        plummer_a_before_rescale=PLUMMER_A_HENON, units="Henon (G=M=1, E=-1/4 unsoftened)",
        T_before=T, W_before=W, T=T1, W=W1, E=T1 + W1, Q=T1 / -W1,
        created_utc=utcnow(), jax_version=jax.__version__,
        sha256=sha256_arrays(pos, vel, mass),
    )
    out = ic_path(N, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pos=pos, vel=vel, mass=mass, meta=json.dumps(meta))
    print(f"[ic] wrote {out}  E={meta['E']:.15f}  Q={meta['Q']:.15f}")
    return out


def load_ic(N: int, seed: int = DEFAULT_SEED) -> dict:
    """Read an IC file (numpy only, no JAX) and verify its checksum."""
    p = ic_path(N, seed)
    with np.load(p) as f:
        pos, vel, mass = f["pos"], f["vel"], f["mass"]
        meta = json.loads(str(f["meta"]))
    sha = sha256_arrays(pos, vel, mass)
    if sha != meta["sha256"]:
        raise RuntimeError(f"IC checksum mismatch for {p}")
    return dict(pos=pos, vel=vel, mass=mass, meta=meta, path=p, sha256=sha)


def ensure_ic(Ns, seed: int = DEFAULT_SEED) -> None:
    """Generate missing IC files in a separate CPU-only process."""
    missing = [N for N in Ns if not ic_path(N, seed).exists()]
    if missing:
        env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        subprocess.run([sys.executable, __file__, "--seed", str(seed), "--N",
                        *map(str, missing)], check=True, env=env)


def softened_E0(N: int, eps: float, seed: int = DEFAULT_SEED) -> float:
    """Softened initial energy (float64), cached next to the IC file."""
    cache = ic_path(N, seed).with_suffix(".E0.json")
    data = json.loads(cache.read_text()) if cache.exists() else {}
    key = repr(float(eps))
    if key not in data:
        ic = load_ic(N, seed)
        data[key] = kinetic_energy(ic["vel"], ic["mass"]) + potential_energy(ic["pos"], ic["mass"], eps)
        cache.write_text(json.dumps(data, indent=1))
    return data[key]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()
    for n in a.N:
        if ic_path(n, a.seed).exists() and not a.force:
            print(f"[ic] exists: {ic_path(n, a.seed)}")
            continue
        generate(n, a.seed)
